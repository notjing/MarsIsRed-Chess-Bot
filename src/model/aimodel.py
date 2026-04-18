import tensorflow as tf
from huggingface_hub import HfApi
import os

def get_dataset(files, batch_size):
    return (
        files
        # allows for several tfrecords to be read at a time
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        # applies parse_tfrecord onto all files
        .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(200_000)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        # allows CPU to fetch more batches while GPU is calculating
        .prefetch(tf.data.AUTOTUNE)
    )

def get_val_dataset(files, batch_size):
    return (
        files
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def pawn_error(y_true, y_pred):
    """
    Converts the output into delta centipawns
    """
    return tf.reduce_mean(abs(y_true - y_pred)) * 1500


def parse_tfrecord(example):
    """
    Unwraps all the TFRecord files
    """

    # Provides the structure of the TFRecord data
    feature_desc = {
        "board": tf.io.FixedLenFeature([8 * 8 * 25], tf.float32),
        "extra": tf.io.FixedLenFeature([19], tf.float32),
        "eval": tf.io.FixedLenFeature([1], tf.float32),
    }

    ex = tf.io.parse_single_example(example, feature_desc)

    # Adjusts eval
    clamped_eval = tf.clip_by_value(ex["eval"][0], -1500.0, 1500.0)
    evalv = clamped_eval / 1500.0

    board = tf.reshape(ex["board"], (8, 8, 25))
    return {"board_input": board, "extra_input": ex["extra"]}, evalv

def res_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([shortcut, x]) # The skip connection
    return tf.keras.layers.Activation("relu")(x)


def main():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TFRECORD_DIR = os.path.join(BASE_DIR, "tfrecords")

    all_files = tf.data.Dataset.list_files(os.path.join(TFRECORD_DIR, "*.tfrecord"), shuffle=True)
    all_files = all_files.shuffle(buffer_size=100, seed=42)

    num_files = len(list(tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "*.tfrecord"))))
    num_test_files = max(1, int(0.1 * num_files))

    test_files = all_files.take(num_test_files)
    train_files = all_files.skip(num_test_files)

    train_ds = get_dataset(train_files, 512)
    test_ds = get_val_dataset(test_files, 512)

    cnn_input = tf.keras.Input(shape=(8, 8, 25), name="board_input")
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(cnn_input)

    for _ in range(12):
        x = res_block(x, 256)

    x = tf.keras.layers.Conv2D(32, (1, 1), activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    dense_input = tf.keras.Input(shape=(19,), name="extra_input")
    y = tf.keras.layers.Dense(128, activation='relu')(dense_input)
    y = tf.keras.layers.Dense(64, activation='relu')(y)
    y = tf.keras.layers.Dense(32, activation='relu')(y)
    y = tf.keras.layers.Dropout(0.3)(y)

    combined = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(128, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.Dense(64, activation='relu')(z)
    output = tf.keras.layers.Dense(1, activation='linear')(z)

    aimodel = tf.keras.Model(inputs=[cnn_input, dense_input], outputs=output)

    batch_size = 512
    epochs = 50

    steps_per_epoch = 6_050_000 // batch_size
    total_steps = steps_per_epoch * epochs
    validation_steps = 600_000 // batch_size

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.002,
        decay_steps=total_steps,
        alpha=0.00001
    )

    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    aimodel.compile(optimizer=optimizer,
                     loss=pawn_error,
                     metrics=["mae"])

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/model_epoch_{epoch:02d}_val{val_loss:.4f}.keras',
            save_freq='epoch',
            save_best_only=False  # keep multiple checkpoints
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train the model
    print("Starting training...")
    aimodel.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,

    )

    print("Evaluating...")
    aimodel.evaluate(test_ds, steps=100, verbose=2)

    model_file = "chessai_model.keras"
    aimodel.save(model_file)

    print("Uploading to Hugging Face...")
    api = HfApi()

    api.upload_file(
        path_or_fileobj=model_file,
        path_in_repo="chessai_model.keras",
        repo_id="notjing/chessai",
        repo_type="model",
        commit_message="Upload trained chess AI model"
    )
    print("Model uploaded successfully!")


if __name__ == "__main__":
    main()
