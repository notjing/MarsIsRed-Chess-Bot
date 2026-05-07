import tensorflow as tf
from huggingface_hub import HfApi
import os

def get_dataset(files, batch_size):
    """ Loads the files into a stream """

    return (
        # files here is a tf.data.Dataset object with string file paths
        files
        # interleave opens multiple files and weaves the records together
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        # map parses each record
        .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        # essentially shuffles the entire thing (kind of)
        .shuffle(200_000)
        # stacks the positions into a multi-dim array grouping batch_size boards together
        .batch(batch_size, drop_remainder=True)
        # makes the dataset loop infinitely
        .repeat()
        # CPU prefetches the next batch
        .prefetch(tf.data.AUTOTUNE)
    )

def get_val_dataset(files, batch_size):
    """ Same as above but for validation dataset specifically"""

    # Why doesn't the validation dataset need shuffle and repeat?
    # The order doesn't really matter since it's just grading how well it responds to new data (shuffle)
    # Training is supposed to loop forever and use steps_per_epoch to shop it into epochs while validation is more of a one time thing

    return (
        files
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def parse_tfrecord(example):
    """ Takes a .tfrecord and breaks it down into the numbers, returning  """

    # creates an object describing what the .tfrecord looks like in that order
    feature_desc = {
        "board": tf.io.FixedLenFeature([8 * 8 * 25], tf.float32),
        "extra": tf.io.FixedLenFeature([19], tf.float32),
        "eval": tf.io.FixedLenFeature([1], tf.float32),
        "policy": tf.io.FixedLenFeature([8 * 8 * 73], tf.float32),
    }

    # parses the record
    ex = tf.io.parse_single_example(example, feature_desc)

    # reshapes the board back into ndarray
    board = tf.reshape(ex["board"], (8, 8, 25))

    return {"board_input": board, "extra_input": ex["extra"]}, {"prob_dist": ex["eval"], "move_dist": ex["policy"]}

def res_block(x, filters):

    """ Defines a block of layers """

    # saves a copy of x which is a ndarray representing the data
    shortcut = x

    # passes x through a bunch of layers
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # creates an add layer which acts as a shortcut
    # essentially adds the original value of x with the value that is determined by the conv2d's to get a new value
    # during backprop, taking the derivative will actually send it down both paths with the same strength, thus "skipping" the conv2d layers
    x = tf.keras.layers.Add()([shortcut, x])
    return tf.keras.layers.Activation("relu")(x)


def main():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # get all the files and create the datasets

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TFRECORD_DIR = os.path.join(BASE_DIR, "tfrecords")

    all_files = tf.data.Dataset.list_files(os.path.join(TFRECORD_DIR, "*.tfrecord"), shuffle=True)
    all_files = all_files.shuffle(buffer_size=100, seed=42)

    num_files = len(list(tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "*.tfrecord"))))
    num_test_files = max(1, int(0.1 * num_files))

    test_files = all_files.take(num_test_files)
    train_files = all_files.skip(num_test_files)

    train_ds = get_dataset(train_files, 256)
    test_ds = get_val_dataset(test_files, 256)

    cnn_input = tf.keras.Input(shape=(8, 8, 25), name="board_input")
    cnn_layers = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(cnn_input)

    # creates 12 layers of the blocks
    for _ in range(12):
        cnn_layers = res_block(cnn_layers, 256)

    shared_features = cnn_layers

    # which move
    p = tf.keras.layers.Conv2D(256, (1, 1), activation="relu")(shared_features)
    p = tf.keras.layers.Conv2D(73, (1, 1), activation="linear")(p)

    p = tf.keras.layers.Flatten()(p)
    p = tf.keras.layers.Softmax(dtype='float32', name='move_dist')(p)

    # winning probability
    v = tf.keras.layers.Conv2D(32, (1, 1), activation="relu")(shared_features)
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(256, activation='relu')(v)
    v = tf.keras.layers.Dropout(0.3)(v)

    dense_input = tf.keras.Input(shape=(19,), name="extra_input")

    norm_dense = tf.keras.layers.BatchNormalization()(dense_input)

    dense_layers = tf.keras.layers.Dense(128, activation='relu')(norm_dense)
    dense_layers = tf.keras.layers.Dense(64, activation='relu')(dense_layers)
    dense_layers = tf.keras.layers.Dense(32, activation='relu')(dense_layers)
    dense_layers = tf.keras.layers.Dropout(0.3)(dense_layers)

    combined = tf.keras.layers.Concatenate()([v, dense_layers])
    z = tf.keras.layers.Dense(128, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.Dense(64, activation='relu')(z)

    win_prob = tf.keras.layers.Dense(1, activation='tanh', name="prob_dist", dtype='float32')(z)

    aimodel = tf.keras.Model(inputs=[cnn_input, dense_input], outputs=[win_prob, p])

    batch_size = 256
    epochs = 30

    steps_per_epoch = 9_000_000 // batch_size
    total_steps = steps_per_epoch * epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0005,
        decay_steps=total_steps,
        alpha=0.00001
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        epsilon=1e-4,
        global_clipnorm=1.0
    )

    aimodel.compile(
        optimizer=optimizer,
        loss={
            "prob_dist": tf.keras.losses.MeanSquaredError(),
            "move_dist": tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        },
        loss_weights={
            "prob_dist": 3.0,
            "move_dist": 1.0
        },
        metrics={
            "prob_dist": ["mae"],
            "move_dist": ["accuracy"]
        }
    )

    aimodel.load_weights("checkpoints/model_epoch_10_val3.3051.keras")
    print("Successfully loaded checkpoint weights!")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/model_epoch_{epoch:02d}_val{val_loss:.4f}.keras',
            save_freq='epoch',
            save_best_only=False
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
    ]

    print("Starting training...")
    aimodel.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_steps=500,
        initial_epoch=10,
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
