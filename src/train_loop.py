import os
import glob
import concurrent.futures
import tensorflow as tf
import tf2onnx

# Import your worker function from your generation script
from self_play import generate_self_play_data


OUTPUT_DIR = "model/tfrecords/self_gen"
MODEL_DIR = "model/model_iteration"
NUM_WORKERS = 4

POSITIONS_PER_WORKER = 50_000
POSITIONS_PER_FILE = 50_000
MAX_BUFFER_FILES = 20
BATCH_SIZE = 128

def parse_tfrecords(example):
    feature_desc = {
        "board": tf.io.FixedLenFeature([8 * 8 * 25], tf.float32),
        "extra": tf.io.FixedLenFeature([19], tf.float32),
        "eval": tf.io.FixedLenFeature([1], tf.float32),
        "policy": tf.io.FixedLenFeature([8 * 8 * 73], tf.float32),
    }
    ex = tf.io.parse_example(example, feature_desc)
    board = tf.reshape(ex["board"], (8, 8, 25))
    return {"board_input": board, "extra_input": ex["extra"]}, {"prob_dist": ex["eval"], "move_dist": ex["policy"]}


def get_dataset(buffer_dir, batch_size, supervised_dir=None):
    self_files = tf.data.Dataset.list_files(
        os.path.join(buffer_dir, "*.tfrecord"),
        shuffle=True
    )

    ds_self = (
        self_files
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        .map(parse_tfrecords, num_parallel_calls=tf.data.AUTOTUNE)
    )

    if supervised_dir is not None:
        sup_files = tf.data.Dataset.list_files(
            os.path.join(supervised_dir, "*.tfrecord"),
            shuffle=True
        )

        sup_files = sup_files.take(4)

        ds_sup = (
            sup_files
            .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
            .map(parse_tfrecords, num_parallel_calls=tf.data.AUTOTUNE)
        )

        # mix datasets
        dataset = tf.data.Dataset.sample_from_datasets(
            [ds_self, ds_sup],
            weights=[0.8, 0.2]
        )
    else:
        dataset = ds_self

    return (
        dataset
        .shuffle(100_000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def prune_replay_buffer(buffer_dir, max_files):
    files = glob.glob(os.path.join(buffer_dir, "*.tfrecord"))
    files.sort(key=os.path.getmtime)

    if len(files) > max_files:
        files_to_delete = len(files) - max_files
        print(f"Replay buffer exceeded {max_files} files. Pruning {files_to_delete} old files...")
        for i in range(files_to_delete):
            try:
                os.remove(files[i])
                print(f"  [Deleted] {os.path.basename(files[i])}")
            except Exception as e:
                print(f"  [Error] Failed to delete {files[i]}: {e}")
    else:
        print(f"Replay buffer holds {len(files)}/{max_files} files. No pruning necessary.")


def train_current_model(iteration, buffer_dir):
    # Calculate file paths
    current_model_path = os.path.join(MODEL_DIR, f"V{iteration - 1}.keras")
    new_model_path = os.path.join(MODEL_DIR, f"V{iteration}.keras")

    print(f"Loading previous model: {current_model_path}")

    try:
        model = tf.keras.models.load_model(current_model_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load {current_model_path}. Did you run the supervised bootstrap? Error: {e}")
        return

    print("Building dataset pipeline from current replay buffer...")
    train_ds = get_dataset(buffer_dir, BATCH_SIZE, supervised_dir="model/tfrecords")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-4, global_clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss={
            "prob_dist": tf.keras.losses.MeanSquaredError(),
            "move_dist": tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            "prob_dist": 2.5,
            "move_dist": 1.0
        }
    )

    print(f"Training V{iteration} for 1 Epoch over the entire Replay Buffer...")

    model.fit(
        train_ds,
        epochs=1,
        verbose=1
    )

    print(f"Saving new generation model: {new_model_path}")
    model.save(new_model_path)

    convert_keras_to_onnx(iteration, MODEL_DIR)

    tf.keras.backend.clear_session()

def convert_keras_to_onnx(iteration, model_dir="model_iteration"):
    keras_path = os.path.join(model_dir, f"V{iteration}.keras")
    onnx_path = os.path.join(model_dir, f"V{iteration}.onnx")

    print(f"Loading Keras model V{iteration} for ONNX conversion...")
    model = tf.keras.models.load_model(keras_path)

    input_signature = [
        tf.TensorSpec((None, 8, 8, 25), tf.float32, name="board_input"),
        tf.TensorSpec((None, 19), tf.float32, name="extra_input")
    ]

    print(f"Converting V{iteration} to ONNX format...")
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=onnx_path
    )
    print(f"Success! Optimized ONNX model saved to {onnx_path}")


def main_orchestrator():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    iteration = 13
    global_batch_counter = 14

    while True:
        print(f"\n========================================================")
        print(f" STARTING ALPHA-ZERO PIPELINE: ITERATION {iteration}")
        print(f"========================================================")

        print("\n>>> STATE 1: GENERATING SELF-PLAY DATA")

        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for worker_id in range(NUM_WORKERS):
                safe_start_batch = global_batch_counter + (worker_id * 1000)
                future = executor.submit(
                    generate_self_play_data,
                    POSITIONS_PER_WORKER,
                    POSITIONS_PER_FILE,
                    OUTPUT_DIR,
                    safe_start_batch,
                    worker_id,
                    iteration - 1
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"CRITICAL: A worker crashed during generation: {e}")

        global_batch_counter += 1

        print("\n>>> STATE 2: PRUNING REPLAY BUFFER")
        prune_replay_buffer(OUTPUT_DIR, MAX_BUFFER_FILES)

        print("\n>>> STATE 3: TRAINING NEURAL NETWORK")
        train_current_model(iteration, OUTPUT_DIR)

        print(f">>> Iteration {iteration} complete. Loop restarting...\n")
        iteration += 1


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # start loop
    main_orchestrator()
