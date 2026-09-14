import os
import glob
import concurrent.futures
import multiprocessing as mp
import tensorflow as tf
import tf2onnx

# Import your worker function from your generation script
from self_play import generate_self_play_data

from arena import run_tournament

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

OUTPUT_DIR = "model/tfrecords/self_gen"
MODEL_DIR = "model/model_iteration"
NUM_WORKERS = 16

POSITIONS_PER_WORKER = 50_000
POSITIONS_PER_FILE = 50_000
MAX_BUFFER_FILES = 48
BATCH_SIZE = 256

EPOCHS = 1
LEARNING_RATE = 1e-5

SUPERVISED_WEIGHT = 0.10
CHAMPION = 23


def parse_tfrecords(example):
    feature_desc = {
        "board": tf.io.FixedLenFeature([8 * 8 * 25], tf.float32),
        "extra": tf.io.FixedLenFeature([19], tf.float32),
        "eval": tf.io.FixedLenFeature([3], tf.float32),
        "policy": tf.io.FixedLenFeature([8 * 8 * 73], tf.float32),
    }
    ex = tf.io.parse_example(example, feature_desc)
    board = tf.reshape(ex["board"], (8, 8, 25))

    return {"board_input": board, "extra_input": ex["extra"]}, {"prob_dist": ex["eval"], "move_dist": ex["policy"]}


def parse_supervised(example):
    """Elite PGN shards still store scalar eval {-1, 0, 1}."""
    feature_desc = {
        "board": tf.io.FixedLenFeature([8 * 8 * 25], tf.float32),
        "extra": tf.io.FixedLenFeature([19], tf.float32),
        "eval": tf.io.FixedLenFeature([1], tf.float32),
        "policy": tf.io.FixedLenFeature([8 * 8 * 73], tf.float32),
    }
    ex = tf.io.parse_example(example, feature_desc)
    board = tf.reshape(ex["board"], (8, 8, 25))
    v = ex["eval"]
    w = tf.maximum(v, 0.0)
    l = tf.maximum(-v, 0.0)
    d = 1.0 - w - l
    wdl = tf.concat([w, d, l], axis=-1)
    return {"board_input": board, "extra_input": ex["extra"]}, {"prob_dist": wdl, "move_dist": ex["policy"]}


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

        ds_sup = (
            sup_files
            .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
            .map(parse_supervised, num_parallel_calls=tf.data.AUTOTUNE)
        )

        # mix datasets
        dataset = tf.data.Dataset.sample_from_datasets(
            [ds_self, ds_sup],
            weights=[1 - SUPERVISED_WEIGHT, SUPERVISED_WEIGHT]
        )
    else:
        dataset = ds_self

    return (
        dataset
        .shuffle(100_000)
        .batch(batch_size, drop_remainder=True)
        .repeat()
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
    current_model_path = os.path.join(MODEL_DIR, f"V{CHAMPION}.keras")
    new_model_path = os.path.join(MODEL_DIR, f"V{iteration}.keras")

    print(f"Loading previous model: {current_model_path}")

    self_play_files = glob.glob(os.path.join(buffer_dir, "*.tfrecord"))

    steps_per_epoch = max(1, (len(self_play_files) * POSITIONS_PER_FILE) // BATCH_SIZE)

    try:
        model = tf.keras.models.load_model(current_model_path, compile=False)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load {current_model_path}. Did you run the supervised bootstrap? Error: {e}")
        return

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    print("Building dataset pipeline from current replay buffer...")
    train_ds = get_dataset(buffer_dir, BATCH_SIZE, supervised_dir="model/tfrecords")

    print(f"Training V{iteration} for {EPOCHS} Epoch over the entire Replay Buffer...")

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * steps_per_epoch,
        alpha=1 / 3,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-4, global_clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss={
            "prob_dist": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            "move_dist": tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            "prob_dist": 1.0,
            "move_dist": 1.0
        }
    )

    model.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
    )

    print(f"Saving new generation model: {new_model_path}")
    model.save(new_model_path)

    convert_keras_to_onnx(iteration, MODEL_DIR)

    tf.keras.backend.clear_session()


def convert_keras_to_onnx(iteration, model_dir="model_iteration"):
    keras_path = os.path.join(model_dir, f"V{iteration}.keras")
    onnx_path = os.path.join(model_dir, f"V{iteration}.onnx")

    print(f"Loading Keras model V{iteration} for ONNX conversion...")
    model = tf.keras.models.load_model(keras_path, compile=False)

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
    global SUPERVISED_WEIGHT, CHAMPION

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    iteration = 22
    global_batch_counter = 22

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
                    CHAMPION,
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

        print("\n>>> STATE 4: RUNNING ARENA")
        promoted, stats = run_tournament(CHAMPION, iteration)

        if promoted:
            CHAMPION = iteration
            print(f"NEW CHAMPION IS V{iteration}")

        with open("model/logs/arena_logs.txt", "a") as f:
            f.write(f"{stats['champion_path']} vs {stats['challenger_path']}:\n")
            f.write(f"\tScore: {stats['champion_points']} - {stats['challenger_points']}\n")
            f.write(f"\tW-L-D: {stats['champion_wins']}-{stats['challenger_wins']}-{stats['draws']}\n")
            if promoted:
                f.write("PROMOTED\n\n")
            else:
                f.write("NOT PROMOTED\n\n")

        SUPERVISED_WEIGHT = max(0.10, SUPERVISED_WEIGHT / 1.50)

        print(f">>> Iteration {iteration} complete. Loop restarting...\n")
        iteration += 1


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # start loop
    main_orchestrator()
