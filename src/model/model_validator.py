import tensorflow as tf
import os
# Assuming your training script is 'train.py' and it contains the loading logic
from aimodel import get_val_dataset, parse_tfrecord


def main():
    # 1. Path to your saved model
    model_path = 'best_model.keras'

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Make sure the file is in the same folder.")
        return

    # 2. Load the model (Standard Keras load)
    print("Loading model...")
    aimodel = tf.keras.models.load_model(model_path)

    # 3. Setup the Validation Dataset
    # This points to your directory containing the .tfrecord files
    TFRECORD_DIR = "tfrecords"

    # We take the files and filter for only the validation/test shards
    all_files = tf.data.Dataset.list_files(os.path.join(TFRECORD_DIR, "*.tfrecord"), shuffle=False)

    # Usually, we take the first 10% for validation. Adjust 'take' if your split was different.
    num_files = len(list(tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "*.tfrecord"))))
    num_test_files = max(1, int(0.1 * num_files))
    test_files = all_files.take(num_test_files)

    print(f"Loading {num_test_files} validation shards...")
    val_ds = get_val_dataset(test_files, batch_size=256)

    # 4. Run Evaluation
    print("Running evaluation on validation data...")
    # 'steps' defines how many batches to check. 200-500 is usually enough for a solid average.
    results = aimodel.evaluate(val_ds, steps=500, verbose=1)

    # 5. Output Results
    # results[1] = Value Head MAE, results[2] = Policy Accuracy
    print("\n" + "=" * 30)
    print("   VALIDATION METRICS")
    print("=" * 30)
    print(f"Value Head MAE:  {results[1]:.4f}")
    print(f"Policy Accuracy: {results[2]:.2%}")
    print("=" * 30)


if __name__ == "__main__":
    main()
