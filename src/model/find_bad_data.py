import tensorflow as tf
import glob
import os


def find_corrupted_shards():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TFRECORD_DIR = os.path.join(BASE_DIR, "tfrecords")

    all_files = glob.glob(os.path.join(TFRECORD_DIR, "*.tfrecord"))
    print(f"Scanning {len(all_files)} TFRecord files for corruption...\n")

    for file_path in all_files:
        try:
            # Attempt to read through the file
            for _ in tf.data.TFRecordDataset(file_path):
                pass
        except tf.errors.DataLossError as e:
            print(f"\n❌ CORRUPTED FILE FOUND: {os.path.basename(file_path)}")
            print(f"Error details: {e}\n")
            print("ACTION: Delete this file and restart training.")
            return  # Stop after finding the bad one

    print("\n✅ All files scanned successfully. No corruption found.")


def check_single_file(filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "tfrecords", filename)

    print(f"Scanning {filename} for corruption...")

    try:
        # Attempt to read through the single file
        for _ in tf.data.TFRecordDataset(file_path):
            pass
        print(f"\n✅ SUCCESS: {filename} is perfectly healthy!")

    except tf.errors.DataLossError as e:
        print(f"\n❌ CORRUPTED FILE FOUND: {filename}")
        print(f"Error details: {e}")
        print("ACTION: Delete this file.")
    except tf.errors.NotFoundError:
        print(f"\n⚠️ ERROR: Could not find {filename}. Check the spelling!")


if __name__ == "__main__":
    check_single_file("lichess_elite_20_04_175.tfrecord")
    # find_corrupted_shards()
