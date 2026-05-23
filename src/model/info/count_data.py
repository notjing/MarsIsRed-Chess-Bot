import tensorflow as tf
import os


def count_tfrecords_in_folder(folder_path):
    # Find all tfrecord files in the target directory
    file_pattern = os.path.join(folder_path, "*.tfrecord")
    tfrecord_files = tf.io.gfile.glob(file_pattern)

    if not tfrecord_files:
        print(f"No .tfrecord files found in: {folder_path}")
        return

    print(f"Found {len(tfrecord_files)} files. Tallying positions...")
    print("-" * 40)

    total_positions = 0

    for file_path in tfrecord_files:
        file_name = os.path.basename(file_path)
        file_count = 0

        try:
            # Load the dataset
            dataset = tf.data.TFRecordDataset(file_path)

            # Iterate through the records to count them
            for _ in dataset:
                file_count += 1

            print(f"[{file_name}]: {file_count} positions")
            total_positions += file_count

        except tf.errors.DataLossError as e:
            # This catches the exact error caused by the OS killing your workers mid-write
            print(f"[{file_name}]: {file_count} positions (WARNING: File truncated at the end due to crash)")
            total_positions += file_count

        except Exception as e:
            print(f"[{file_name}]: ERROR - {e}")

    print("-" * 40)
    print(f"GRAND TOTAL: {total_positions} valid positions ready for training.")


if __name__ == "__main__":
    # Point this to the directory where your workers are saving the data
    TARGET_FOLDER = r"C:\Users\ethan\EthansCode\PycharmProjects\Chess\ChessAI\src\model\tfrecords\self_gen"

    # Run the counter
    count_tfrecords_in_folder(TARGET_FOLDER)
