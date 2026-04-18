import os
import chess
import numpy as np
import tensorflow as tf
import pandas as pd

from createparams import square_control, makeboards, board_parameters, get_mapped_coords

os.makedirs("tfrecords", exist_ok=True)

# -----------------------------
# Utils
# -----------------------------
def float_feature(x):
    """
    Accepts a numpy array and converts it into a feature
    Just to help speed up this process, not exactly sure how yet
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))

def serialize_example(board_input, extra_input, eval_value):
    """
    Converts the features and evaluation, x and y respectively into the TFRecord file format
    """
    feature = {
        "board": float_feature(board_input.flatten()),
        "extra": float_feature(extra_input),
        "eval": float_feature([eval_value])
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def convert_eval(e, flip):
    """
    Given an eval, forces the evaluation to be within 1500 cp
    If there is an error ("+-#x") just sets it to be +-1500 depending on who has mate
    """

    ret = 0

    try:
        ret = float(np.clip(float(e), -1500, 1500))
    except:
        ret = 1500 if "+" in str(e) else -1500

    return ret * (-1 if flip else 1)

# -----------------------------
# Core processing
# -----------------------------
def process_fen(fen):
    """
    Takes the fen and retreives all the needed parameters
    """
    board = chess.Board(fen)

    extra = board_parameters(board)

    # Gets the CNN parameters
    layers_12 = makeboards(board)
    s_control = square_control(board)

    # depending on whos turn it is gets the enpassant squares
    ep_grid = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        flip = (board.turn == chess.BLACK)
        r, c = get_mapped_coords(board.ep_square, flip)
        ep_grid[r][c] = 1.0

    # stacks all the board layers into a single nparray
    layers = np.array(layers_12 + s_control + [ep_grid], dtype=np.float32)

    # tranposes to (8,8,25)
    layers = np.transpose(layers, (1, 2, 0))

    return layers, np.array(extra, dtype=np.float32)

def build_tfrecord(csv_path, tfrecord_prefix, max_rows=None, shard_size=50_000):
    """
    Builds all the tfrecord shards given a file
    """

    # parses file
    df = pd.read_csv(csv_path, nrows=max_rows)
    fens = df.iloc[:, 0].tolist()
    evals = df.iloc[:, 1].tolist()

    shard_idx = 0
    count = 0
    writer = tf.io.TFRecordWriter(
        f"tfrecords/{tfrecord_prefix}_{shard_idx:03d}.tfrecord"
    )

    cnt = 0
    for fen, ev in zip(fens, evals):
        cnt = cnt + 1
        if cnt % 10000 == 0:
            print(cnt)

        try:

            ev = convert_eval(ev, (chess.Board(fen).turn == chess.BLACK))

            b, e = process_fen(fen)
            writer.write(serialize_example(b, e, ev))
            count += 1

            # reached maximum shard size, close it and move onto the next shard
            if count % shard_size == 0:
                writer.close()
                shard_idx += 1
                writer = tf.io.TFRecordWriter(
                    f"tfrecords/{tfrecord_prefix}_{shard_idx:03d}.tfrecord"
                )

        except Exception as e:
            print(e)
            continue

    writer.close()
    print(f"Finished {csv_path}")

if __name__ == "__main__":
    build_tfrecord("chess_evaluations/random_evals.csv", "random", 1_000_000)
    build_tfrecord("chess_evaluations/mate_in_one_0.csv", "mate0")
    build_tfrecord("chess_evaluations/mate_in_one_1.csv", "mate1")
    build_tfrecord("chess_evaluations/chessData.csv", "data", 2_500_000)
    build_tfrecord("chess_evaluations/tactic_evals.csv", "tactics", 2_500_000)
