import chess
import numpy as np
import tensorflow as tf

from utils.board_utils import get_mapped_coords


def move_to_index(move, turn):
    QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    KNIGHT_DIRS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    UNDER_PROMOS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    flip = (turn == chess.BLACK)

    from_r, from_c = get_mapped_coords(move.from_square, flip)
    to_r, to_c = get_mapped_coords(move.to_square, flip)

    dc = to_c - from_c
    dr = from_r - to_r

    if move.promotion and move.promotion != chess.QUEEN:
        promo_idx = UNDER_PROMOS.index(move.promotion)
        plane = 64 + ((dc + 1) * 3) + promo_idx
        return from_r, from_c, plane

    if (dc, dr) in KNIGHT_DIRS:
        plane = 56 + KNIGHT_DIRS.index((dc, dr))
        return from_r, from_c, plane

    distance = max(abs(dc), abs(dr))
    direction_u = (dc // distance, dr // distance)
    dir_idx = QUEEN_DIRS.index(direction_u)

    plane = (dir_idx * 7) + (distance - 1)
    return from_r, from_c, plane


def make_policy_target(prob_dist, board):
    # Initialize a blank 8x8x73 tensor
    target = np.zeros((8, 8, 73), dtype=np.float32)

    for move, prob in prob_dist:
        x, y, p = move_to_index(move, board.turn)
        target[x, y, p] = prob

    return target


def float_feature(x):
    """Converts a numpy array into a TF FloatList Feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))


def serialize_example(board_input, extra_input, eval_value, policy_target):
    """
    Converts features and dual-head targets into the TFRecord file format.
    """
    feature = {
        "board": float_feature(board_input.flatten()),
        "extra": float_feature(extra_input),
        "eval": float_feature([eval_value]),
        "policy": float_feature(policy_target.flatten())  # NEW: 4672-length array
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
