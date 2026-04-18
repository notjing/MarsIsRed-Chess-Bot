import math
import sys
import numpy as np
import chess
import tensorflow as tf
from huggingface_hub import hf_hub_download
from model.createparams import square_control, makeboards, board_parameters, get_mapped_coords


def pawn_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred)) * 1500


model_dir = "model_cache"
model_path = hf_hub_download(
    repo_id="notjing/chessai",
    filename="chessai_model.keras",
    local_dir=model_dir
)

mod = tf.keras.models.load_model(
    model_path,
    custom_objects={'pawn_error': pawn_error}
)


@tf.function
def predict_fn(inputs):
    return mod(inputs, training=False)

board_cache = {}

cache_hits = 0
cache_misses = 0


def evaluate_board(boards):
    global cache_hits, cache_misses

    batch_planes = []
    batch_vecs = []

    for board in boards:

        fen = board.fen()

        if fen in board_cache:
            planes, dense = board_cache[fen]
            cache_hits += 1
        else:
            cache_misses += 1
            flip = (board.turn == chess.BLACK)
            dense = board_parameters(board)
            layers = makeboards(board)
            s_control = square_control(board)

            ep_grid = np.zeros((8, 8), dtype=np.float32)
            if board.ep_square is not None:
                r, c = get_mapped_coords(board.ep_square, flip)
                ep_grid[r][c] = 1.0

            planes = np.array(layers + s_control + [ep_grid], dtype="float32")
            planes = np.transpose(planes, (1, 2, 0))

            board_cache[fen] = (planes, dense)

        batch_vecs.append(dense)
        batch_planes.append(planes)

    input_planes = np.array(batch_planes, dtype="float32")
    input_vecs = np.array(batch_vecs, dtype="float32")

    preds = predict_fn([input_planes, input_vecs])
    raw_scores = preds.numpy().flatten()

    win_probabilities = 2 / (1 + np.exp(-0.00368208 * raw_scores * 1500)) - 1

    if (cache_hits + cache_misses) % 1000 == 0:
        hit_rate = cache_hits / (cache_hits + cache_misses) * 100
        print(f"Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)")

    return win_probabilities

print(evaluate_board([chess.Board("2kr1b1r/ppp1pppp/2b5/3q4/P1P5/1P3N2/1B1PQPPP/R3R1K1 b - - 0 13")]))
print(evaluate_board([chess.Board("2kr1b1r/ppp1pppp/2b5/8/P1P5/1P3q2/1B1PQPPP/R3R1K1 w - - 0 14")]))

