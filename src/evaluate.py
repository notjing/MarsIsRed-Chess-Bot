import numpy as np
import chess
import onnxruntime as ort
from model.createparams import square_control, makeboards, board_parameters, get_mapped_coords
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, "model_cache", "chessai_model.onnx")

try:
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
except Exception as e:
    print(f"Failed to load ONNX model. Error: {e}")
    raise

input_name_board = session.get_inputs()[0].name
input_name_extra = session.get_inputs()[1].name

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

    input_planes = np.array(batch_planes, dtype=np.float32)
    input_vecs = np.array(batch_vecs, dtype=np.float32)

    ort_inputs = {
        input_name_board: input_planes,
        input_name_extra: input_vecs
    }

    outputs = session.run(None, ort_inputs)

    for i, output in enumerate(session.get_outputs()):
        print(f"Output {i}: {output.name} - Shape: {output.shape}")

    win_prob = outputs[0]
    policy = outputs[1]

    return win_prob, policy
