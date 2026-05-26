import numpy as np
import chess
import onnxruntime as ort
from utils.board_utils import get_mapped_coords, square_control, piece_positions, dense_params
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, "model_cache/" "og.onnx")

cuda_options = {
    "device_id": 0,
    "gpu_mem_limit": int(1.5 * 1024 * 1024 * 1024),
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "DEFAULT"
}

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

try:
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=[
            ('CUDAExecutionProvider', cuda_options),
            'CPUExecutionProvider'
        ]
    )
except Exception as e:
    print(f"Failed to load ONNX model. Error: {e}")
    raise

# 3. Get input names
input_name_board = session.get_inputs()[0].name
input_name_extra = session.get_inputs()[1].name

board_cache = {}

cache_hits = 0
cache_misses = 0


def evaluate_board(planes, dense):
    global cache_hits, cache_misses

    input_planes = np.array(planes, dtype=np.float32)
    input_vecs = np.array(dense, dtype=np.float32)

    ort_inputs = {
        input_name_board: input_planes,
        input_name_extra: input_vecs
    }

    outputs = session.run(None, ort_inputs)

    win_prob = outputs[0]
    policy = outputs[1]

    return win_prob, policy

def clear_cache():
    global board_cache, cache_hits, cache_misses
    board_cache = {}
    cache_hits = 0
    cache_misses = 0
