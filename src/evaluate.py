import numpy as np
import onnxruntime as ort
import os

session = None
input_name_board = None
input_name_extra = None
def load_model_for_worker(iteration):
    """
    loads the most recent iteration for the worker
    """
    global session, input_name_board, input_name_extra

    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, "model", "model_iteration", f"V{iteration}.onnx")

    cuda_options = {
        "device_id": 0,
        "gpu_mem_limit": int(1.5 * 1024 * 1024 * 1024),
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT"
    }

    # locks each worker to a single thread
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    try:
        # loads onnx model
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=[
                ('CUDAExecutionProvider', cuda_options),
                'CPUExecutionProvider'
            ]
        )

        # Get input names dynamically
        input_name_board = session.get_inputs()[0].name
        input_name_extra = session.get_inputs()[1].name

        print(f"ONNX Session successfully initialized for V{iteration}.onnx")

    except Exception as e:
        print(f"Failed to load ONNX model {onnx_path}. Error: {e}")
        raise


def evaluate_board(planes, dense):
    """ Evaluates a batch of board states """

    global session, input_name_board, input_name_extra

    if session is None:
        raise RuntimeError("ONNX Session not initialized! Call load_model_for_worker() first.")

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
