import os
import glob
import site

# MUST run before importing onnxruntime
def _setup_nvidia_lib_path():
    dirs = []
    for sp in site.getsitepackages():
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    for d in ["/usr/local/cuda/lib64", "/usr/lib/wsl/lib"]:
        if os.path.isdir(d):
            dirs.append(d)

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = ":".join(dict.fromkeys([d for d in dirs if d] + ([existing] if existing else [])))
    os.environ["LD_LIBRARY_PATH"] = merged

_setup_nvidia_lib_path()

import numpy as np
import onnxruntime as ort

session = None
input_name_board = None
input_name_extra = None

#def load_model_for_worker(iteration):
#global session, input_name_board, input_name_extra

script_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(script_dir, "model", "model_iteration", f"V{26}.onnx")

cuda_options = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "EXHAUSTIVE",
}

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

try:
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=[
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",  # safe fallback
        ],
    )
    print("ORT providers:", session.get_providers())

    input_name_board = session.get_inputs()[0].name
    input_name_extra = session.get_inputs()[1].name
    print(f"ONNX Session successfully initialized for V{26}.onnx")

except Exception as e:
    print(f"Failed to load ONNX model {onnx_path}. Error: {e}")
    raise

def set_session(ort_session):
    global session, input_name_board, input_name_extra
    session = ort_session
    input_name_board = session.get_inputs()[0].name
    input_name_extra = session.get_inputs()[1].name

def evaluate_board(planes, dense):
    global session, input_name_board, input_name_extra

    if session is None:
        raise RuntimeError("ONNX Session not initialized! Call load_model_for_worker() first.")

    input_planes = np.array(planes, dtype=np.float32)
    input_vecs = np.array(dense, dtype=np.float32)

    outputs = session.run(
        None,
        {
            input_name_board: input_planes,
            input_name_extra: input_vecs,
        },
    )
    return outputs[0], outputs[1]
