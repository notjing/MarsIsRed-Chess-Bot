import chess
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
# Import your state converters from your utils
from c_bindings import mcts_exts


def test_policy_head(model_path, fen):
    print(f"\ncdLoading Model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

    board = chess.Board(fen)
    print(f"Testing Position:\n{board}\n")

    # Convert FEN to your model's input format
    board_input = np.expand_dims(mcts_exts.py_board_params(board.fen()), axis=0)
    extra_input = np.expand_dims(mcts_exts.py_dense_params(board.fen()), axis=0)

    # Run raw prediction (NO SEARCH)
    predictions = model.predict({"board_input": board_input, "extra_input": extra_input}, verbose=0)

    # predictions is a tuple/list depending on your exact Keras output order.
    # Assume [0] is value, [1] is policy. Adjust if yours is reversed!
    value = predictions[0][0][0]
    policy = predictions[1][0]

    print(f"Value Head Evaluation: {value:.3f}")

    # In AlphaZero, the policy array is 8*8*73. You likely have a function
    # to decode this, but we can just look at the raw max index for now.
    best_move_index = np.argmax(policy)
    confidence = policy[best_move_index] * 100

    print(f"Policy Head's Top Choice Index: {best_move_index} (Confidence: {confidence:.1f}%)")


if __name__ == "__main__":
    mate_in_one_fen = "r2q1rk1/ppp1b1pp/2n1bp2/3p4/5B2/2PB1N2/PP3PPP/R2Q1RK1 w - - 0 1"

    test_policy_head("model/model_iteration/V24.keras", mate_in_one_fen)
