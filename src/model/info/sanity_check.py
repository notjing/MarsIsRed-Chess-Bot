import chess
import numpy as np

# Import your evaluation file (make sure evaluate.py is pointing to v2.onnx!)
import evaluate
from utils.data_utils import move_to_index


def check_brain():
    board = chess.Board()
    print("Checking raw neural network intuition (No MCTS)...\n")

    # 1. Ask the Neural Network to evaluate the starting board
    win_prob, policies = evaluate.evaluate_board([board])

    # 2. Reshape the policy to match your engine's logic
    policy = policies[0].reshape(8, 8, 73)

    # 3. Filter only legal moves
    legal_moves = list(board.legal_moves)
    move_probs = []

    for move in legal_moves:
        idx = move_to_index(move, board.turn)
        prob = policy[idx]
        move_probs.append((move, prob))

    # Sort by highest probability
    move_probs.sort(key=lambda x: x[1], reverse=True)

    print(
        f"Board Evaluation (Value Head): {win_prob[0][0] if isinstance(win_prob[0], (list, np.ndarray)) else win_prob[0]:.3f}")
    print("\nTop 5 Raw 'Gut' Moves (Policy Head):")
    for move, prob in move_probs[:5]:
        print(f"{move}: {prob:.4f} ({prob * 100:.1f}%)")


if __name__ == "__main__":
    check_brain()
