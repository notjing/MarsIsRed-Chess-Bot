import chess
from evaluate import evaluate_board
from utils.board_utils import board_params, dense_params
from utils.data_utils import move_to_index

board = chess.Board() # Starting position

# 1. Ask the GPU directly
win_probs, policies = evaluate_board([board_params(board)], [dense_params(board)])
policy = policies[0].reshape(8, 8, 73)

print(f"White Win Probability: {win_probs[0][0]:.3f}")
print("--- Raw Policy Top 5 ---")

# 2. Extract the probabilities using your index math
move_probs = []
for move in board.legal_moves:
    try:
        r, c, p = move_to_index(move, board.turn)
        move_probs.append((move, policy[r, c, p]))
    except Exception as e:
        print(f"Math Error on {move}: {e}")

# 3. Sort and print
move_probs.sort(key=lambda x: x[1], reverse=True)
for m, prob in move_probs[:10]:
    print(f"{m}: {prob:.4f}")
