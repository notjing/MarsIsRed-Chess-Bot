import chess
import onnxruntime as ort
import os
import time

# Import your existing engine files
import evaluate
import search_MCTS

OPENING_BOOK = [
    [],
    ["e2e4", "e7e5"],
    ["e2e4", "c7c5"],
    ["e2e4", "d7d5"],
    ["e2e4", "c7c6"],
    ["e2e4", "e7e6"],
    ["e2e4", "g8f6"],
    ["e2e4", "b8c6"],
    ["d2d4", "g8f6"],
    ["d2d4", "d7d5"],
    ["g1f3", "d7d5"],
    ["g1f3", "g8f6"],
    ["g1f3", "c7c5"],
    ["g1f3", "g7g6"],
    ["c2c4", "c7c5"],
    ["c2c4", "e7e5"],
    ["b2b3", "e7e5"]
]


def load_session(model_path):
    """Loads an ONNX session with memory limits to fit both on the GPU."""
    cuda_options = {
        "device_id": 0,
        "gpu_mem_limit": int(1.5 * 1024 * 1024 * 1024),  # 1.5GB per model (3GB total)
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT"
    }
    return ort.InferenceSession(
        model_path,
        providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
    )


def play_match(model_white, model_black, game_num, board):
    """Plays a single game between two loaded ONNX sessions."""
    search_MCTS.clear_tree()

    print(f"\n--- Game {game_num} Started ---")

    # Optional: Lower this if the arena is taking too long
    # (Requires changing NUM_NODES inside search_MCTS.py to use a global variable first)

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            evaluate.session = model_white  # Swap brain to White
            active_name = "White"
        else:
            evaluate.session = model_black  # Swap brain to Black
            active_name = "Black"

        # We must clear the tree every turn because the priors belong to different brains!
        search_MCTS.clear_tree()
        evaluate.clear_cache()

        # Search and get best move
        start_time = time.time()
        best_move = search_MCTS.search(board, 0, add_noise=False)

        if best_move is None:
            break  # Failsafe

        board.push(best_move)

        move_time = time.time() - start_time
        print(f"Move {board.fullmove_number} ({active_name}): {best_move} | Time: {move_time:.1f}s")

    res = board.result(claim_draw=True)
    print(f"Game Over! Result: {res}")
    return res


def run_tournament():
    # 1. Paths to your two models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    v1_path = os.path.join(script_dir, "model_cache", "modeL_v1.onnx")
    v2_path = os.path.join(script_dir, "model_cache", "model_v7.onnx")

    print("Loading AI Brains into GPU...")
    session_v1 = load_session(v1_path)
    session_v2 = load_session(v2_path)
    print("Brains loaded successfully!\n")

    # Tournament Stats
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for i in range(1, len(OPENING_BOOK) * 2 + 1):
        book_index = (i - 1) % len(OPENING_BOOK)

        board = chess.Board()
        for move_str in OPENING_BOOK[book_index]:
            board.push_uci(move_str)

        if i % 2 != 0:
            print(f">>> MATCH {i}: V1 (White) vs V2 (Black) <<<")
            result = play_match(model_white=session_v1, model_black=session_v2, game_num=i, board=board)
            if result == "1-0":
                p1_wins += 1
            elif result == "0-1":
                p2_wins += 1
            else:
                draws += 1
        else:
            print(f">>> MATCH {i}: V2 (White) vs V1 (Black) <<<")
            result = play_match(model_white=session_v2, model_black=session_v1, game_num=i, board=board)
            if result == "1-0":
                p2_wins += 1
            elif result == "0-1":
                p1_wins += 1
            else:
                draws += 1

    print("\n" + "=" * 30)
    print("🏆 TOURNAMENT RESULTS 🏆")
    print("=" * 30)
    print(f"V1 (Iteration 1) Wins: {p1_wins}")
    print(f"V2 (Iteration 2) Wins: {p2_wins}")
    print(f"Draws:                 {draws}")

    if p2_wins > p1_wins:
        print("\nhuge ups.")
    elif p2_wins == p1_wins:
        print("\nmeh")
    else:
        print("\nbruh")


if __name__ == "__main__":
    run_tournament()
