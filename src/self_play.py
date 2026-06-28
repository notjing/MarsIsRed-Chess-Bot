import chess
import numpy as np
import os
import tensorflow as tf
import concurrent.futures
import evaluate
import search_MCTS
from utils.math_utils import get_policy
from utils.board_utils import board_params, dense_params
from utils.data_utils import make_policy_target, serialize_example
import random
from c_bindings import mcts_exts

OPENING_BOOK = [
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    ["e2e4", "e7e5"], ["e2e4", "c7c5"], ["e2e4", "d7d5"], ["e2e4", "c7c6"], ["e2e4", "e7e6"],
    ["e2e4", "g8f6"], ["e2e4", "b8c6"], ["d2d4", "g8f6"], ["d2d4", "d7d5"], ["g1f3", "d7d5"],
    ["g1f3", "g8f6"], ["g1f3", "c7c5"], ["g1f3", "g7g6"], ["c2c4", "c7c5"], ["c2c4", "e7e5"],
    ["b2b3", "e7e5"]
]


def play_single_game():
    board = chess.Board()

    opening = random.choice(OPENING_BOOK)
    for move_str in opening:
        board.push_uci(move_str)

    mcts_exts.free_tree()

    game_history = []

    # Added a hard 200-move cap to prevent infinite stalling loops
    while not board.is_game_over(claim_draw=True) and len(game_history) < 200:
        if len(game_history) <= 30:
            T = 1.0
        else:
            T = 0.0

        search_MCTS.search(board, 0, True)
        raw_policy = mcts_exts.get_root_policy(T)

        policy = [(chess.Move.from_uci(m), p) for m, p in raw_policy]

        game_history.append({
            "board_layers": board_params(board),
            "dense_layers": dense_params(board),
            "policy_target": make_policy_target(policy, board),
            "turn": board.turn
        })

        moves = [pair[0] for pair in policy]
        probs = [pair[1] for pair in policy]

        chosen_move = np.random.choice(moves, p=probs)

        board.push(chosen_move)

        mcts_exts.promote_root(chosen_move.uci(), board.fen())

    mcts_exts.free_tree()

    res = board.result()
    if res == "1-0":
        game_value = 1.0
    elif res == "0-1":
        game_value = -1.0
    else:
        game_value = 0.0

    tfrecord_examples = []

    for state in game_history:
        state_value = game_value if state["turn"] == chess.WHITE else -game_value

        example_str = serialize_example(
            state["board_layers"],
            state["dense_layers"],
            state_value,
            state["policy_target"]
        )
        tfrecord_examples.append(example_str)

    return tfrecord_examples


def generate_self_play_data(target_positions, positions_per_file, output_dir, start_batch, worker_id, iteration=0):
    os.makedirs(output_dir, exist_ok=True)

    # Instruct evaluate to load the specific model version for this generation cycle
    if hasattr(evaluate, 'load_model_for_worker'):
        evaluate.load_model_for_worker(iteration)

    batch_num = start_batch
    positions_in_current_file = 0
    writer = None
    total_positions_generated = 0
    game_count = 0

    print(f"[Worker {worker_id}] Starting Iteration {iteration} | Target: {target_positions:,} pos.")

    while total_positions_generated < target_positions:
        if writer is None:
            output_path = os.path.join(output_dir, f"self_play_batch_{batch_num:03d}_w{worker_id}.tfrecord")
            writer = tf.io.TFRecordWriter(output_path)

        game_examples = play_single_game()
        game_count += 1

        for example in game_examples:
            writer.write(example)

        positions_yielded = len(game_examples)
        positions_in_current_file += positions_yielded
        total_positions_generated += positions_yielded

        print(
            f"[Worker {worker_id}] Game {game_count} finished ({positions_yielded} moves). Total: {total_positions_generated:,}/{target_positions:,}")

        if positions_in_current_file >= positions_per_file:
            writer.close()
            writer = None
            print(f"[Worker {worker_id}] -> Batch {batch_num:03d}_w{worker_id} secured to disk.")
            positions_in_current_file = 0
            batch_num += 1

    if writer is not None:
        writer.close()
        print(f"[Worker {worker_id}] -> Final batch {batch_num:03d}_w{worker_id} secured to disk.")

    print(f"\n[Worker {worker_id}] Generation Complete!")


if __name__ == "__main__":

    # Protect VRAM when testing this file standalone
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    output_dir = "model/tfrecords/self_gen"

    NUM_WORKERS = 4
    TOTAL_POSITIONS_PER_WORKER = 50_000
    POSITIONS_PER_FILE = 50_000
    STARTING_BATCH = 8
    CURRENT_ITERATION = 0  # Default to 0 for standalone testing

    print(f"Launching {NUM_WORKERS} Parallel Workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []

        for worker_id in range(NUM_WORKERS):
            future = executor.submit(
                generate_self_play_data,
                TOTAL_POSITIONS_PER_WORKER,
                POSITIONS_PER_FILE,
                output_dir,
                STARTING_BATCH,
                worker_id,
                CURRENT_ITERATION
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A worker crashed with error: {e}")

    print("\nAll Parallel Generation Complete!")
