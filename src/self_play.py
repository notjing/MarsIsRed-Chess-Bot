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
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
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

def play_single_game():
    board = chess.Board()

    opening = random.choice(OPENING_BOOK)
    for move_str in opening:
        board.push_uci(move_str)

    mcts_exts.free_tree()

    game_history = []
    evaluate.clear_cache()

    while not board.is_game_over(claim_draw=True):
        T = 1 - min(len(game_history)/24, 1)

        search_MCTS.search(board, 0, True)
        raw_policy = mcts_exts.get_root_policy(T)

        policy = [(chess.Move.from_uci(m), p) for m, p in raw_policy]

        mcts_exts.free_tree()

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


def generate_self_play_data(total_games, positions_per_file, output_dir, start_batch, worker_id):
    os.makedirs(output_dir, exist_ok=True)

    batch_num = start_batch
    positions_in_current_file = 0
    writer = None
    total_positions_generated = 0

    print(f"Starting Self-Play: {total_games} games total.")
    print(f"Chunking every ~{positions_per_file} positions.")

    for i in range(total_games):
        if writer is None:
            output_path = os.path.join(output_dir, f"self_play_batch_{batch_num:03d}_w{worker_id}.tfrecord")
            writer = tf.io.TFRecordWriter(output_path)
            print(f"\n--- Started new batch: {output_path} ---")

        print(f"Game {i + 1}/{total_games} in progress...")
        game_examples = play_single_game()

        for example in game_examples:
            writer.write(example)

        positions_yielded = len(game_examples)
        positions_in_current_file += positions_yielded
        total_positions_generated += positions_yielded

        print(
            f"-> Game {i + 1} finished. Batch {batch_num:03d} now holds {positions_in_current_file}/{positions_per_file} positions.")

        if positions_in_current_file >= positions_per_file:
            writer.close()
            writer = None
            print(f"-> Batch {batch_num:03d}_w{worker_id} secured to disk.")
            positions_in_current_file = 0
            batch_num += 1

    if writer is not None:
        writer.close()
        print(f"-> Final batch {batch_num:03d}_w{worker_id} secured to disk.")

    print(f"\nGeneration Complete! Total positions saved: {total_positions_generated}")


if __name__ == "__main__":
    output_dir = "model/tfrecords/self_gen"

    NUM_WORKERS = 4

    TOTAL_GAMES_PER_WORKER = 450
    POSITIONS_PER_FILE = 50_000
    STARTING_BATCH = 1

    print(f"Launching {NUM_WORKERS} Parallel Workers...")

    # 2. Launch the Pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        # We store the "future" (the running process) so we can catch any errors
        futures = []

        for worker_id in range(NUM_WORKERS):
            # Tell the executor to run the function with these specific arguments
            future = executor.submit(
                generate_self_play_data,
                TOTAL_GAMES_PER_WORKER,
                POSITIONS_PER_FILE,
                output_dir,
                STARTING_BATCH,
                worker_id
            )
            futures.append(future)

        # 3. Wait for all of them to finish and catch any silent crashes
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise an exception if the worker crashed
            except Exception as e:
                print(f"A worker crashed with error: {e}")

    print("\nAll Parallel Generation Complete!")

