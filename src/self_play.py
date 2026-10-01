import chess
import numpy as np
import os
import concurrent.futures
import evaluate
import search_MCTS
from utils.math_utils import get_policy
from utils.board_utils import board_params, dense_params
from utils.data_utils import make_policy_target
from model.shard_io import ShardWriter
import random
from c_bindings import mcts_exts

OPENING_BOOK = [
    [], [], [], [],

    # -----------------------------
    # King's Pawn (1.e4)
    # -----------------------------
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],             # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],             # Italian
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],             # Scotch
    ["e2e4", "e7e5", "g1f3", "b8c6", "c2c3"],             # Ponziani
    ["e2e4", "e7e5", "f2f4"],                             # King's Gambit

    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"],             # Open Sicilian
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4"],
    ["e2e4", "c7c5", "c2c3"],                             # Alapin
    ["e2e4", "c7c5", "g2g3"],                             # Closed Sicilian
    ["e2e4", "c7c5", "b2b4"],                             # Wing Gambit

    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"],             # French Classical
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2"],             # Tarrasch
    ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"],             # Advance

    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"],             # Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"],             # Advance Caro

    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"],             # Scandinavian
    ["e2e4", "g8f6", "e4e5", "f6d5"],                     # Alekhine
    ["e2e4", "g7g6", "d2d4", "f8g7"],                     # Modern
    ["e2e4", "d7d6", "d2d4", "g8f6"],                     # Pirc

    # -----------------------------
    # Queen's Pawn (1.d4)
    # -----------------------------
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3"],             # QGD
    ["d2d4", "d7d5", "c2c4", "c7c6"],                     # Slav
    ["d2d4", "d7d5", "g1f3", "g8f6"],                     # London setup
    ["d2d4", "g8f6", "c2c4", "e7e6"],                     # Indian setup
    ["d2d4", "g8f6", "c2c4", "g7g6"],                     # King's Indian
    ["d2d4", "g8f6", "c2c4", "e7e5"],                     # Budapest
    ["d2d4", "f7f5"],                                    # Dutch
    ["d2d4", "d7d6", "e2e4"],                            # Old Indian

    # -----------------------------
    # English
    # -----------------------------
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    ["c2c4", "c7c5", "g2g3", "g7g6"],
    ["c2c4", "e7e6", "g2g3", "d7d5"],
    ["c2c4", "g8f6", "b1c3", "e7e5"],

    # -----------------------------
    # Réti
    # -----------------------------
    ["g1f3", "d7d5", "c2c4"],
    ["g1f3", "g8f6", "c2c4"],
    ["g1f3", "c7c5", "g2g3"],
    ["g1f3", "d7d5", "g2g3"],

    # -----------------------------
    # Bird
    # -----------------------------
    ["f2f4", "d7d5"],
    ["f2f4", "g8f6"],
    ["f2f4", "d7d5", "g1f3"],

    # -----------------------------
    # Larsen
    # -----------------------------
    ["b2b3", "e7e5", "c1b2"],
    ["b2b3", "d7d5", "c1b2"],

    # -----------------------------
    # Grob (for diversity)
    # -----------------------------
    ["g2g4", "d7d5"],

    # -----------------------------
    # Polish
    # -----------------------------
    ["b2b4", "e7e5"],

    # -----------------------------
    # Sokolsky
    # -----------------------------
    ["b2b4", "g8f6"],

    # -----------------------------
    # Random but sound transpositions
    # -----------------------------
    ["e2e4", "e7e5", "g1f3", "g8f6"],
    ["e2e4", "c7c6", "g1f3"],
    ["d2d4", "g8f6", "g1f3"],
    ["c2c4", "g8f6", "g1f3"],

    # -----------------------------
    # King's Gambit
    # -----------------------------
    ["e2e4", "e7e5", "f2f4", "e5f4"],
    ["e2e4", "e7e5", "f2f4", "d7d5"],

    # Vienna Gambit
    ["e2e4", "e7e5", "b1c3", "g8f6", "f2f4"],

    # Danish Gambit
    ["e2e4", "e7e5", "d2d4", "e5d4", "c2c3"],

    # Evans Gambit
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"],

    # Smith-Morra Gambit
    ["e2e4", "c7c5", "d2d4", "c5d4", "c2c3"],

    # Wing Gambit
    ["e2e4", "c7c5", "b2b4"],

    # Blackmar-Diemer Gambit
    ["d2d4", "d7d5", "e2e4", "d5e4", "b1c3"],

    # Benko Gambit
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5"],

    # Benoni
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6"],

    # King's Indian
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4"],

    # Dutch
    ["d2d4", "f7f5", "g2g3"],

    # Dutch Stonewall
    ["d2d4", "f7f5", "g2g3", "e7e6", "f1g2", "d7d5"],

    # Sicilian Dragon
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4",
     "f3d4", "g8f6", "b1c3", "g7g6"],

    # Sicilian Najdorf
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4",
     "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],

    # Sicilian Accelerated Dragon
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4",
     "c5d4", "f3d4", "g7g6"],

    # Scandinavian Portuguese
    ["e2e4", "d7d5", "e4d5", "g8f6"],

    # Latvian Gambit
    ["e2e4", "e7e5", "g1f3", "f7f5"],

    # Elephant Gambit
    ["e2e4", "e7e5", "g1f3", "d7d5"],

    # Englund Gambit
    ["d2d4", "e7e5"],

    # Budapest Gambit
    ["d2d4", "g8f6", "c2c4", "e7e5"],

    # From Gambit
    ["f2f4", "e7e5"],

    # Bird's Gambit
    ["f2f4", "d7d5", "g1f3", "g8f6", "e2e3"],

    # Owen's Defense
    ["e2e4", "b7b6"],

    # Nimzowitsch Defense
    ["e2e4", "b8c6"],

    # Polish Defense
    ["d2d4", "b7b5"],

    # St. George Defense
    ["e2e4", "a7a6"],

    # Orangutan
    ["b2b4", "e7e5", "c1b2"],

    # Grob
    ["g2g4", "d7d5", "f1g2"],

    # Bird vs Dutch setup
    ["f2f4", "f7f5"]
]

BLEND = 0.5


def play_single_game():
    board = chess.Board()

    opening = random.choice(OPENING_BOOK)
    for move_str in opening:
        board.push_uci(move_str)

    mcts_exts.free_tree()

    game_history = []

    # Added a hard 200-move cap to prevent infinite stalling loops
    while not board.is_game_over() and len(game_history) < 250:

        chosen_move = search_MCTS.search(board, 0, False, add_noise=True)
        raw_policy = mcts_exts.get_root_policy()
        raw_value = mcts_exts.get_root_value()

        policy = [(chess.Move.from_uci(m), p) for m, p in raw_policy]

        game_history.append({
            "board_layers": mcts_exts.py_board_params(board.fen()),
            "dense_layers": mcts_exts.py_dense_params(board.fen()),
            "policy_target": make_policy_target(policy, board),
            "value_target": raw_value,
            "turn": board.turn
        })

        moves = [pair[0] for pair in policy]
        probs = [pair[1] for pair in policy]

        board.push(chosen_move)

    mcts_exts.free_tree()

    res = board.result()
    if res == "1-0":
        game_value = 1.0
    elif res == "0-1":
        game_value = -1.0
    elif res == "1/2-1/2":
        game_value = 0.0
    else:
        return None

    examples = []

    for state in game_history:
        state_value = game_value if state["turn"] == chess.WHITE else -game_value

        wdl_z = []
        if state_value == 1.0:
            wdl_z = [1, 0, 0]
        elif state_value == -1.0:
            wdl_z = [0, 0, 1]
        else:
            wdl_z = [0, 1, 0]

        q = state["value_target"]
        wdl_q = [q, 1-q, 0] if q >= 0 else [0, 1+q, -q]

        wdl = [(1 - BLEND) * a + BLEND * b for a, b in zip(wdl_z, wdl_q)]

        examples.append(
            (
                state["board_layers"],
                state["dense_layers"],
                np.asarray(wdl, dtype=np.float32),
                state["policy_target"],
            )
        )

    return examples


def generate_self_play_data(target_positions, positions_per_file, output_dir, start_batch, worker_id, champion, iteration=0):
    os.makedirs(output_dir, exist_ok=True)

    # Instruct evaluate to load the specific model version for this generation cycle
    if hasattr(evaluate, 'load_model_for_worker'):
        evaluate.load_model_for_worker(champion)

    batch_num = start_batch
    positions_in_current_file = 0
    writer = None
    total_positions_generated = 0
    game_count = 0

    print(f"[Worker {worker_id}] Starting Iteration {iteration} | Target: {target_positions:,} pos.")

    while total_positions_generated < target_positions:
        if writer is None:
            output_path = os.path.join(output_dir, f"self_play_batch_{batch_num:03d}_w{worker_id}.bin")
            writer = ShardWriter(output_path)

        game_examples = play_single_game()

        if game_examples is None:
            continue

        game_count += 1

        for board, extra, wdl, policy in game_examples:
            writer.write(board, extra, wdl, policy)

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
