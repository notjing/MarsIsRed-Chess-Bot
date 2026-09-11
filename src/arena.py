from __future__ import annotations

import os
import time
from typing import Any

import chess
import onnxruntime as ort

import evaluate
import search_MCTS
from c_bindings import mcts_exts
import random

OPENING_BOOK = [
    [],
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

DEFAULT_TIME_LIMIT = 2.0
DEFAULT_MAX_PLIES = 400  # 200 full moves; long games count as draws


def points_needed_for_promotion(num_games):
    return (num_games / 2.0) + 1.0


def should_promote(challenger_points, num_games):
    return challenger_points >= points_needed_for_promotion(num_games)


def load_session(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    cuda_options = {
        "device_id": 0,
        "gpu_mem_limit": int(1.5 * 1024 * 1024 * 1024),
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT",
    }
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
    )


def _apply_opening(board, book_index):
    for move_str in OPENING_BOOK[book_index % len(OPENING_BOOK)]:
        board.push_uci(move_str)


def play_game(model_white, model_black, board, time_limit=DEFAULT_TIME_LIMIT, max_plies=DEFAULT_MAX_PLIES, verbose=False, game_num=0):

    mcts_exts.free_tree()

    if verbose:
        print(f"\n--- Game {game_num} | start FEN: {board.fen()} ---")

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == chess.WHITE:
            evaluate.set_session(model_white)
            side = "White"
        else:
            evaluate.set_session(model_black)
            side = "Black"

        mcts_exts.free_tree()

        t0 = time.time()
        best_move = search_MCTS.search(board, 0, useTime=False, add_noise=False)

        if best_move is None or best_move not in board.legal_moves:
            if verbose:
                print(f"  Illegal/None move from {side}: {best_move} — treating as draw")
            mcts_exts.free_tree()
            return "1/2-1/2"

        board.push(best_move)
        plies += 1

        if verbose:
            print(f"  ply {plies} ({side}): {best_move}  ({time.time() - t0:.2f}s)")

    mcts_exts.free_tree()

    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True)
    return "1/2-1/2"


def compare_models(
    champion_path: str,
    challenger_path: str,
    num_games: int | None = None,
    time_limit: float = DEFAULT_TIME_LIMIT,
    verbose: bool = True,
) -> tuple[bool, dict[str, Any]]:

    if num_games is None:
        num_games = len(OPENING_BOOK) * 2

    if num_games < 1:
        raise ValueError("num_games must be >= 1")

    need = points_needed_for_promotion(num_games)

    if verbose:
        print(f"Arena: champion   = {champion_path}")
        print(f"       challenger = {challenger_path}")
        print(f"       games={num_games}, time/move={time_limit}s")
        print(f"       promote if challenger points >= {need:.1f}  (50% + 1)")

    session_champ = load_session(champion_path)
    session_chall = load_session(challenger_path)

    chall_wins = 0
    champ_wins = 0
    draws = 0

    for i in range(1, num_games + 1):
        book_index = (i - 1) % len(OPENING_BOOK)
        board = chess.Board()
        _apply_opening(board, book_index)

        challenger_is_white = (i % 2 != 0)

        if challenger_is_white:
            if verbose:
                print(
                    f">>> Game {i}/{num_games}: Challenger (W) vs Champion (B) "
                    f"[book {book_index}]"
                )
            result = play_game(
                session_chall,
                session_champ,
                board,
                time_limit=time_limit,
                verbose=False,
                game_num=i,
            )
            if result == "1-0":
                chall_wins += 1
            elif result == "0-1":
                champ_wins += 1
            else:
                draws += 1
        else:
            if verbose:
                print(
                    f">>> Game {i}/{num_games}: Champion (W) vs Challenger (B) "
                    f"[book {book_index}]"
                )
            result = play_game(
                session_champ,
                session_chall,
                board,
                time_limit=time_limit,
                verbose=False,
                game_num=i,
            )
            if result == "1-0":
                champ_wins += 1
            elif result == "0-1":
                chall_wins += 1
            else:
                draws += 1

        chall_points_so_far = chall_wins + 0.5 * draws
        if verbose:
            print(
                f"    running: chall {chall_wins}W / champ {champ_wins}W / "
                f"{draws}D  |  chall points={chall_points_so_far:.1f}"
            )

    chall_points = chall_wins + 0.5 * draws
    champ_points = champ_wins + 0.5 * draws
    score = chall_points / num_games
    promoted = should_promote(chall_points, num_games)

    stats: dict[str, Any] = {
        "challenger_wins": chall_wins,
        "champion_wins": champ_wins,
        "draws": draws,
        "challenger_points": chall_points,
        "champion_points": champ_points,
        "num_games": num_games,
        "score": score,
        "points_needed": need,
        "promoted": promoted,
        "champion_path": champion_path,
        "challenger_path": challenger_path,
        "time_limit": time_limit,
    }

    if verbose:
        print("\n" + "=" * 34)
        print("ARENA RESULTS")
        print("=" * 34)
        print(f"Challenger: {chall_wins}W  {draws}D  {champ_wins}L")
        print(
            f"Points:     {chall_points:.1f} / {num_games}  "
            f"(need >= {need:.1f})"
        )
        print(f"Score:      {score:.3f}")
        print("PROMOTED" if promoted else "REJECTED — keep champion")

    return promoted, stats


def run_tournament(champ, chall):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model", "model_iteration")

    champion = os.path.join(model_dir, f"V{champ}.onnx")
    challenger = os.path.join(model_dir, f"V{chall}.onnx")

    return compare_models(
        champion_path=champion,
        challenger_path=challenger,
        num_games=len(OPENING_BOOK) * 2,
        time_limit=DEFAULT_TIME_LIMIT,
        verbose=True,
    )
