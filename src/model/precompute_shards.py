"""Stream a PGN into fixed-size CHAI training shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = MODEL_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.shard_io import ShardWriter
from utils.board_utils import board_params, dense_params
from utils.data_utils import make_policy_target


def _wdl_from_result(base_value: float, turn) -> list[float]:
    current = base_value if turn == chess.WHITE else -base_value
    if current == 1:
        return [1.0, 0.0, 0.0]
    if current == -1:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def build_shards_from_pgn(pgn_path, out_dir, prefix, max_games=None, shard_size=50_000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    count = 0
    games_processed = 0
    writer = None

    with open(pgn_path, "r", encoding="utf-8") as pgn_handle:
        while True:
            if max_games and games_processed >= max_games:
                break

            game = chess.pgn.read_game(pgn_handle)
            if game is None:
                break

            result = game.headers.get("Result")
            if result == "1-0":
                base_value = 1.0
            elif result == "0-1":
                base_value = -1.0
            else:
                base_value = 0.0

            board = game.board()
            for move in game.mainline_moves():
                win_dist = _wdl_from_result(base_value, board.turn)
                try:
                    board_layers = board_params(board)
                    dense_layers = np.array(dense_params(board), dtype=np.float32)
                    policy_target = make_policy_target([(move, 1.0)], board)
                    if writer is None:
                        writer = ShardWriter(str(out_dir / f"{prefix}_{shard_idx:03d}.bin"))
                    writer.write(board_layers, dense_layers, win_dist, policy_target)
                    count += 1
                    if count % shard_size == 0:
                        writer.close()
                        writer = None
                        shard_idx += 1
                        print(f"Created Shard {shard_idx} ({count} total positions)")
                except Exception as e:
                    print(f"Skipped position due to error: {e}")
                board.push(move)

            games_processed += 1
            if games_processed % 1000 == 0:
                print(f"Processed {games_processed} games from {pgn_path}...")

    if writer is not None:
        writer.close()
    print(f"Finished parsing {pgn_path}. Total positions extracted: {count}")


def main():
    parser = argparse.ArgumentParser(description="Build CHAI shards from a PGN")
    parser.add_argument(
        "--pgn",
        default=str(MODEL_DIR / "data" / "lichess_games" / "lichess_elite_2020-07.pgn"),
    )
    parser.add_argument("--out-dir", default=str(MODEL_DIR / "shards"))
    parser.add_argument("--prefix", default="lichess_elite_2020_07")
    parser.add_argument("--max-games", type=int, default=300_000)
    parser.add_argument("--shard-size", type=int, default=50_000)
    args = parser.parse_args()
    build_shards_from_pgn(args.pgn, args.out_dir, args.prefix, args.max_games, args.shard_size)


if __name__ == "__main__":
    main()
