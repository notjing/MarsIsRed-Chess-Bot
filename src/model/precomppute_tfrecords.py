import os
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
import io

from utils.board_utils import dense_params, board_params

from ChessAI.src.utils.data_utils import make_policy_target, serialize_example

os.makedirs("tfrecords", exist_ok=True)


# -----------------------------
# Core Processing
# -----------------------------
def build_tfrecord_from_pgn(pgn_path, tfrecord_prefix, max_games=None, shard_size=50_000):
    """
    Streams a PGN file and builds TFRecord shards.
    """
    shard_idx = 0
    count = 0
    games_processed = 0

    writer = tf.io.TFRecordWriter(
        f"tfrecords/{tfrecord_prefix}_{shard_idx:03d}.tfrecord"
    )

    with open(pgn_path, "r", encoding="utf-8") as pgn_handle:
        while True:
            if max_games and games_processed >= max_games:
                break

            game = chess.pgn.read_game(pgn_handle)
            if game is None:
                break  # End of file

            # 1. Base Game Value (from White's absolute perspective)
            result = game.headers.get("Result")
            if result == "1-0":
                base_value = 1.0
            elif result == "0-1":
                base_value = -1.0
            else:
                base_value = 0.0

            board = game.board()

            # 2. Iterate Moves
            for move in game.mainline_moves():
                turn = board.turn

                # PERSPECTIVE FLIP: Value must be from current player's POV
                current_value = base_value if turn == chess.WHITE else -base_value

                try:

                    board_layers = board_params(board)

                    # board_parameters returns the flat list of 19 items
                    dense_layers = np.array(dense_params(board), dtype=np.float32)

                    policy = [(move, 1.0)]
                    policy_target = make_policy_target(policy, board)

                    # Serialize and Write
                    writer.write(serialize_example(board_layers, dense_layers, current_value, policy_target))
                    count += 1

                    # Handle Sharding
                    if count % shard_size == 0:
                        writer.close()
                        shard_idx += 1
                        writer = tf.io.TFRecordWriter(
                            f"tfrecords/{tfrecord_prefix}_{shard_idx:03d}.tfrecord"
                        )
                        print(f"Created Shard {shard_idx} ({count} total positions)")

                except Exception as e:
                    print(f"Skipped position due to error: {e}")

                # Move board forward
                board.push(move)

            games_processed += 1
            if games_processed % 1000 == 0:
                print(f"Processed {games_processed} games from {pgn_path}...")

    writer.close()
    print(f"Finished parsing {pgn_path}. Total positions extracted: {count}")


if __name__ == "__main__":
    ## debug_single_game()
    build_tfrecord_from_pgn("data/lichess_games/lichess_elite_2020-07.pgn", "lichess_elite_2020_07", max_games=300_000)
