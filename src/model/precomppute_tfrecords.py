import os
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
import io

from utils.board_utils import board_parameters, square_control, makeboards, get_mapped_coords

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

                    planes = makeboards(board)

                    s_control = square_control(board)

                    ep_grid = np.zeros((8, 8), dtype=np.float32)
                    if board.ep_square is not None:
                        flip = (board.turn == chess.BLACK)
                        r, c = get_mapped_coords(board.ep_square, flip)
                        ep_grid[r][c] = 1.0

                    # Combine into 25 planes
                    board_layers = np.array(planes + s_control + [ep_grid], dtype=np.float32)

                    board_layers = np.transpose(board_layers, (1, 2, 0))

                    # board_parameters returns the flat list of 19 items
                    dense_layers = np.array(board_parameters(board), dtype=np.float32)

                    policy_target = make_policy_target(move, board)

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
    build_tfrecord_from_pgn("data/lichess_games/lichess_elite_2020-04.pgn", "lichess_elite_20_04", max_games=200000)
