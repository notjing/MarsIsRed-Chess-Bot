import os
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
import io

from model.createparams import board_parameters, square_control, makeboards, get_mapped_coords

os.makedirs("tfrecords", exist_ok=True)


def move_to_index(move, turn):
    QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    KNIGHT_DIRS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    UNDER_PROMOS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    flip = (turn == chess.BLACK)

    from_r, from_c = get_mapped_coords(move.from_square, flip)
    to_r, to_c = get_mapped_coords(move.to_square, flip)

    dc = to_c - from_c
    dr = from_r - to_r

    if move.promotion and move.promotion != chess.QUEEN:
        promo_idx = UNDER_PROMOS.index(move.promotion)
        plane = 64 + ((dc + 1) * 3) + promo_idx
        return from_r, from_c, plane

    if (dc, dr) in KNIGHT_DIRS:
        plane = 56 + KNIGHT_DIRS.index((dc, dr))
        return from_r, from_c, plane

    distance = max(abs(dc), abs(dr))
    direction_u = (dc // distance, dr // distance)
    dir_idx = QUEEN_DIRS.index(direction_u)

    plane = (dir_idx * 7) + (distance - 1)
    return from_r, from_c, plane


def make_policy_target(move, board):
    # Initialize a blank 8x8x73 tensor
    target = np.zeros((8, 8, 73), dtype=np.float32)

    try:
        x, y, p = move_to_index(move, board.turn)
        target[x, y, p] = 1.0
    except (ValueError, IndexError):
        pass

    return target


def float_feature(x):
    """Converts a numpy array into a TF FloatList Feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))


def serialize_example(board_input, extra_input, eval_value, policy_target):
    """
    Converts features and dual-head targets into the TFRecord file format.
    """
    feature = {
        "board": float_feature(board_input.flatten()),
        "extra": float_feature(extra_input),
        "eval": float_feature([eval_value]),
        "policy": float_feature(policy_target.flatten())  # NEW: 4672-length array
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


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


def count_games_in_pgn(file_path):
    """
    Counts the number of games in a PGN file by looking for the standard
    starting tag. This is hundreds of times faster than parsing the chess logic.
    """
    count = 0

    print(f"Counting games in {file_path}...")

    # errors='ignore' prevents crashes if there's a weird character in a player's name
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Every new game in a standard PGN starts with the Event tag
            if line.startswith('[Event '):
                count += 1

    print(f"Total Games in {file_path}: {count}")
    return count

def debug_single_game():
    # A short 4-move Scholar's Mate game
    sample_pgn = (
        '[Event "Debug Check and Material"]\n'
        '[Result "1-0"]\n'
        '\n'
        '1. e4 d5 2. exd5 Nf6 3. Bb5+ c6 4. dxc6 Nxc6 5. Bxc6+ bxc6 1-0\n'
    )
    pgn_handle = io.StringIO(sample_pgn)
    game = chess.pgn.read_game(pgn_handle)

    # 1. Base Game Value
    result = game.headers.get("Result")
    base_value = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0

    board = game.board()
    print("=== STARTING DEBUG RUN ===")

    for i, move in enumerate(game.mainline_moves()):
        turn = board.turn
        current_value = base_value if turn == chess.WHITE else -base_value

        print(f"\n--- Move {i + 1}: {board.san(move)} ({move.uci()}) ---")
        print(f"Perspective: {'White' if turn else 'Black'} to move")
        print(f"Target Value (Y_val): {current_value}")

        # --- X1: CNN Spatial Features ---
        planes = makeboards(board)  # 12 layers
        s_control = square_control(board)  # 12 layers

        ep_grid = np.zeros((8, 8), dtype=np.float32)
        if board.ep_square is not None:
            flip = (board.turn == chess.BLACK)
            r, c = get_mapped_coords(board.ep_square, flip)
            ep_grid[r][c] = 1.0

        # Combine into 25 planes and transpose
        board_layers = np.array(planes + s_control + [ep_grid], dtype=np.float32)
        board_layers = np.transpose(board_layers, (1, 2, 0))

        print(f"Board Input Shape (X1): {board_layers.shape}  --> Expected: (8, 8, 25)")

        # board_parameters returns a single flat list of 19 values
        dense_features = board_parameters(board)
        dense_layers = np.array(dense_features, dtype=np.float32)

        print(f"Dense Input Shape (X2): {dense_layers.shape}   --> Expected: (19,)")
        # Print rounded values for readability
        print(f"Dense Values: {[round(x, 2) for x in dense_layers]}")

        # --- Y: Policy Target ---
        policy_target = make_policy_target(move, board)
        print(f"Policy Target Shape (Y_pol): {policy_target.shape} --> Expected: (8, 8, 73)")

        # Find where the 1.0 is in the policy target
        activated_indices = np.where(policy_target == 1.0)
        if len(activated_indices[0]) > 0:
            # FIXED: Properly unpacking the arrays returned by np.where
            x = activated_indices[0][0]
            y = activated_indices[1][0]
            p = activated_indices[2][0]
            print(f"Policy Activated at [x={x}, y={y}, plane={p}] for move {move.uci()}")
        else:
            print("WARNING: Policy target is all zeros. Move encoding failed.")

            # FIXED: Pushing the move to the board so the next loop has the correct position
        board.push(move)

if __name__ == "__main__":
    ## debug_single_game()
    build_tfrecord_from_pgn("data/lichess_games/lichess_elite_2020-04.pgn", "lichess_elite_20_04", max_games=200000)
