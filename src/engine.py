import sys
import time
import chess
import traceback

# Import the MCTS search logic
import search_MCTS as search


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _uci_seconds(tokens, key):
    try:
        return float(tokens[tokens.index(key) + 1]) / 1000.0
    except (ValueError, IndexError):
        return None


def allocate_time(T, I, board):
    """Rising spend: cheap opening, more in the ending, never dump the clock."""
    fullmove = board.fullmove_number
    pieces = len(board.piece_map())

    if fullmove <= 12:
        moves_left = 32
    elif fullmove <= 30:
        moves_left = 24
    elif pieces > 10:
        moves_left = 18
    else:
        moves_left = 14
    moves_left += 8

    if fullmove <= 12:
        phase = 0.45
    elif pieces <= 14:
        phase = 1.50
    elif fullmove <= 22:
        phase = 0.80
    else:
        phase = 1.10

    think = (T / moves_left + 0.8 * I) * phase
    think = min(think, 0.20 * T)
    think = min(think, max(0.05, T - max(0.4, 0.08 * T)))

    if T < 1.5:
        think = 0.10
    elif T < 4.0:
        think = min(think, 0.12 * T)

    if fullmove <= 12 and T > 4.0:
        think = max(think, 0.5)
    elif T >= 1.5:
        think = max(think, 0.15)

    return min(think, max(0.05, T - 0.05))


def parse_time_limit(tokens, board):
    movetime = _uci_seconds(tokens, "movetime")
    if movetime is not None:
        return movetime

    time_key = "wtime" if board.turn == chess.WHITE else "btime"
    inc_key = "winc" if board.turn == chess.WHITE else "binc"
    remaining = _uci_seconds(tokens, time_key)
    if remaining is None:
        return 1.0

    increment = _uci_seconds(tokens, inc_key) or 0.0
    return allocate_time(remaining, increment, board)


def warmup():
    """
    Forces TensorFlow to compile graphs.
    """
    log(">> WARMING UP ENGINE (Compiling TensorFlow)... Please wait...")
    dummy_board = chess.Board()
    try:
        # Run a tiny search to trigger JIT compilation
        search.search(dummy_board, time_limit=1.0)
        log(">> WARMUP COMPLETE. Engine is ready.")
    except Exception as e:
        log(f">> Warmup failed: {e}")


# -------------------------
# UCI Main Loop
# -------------------------
board = chess.Board()


def main():
    global board

    # Track if we have warmed up yet
    is_warmed_up = False

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                continue

            line = line.strip()
            tokens = line.split()
            if not tokens: continue

            cmd = tokens[0]

            if cmd == "uci":
                print("id name MarsIsRed MCTS")
                print("id author RedIsMars")
                print("uciok", flush=True)

            elif cmd == "isready":
                # DO THE WARMUP HERE
                # The GUI/Bot is waiting for "readyok", so it's safe to block now.
                if not is_warmed_up:
                    warmup()
                    is_warmed_up = True
                print("readyok", flush=True)

            elif cmd == "position":
                if "startpos" in tokens:
                    board = chess.Board()
                    if "moves" in tokens:
                        moves_idx = tokens.index("moves") + 1
                        for move in tokens[moves_idx:]:
                            board.push_uci(move)
                elif "fen" in tokens:
                    fen_idx = tokens.index("fen") + 1
                    fen_parts = tokens[fen_idx:fen_idx + 6]
                    board = chess.Board(" ".join(fen_parts))
                    if "moves" in tokens:
                        moves_idx = tokens.index("moves") + 1
                        for move in tokens[moves_idx:]:
                            board.push_uci(move)


            elif cmd == "go":

                limit = parse_time_limit(tokens, board)

                log(f"Searching with time limit: {limit:.2f}s")

                try:

                    move = search.search(board, limit, True, False, verbose=True)

                    if move is None:
                        # Fallback if no moves found

                        move = list(board.legal_moves)[0]

                        print(f"info string EMERGENCY FALLBACK (MCTS returned None)", flush=True)

                    print(f"bestmove {move.uci()}", flush=True)


                except Exception as e:


                    error_str = traceback.format_exc()

                    for err_line in error_str.split('\n'):

                        if err_line.strip():
                            print(f"CRASH: {err_line}", flush=True)


                    import random

                    fallback_move = random.choice(list(board.legal_moves))

                    print(f"bestmove {fallback_move.uci()}", flush=True)

            elif cmd == "quit":
                break

        except Exception:
            log(traceback.format_exc())


if __name__ == "__main__":
    main()
