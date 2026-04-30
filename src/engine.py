import sys
import time
import chess
import traceback

# Import the MCTS search logic
import search_MCTS as search


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def parse_time_limit(tokens, turn):
    # 1. Fixed move time
    if "movetime" in tokens:
        idx = tokens.index("movetime") + 1
        return float(tokens[idx]) / 1000.0

    # 2. Dynamic time management
    time_key = "wtime" if turn == chess.WHITE else "btime"
    if time_key in tokens:
        idx = tokens.index(time_key) + 1
        time_left_ms = float(tokens[idx])
        return 20
        # Use 1/20th of remaining time, min 0.1s
        #return max(0.1, (time_left_ms / 1000.0) / 20.0)

    # 3. Default
    return 10.0


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
                limit = parse_time_limit(tokens, board.turn)
                log(f"Searching with time limit: {limit:.2f}s")

                move = search.search(board, limit)

                if move is None:
                    print("bestmove (none)", flush=True)
                else:
                    print(f"bestmove {move.uci()}", flush=True)

            elif cmd == "quit":
                break

        except Exception:
            log(traceback.format_exc())


if __name__ == "__main__":
    main()
