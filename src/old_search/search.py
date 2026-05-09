import chess.engine
import time
import evaluate as model
from ChessAI.src.old_search.utils import compute_child_hash, compute_zobrist, TT, TTEntry
from move_ordering import score_move

search_deadline = None



# -----------------------------
# Transposition Table
# -----------------------------

nodes = 0
leaf_nodes = 0

real_eval=True

def evaluate(board):
    global leaf_nodes
    leaf_nodes += 1
    engine_start = time.time()
    if real_eval:

        #info = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME))
        val = model.evaluate_board(board)
        #val = info["score"].white().score(mate_score=1000000)

        return val
    else:
        return 0
    engine_end = time.time()
    print(str(engine_end - engine_start))

    #time.sleep(0.001)  # 5 ms
    #return 0

# -----------------------------
# Global to store PV from previous depth
# -----------------------------
pv_move = None


def search(board, depth, alpha, beta, zobrist_hash):
    global nodes, pv_move
    nodes += 1

    # Check for cancellation
    if search_deadline is not None and time.time() >= search_deadline:
        print("TIME IS UP")
        return None, None

    # Transposition table lookup
    entry = TT.get(zobrist_hash)
    if entry and entry.depth >= depth:
        if entry.flag == "EXACT":
            return entry.eval, entry.move
        elif entry.flag == "LOWERBOUND":
            alpha = max(alpha, entry.eval)
        elif entry.flag == "UPPERBOUND":
            beta = min(beta, entry.eval)
        if alpha >= beta:
            return entry.eval, entry.move

    if depth == 0 or board.is_game_over():
        val = evaluate(board)
        return val, None

    best_move = None
    maximizing = board.turn
    best_eval = -1e9 if maximizing else 1e9

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: score_move(board, m, pv_move, zobrist_hash), reverse=True)

    orig_alpha = alpha
    orig_beta = beta

    for move_index, move in enumerate(moves):
        new_hash = compute_child_hash(board, move, zobrist_hash)

        # Late Move Reductions
        reduction = 0
        if (
            depth >= 3
            and move_index >= 3
            and not board.is_capture(move)
            and not board.gives_check(move)
            and move.promotion is None
            and not board.is_check()
        ):
            reduction = 1

        board.push(move)
        if reduction > 0:
            # first search on a small alpha-beta window (can this move do better/worse than alpha/beta)
            if maximizing:
                val, mov = search(board, depth - 1 - reduction, alpha, alpha + 1, new_hash)
            else:
                val, mov = search(board, depth - 1 - reduction, beta - 1, beta, new_hash)

            # not clear if move is bad or good --> research
            if val is not None and alpha < val < beta:
                val, mov = search(board, depth - 1, alpha, beta, new_hash)
        else:
            val, mov = search(board, depth - 1, alpha, beta, new_hash)

        board.pop()

        # Propagate cancellation
        if val is None:
            return None, None

        if maximizing:
            if val > best_eval:
                best_eval = val
                best_move = move
            alpha = max(alpha, val)
        else:
            if val < best_eval:
                best_eval = val
                best_move = move
            beta = min(beta, val)

        if alpha >= beta:
            break

    # Store in TT
    flag = "EXACT"
    if best_eval <= orig_alpha:
        flag = "UPPERBOUND"
    elif best_eval >= orig_beta:
        flag = "LOWERBOUND"
    TT[zobrist_hash] = TTEntry(depth, best_eval, flag, best_move)

    # Update PV
    # if depth == 1:
    #   pv_move = best_move

    return best_eval, best_move


def find_move(board, max_depth):
    global pv_move, search_deadline
    search_deadline = time.time() + 10.0
    #if ctx.timeLeft<10000:
    #    search_deadline = time.time() + 0.5
    #elif ctx.timeLeft<5000:
    #    search_deadline = time.time() + 0.01
    zob_hash = compute_zobrist(board)
    best_eval, best_move = None, None

    pv_move = None

    time_start = time.time()
    #if ctx.timeLeft<30000:
    #    max_depth -=1
    #if ctx.timeLeft<20000:
    #    max_depth -=1

    for depth in range(1, max_depth + 1):
        print(search_deadline - time.time())
        print(f"\n=== Starting depth {depth} ===")
        eval, move = search(board, depth, float('-inf'), float('inf'), zob_hash)

        if eval is None:
            print("Search cancelled due to timeout")
            break

        best_eval, best_move = eval, move
        pv_move = best_move

        print(f"Depth {depth} finished. Best move found: {best_move}\n")

    time_end = time.time()
    elapsed = time_end - time_start
    nps = nodes / elapsed

    print("===================================")
    print(f"Depth: {depth}")
    print(f"Best Move: {best_move}")
    print(f"Eval: {best_eval}")
    print(f"Nodes: {nodes:,}")
    print(f"Leaf Nodes: {leaf_nodes:,}")
    print(f"Time: {elapsed:.4f}s")
    print(f"NPS: {nps:,.0f}")
    print("===================================\n")

    return best_eval, best_move

board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
print(evaluate(board))
