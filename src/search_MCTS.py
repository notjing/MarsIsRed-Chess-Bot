import time
import chess
import math
import scipy.special
import numpy as np
from evaluate import evaluate_board
from utils.data_utils import move_to_index
from utils.math_utils import PUCT
from utils.board_utils import board_params, dense_params
from c_bindings import mcts_exts

BATCH_SIZE = 32
NUM_NODES = 800


def value_to_scalar(raw_value):
    """Map value head to (N, 1) in [-1, 1]. tanh scalar, or WDL [W, D, L] as W - L."""
    arr = np.asarray(raw_value, dtype=np.float32)
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        return (arr[..., 0] - arr[..., 2]).reshape(-1, 1)
    return arr.reshape(-1, 1)


def clear_tree():
    mcts_exts.init_tree(chess.Board().fen())

def policy_index(move, turn):
    r, c, p = move_to_index(move, turn)
    return (r * 8 + c) * 73 + p

def _san(board, move):
    try:
        return board.san(move)
    except ValueError:
        return move.uci()


def get_network_heads(root_board):
    """Raw value/policy heads for the current root, legal moves only."""

    planes = np.expand_dims(mcts_exts.py_board_params(root_board.fen()), axis=0)
    dense = np.expand_dims(mcts_exts.py_dense_params(root_board.fen()), axis=0)
    raw_value, raw_policy = evaluate_board(planes, dense)

    value = float(value_to_scalar(raw_value).reshape(-1)[0])
    policy = scipy.special.softmax(np.asarray(raw_policy, dtype=np.float32).reshape(1, -1), axis=1)[0]

    scored = []
    for move in root_board.legal_moves:
        scored.append((move, float(policy[policy_index(move, root_board.turn)])))
    scored.sort(key=lambda item: item[1], reverse=True)
    legal_mass = sum(prob for _, prob in scored)
    return value, scored, legal_mass


def _compact_moves(board, pairs, top_k=5):
    parts = []
    for move, prob in pairs[:top_k]:
        parts.append(f"{_san(board, move)} {prob * 100:.1f}%")
    return ", ".join(parts)


def _emit_play_info(root_board, best_move, top_k=5):
    """UCI info that lichess-bot surfaces. One string only — python-chess keeps the last."""

    value, scored, legal_mass = get_network_heads(root_board)
    side = "White" if root_board.turn == chess.WHITE else "Black"
    nn_moves = _compact_moves(root_board, scored, top_k)

    raw = mcts_exts.get_root_policy(temperature=1.0)
    mcts_pairs = []
    if raw:
        ranked = sorted(raw, key=lambda item: item[1], reverse=True)[:top_k]
        mcts_pairs = [(chess.Move.from_uci(uci), float(prob)) for uci, prob in ranked]
    mcts_moves = _compact_moves(root_board, mcts_pairs, top_k) if mcts_pairs else "n/a"

    summary = (
        f"[NN] {side} value={value:+.3f} | policy: {nn_moves} "
        f"(legal {legal_mass * 100:.0f}%) || "
        f"[MCTS] played {_san(root_board, best_move)} | visits: {mcts_moves}"
    )

    # score cp is what print_stats shows as Evaluation (divides by 100)
    cp = int(round(value * 100))
    print(f"info score cp {cp}", flush=True)
    print(f"info pv {best_move.uci()}", flush=True)
    print(f"info string {summary}", flush=True)


def search(root_board, time_limit, useTime=False, add_noise=False, verbose=False):
    """ main search func that returns the best move from the given position """

    tmp_board = root_board.copy()

    # checks if the root_board has had any moves made
    if root_board.move_stack:
        # promote the old root to root_board

        missing_moves = []

        reset = True
        if tmp_board.fen() == mcts_exts.get_fen():
            reset = False

        while tmp_board.move_stack and tmp_board.fen() != mcts_exts.get_fen():
            missing_moves.append(tmp_board.pop())
            if tmp_board.fen() == mcts_exts.get_fen():
                reset = False

        if reset:
            mcts_exts.init_tree(root_board.fen())
        else:
            for move in reversed(missing_moves):
                tmp_board.push(move)
                mcts_exts.promote_root(move.uci(), tmp_board.fen())
    else:
        mcts_exts.init_tree(root_board.fen())

    # does a batch of 1 to initialize the root_board (? check if still neccessary)
    board_list, dense_list = mcts_exts.get_leaf_batch(1, [m.uci() for m in root_board.move_stack])

    if board_list:
        batch_board_layers = np.stack(board_list)
        batch_dense_layers = np.stack(dense_list)

        win_probs, policies = evaluate_board(batch_board_layers, batch_dense_layers)
        policies = scipy.special.softmax(policies, axis=1)

        legal_prior_sum = 0.0
        legal_priors = []

        for move in root_board.legal_moves:
            prob = float(policies[0][policy_index(move, root_board.turn)])
            legal_prior_sum += prob
            legal_priors.append((move.uci(), prob))

        legal_priors.sort(key=lambda x: x[1], reverse=True)

        # print("--- Raw NN Legal Priors ---")
        # print(f"Legal prior mass: {legal_prior_sum:.6f}")
        # print(f"Top legal priors: {legal_priors[:10]}")

        win_probs_formatted = value_to_scalar(win_probs)
        policies_formatted = np.array(policies, dtype=np.float32).reshape(len(policies), 4672)

        mcts_exts.expand_and_backprop(win_probs_formatted, policies_formatted)

    # adds noise to the root
    if add_noise:
        mcts_exts.apply_dirichlet_noise(alpha=0.3, epsilon=0.25)

    start_time = time.time()
    nodes_visited = 0
    last_batch_dt = 0.0
    pad = min(0.5, 0.15 * time_limit)
    safe_limit = max(0.05, time_limit - pad)
    deadline = start_time + safe_limit

    mcts_exts.log_visits()

    while True:
        if useTime:
            if time.time() + last_batch_dt > deadline:
                break
        elif nodes_visited >= NUM_NODES:
            break

        batch_start = time.time()
        # gets a batch of leaves
        board_list, dense_list = mcts_exts.get_leaf_batch(BATCH_SIZE, [m.uci() for m in root_board.move_stack])
        nodes_visited += BATCH_SIZE

        if not board_list:
            last_batch_dt = time.time() - batch_start
            continue

        batch_board_layers = np.stack(board_list)
        batch_dense_layers = np.stack(dense_list)

        # evals the batch
        win_probs, policies = evaluate_board(batch_board_layers, batch_dense_layers)
        policies = scipy.special.softmax(policies, axis=1) # logits is True

        # normalises the shape
        win_probs_formatted = value_to_scalar(win_probs)
        policies_formatted = np.array(policies, dtype=np.float32).reshape(len(board_list), 4672)

        # sends it back to backprop w the NN eval
        mcts_exts.expand_and_backprop(win_probs_formatted, policies_formatted)
        last_batch_dt = time.time() - batch_start

    best_move_uci = mcts_exts.get_best_move()
    best_move = chess.Move.from_uci(best_move_uci)

    if verbose:
        try:
            _emit_play_info(root_board, best_move, top_k=5)
        except Exception as exc:
            print(f"info string [NN] failed to dump heads: {exc}", flush=True)

    return best_move


