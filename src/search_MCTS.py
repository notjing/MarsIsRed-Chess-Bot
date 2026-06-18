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

BATCH_SIZE = 128
NUM_NODES = 1000

def clear_tree():
    mcts_exts.init_tree(chess.Board().fen())

def search(root_board, time_limit, add_noise):
    """ main search func that returns the best move from the given position """

    # checks if the root_board has had any moves made
    if root_board.move_stack:
        # promote the old root to root_board
        mcts_exts.promote_root(root_board.peek().uci(), root_board.fen())
    else:
        mcts_exts.init_tree(root_board.fen())

    # does a batch of 1 to initialize the root_board (? check if still neccessary)
    board_list, dense_list = mcts_exts.get_leaf_batch(1)

    batch_board_layers = np.stack(board_list)
    batch_dense_layers = np.stack(dense_list)

    win_probs, policies = evaluate_board(batch_board_layers, batch_dense_layers)
    policies = scipy.special.softmax(policies, axis=1)

    win_probs_formatted = np.array(win_probs, dtype=np.float32).reshape(len(win_probs), 1)
    policies_formatted = np.array(policies, dtype=np.float32).reshape(len(policies), 4672)

    mcts_exts.expand_and_backprop(win_probs_formatted, policies_formatted)

    # adds noise to the root
    if add_noise:
        mcts_exts.apply_dirichlet_noise(alpha=0.3, epsilon=0.15)

    start_time = time.time()
    nodes_visited = 0

    safe_limit = max(0.1, time_limit - 0.5)

    while nodes_visited < NUM_NODES:
    # while time.time() - start_time <= safe_limit:
        # gets a batch of leaves
        board_list, dense_list = mcts_exts.get_leaf_batch(BATCH_SIZE)

        batch_board_layers = np.stack(board_list)
        batch_dense_layers = np.stack(dense_list)

        # evals the batch
        win_probs, policies = evaluate_board(batch_board_layers, batch_dense_layers)
        policies = scipy.special.softmax(policies, axis=1) # logits is True

        # normalises the shape
        win_probs_formatted = np.array(win_probs, dtype=np.float32).reshape(len(board_list), 1)
        policies_formatted = np.array(policies, dtype=np.float32).reshape(len(board_list), 4672)

        mcts_exts.expand_and_backprop(win_probs_formatted, policies_formatted)

        nodes_visited += BATCH_SIZE

    # elapsed_time = time.time() - start_time
    # nps = nodes_visited / elapsed_time if elapsed_time > 0 else 0

    # get the visit distribution from the root
    # policy = mcts_exts.get_root_policy(temperature=1)

    # top_moves = sorted(policy, key=lambda x: x[1], reverse=True)

    # print(f"\n--- Search Complete ---")
    # print(f"Nodes Visited: {nodes_visited:,}")
    # print(f"Engine Speed:  {nps:,.0f} NPS")
    # print("Top Candidate Moves:")

    # Print the top 3 moves (or fewer if there aren't 3 legal moves)
    # for i in range(min(3, len(top_moves))):
    #     move_uci, prob = top_moves[i]
    #     print(f"  {i + 1}. {move_uci} ({(prob * 100):.1f}%)")
    # print(f"-----------------------\n")

    best_move_uci = mcts_exts.get_best_move()

    return chess.Move.from_uci(best_move_uci)


