import time
import chess
import math
from evaluate import evaluate_board
from model.precomppute_tfrecords import move_to_index

BATCH_SIZE = 128

# the current node

TREE_ROOT = None


class Node:
    """ node """

    def __init__(self, parent=None, prob=None, move=None, turn=None):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.prob = prob
        self.parent = parent
        self.is_expanded = False
        self.move = move
        self.turn = turn


def clear_tree():
    global TREE_ROOT
    TREE_ROOT = None


def PUCT(node):
    """ evaluates the node using PUCT """
    # Q is avg value of past simuations using this move
    # P is the probability from the NN (not impl. yet
    # N is the visit count of the parent node
    # C is just a constant to balance exploitation vs exploration

    Q = 0
    if node.visit_count > 0:
        Q = node.value_sum / node.visit_count

    if node.parent and node.parent.turn == chess.BLACK:
        Q = -Q

    # when c is higher, it increases exploration
    C = 3.5

    N = node.parent.visit_count if node.parent else 1
    U = C * node.prob * math.sqrt(N) / (1 + node.visit_count)

    return Q + U


def select_leaf(node, board):
    """ continues going down the tree until we find the best leaf node (unexplored child)"""
    current = node
    path = [current]

    # loops as long as node is unexplored
    while current.is_expanded and current.children:
        best_move = max(
            current.children,
            key=lambda m: PUCT(current.children[m])
        )
        current = current.children[best_move]
        board.push(best_move)
        path.append(current)

    return current, path


def search(root_board, time_limit):
    """ main search func that returns the best move from the given position """
    global TREE_ROOT

    # if TREE_ROOT is a move behind the new board then make the move
    if TREE_ROOT is not None and root_board.move_stack:
        last_move = root_board.peek()
        if last_move in TREE_ROOT.children:
            TREE_ROOT = TREE_ROOT.children[last_move]
            TREE_ROOT.parent = None
        else:
            TREE_ROOT = Node(parent=None, prob=1.0, turn=root_board.turn)
    # otherwise just make a new node
    else:
        TREE_ROOT = Node(parent=None, prob=1.0, turn=root_board.turn)

    start_time = time.time()
    nodes_visited = 0

    while time.time() - start_time <= time_limit:
        batch_nodes = []
        batch_boards = []
        batch_paths = []

        # batches the positions to efficiently eval them
        while len(batch_nodes) < BATCH_SIZE and time.time() - start_time <= time_limit:
            leaf, path = select_leaf(TREE_ROOT, root_board)

            if root_board.is_game_over():
                res = root_board.outcome()
                val = 1.0 if res.winner == chess.WHITE else -1.0 if res.winner == chess.BLACK else 0.0

                # increments visit and values of each node in the path
                for node in path:
                    node.visit_count += 1
                    node.value_sum += val

                # empties out root_board
                for _ in range(len(path) - 1):
                    root_board.pop()
                continue

            for node in path:
                node.visit_count += 1
                # virtual loss to keep batching from sending the same board
                v_loss = -1.0 if node.turn == chess.BLACK else 1.0
                node.value_sum += v_loss

            batch_nodes.append(leaf)
            batch_boards.append(root_board.copy())
            batch_paths.append(path)

            for _ in range(len(path) - 1):
                root_board.pop()

        if not batch_nodes:
            print("no nodes in the batch bruh")
            continue

        win_probs, policies = evaluate_board(batch_boards)

        # goes through all the batch_nodes
        for i, leaf_node in enumerate(batch_nodes):
            win_prob = win_probs[i][0]
            policy = policies[i]
            board = batch_boards[i]
            path = batch_paths[i]

            if not leaf_node.is_expanded:
                # loops through all the children and creates nodes and assigns the probability for them
                for move in list(board.legal_moves):
                    prob = policy[move_to_index(move, board.turn)]
                    leaf_node.children[move] = Node(parent=leaf_node, prob=prob, move=move, turn=not board.turn)
                leaf_node.is_expanded = True


            # removes the virtual loss
            for node in path:
                v_loss = -1.0 if node.turn == chess.BLACK else 1.0
                node.value_sum -= v_loss
                node.value_sum += win_prob

            nodes_visited += 1

    if not TREE_ROOT.children:
        return None

    best_move = max(TREE_ROOT.children.keys(), key=lambda m: TREE_ROOT.children[m].visit_count)

    sorted_moves = sorted(TREE_ROOT.children.items(), key=lambda x: x[1].visit_count, reverse=True)[:5]

    print(f"Nodes: {nodes_visited} NPS: {int(nodes_visited / (time.time() - start_time))}")
    for move, node in sorted_moves:
        avg_v = node.value_sum / node.visit_count if node.visit_count > 0 else 0
        print(f" {move}: visits={node.visit_count}, white_val={avg_v:.3f}, prior={node.prob:.3f}")

    print(f"bestmove {best_move}", flush=True)
    return best_move
