import time
import chess
import math
from evaluate import evaluate_board
from search_heuristic import get_heuristic_policy

BATCH_SIZE = 256

class Node:
    def __init__(self, state, parent=None, prob=None, move=None):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.state = state
        self.prob = prob # probability of this move being chosen
        self.parent = parent
        self.is_expanded = False # have we explored this yet?
        self.move = move


def PUCT(node):
    # calculates the probability of choosing this move, balancing exploration and exploitation
    Q = 0
    if node.visit_count > 0:
        Q = node.value_sum / node.visit_count

    C = 2.0
    parent_visits = node.parent.visit_count if node.parent else 1
    sqrt_visits = math.sqrt(max(1, parent_visits))

    U = C * node.prob * sqrt_visits / (1 + node.visit_count)

    return Q + U


def select_leaf(node):
    # continues to go down the tree w highest PUCT until we reach a leaf node
    current = node
    path = [current]

    while current.is_expanded:
        if not current.children:
            return current, path

        is_white = (current.state.turn == chess.WHITE)

        best_move = max(
            current.children,
            key=lambda m: PUCT(current.children[m])
        )
        current = current.children[best_move]
        path.append(current)

    return current, path


def search(root_board, time_limit):
    root_node = Node(state=root_board, parent=None, prob=1.0)

    # todo: update heuristic policy into taking from NN that generates prob dist
    # creates nodes for all of its children
    for move, prob in get_heuristic_policy(root_board):
        next_state = root_board.copy()
        next_state.push(move)
        root_node.children[move] = Node(state=next_state, parent=root_node, prob=prob, move=move)
    root_node.is_expanded = True

    start_time = time.time()
    nodes_visited = 0

    while time.time() - start_time <= time_limit:
        leaf_nodes = []
        paths = []

        # runs until batch is complete or time up
        while len(leaf_nodes) < BATCH_SIZE and time.time() - start_time <= time_limit:
            leaf, path = select_leaf(root_node)

            # handles terminal nodes
            if leaf.state.is_game_over():
                result = leaf.state.outcome()
                if result.winner == chess.WHITE:
                    val = 1.0
                elif result.winner == chess.BLACK:
                    val = -1.0
                else:
                    val = 0.0

                # val is from whites perspective, backprop with correct signs
                for node in path:
                    node.visit_count += 1
                    # each node stores value from the perspective of who moved to reach it
                    if node.state.turn == chess.WHITE:
                        node.value_sum -= val
                    else:
                        node.value_sum += val
                continue

            # apply virtual loss
            for node in path:
                node.visit_count += 1
                node.value_sum -= 1.0

            leaf_nodes.append(leaf)
            paths.append(path)

        if not leaf_nodes:
            continue

        boards = [n.state for n in leaf_nodes]
        values = evaluate_board(boards)  # returns value from current player's perspective

        for i, leaf in enumerate(leaf_nodes):
            val = values[i]  # Value from the perspective of leaf.state.turn
            path = paths[i]

            # convert to White's perspective
            if leaf.state.turn == chess.BLACK:
                val = -val
            # now val is from White's perspective

            # expand leaf
            if not leaf.is_expanded:
                for move, prob in get_heuristic_policy(leaf.state):
                    next_state = leaf.state.copy()
                    next_state.push(move)
                    leaf.children[move] = Node(state=next_state, parent=leaf, prob=prob, move=move)
                leaf.is_expanded = True

            # backpropagate (val is from White's perspective)
            for node in path:
                node.value_sum += 1.0  # Remove virtual loss

                # Add real value from the perspective of who chose this node
                if node.parent is None or node.parent.state.turn == chess.WHITE:
                    node.value_sum += val
                else:
                    node.value_sum -= val

            nodes_visited += 1

    if not root_node.children:
        return None

    best_move = max(
        root_node.children.keys(),
        key=lambda m: root_node.children[m].visit_count
    )

    # Debug: print top moves
    sorted_moves = sorted(root_node.children.items(),
                          key=lambda x: x[1].visit_count,
                          reverse=True)[:5]
    print(f"Nodes: {nodes_visited}, Best: {best_move}, Visits: {root_node.children[best_move].visit_count}")
    for move, node in sorted_moves:
        avg_val = node.value_sum / node.visit_count if node.visit_count > 0 else 0
        print(f"  {move}: visits={node.visit_count}, avg_value={avg_val:.3f}, prior={node.prob:.3f}")

    return best_move



