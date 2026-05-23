import math

import chess


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
    C = 1

    N = node.parent.visit_count if node.parent else 1
    U = C * node.prob * math.sqrt(N) / (1 + node.visit_count)

    return Q + U


def get_policy(root_board, T=1.0):
    total_visits = 0
    prob_dist = []

    if T == 0:
        mx = 0
        mv = None
        for move, child in root_board.children.items():
            if child.visit_count > mx:
                mv = move
                mx = child.visit_count

        return [(mv, 1)]

    else:
        for child in root_board.children.values():
            total_visits += child.visit_count ** (1/T)

        for move, child in root_board.children.items():
            prob_dist.append((move, child.visit_count ** (1/T) / total_visits))

        return prob_dist
