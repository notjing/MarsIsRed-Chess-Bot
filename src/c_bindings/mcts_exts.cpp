#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "chess.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace py = pybind11;
using namespace chess;

// Node
struct Node {
    int visit_count;
    double value_sum;
    double prob;
    Move move;
    Color turn;
    bool is_expanded;

    Node* parent;
    std::vector<Node*> children;

    //initializer
    Node(Node* p = nullptr, double pr = 0.0, Move m = Move::NULL_MOVE, Color t = Color::WHITE) {
        visit_count = 0;
        value_sum = 0.0;
        prob = pr;
        move = m;
        turn = t;
        is_expanded = false;
        parent = p;
    }

    // destructor
    ~Node() {
        for (Node* child : children) {
            delete child;
        }
    }
};

Node* TREE_ROOT = nullptr;
std::string ROOT_FEN = "";
std::vector<std::vector<Node*>> batch_paths;

// this is the same thing as the one in python
int move_to_index(chess::Move move, chess::Color turn) {
    bool flip = (turn == chess::Color::BLACK);

    // 1. Get raw rank (0-7) and file (0-7) from chess.hpp
    int from_rank = move.from().rank();
    int from_file = move.from().file();
    int to_rank = move.to().rank();
    int to_file = move.to().file();

    // 2. Mapped coords (Flip for Black perspective)
    int from_r = flip ? from_rank : 7 - from_rank;
    int from_c = flip ? 7 - from_file : from_file;
    int to_r = flip ? to_rank : 7 - to_rank;
    int to_c = flip ? 7 - to_file : to_file;

    // 3. Deltas
    int dc = to_c - from_c;
    int dr = from_r - to_r;

    int plane = 0;

    // 4. Underpromotions
    if (move.promotionType() != chess::PieceType::NONE && move.promotionType() != chess::PieceType::QUEEN) {
        int promo_idx = 0;
        if (move.promotionType() == chess::PieceType::KNIGHT) promo_idx = 0;
        else if (move.promotionType() == chess::PieceType::BISHOP) promo_idx = 1;
        else if (move.promotionType() == chess::PieceType::ROOK) promo_idx = 2;

        plane = 64 + ((dc + 1) * 3) + promo_idx;
    }
    // 5. Normal Moves
    else {
        // Knight moves array
        int knight_dirs[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        bool is_knight_move = false;

        for (int i = 0; i < 8; ++i) {
            if (dc == knight_dirs[i][0] && dr == knight_dirs[i][1]) {
                plane = 56 + i;
                is_knight_move = true;
                break;
            }
        }

        // Queen moves array
        if (!is_knight_move) {
            int queen_dirs[8][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
            int distance = std::max(std::abs(dc), std::abs(dr));

            // SAFETY FIX: Prevent division by zero
            int dir_u_c = (distance > 0) ? (dc / distance) : 0;
            int dir_u_r = (distance > 0) ? (dr / distance) : 0;

            int dir_idx = 0;
            for (int i = 0; i < 8; ++i) {
                if (dir_u_c == queen_dirs[i][0] && dir_u_r == queen_dirs[i][1]) {
                    dir_idx = i;
                    break;
                }
            }
            plane = (dir_idx * 7) + std::max(0, distance - 1);
        }
    }

    // 6. Flatten the 3D index (x, y, p) into a 1D array index (0 to 4671)
    return (from_r * 8 + from_c) * 73 + plane;
}

//also the same PUCT function
double calculate_PUCT(Node* parent, Node* child) {
    double q_value = 0.0;
    if (child->visit_count > 0) {
        q_value = child->value_sum / child->visit_count;

        if (parent->turn == Color::BLACK) q_value = -q_value;
    }

    double C = 1.25;

    double u_value = C * child->prob * std::sqrt(parent->visit_count) / (1.0 + child->visit_count);
    return q_value + u_value;
}

// initializes the ROOT_FEN and creates its node
void init_tree(std::string fen) {
    ROOT_FEN = fen;
    Board board(fen);

    if (TREE_ROOT != nullptr) {
        delete TREE_ROOT;
    }

    TREE_ROOT = new Node(nullptr, 1.0, Move::NULL_MOVE, board.sideToMove());
}


// gets a batch of leaves from the TREE_ROOT
std::vector<std::string> get_leaf_batch(int batch_size) {
    std::vector<std::string> leaf_fens;
    batch_paths.clear();

    Board root_board(ROOT_FEN);

    //loops starting from the TREE_ROOT until you hit a leaf
    while (leaf_fens.size() < batch_size) {
        Node* current = TREE_ROOT;
        std::vector<Node*> current_path = {current};
        Board board = root_board;

        // selects the leaf
        while (current->is_expanded && !current->children.empty()) {
            Node* best_child = nullptr;
            double best_puct = -9999999.0;

            for (Node* child : current->children) {
                double score = calculate_PUCT(current, child);
                if (score > best_puct) {
                    best_puct = score;
                    best_child = child;
                }
            }

            current = best_child;
            board.makeMove(current->move);
            current_path.push_back(current);
        }

        // adds virtual loss to it
        for (Node* n : current_path) {
            n->visit_count += 1;
            double v_loss = (n->turn == Color::BLACK) ? -1.0 : 1.0;
            n->value_sum += v_loss;
        }

        batch_paths.push_back(current_path);

        leaf_fens.push_back(board.getFen());
    }

    return leaf_fens;
}


void expand_and_backprop(py::array_t<float> win_probs, py::array_t<float> policies) {

    // doesn't do any safety checks when accessing memory, 2 is the dimension
    auto win_probs_buf = win_probs.unchecked<2>();
    auto policies_buf = policies.unchecked<2>();

    for (int i = 0; i < batch_paths.size(); ++i) {
        std::vector<Node*> path = batch_paths[i];
        Node* leaf = path.back();

        // Read directly from the 2D buffer: win_probs_buf(batch_index, 0)
        float win_prob = win_probs_buf(i, 0);

        // Reconstruct the board state at the leaf
        Board board(ROOT_FEN);
        for (size_t j = 1; j < path.size(); ++j) {
            board.makeMove(path[j]->move);
        }

        if (!leaf->is_expanded) {
            Movelist moves;
            movegen::legalmoves(moves, board);

            // Handle Checkmates/Draws
            if (moves.empty()) {
                if (board.inCheck()) { // Someone got mated
                    win_prob = (board.sideToMove() == Color::WHITE) ? -1.0 : 1.0;
                } else { // Draw
                    win_prob = 0.0;
                }
            }
            // Normal Expansion
            else {
                for (const Move& m : moves) {
                    int policy_idx = move_to_index(m, board.sideToMove());

                    // Read directly from the 2D buffer: policies_buf(batch_index, policy_index)
                    double move_prob = policies_buf(i, policy_idx);

                    Node* child = new Node(leaf, move_prob, m, ~board.sideToMove());
                    leaf->children.push_back(child);
                }
                leaf->is_expanded = true;
            }
        }

        // 3. BACKPROPAGATE
        for (Node* n : path) {
            // MATCH THE SIGNS EXACTLY WITH get_leaf_batch
            double v_loss = (n->turn == Color::BLACK) ? -1.0 : 1.0;
            n->value_sum -= v_loss; // Remove virtual loss
            n->value_sum += win_prob; // Add real evaluation
        }
    }
}


std::string get_best_move() {
    if (TREE_ROOT == nullptr || TREE_ROOT->children.empty()) return "0000";

    Node* best_child = nullptr;
    int max_visits = -1;

    for (Node* child : TREE_ROOT->children) {
        if (child->visit_count > max_visits) {
            max_visits = child->visit_count;
            best_child = child;
        }
    }

    return uci::moveToUci(best_child->move);
}

void free_tree() {
    if (TREE_ROOT != nullptr) {
        delete TREE_ROOT;
        TREE_ROOT = nullptr;
    }
}

PYBIND11_MODULE(mcts_ext, m) {
    m.def("init_tree", &init_tree);
    m.def("get_leaf_batch", &get_leaf_batch);
    m.def("expand_and_backprop", &expand_and_backprop);
    m.def("get_best_move", &get_best_move);
    m.def("free_tree", &free_tree);
}
