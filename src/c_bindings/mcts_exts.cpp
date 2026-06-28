#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "chess.hpp"
#include "headerFiles/zobristHashing.hpp"
#include "headerFiles/feature_extraction.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

namespace py = pybind11;
using namespace chess;

// Node :)
struct Node {
    int visit_count;
    double value_sum;
    double prob;
    Move move;
    Color turn;
    bool is_expanded;
    u64 hash;

    Node* parent;
    std::vector<Node*> children;

    //initializer
    Node(Node* p = nullptr, double pr = 0.0, Move m = Move::NULL_MOVE, Color t = Color::WHITE, u64 hsh = 0) {
        visit_count = 0;
        value_sum = 0.0;
        prob = pr;
        move = m;
        turn = t;
        is_expanded = false;
        parent = p;
        hash = hsh;
    }

    // destructor
    ~Node() {
        for (Node* child : children) {
            delete child;
        }
    }
};

struct TableEntry {
    u64 hash;
    float winProbabilities;
    std::array<float, 4672> policy;

    TableEntry(){
        hash = 0;
    }

};

const int TABLE_SIZE = 1 << 16;

TableEntry* transTable = new TableEntry[TABLE_SIZE];
Node* TREE_ROOT = nullptr;
std::string ROOT_FEN = "";
std::vector<std::vector<Node*>> batch_paths;

// initializes the ROOT_FEN & TREE_ROOT and creates its node
void init_tree(std::string fen) {
    ROOT_FEN = fen;
    Board board(fen);

    if (TREE_ROOT != nullptr) {
        delete TREE_ROOT;
    }

    u64 initHash = generateHash(board);

    TREE_ROOT = new Node(nullptr, 1.0, Move::NULL_MOVE, board.sideToMove(), initHash);
}


void promoteRoot(std::string moveUci, std::string fen){

    chess::Board board(fen);
    chess::Move playedMove = chess::uci::uciToMove(board, moveUci);

    if(TREE_ROOT == nullptr){
        init_tree(fen);
        return;
    }

    for(auto child : TREE_ROOT->children){
        if(child->move == playedMove){
            Node* tmp = TREE_ROOT;
            TREE_ROOT->children.erase(

            std::remove(TREE_ROOT->children.begin(),
                        TREE_ROOT->children.end(),
                        child),
            TREE_ROOT->children.end()
            );

            TREE_ROOT = child;
            ROOT_FEN = fen;
            delete tmp;
            TREE_ROOT->parent = nullptr;
            return;
        }
    }

    init_tree(fen);

}

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
    int from_c = from_file;
    int to_r = flip ? to_rank : 7 - to_rank;
    int to_c = to_file;

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

void applyEvaluation(std::vector<Node*>& path, float relative_win_prob, const float* policy, chess::Board& board){

    Node* leaf = path.back();
    float win_prob = (leaf->turn == Color::WHITE) ? relative_win_prob : -relative_win_prob;

    if (!leaf->is_expanded) {
        Movelist moves;
        movegen::legalmoves(moves, board);

        if (moves.empty()) {
            if (board.inCheck()) {
                win_prob = (board.sideToMove() == Color::WHITE) ? -1.0 : 1.0;
            } else {
                win_prob = 0.0;
            }
        }
        // Normal Expansion
        else {
            for (const Move& m : moves) {
                int policy_idx = move_to_index(m, board.sideToMove());
                u64 newHash = updateZobristMove(leaf->hash, m, board);
                Node* child = new Node(leaf, policy[policy_idx], m, ~board.sideToMove(), newHash);

                leaf->children.push_back(child);
            }
        }

        leaf->is_expanded = true;

    }

    for (Node* n : path) {
        double v_loss = (n->turn == Color::BLACK) ? -0.25 : 0.25;
        n->value_sum -= v_loss; // Remove virtual loss
        n->value_sum += win_prob; // Add real evaluation
    }
}

// gets a batch of leaves from the TREE_ROOT
py::tuple get_leaf_batch(int batch_size) {
    py::list board_features;
    py::list dense_features;

    batch_paths.clear();
    Board root_board(ROOT_FEN);

    int cacheHits = 0;

    // loops starting from the TREE_ROOT until you hit a leaf
    while (cacheHits < batch_size) {
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

        size_t idx = current->hash & (TABLE_SIZE - 1);

        // adds virtual loss to it
        for (Node* n : current_path) {
            n->visit_count += 1;
            double v_loss = (n->turn == Color::BLACK) ? -0.25 : 0.25;
            n->value_sum += v_loss;
        }

        cacheHits++;

        // not a collision
        if(transTable[idx].hash == current->hash) {
            applyEvaluation(current_path, transTable[idx].winProbabilities, transTable[idx].policy.data(), board);

            continue;
        }

        batch_paths.push_back(current_path);

        board_features.append(boardParams(board));
        dense_features.append(denseParams(board));
    }

    return py::make_tuple(board_features, dense_features);
}


void expand_and_backprop(py::array_t<float> win_probs, py::array_t<float> policies) {

    auto win_probs_buf = win_probs.unchecked<2>();

    // loops through the paths
    for (int i = 0; i < batch_paths.size(); i++) {
        std::vector<Node*> path = batch_paths[i];
        Node* leaf = path.back();

        // reconstructs the board
        Board board(ROOT_FEN);
        for (size_t j = 1; j < path.size(); ++j) {
            board.makeMove(path[j]->move);
        }

        float relative_win_prob = win_probs_buf(i, 0);
        const float* policy = policies.data(i,0);

        applyEvaluation(path, relative_win_prob, policy, board);

        size_t idx = leaf->hash & (TABLE_SIZE - 1);
        transTable[idx].hash = leaf->hash;
        transTable[idx].winProbabilities = relative_win_prob;
        std::copy(policy, policy + 4672, transTable[idx].policy.begin());

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

void apply_dirichlet_noise(double alpha = 0.3, double epsilon = 0.25) {
    // If the root doesn't exist or hasn't been expanded yet, we can't add noise
    if (TREE_ROOT == nullptr || TREE_ROOT->children.empty()) return;

    // Set up the random number generator and the Gamma distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<double> gamma_dist(alpha, 1.0);

    std::vector<double> noise;
    double sum = 0.0;

    // 1. Generate a Gamma sample for every single legal move
    for (size_t i = 0; i < TREE_ROOT->children.size(); ++i) {
        double n = gamma_dist(gen);
        noise.push_back(n);
        sum += n;
    }

    // 2. Normalize the samples to create the Dirichlet distribution and apply it
    if (sum > 1e-8) { // Safety check to prevent division by zero
        for (size_t i = 0; i < TREE_ROOT->children.size(); ++i) {
            double normalized_noise = noise[i] / sum;

            // The standard AlphaZero blend formula: (1 - epsilon) * prior + epsilon * noise
            TREE_ROOT->children[i]->prob = (1.0 - epsilon) * TREE_ROOT->children[i]->prob + (epsilon * normalized_noise);
        }
    }
}

py::list get_root_policy(double temperature = 1.0) {
    py::list policy;
    if (TREE_ROOT == nullptr || TREE_ROOT->children.empty()) return policy;

    double total_weight = 0.0;
    std::vector<double> weights;

    for (Node* child : TREE_ROOT->children) {
        double weight = (temperature < 1e-3) ?
                        ((child->visit_count == TREE_ROOT->children[0]->visit_count) ? 1.0 : 0.0) :
                        std::pow(child->visit_count, 1.0 / temperature);
        weights.push_back(weight);
        total_weight += weight;
    }

    if (temperature < 1e-3) {
        int best_idx = 0;
        int max_visits = -1;
        for (size_t i = 0; i < TREE_ROOT->children.size(); ++i) {
            if (TREE_ROOT->children[i]->visit_count > max_visits) {
                max_visits = TREE_ROOT->children[i]->visit_count;
                best_idx = i;
            }
        }
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = (i == best_idx) ? 1.0 : 0.0;
        }
        total_weight = 1.0;
    }

    for (size_t i = 0; i < TREE_ROOT->children.size(); ++i) {
        std::string move_uci = uci::moveToUci(TREE_ROOT->children[i]->move);
        double prob = (total_weight > 0) ? (weights[i] / total_weight) : 0.0;
        policy.append(py::make_tuple(move_uci, prob));
    }

    return policy;
}

// Add this above PYBIND11_MODULE
void print_root_stats() {
    if (TREE_ROOT == nullptr || TREE_ROOT->children.empty()) {
        std::cout << "Tree root is empty." << std::endl;
        return;
    }

    std::cout << "\n--- Root Node Children Stats ---" << std::endl;
    for (Node* child : TREE_ROOT->children) {
        double q_val = (child->visit_count > 0) ? (child->value_sum / child->visit_count) : 0.0;

        std::cout << "Move: " << chess::uci::moveToUci(child->move)
                  << " | N (Visits): " << child->visit_count
                  << " | W (Total Val): " << child->value_sum
                  << " | Q (Avg Val): " << q_val
                  << " | P (Policy): " << child->prob << std::endl;
    }
    std::cout << "--------------------------------\n" << std::endl;
}

// sends these back to python
PYBIND11_MODULE(mcts_exts, m) {
    m.def("init_tree", &init_tree);
    m.def("get_leaf_batch", &get_leaf_batch);
    m.def("expand_and_backprop", &expand_and_backprop);
    m.def("get_best_move", &get_best_move);
    m.def("free_tree", &free_tree);
    m.def("apply_dirichlet_noise", &apply_dirichlet_noise,
          py::arg("alpha") = 0.3,
          py::arg("epsilon") = 0.25);
    m.def("get_root_policy", &get_root_policy, py::arg("temperature") = 1.0);
    m.def("promote_root", &promoteRoot);
    m.def("print_root_stats", &print_root_stats);
}
