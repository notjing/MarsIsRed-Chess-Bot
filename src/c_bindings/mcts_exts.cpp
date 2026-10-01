#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "chess.hpp"
#include "headerFiles/zobristHashing.hpp"
#include "headerFiles/feature_extraction.hpp"
#include "headerFiles/node.hpp"
#include "sequentialHalving.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <cstring>

namespace py = pybind11;
using namespace chess;

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

SequentialHalving seqHalve;

void logVisits(){
    if(TREE_ROOT == nullptr) std::cout << "TREE_ROOT is null" << "\n";
    else std::cout << "ROOT VISITS: " << TREE_ROOT->visit_count << "\n";
}

std::string getFen(){
    return ROOT_FEN;
}

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

/*
    moveUCI is the move that we want to promote onto the current board
    fen is the current state of the board --> when called at the beginning of search,
*/
void promoteRoot(std::string moveUci, std::string fen){

    seqHalve.reset();
    if(TREE_ROOT == nullptr){
        init_tree(fen);
        return;
    }

    // looks through the children of TREE_ROOT to find the move and promote it w/o reiniting the whole thing
    for(auto child : TREE_ROOT->children){
        if(uci::moveToUci(child->move) == moveUci){
            Node* tmp = TREE_ROOT;
            TREE_ROOT->children.erase(
                std::remove(TREE_ROOT->children.begin(), TREE_ROOT->children.end(),child),
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

    if (move.typeOf() == chess::Move::CASTLING) {
        const bool kingSide = move.to() > move.from();
        const auto kingTo =
            chess::Square::castling_king_square(kingSide, turn);

        to_rank = kingTo.rank();
        to_file = kingTo.file();
    }

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
    if (move.typeOf() == chess::Move::PROMOTION && move.promotionType() != chess::PieceType::QUEEN) {
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
double calculate_PUCT(Node* parent, Node* child, double logitSum, double maxLogit) {
    double q_value = 0.0;
    if (child->visit_count > 0) {
        q_value = child->value_sum / child->visit_count;

        if (parent->turn == Color::BLACK) q_value = -q_value;
    }

    double C = 1.25;

    double prob = std::exp(child->prob - maxLogit) / logitSum;

    double u_value = C * prob * std::sqrt(parent->visit_count) / (1.0 + child->visit_count);
    return q_value + u_value;
}

// returns relative evaluation for the TT
float applyEvaluation(std::vector<Node*>& path, float relative_win_prob, const float* policy, chess::Board& board){

    Node* leaf = path.back();
    float win_prob = (leaf->turn == Color::WHITE) ? relative_win_prob : -relative_win_prob;

    if (!leaf->is_expanded) {
        // get_leaf_batch already decided this leaf is a 2-fold in THIS search.
        if (policy == nullptr) {
            relative_win_prob = 0;
            win_prob = 0;

            leaf->eval = 0;
        } else {
            Movelist moves;
            movegen::legalmoves(moves, board);

            if (moves.empty()) {
                if (board.inCheck()) {
                    win_prob = (board.sideToMove() == Color::WHITE) ? -1.0 : 1.0;
                    relative_win_prob = -1.0f;
                } else {
                    win_prob = 0.0;
                    relative_win_prob = 0.0;
                }
            } else {
                for (const Move& m : moves) {
                    int policy_idx = move_to_index(m, board.sideToMove());
                    u64 newHash = updateZobristMove(leaf->hash, m, board);
                    Node* child = new Node(leaf, policy[policy_idx], m, ~board.sideToMove(), newHash);

                    leaf->children.push_back(child);
                }
            }

            leaf->eval = relative_win_prob;
            leaf->is_expanded = true;
        }
    }

    for (Node* n : path) {
        double v_loss = (n->turn == Color::BLACK) ? -0.25 : 0.25;
        n->value_sum -= v_loss; // Remove virtual loss
        n->value_sum += win_prob; // Add real evaluation
    }

    return relative_win_prob;
}

// gets a batch of leaves from the TREE_ROOT
py::tuple get_leaf_batch(int batch_size, std::vector<std::string> gameMoves) {
    py::list board_features;
    py::list dense_features;

    Board root_board(constants::STARTPOS);

    batch_paths.clear();

    if (batch_size == 1) seqHalve.reset();

    for(std::string uci : gameMoves){
        chess::Move m = chess::uci::uciToMove(root_board, uci);
        root_board.makeMove(m);
    }

    int cacheHits = 0;

    // loops starting from the TREE_ROOT until you hit a leaf
    while (cacheHits < batch_size) {
        Node* current = TREE_ROOT;
        std::vector<Node*> current_path = {current};
        Board board = root_board;

        if (current->is_expanded) {
            if(seqHalve.candidates.empty()) return py::make_tuple(board_features, dense_features);

            Node* child = seqHalve.getNext();
            current = child;

            current_path.push_back(child);
            board.makeMove(child->move);

            // selects the leaf
            while (current->is_expanded && !current->children.empty()) {
                Node* best_child = nullptr;
                double best_puct = -9999999.0;

                double totalLogits = 0.0;
                double maxLogit = -1e9;
                for (Node* child: current->children) maxLogit = std::max(maxLogit, child->prob);

                for (Node* child: current->children) totalLogits += std::exp(child->prob - maxLogit);

                for (Node* child : current->children) {
                    double score = calculate_PUCT(current, child, totalLogits, maxLogit);
                    if (score > best_puct) {
                        best_puct = score;
                        best_child = child;
                    }
                }

                if (best_child == nullptr) break;

                current = best_child;
                board.makeMove(current->move);
                current_path.push_back(current);
            }
        }

        // adds virtual loss to it
        for (Node* n : current_path) {
            n->visit_count += 1;
            double v_loss = (n->turn == Color::BLACK) ? -0.25 : 0.25;
            n->value_sum += v_loss;
        }

        cacheHits++;

        if(board.isRepetition(1) && current_path.size() > 1){
            applyEvaluation(current_path, 0, nullptr, board);
            continue;
        }

        size_t idx = current->hash & (TABLE_SIZE - 1);

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

// policies is passed in as logits
void expand_and_backprop(py::array_t<float> win_probs, py::array_t<float> policies) {

    auto win_probs_buf = win_probs.unchecked<2>();
    auto policies_buf = policies.unchecked<2>();

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
        const float* policy = policies_buf.data(i, 0);

        float rel_prob = applyEvaluation(path, relative_win_prob, policy, board);

        size_t idx = leaf->hash & (TABLE_SIZE - 1);
        transTable[idx].hash = leaf->hash;
        transTable[idx].winProbabilities = rel_prob;
        std::copy(policy, policy + 4672, transTable[idx].policy.begin());

    }
}


std::string get_best_move() {
    if(seqHalve.candidates.empty()) return "0000";
    else return chess::uci::moveToUci(seqHalve.candidates[0]->move);
}

void free_tree() {
    if (TREE_ROOT != nullptr) {
        delete TREE_ROOT;
        TREE_ROOT = nullptr;
    }

    std::memset(transTable, 0, TABLE_SIZE * sizeof(TableEntry));
}

// adds gumbel noise to the logits
void apply_gumbel_noise() {
    // If the root doesn't exist or hasn't been expanded yet, we can't add noise
    if (TREE_ROOT == nullptr || TREE_ROOT->children.empty()) return;

    // setting up gumbel dist. polling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::extreme_value_distribution<double> gumbel_dist(0.0, 1.0);

    std::vector<double> noise;

    for (size_t i = 0; i < TREE_ROOT->children.size(); ++i) {
        TREE_ROOT->children[i]->gumbelNoise += gumbel_dist(gen);
    }
}

py::list get_root_policy() {
    py::list policy;
    if (TREE_ROOT == nullptr || TREE_ROOT->children.empty()) return policy;

    int cVisit = 50;
    double scale = 1.0;
    int maxVisits = 0;
    double totalVisits = 0.0;
    double maxLogit = 0.0;
    bool anyVisited = false;

    for (Node* child : TREE_ROOT->children) {
        if (child->visit_count == 0) continue;
        totalVisits += child->visit_count;
        maxVisits = std::max(maxVisits, child->visit_count);
        if (!anyVisited || child->prob > maxLogit) {
            maxLogit = child->prob;
            anyVisited = true;
        }
    }

    // prior sum
    double sumPrior = 0.0;
    if (anyVisited) {
        for (Node* child : TREE_ROOT->children) {
            if (child->visit_count == 0) continue;
            sumPrior += std::exp(child->prob - maxLogit);
        }
    }

    double Q = 0.0;

    // calculates cumulative Q
    if (anyVisited) {
        for (Node* child : TREE_ROOT->children) {
            if (child->visit_count == 0) continue;
            int turn = child->turn == chess::Color::WHITE ? -1 : 1;
            double q = turn * child->value_sum / child->visit_count;
            double prior = std::exp(child->prob - maxLogit) / sumPrior;
            Q += prior * q;
        }
    }

    double v = (TREE_ROOT->eval + totalVisits * Q) / (totalVisits + 1.0);

    std::vector<double> scores;
    scores.reserve(TREE_ROOT->children.size());

    // assigns all scores to root children
    for (Node* child : TREE_ROOT->children) {
        int turn = child->turn == chess::Color::WHITE ? -1 : 1;
        double qHat = v;
        if (child->visit_count != 0) {
            qHat = turn * child->value_sum / child->visit_count;
        }
        double score = child->prob + (cVisit + maxVisits) * scale * qHat;
        scores.push_back(score);
    }

    // softmax
    double maxScore = *std::max_element(scores.begin(), scores.end());
    double sumExp = 0.0;
    for (double& score : scores) {
        score = std::exp(score - maxScore);
        sumExp += score;
    }

    for (size_t i = 0; i < TREE_ROOT->children.size(); ++i) {
        policy.append(py::make_tuple(
            uci::moveToUci(TREE_ROOT->children[i]->move),
            scores[i] / sumExp));
    }

    return policy;
}

double getRootValue(){
    if(TREE_ROOT == nullptr || TREE_ROOT->visit_count == 0) return 0;

    double qWhite = TREE_ROOT->value_sum / TREE_ROOT->visit_count;

    return (TREE_ROOT->turn == Color::WHITE) ? qWhite : -qWhite;
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

void initialize_sequential_halving(){
    if (TREE_ROOT == nullptr || TREE_ROOT->children.empty()) return;
    seqHalve.reset();

    std::sort(TREE_ROOT->children.begin(), TREE_ROOT->children.end(), [](Node* a, Node* b){
        return (a->prob + a->gumbelNoise) > (b->prob + b->gumbelNoise);
    });

    for(int i = 0; i < std::min(seqHalve.k, (int)TREE_ROOT->children.size()); i++){
        seqHalve.candidates.push_back(TREE_ROOT->children[i]);
    }

    seqHalve.k = std::min(seqHalve.k, (int)TREE_ROOT->children.size());
}

void updatePhase() {
    if (TREE_ROOT != nullptr) {
        seqHalve.updatePhaseIfNeeded();
    }
}

// sends these back to python
PYBIND11_MODULE(mcts_exts, m) {
    m.def("get_fen", &getFen);
    m.def("log_visits", &logVisits);
    m.def("init_tree", &init_tree);
    m.def("get_leaf_batch", &get_leaf_batch);
    m.def("expand_and_backprop", &expand_and_backprop);
    m.def("get_best_move", &get_best_move);
    m.def("free_tree", &free_tree);
    m.def("apply_gumbel_noise", &apply_gumbel_noise);
    m.def("get_root_policy", &get_root_policy);
    m.def("promote_root", &promoteRoot);
    m.def("print_root_stats", &print_root_stats);
    m.def("py_board_params", &py_board_params);
    m.def("py_dense_params", &py_dense_params);
    m.def("get_root_value", &getRootValue);
    m.def("initialize_sequential_halving", &initialize_sequential_halving);
    m.def("update_phase", &updatePhase);
}
