#pragma once

#include "chess.hpp"
#include <cstdint>

typedef std::uint64_t u64;
// Node :)
struct Node {
    int visit_count;
    double value_sum;
    double eval;
    double gumbelNoise;
    double prob; // actually contains the logits (kinda of represent probability)
    chess::Move move;
    chess::Color turn;
    bool is_expanded;
    u64 hash;

    Node* parent;
    std::vector<Node*> children;

    //initializer
    Node(Node* p = nullptr, double pr = 0.0, chess::Move m = chess::Move::NULL_MOVE, chess::Color t = chess::Color::WHITE, u64 hsh = 0) {
        visit_count = 0;
        value_sum = 0.0;
        eval = 0.0;
        gumbelNoise = 0.0;
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
