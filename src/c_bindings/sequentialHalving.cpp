#include "headerFiles/node.hpp"
#include "headerFiles/sequentialHalving.hpp"

void SequentialHalving::reset(){
    candidates.clear();
    visits = 0;
    round = 1;
    k = 8;
}

Node* SequentialHalving::getNext() {
    if(candidates.empty()) return nullptr;
    return candidates[visits++ % candidates.size()];
}

int SequentialHalving::getNumCandidates(){
    return candidates.size();
}

double SequentialHalving::scoreQ(Node* node, int maxVisits) {
    if(node == nullptr){
        std::cout << "WEE WOO SOMETHING BROKEN" << "\n";
    }

    if (node->visit_count == 0) {
        return node->prob + node->gumbelNoise;
    }

    const int cVisit = 50;
    const double scale = 0.3;
    const int turn = node->turn == chess::Color::WHITE ? -1 : 1;

    double valHead = (cVisit + maxVisits) * scale * node->value_sum * turn / node->visit_count;
    return valHead + node->prob + node->gumbelNoise;
}

void SequentialHalving::updatePhaseIfNeeded() {
    if (candidates.size() <= 1) return;

    round++;
    visits = 0;

    int maxVisits = 0;
    for (Node* n : candidates) {
        maxVisits = std::max(maxVisits, n->visit_count);
    }

    std::vector<std::pair<double, Node*>> scored;
    scored.reserve(candidates.size());
    for (Node* n : candidates) {
        scored.emplace_back(scoreQ(n, maxVisits), n);
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    candidates.clear();
    for (const auto& [score, node] : scored) {
        candidates.push_back(node);
    }
    candidates.resize(candidates.size() / 2);
}
