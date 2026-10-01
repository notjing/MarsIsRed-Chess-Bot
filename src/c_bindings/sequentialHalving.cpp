#include "headerFiles/node.hpp"
#include "headerFiles/sequentialHalving.hpp"

void SequentialHalving::reset(){
    candidates.clear();
    visits = 0;
    round = 1;
    k = 4;
}

Node* SequentialHalving::getNext() {
    if(candidates.empty()) return nullptr;
    return candidates[visits++ % candidates.size()];
}

void SequentialHalving::updatePhaseIfNeeded(){
    if(candidates.size() == 1) return;

    round++;
    visits = 0;

    std::sort(candidates.begin(), candidates.end(), [this](Node* a, Node* b){ return calculateQ(a) > calculateQ(b); });
    candidates.resize(candidates.size() / 2);

}

double SequentialHalving::calculateQ(Node *node){
    int cVisit = 50;
    double scale = 1.0;
    int maxVisits = 0;

    for(Node* n : candidates){
        maxVisits = std::max(maxVisits, n->visit_count);
    }

    int turn = node->turn == chess::Color::WHITE ? -1 : 1;


    double valHead = (cVisit + maxVisits) * scale * node->value_sum * turn / node->visit_count;

    return valHead + node->prob + node->gumbelNoise;
}


