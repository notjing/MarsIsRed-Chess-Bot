#include "headerFiles/node.hpp"

struct SequentialHalving{

    std::vector<Node*> candidates;
    int totalVisits = 64; // should be a pow of 2
    int visits = 0;
    int k = 8; // should be a pow of 2
    int round = 1;

    void reset();

    Node* getNext();

    void updatePhaseIfNeeded();

    int getNumCandidates();

    private:

        double scoreQ(Node* node, int maxVisits);



};
