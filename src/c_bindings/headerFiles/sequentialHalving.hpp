#include "headerFiles/node.hpp"

struct SequentialHalving{

    std::vector<Node*> candidates;
    int totalVisits = 32; // should be a pow of 2
    int visits = 0;
    int k = 4; // should be a pow of 2
    int round = 1;

    void reset();

    Node* getNext();

    void updatePhaseIfNeeded();

    private:

        double calculateQ(Node* node);



};
