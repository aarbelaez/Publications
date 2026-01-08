#ifndef MOVE__H
#define MOVE__H

#include "string"


class Move {
public:
    Node *nNode;
    pair<Node **, double> n;
    Move(void) {
        nNode = NULL;
        n = make_pair( static_cast<Node**>(NULL), -1);
    }
    Move( const Move & m) {
        nNode = m.nNode;
        n.first = m.n.first;
        n.second = m.n.second;
    }
    Move(Node *_nNode, pair<Node **, double> _n) {
        nNode = _nNode;
        n = make_pair(_n.first, _n.second);
    }
    
};

#endif
