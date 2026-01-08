#ifndef TWO_PRED_H
#define TWO_PRED_H

#include "Node.h"

class TwoPred {
    friend class Node;
private:
    bool checkDisjointNess(void);
    
	void addPred(Node *, bool);
    
	void delPred(Node *, bool);
    
	//non existing link ... only for disjointness
	bool validPred(Node *, bool);

public:
	int id_node;
	/*int pred1;
	int pred2;*/
    int pred[2];

	TwoPred(void);
    ~TwoPred(void);
	
    void copy(TwoPred *, bool);
	void copy(TwoPred *);
    
    void reset(void);
    
    string toString(void);
};

#endif