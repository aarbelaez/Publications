#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "BH.h"
#include "FirstLevelList.h"
#include "TwoPred.h"

using namespace std;



struct NodeSDistance;


class Node {
	friend class MetroNode;
    friend struct NodeSDistance;
private:
    double getRoadFactor(void); 
    int id_metro_node;
    int id_node;
	int pos_mn; 
	int input_id;
    //predecessor
    Node *pred;
    //sucessors
    int mark;
    double distance;
    double sdistance;
	MetroNode *metro_node;
    //vector<MetroNode> metro_node;
    int **matrix;
    bool junction;

    void addPredForest(Node *);
    void delPredForest(Node *);
    bool validPredForest(Node *);
    
    int sizeTree;
    int treeLevel;
    FirstLevelNode *firstLevel;
    
    Node *fnode;

public:
    //shorted path predecessor
    Node *spred;

    double getLambda(void);
    double getLambda(Node *a, Node *b);
    int getSizeTree(void) { return sizeTree; }
    void isFirstLevel(void);
    FirstLevelNode *getFirstLevel(void);
    void setFirstLevel(FirstLevelNode *);
    void increaseFirstLevelSize(void);
    void decreaseFirstLevelSize(void);
    
    Node *getFNode(void) { return fnode; }
    
    bool isMetroNode(void);
    MetroNode* getMetro(void) { return metro_node; }
    bool isPrimary;
    pstate state;
    void setJunction();
    bool isJunction();
    void setSDistance(double dist) { sdistance = dist; }
    double getSDistance(void) { return sdistance; }
    void setMaxSDistance(void) {
        sdistance = numeric_limits<double>::max();
    }
    vector<SiteNode> site_node;
	void setMetroNode(MetroNode *mn) { metro_node = mn; } 
    //void setMetroNodeVector(vector<MetroNode> mn) { metro_node = mn; }
    //void setSiteNodeVector(vector<SiteNode> &sn) { site_node = sn; }
    void setMatrix(int **m) { matrix = m; }

    static bool firstImprovement;
    
    int random(int i);

    int usage;
    vector<Node*> suc;
    double dist2leaf;
    vector<Node*> dom;

    struct NodeGreater {
        bool operator() (const Node* a, const Node* b) { 
            return ( a->sdistance < b->sdistance );
        }
    };

    priority_queue<Node*, vector <Node*>, NodeGreater> invalid;
    //priority_queue<Node*> invalid;

    Node(void);
    Node(int id_mn, int _id);

    Node(int id_mn);

    /*const bool operator<(const Node &n1) {
        return getId() > n1.getId();
    }*/

    /*bool operator() (const Node &n1, const Node &n2) const {
        return n1->getId() < (*n2).getId();
    }*/

	TwoPred *getTP(void);

    //bool isMetroNode(void) { return id_node == -1; } 

    void markNode(void) { mark = 1; }

    void unMarkNode(void) { mark = 0; }

    bool isMarked(void) { return mark==1; }

    int getMetroNodeId(void) { return id_metro_node; }

    void setId(int id) { id_node = id; }

	int getIdMetroNode(void) { return id_metro_node;}
    int getId(void) const { return id_node==-1?id_metro_node:id_node; }

	void setInputId(int _id) { input_id = _id; }
	int getInputId(void) { return input_id; }
	string getIdString(void) { 
		string tmp; 
		stringstream ss; 
		ss<<input_id;
		if(this == getMetroNode()) {
			tmp+="m";
		}
		tmp+=ss.str(); 
		return tmp;
	}


    void setMetroNode(void) { id_node = -1; }

    void setPred(Node *_pred);
    void initPred(Node *_pred);
    void initPred(void);

    void addSuccessor(Node *s);
    void delSuccessor(Node *s);
    void updateDistance(Node *);

    bool validMove(Node *n);

    Node *getPred(void) { return pred; }

    Node *getMetroNode(void);

    void printInvalidNodes(void) {
        cout<<"Invalid nodes for NODE["<<getId()<<"]: ";
        //priority_queue<Node*> tmp;
        priority_queue<Node*, vector<Node*>, NodeGreater> tmp;
        while( !invalid.empty() ) {
            cout<<invalid.top()->getId()<<", ";
            tmp.push(invalid.top());
            invalid.pop();
        }
        invalid = tmp;
        cout<<endl;
    }

    void move(Node *nPred);
    void reset(bool);

    
    void restoreNode(Node *oldPred);
    bool validMoveNode(Node *a, Node *b);
    Node *randomMove(void);
    Node *bestMove(void);
     
    bool validMoveNode2(Node *a, Node *b);

    void changePred(Node *);
    
    void restoreSubTree(Node *);
    void delSubTree(void);
    double moveSubTree(Node *a, Node *b);
    bool validMoveSubTree(Node *a, Node *b);
    pair<Node **, double> bestMoveSubTree(void);
    //Node **bestMoveST(double); 
    pair<Node**, double> bestMoveST(double);
    pair<Node **, double> randomMoveSubTree(void);
    Node **randomMoveST(void);

    double costMove2(Node *a, Node *b);
    //Node **bestMoveNode2(void);
    void moveNode2(Node *a, Node *b);
    bool delNode(void);

    double moveNode(Node *a, Node *b);
    pair<Node **, double> bestMoveNode(void);
    Node **_bestMoveNode(void);
    double costMove(Node *a, Node *b);
    pair<Node **, double> randomMoveNode(void);

    void checkLoop(void);
    void testDistance(void);

    double getEuclideanDistance(Node *_n);
    double getDistance(Node *_n);
    double getDistance(void);
    
    double getCost(Node *_n);

    void copy(Node *); 
	void delTree(void);
    
    pair<Node**,double> add2Tree(void);
    pair<Node**,double> bestDistanceMoveNode(void);
private:

    void copyAtr(Node *);
    int totalNeighbours(void);
    //this should used only after re-allocating the tree
    void updateInvalidNodes(Node *oldPred);
    //remove vNodes elements from invalid (domain)
    void updateInvalidVector(vector<Node*> vNodes);

    void printValidNeighbours(void);

    void setDistance(double);

    double getDist2leaf(void);
};

//bool NodeGreater (Node *i, Node *j) { return (i->getInputId() < j->getInputId()); }
bool NodeGreater(Node *, Node *);


struct NodeSDistance {
    bool operator() (const Node* a, const Node* b) const { 
        return ( a->sdistance < b->sdistance );
    }
};


#endif