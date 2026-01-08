#ifndef METRONODE__H
#define METRONODE__H

#include "Node.h"
#include "FirstLevelList.h"
#include "TwoPred.h"
#include "mt19937p.h"

class MetroNode {
    friend class Node;
private:
    //Node *node;
    int id;
	int input_id;
    struct mt19937p mt;
public:
    double road_factor;

    FirstLevelList fLevelList;
    FirstLevelNode *getNewFirstLevelNode(Node *);
    void decFirstLevelNode(FirstLevelNode *&);
    void checkSize(void);
    void checkDistance(void);
    
    int usage;
    pstate state;
    double tCost;
	double **distMatrix;
    double **costMatrix;

    bool **validEdges;
    //shortest path from root to i^th node
    vector< vector<Node*> > spath;
	vector<TwoPred*> tps;

    vector<Node *> active;
    vector<Node *> inactive;
    
    bool isActive(void);
    Node *getActiveNode(int);
    int getTotalActiveNodes(void);
    void resetActiveAndInactive(void);
    void activeNodes(void);
    void setActiveNodes(void);
    bool setInactiveNode(int pos_node); //indicates whether I should also inactive the metro node
    void printActiveAndInactiveNodes(void);

	bool comparePtrToNode(Node* a, Node* b) { return ( a->getId() < b->getId() ); }

    vector<Node*> node;
    int size;

    int random(int i);
    void randomSeed(int s);
    
    MetroNode(void);
	~MetroNode(void);

    string getIdString(void) {
        return getMetroNode()->getIdString();
    }
    void printNodes(void) {
        cout<<node[0]->getId()<<" -> ";
        for(int i=1;i<(int)node.size();i++) cout<<node[i]->getId()<<", ";
        cout<<endl;
    }

    void dijkstra(void);
    void InitialTree(void);
    void BDBHeuristic(void);
    void checkTree(void);
    
    void setDistValueMN2LE(int idLE, double dist);
    void setDistMatrixValue(int id1, int id2, double cost);
    void setCostValueMN2LE(int idLE, double cost);
    void setCostMatrixValue(int id1, int id2, double dist);
    void setDistAndCostMatrixValue(int id1, int id2, double dist, double cost);
    void setDistAndCostValueMN2LE(int idLE, double dist, double cost);
	void initDistMatrix(vector<SiteNode> sn);
    void initCostMatrix(vector<SiteNode> sn);
    
    void setDistAndCostMatrixValue(Node *n1, Node *n2, double dist, double cost);

    bool validEdge(Node *, Node *);
    void setValidAllEdges(void);
    void setValidEdge(Node *, Node*);
    void setInValidEdge(Node *, Node*);
    void initValidEdges(void);
	
	void setPosNodes(void);
	void setInputId(int _id) { input_id = _id; }
	int getInputId(void) { return input_id; }
    void setId(int _id);
    int getId(void) { return id; }

    int getSize(void);

    void setNodes(void);

    void addNode(int i, Node *n);

    Node *getNode(int i);

    void printTree(void);

    void checkLoop(void);

    void testDistance(void);
    double testDist2Leaf(Node *);

    Node *getMetroNode(void) { return node[0]; }

	void copy(MetroNode *);
    
private:
    pair<Node*, double> getNewEdge(Node *root, Node *n);
    Node *getNextBDBNode(Node *root, vector<Node*> &v);
};


#endif