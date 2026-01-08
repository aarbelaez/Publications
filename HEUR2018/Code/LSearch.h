#ifndef LSEARCH_H
#define LSEARCH_H

#include <vector>
#include <iostream>


#include "BH.h"
#include "MetroNode.h"
#include "Parameters.h"
#include "Solution.h"
#include "timer.h"
#include "MNGraph.h"
#include "Move.h"

#include "read_files.h"

class LSearch {
private:
    bool g_conflict;
public:
    timer pclock;
    int maxLocalTime;
    int nbThreads;
    double totalCost;
    double targetSol;
    double distTmp;
    int moveOpe;
    int lastChange;
    
    double bestSol;
    double lastChangeSol;
    int maxNoImp;
    int maxIter;
	int maxTime;
    int totalPerturbations;
    Parameters param;
    vector<MetroNode*> metro_node;
    vector<SiteNode>  site_node;

    /*
	vector<Node*> valid_nodes;
	vector<Node*> invalid_nodes;
	
	vector< vector<Node *> > inv_nodes; 
     */
    
    vector<MetroNode*> activeMetroNode;
    vector<MetroNode*> inactiveMetroNode;
	
	//MetroNode *mnIter;
    MetroNode *activeMN;
    
    int nbMetroNodes;
    int nbExchangeSites;
    int nbNodes;

    Solution best_sol;
    Solution global_best;
    timer mclock;
    
	vector<TwoPred*> tps;
    int **matrix;
	int iter;
    
    bool initData;
	
	MNGraph g;

public:
    
    virtual bool isDiversification(void);
	LSearch(Parameters _param);
    
    double costTrivialSolution(void);
    
	void InitMatrix(void);

    void initMST(void);

	void init(void);
    
    void createTreePrimAlg(MetroNode *);
    void createGreedyTree(MetroNode *);
    void createTrivialTree(MetroNode *);
    void createTree(MetroNode *);
	void InitialSolution(void);

	void initFromBestSol(void);

	void testAllDistances(void);
    
    virtual void SelElementDiversication() = 0;
    virtual void SelElementIntensification(Move m) = 0;
    
    //diversification step?
    virtual Move divStep(int) = 0;

	bool stopSearch;
    void search(void);

    virtual void solve(void);
	void ILS(void);
	virtual void updateBest(void);
    virtual void updateBest(Solution &, Solution &, bool) =0;

    void ILSPar(void);
    //ILS for the minimum spanning tree
    void ILS_MST(void);
    void ILS_MST(int);
	
    virtual void selectMetroNodes(void) {}
    virtual void clearSelectMetroNodes(void) {}
    
    virtual void InitData(void) {}
    virtual void localStart(void);
    virtual bool localStop(void);
	virtual bool stop(void);
	virtual void searchIntensification(void);
	virtual void searchDiversification(void);
    
    
    virtual void unLockNode(Node *) = 0;
    virtual void unLockMetroNode(MetroNode *) = 0;
    
private:
    void setMetroNodeRandomSeed(void);
    void readData(void);
    void initLocalInfoMST(MetroNode *);
    double MSTCost(MetroNode *);

protected:
    vector<MetroNode*> metro_node_core;
    struct mt19937p mt;
    void randomSeed(int s);
    int random(int i);

};

//bool EdgeGreater( pair<Node*, Node*> i, pair<Node*,Node*> j);

typedef pair<Node*,Node*> edgeGraph;
Node *startNode(edgeGraph g);
Node *endNode(edgeGraph g);
double distanceEdge(edgeGraph);
double costEdge(edgeGraph);

//double distanceEdge(edgeGraph g) {return startNode(g)->getDistance(endNode(g)); }
#define printEdge(g) cout<<startNode(g)->getIdString()<<" -> "<<endNode(g)->getIdString();
struct EdgeGreater {
    bool operator() (const edgeGraph &a, const edgeGraph & b) const {
        /*cout<<"test: ";
        printEdge(a);
        cout<<" ";
        printEdge(b);
        cout<<endl;*/
        //return ( distanceEdge(a) > distanceEdge(b) );
        return (costEdge(a) > costEdge(b));
    }
};


#endif