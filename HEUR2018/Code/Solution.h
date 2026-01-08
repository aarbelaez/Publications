#ifndef SOLUTION_H
#define SOLUTION_H

#include <iostream>
#include <vector>

#include "BH.h"
#include "MetroNode.h"

using namespace std;

class Solution {
public:
    string name;
	int ***sol;
	double cost;
	int iter;
	int time2sol;
	int nbNodes;

	vector<MetroNode*> metro;
	vector<SiteNode> site_node;
	vector<TwoPred*> tps;

	int **matrix;
	
	Solution(void);
	~Solution(void);

	void init(int nbNodes, vector<MetroNode*>, vector<SiteNode>);
	double copySolution(vector<MetroNode*> metro, vector<TwoPred*> _tps, int **m);
	void setSolution(vector<MetroNode*> metro, vector<TwoPred*> _tps, int **m, int nbNodes, double _cost, int time);
	void delCurrentSolution(void);

    double copySolution(MetroNode *mn);
    void setSolution(MetroNode *mn);
    double aggregateCost(void);
    
	void print(MetroNode *, string);
    void print(string);
	void printTree(void);
	double solCost(void);
	double DIST(Node *, Node*);
};

#endif