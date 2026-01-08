#ifndef BH_HH
#define BH_HH


#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <functional>

#include <signal.h>
#include <limits>
#include <assert.h>
#include <math.h>
//#include "Node.h"

//#include <time.h>
#include <sys/time.h>

//#include "tools.h"

#include <omp.h>

using namespace std;

#define NODE_OPERATOR 1
#define ARC_OPERATOR 2
#define ARC_NODE_OPERATOR 3
#define SUBTREE_OPERATOR 4

#define INDSET 6
#define RANDCONF 7
#define PARNONE 8

#define EPSILON 0.00001
#define FACTOR 1000

/*
#ifndef LAMBDA

#ifdef SCALE2
	#define LAMBDA 67  (120000/1000)^2 
#else 
	#define LAMBDA 120000
#endif

#endif

*/

typedef enum { available,busy } pstate;


inline bool EqualDouble(double d1, double d2) {
    return (fabs(d1 - d2) < EPSILON);
}

extern int maxTreeSize;
extern double LAMBDA;
//extern double totalCost;
extern bool forceStop;
extern int method;

#define getX(id) site_node[id].x
#define getY(id) site_node[id].y

#define massert(cond) { if(!(cond)) { exit(EXIT_FAILURE); } }

#define PCLOCK

class TwoPred;
class MetroNode;
class Node;
class MNGraph;
class Solution;
class Parameters;
class timer;
class FirstLevelNode;
class FirstLevelList;

class Pair {
public:
    int id;
    double x;
    double y;
};

class SiteNode {
public:
    Node *n1;
    Node *n2;
    int id_mn[2];
    int id;
	int input_id;
    double x;
    double y;
    

	SiteNode(void) { 
        id_mn[0]=-1; id_mn[1]=-1; 
    }
    SiteNode(double _x, double _y) {
        x = _x;
        y = _y;
		id_mn[0]=-1; id_mn[1]=-1;
    }
    ~SiteNode(void) {
    }
};



#endif
