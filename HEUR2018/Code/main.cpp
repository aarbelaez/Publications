#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <limits>
#include <utility>

#include <math.h>
#include <assert.h>


#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>


using namespace std;

int maxTreeSize = numeric_limits<int>::max();

double LAMBDA = 120.0;
//double totalCost;
bool forceStop = false;
int method = 2; 
void signalHandler( int signum )
{
    cout << "Interrupt signal (" << signum << ") received.\n";
	forceStop = true;
    // cleanup and close up stuff here  
    // terminate program  

   //exit(signum); 
}

#include "read_files.h"

#include "timer.h"
#include "BH.h"
#include "Node.h"
#include "MetroNode.h"
#include "MNGraph.h"
#include "Parameters.h"
#include "Solution.h"
#include "Search.h"

//#include "read_files.cpp"



//#include "ad_solver.h"


/*static MetroNode *metro_node;
static SiteNode *site_node;
static double totalDistance;*/

/*static vector<MetroNode> metro_node;
static vector<SiteNode> site_node;
static double totalDistance;

static int **matrix;

static int size;
static int *sol;

static int nbMetroNodes;
static int nbExchangeSites;

static int nbNodes;*/

Parameters param;
DEFINE_int64(seed, -1, "seed number");
DEFINE_int64(max_iter, numeric_limits<int>::max(), "max iter");
DEFINE_string(data_exchange_site,"../../data/Ireland/ire_data_100_cover.txt"," file exchange site nodes");
DEFINE_string(data_metro_node,"m20_mnsTable.txt"," file metro nodes");
DEFINE_string(data_exchange_site_mn,"m20_primsAndSecs.txt","file exchange site -> metro nodes");
DEFINE_int64(max_time, numeric_limits<int>::max(), "max time");
DEFINE_int64(input_format, 2, "input format");
DEFINE_string(first_improvement, "false", "first improvement");
DEFINE_string(move_operator, "node", "move operator");
DEFINE_int64(iter_no_imp, 50, "input format");
DEFINE_int64(perturbation_steps, 1, "input format");
DEFINE_string(sol_file,"console","solution output file");
DEFINE_double(lambda, -1, "LAMBDA (length constraint)");
DEFINE_int64(method,1,"Method");
DEFINE_int64(sol_metro,-1, "Sol metro node");
DEFINE_int64(nbCores, 1, "Number of cores");
DEFINE_double(target_sol, -1, "Target solution");
DEFINE_int64(local_time,2,"Local time");
DEFINE_double(road_factor,1,"Road factor");

DEFINE_string(cost_dist_es2es,"none", "Cost and distance file exchange site to exchange -- site");
DEFINE_string(cost_dist_mn2es,"none", "Cost and distance metro node to exchange site -- file");

DEFINE_string(initial_sol_only,"false","Compute initial solution only");

DEFINE_string(par_algorithm,"none","Independent sets for parallel computation");


//Solution best_sol;
//timer mclock;


int main(int argc, char *argv[]) {

	signal(SIGINT, signalHandler);
    string usage("usage\n");
    usage+=argv[0];
    google::SetUsageMessage(usage);
	google::ParseCommandLineFlags(&argc, &argv, true);
	param.seed = FLAGS_seed==-1?time(NULL):FLAGS_seed;
	param.max_iter = FLAGS_max_iter;
	param.data_exchange_site = FLAGS_data_exchange_site;
	param.data_metro_node = FLAGS_data_metro_node;
	param.data_exchange_site_mn = FLAGS_data_exchange_site_mn;
	param.max_time = FLAGS_max_time;
	param.input_format = FLAGS_input_format;
    param.iter_no_imp = FLAGS_iter_no_imp;
	param.perturbation_steps = FLAGS_perturbation_steps;
	param.sol_file = FLAGS_sol_file;
    param.sol_metro = FLAGS_sol_metro;
    param.nbCores = FLAGS_nbCores;
    param.target_sol = FLAGS_target_sol;
    param.local_time = FLAGS_local_time;
    LAMBDA = FLAGS_lambda;
    method = FLAGS_method;
    param.road_factor = FLAGS_road_factor;
    
    //param.initial_sol_only = FLAGS_initial_sol_only;
    if( FLAGS_initial_sol_only.compare("true") == 0 ) {
        param.initial_sol_only = true;
    } else if( FLAGS_initial_sol_only.compare("false") == 0 ) {
        param.initial_sol_only = false;
    } else assert(false);

    if(param.sol_metro != -1) method=2;
    if(param.input_format == 3) {
        method=2;
        param.sol_metro = 0;
    }

    param.cost_dist_es2es = FLAGS_cost_dist_es2es;
    param.cost_dist_mn2es = FLAGS_cost_dist_mn2es;
    
	if( FLAGS_move_operator.compare("node") == 0) {
	    param.move_operator = NODE_OPERATOR;
	}
    else if( FLAGS_move_operator.compare("arc") == 0) {
        param.move_operator = ARC_OPERATOR;
    }
	else if( FLAGS_move_operator.compare("arc-node") == 0) {
		param.move_operator = ARC_NODE_OPERATOR;
	}
    else if( FLAGS_move_operator.compare("subtree") == 0) {
        param.move_operator = SUBTREE_OPERATOR;
        if(FLAGS_par_algorithm.compare("indset") == 0) {
            param.par_algorithm = INDSET;
        }
        else if(FLAGS_par_algorithm.compare("randconf") ==0 ) {
            param.par_algorithm = RANDCONF;
        }
        else if(FLAGS_par_algorithm.compare("none") == 0 ) {
            param.par_algorithm = PARNONE;
        } else {
            cout<<"Error invalid parallel algorithm"<<endl;
            return 0;
        }
    }
    /*else if( FLAGS_move_operator.compare("subtree-par") == 0) {
        param.move_operator = SUBTREE_PAR_OPERATOR;
    }*/
    else {
        cout<<"Error invalid move operator"<<endl;
        return 0;
    }

	if( FLAGS_first_improvement.compare("true") == 0 ) {
	    Node::firstImprovement = true;
	}
	else if ( FLAGS_first_improvement.compare("false") == 0 ) {
	    Node::firstImprovement = false;
	}
	else {
	    cout<<"Error unvalid first improvement parameter ("<<FLAGS_first_improvement<<")"<<endl;
	    return 0;
	}

    cout<<"SEED: "<<param.seed<<endl;
    //Randomize_Seed(param.seed);

    LSearch *s;
    if(param.move_operator == NODE_OPERATOR) {
        s = new NodeSearch(param);
	}
    /*else if (param.move_operator == ARC_OPERATOR) {
        s = new ReWireSearch(param);    
	}*/
    else if (param.move_operator == SUBTREE_OPERATOR) {
        if(param.par_algorithm == INDSET) {
            s = new SubTreeSearchPar(param);
        }
        else if(param.par_algorithm == RANDCONF) {
            s = new SubTreeSearchPar2(param);
        }
        else s = new SubTreeSearch(param);
    }
    /*else if (param.move_operator == SUBTREE_PAR_OPERATOR) {
        s = new SubTreeSearchPar2(param);
    }*/
	/*else if(param.move_operator == ARC_NODE_OPERATOR)
		s = new ReWire_Node(param);*/
		//s = new ReWireExhaustive(param);

    //ReWireSearch s(param);

	param.print();
	//mclock.init();
    //Solve();
    //s->search();
	//s->ILS();
    //s->solve();
    //s->ILSPar();
    /*
    if(method == 1)
        s->ILS();
    else
        s->ILS_MST();
    */
    if(method == 1)
        s->solve();
    else
        s->ILS_MST();
    return 1;
}
