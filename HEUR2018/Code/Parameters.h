#ifndef PARAMETERS__H
#define PARAMETERS__H

#include <iostream>
#include <string>

class Parameters {
public:
	int seed;
	int max_iter;
	int max_time;
	string data_exchange_site;
	string data_metro_node;
	string data_exchange_site_mn;
    string cost_dist_es2es;
    string cost_dist_mn2es;
    
	int input_format;
	string first_improvement;
	int move_operator;
    int iter_no_imp;
	int perturbation_steps;
	string sol_file;
    int nbCores;
    double targetSolution;
    int local_time;
    double target_sol;
    
    double road_factor;
    int par_algorithm;
    
    int sol_metro;
    
    bool initial_sol_only;
    
	string getOperator(void) {
		if(move_operator == NODE_OPERATOR) return string("Node");
		if(move_operator == ARC_OPERATOR) return string("Arc");
		if(move_operator == ARC_NODE_OPERATOR) return string("Arc-Node");
        if(move_operator == SUBTREE_OPERATOR) return string("Subtree");
		assert(false);
		return string();
	}
	void print(void) {
		cout<<"Parameters ==================="<<endl
			<<"Lambda: "<<LAMBDA<<endl
			<<"seed: "<<seed<<endl
			//<<"move operator: "<<(move_operator==NODE_OPERATOR?"Node":"Arc")<<endl  
			<<"move operator: "<<getOperator()<<endl
			<<"first improvement: "<<(Node::firstImprovement?"true":"false")<<endl
            <<"Iter no improvement: "<<iter_no_imp<<endl
			<<"Perturbation steps: "<<perturbation_steps<<endl
			<<"max iter: "<<max_iter<<endl
			<<"max time: "<<max_time<<endl
			<<"data exchange site: "<<data_exchange_site<<endl
			<<"data metro node: "<<data_metro_node<<endl
			<<"data exchange site mn: "<<data_exchange_site_mn<<endl
            <<"data cost distance es2es: "<<cost_dist_es2es<<endl
            <<"data cost distance mn2es: "<<cost_dist_mn2es<<endl
			<<"input format "<<input_format<<" -> "<<(input_format==1?"Ireland":"Italy")<<endl
            <<"Target solution: "<<target_sol<<endl
            <<"Road factor: "<<road_factor<<endl
            <<"sol_file: "<<sol_file<<endl;
        if(par_algorithm!=PARNONE) {
            cout<<"parallel algorithm: "<<(par_algorithm==INDSET?"Independent Set":"Random Conflict")<<endl
                <<"nbCores: "<<nbCores<<endl;
        }
        if(method==2)
            cout<<"Solving metro node: "<<sol_metro<<endl;
			cout<<"============================="<<endl;
	}
};


#endif