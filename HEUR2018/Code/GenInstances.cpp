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

double totalCost;
bool forceStop = false;

#include "read_files.h"

#include "timer.h"
#include "BH.h"
#include "Node.h"
#include "MetroNode.h"
#include "MNGraph.h"
#include "Parameters.h"
#include "Solution.h"
#include "Search.h"


#include <string>

Parameters param;

double DIST(Node *N1, Node *N2, vector<SiteNode> site_node) {
#ifdef SCALE2
	return sqrt(pow(getX(N1->getId())*FACTOR - getX(N2->getId())*FACTOR, 2) + pow(getY(N1->getId())*FACTOR - getY(N2->getId() )*FACTOR, 2) );
#else
	return sqrt(pow(getX(N1->getId()) - getX(N2->getId()), 2) + pow(getY(N1->getId()) - getY(N2->getId() ), 2) );
#endif
}


bool isPrimary(Node * n, MetroNode* mn, vector<MetroNode*> metro_node, vector<SiteNode> sn) {
	int mn1 = sn[n->getId()].id_mn[0];
	int mn2 = sn[n->getId()].id_mn[1];
	double d1 = DIST(n, metro_node[mn1]->getMetroNode(), sn);
	double d2 = DIST(n, metro_node[mn2]->getMetroNode(), sn);
	if(d1 < d2 && mn == metro_node[mn1]) return true;
	if(d2 < d1 && mn == metro_node[mn2]) return true;
	return false;
}

int TotalPrimary(MetroNode *mn, vector<MetroNode*> metro_node, vector<SiteNode> sn) {
	int t=0;
	for(int i=1;i<(int) mn->node.size(); i++) {
		if( isPrimary(mn->getNode(i), mn, metro_node, sn) ) t++;
	}
	return t;
}

vector<Node*> getPrimary(MetroNode *mn, vector<MetroNode*> metro_node, vector<SiteNode> sn) {
	vector<Node*> nodes;
	for(int i=1;i<(int)mn->node.size(); i++) {
		if( isPrimary(mn->getNode(i), mn, metro_node, sn) ) nodes.push_back(mn->getNode(i));
	}
	return nodes;
}

void delNode(Node *n, vector<MetroNode*> metro_node, vector<SiteNode> sn) {
	int mn1 = sn[n->getId()].id_mn[0];
	int mn2 = sn[n->getId()].id_mn[1];
	int t=mn1;
	bool deleted=false;
	for(int i=0;i<(int)metro_node[t]->node.size(); i++) {
		if(n->getId() == metro_node[t]->getNode(i)->getId()) {
			metro_node[t]->node.erase(metro_node[t]->node.begin()+i);
			deleted=true;
			break;
		}
	}
	assert(deleted);
	deleted=false;
	t=mn2;
	for(int i=0;i<(int)metro_node[t]->node.size(); i++) {
		if(n->getId() == metro_node[t]->getNode(i)->getId()) {
			metro_node[t]->node.erase(metro_node[t]->node.begin()+i);
			deleted=true;
			break;
		}
	}
	assert(deleted);	
}


bool isIn(Node *n, vector<MetroNode*> metro_node) {
	for(int i=0;i<(int)metro_node.size();i++) {
		for(int j=0;j<(int)metro_node[i]->node.size();j++) {
			if(n->getId() == metro_node[i]->node[j]->getId()) {
				return true;
			}
		}
	}
	return false;
}

void inst2file(vector<MetroNode*> metro_node, vector<SiteNode> sn, string fname) {
	ofstream mfile;
	string fname1=fname+"_primsAndSecs.txt";
	mfile.open(fname1.c_str());
	
	ofstream mfile2;
	string fname3=fname+"_mnsTable.txt";
	mfile2.open(fname3.c_str());

	for(int i=0;i<(int)metro_node.size();i++) {
		mfile2<<sn[metro_node[i]->getNode(0)->getId()].x<<"\t"<<sn[metro_node[i]->getNode(0)->getId()].y<<endl;
		
		while(metro_node[i]->node.size() > 1) {
			Node *tmp = metro_node[i]->getNode(1);
			int mn1 = sn[tmp->getId()].id_mn[0];
			int mn2 = sn[tmp->getId()].id_mn[1];
			
			stringstream ssMN1;
			//ssMN1<<metro_node[mn1]->getMetroNode()->getInputId();
			ssMN1<<metro_node[mn1]->getMetroNode()->getId();
			stringstream ssMN2;
			//ssMN2<<metro_node[mn2]->getMetroNode()->getInputId();
			ssMN2<<metro_node[mn2]->getMetroNode()->getId();
			stringstream ssNode;
			ssNode<<tmp->getInputId();


			mfile<<ssNode.str()<<"\t"<<ssMN1.str()<<"\t"<<ssMN2.str()<<"\t-1.0\t-1.0"<<endl; 
			//mfile<<ssMN1.str()<<"\t"<<ssMN2.str()<<endl;
			delNode(tmp, metro_node, sn);
		}
	}

	mfile.close();
	mfile2.close();
}


void GenInstances(int maxNumber, string fname) {

	vector<MetroNode*> metro_node;
	vector<MetroNode*> metro_node_min;
	vector<SiteNode> site_node;

	vector<Pair> metro_node1;
	vector<Pair> site_node1;
	//read_sites_nodes_format2("../data/ire_data_100_cover.txt", site_node1);
    //read_metro_nodes_format2("../data/HEAnet_metronodes.txt", metro_node1);

	read_metro_nodes(param.data_metro_node.c_str(), metro_node, site_node);
	read_site_nodes(param.data_exchange_site.c_str(), metro_node.size(), site_node);
	Set_Init_ExchangeSites(param.data_exchange_site_mn.c_str(), site_node, metro_node);
	
	cout<<"MNS: "<<metro_node.size()<<endl;
	
	int nbMinNodesMN = 10000;
	
	for(int i=0; i<(int)metro_node.size(); i++) {
		for(int j=0;j<(int)metro_node[i]->node.size(); j++) {
			Node *tmp = metro_node[i]->getNode(j); 
			tmp->setMetroNode(metro_node[i]);
		}
		metro_node[i]->setPosNodes();
		metro_node[i]->initDistMatrix(site_node);
		cout<<"MN["<<i<<"]: "<<metro_node[i]->node.size()<<endl;
	}
	
	for(int i=0; i<(int)metro_node.size(); i++) {
		int tPrimary = TotalPrimary(metro_node[i], metro_node, site_node);
		cout<<"MN["<<i<<"]: "<<tPrimary<<endl;
		if(nbMinNodesMN > tPrimary) nbMinNodesMN = tPrimary;
	}
	
	if(nbMinNodesMN > maxNumber/metro_node.size()) nbMinNodesMN = maxNumber/metro_node.size();
	
	cout<<"nbMinNodesMN: "<<nbMinNodesMN<<endl;
	
	int delInstances = (site_node.size() - metro_node.size()) - maxNumber;
	cout<<"delInstances: "<<delInstances<<endl;
	vector<int> valid_mn;

	for(int i=0;i<(int)metro_node.size();i++) valid_mn.push_back(i);
	while(delInstances > 0) {
		//int rMN = Random(metro_node.size());
		int rMN = valid_mn[ Random(valid_mn.size()) ];
		//int rNode = Random(metro_node[rMN]->node.size()-1)+1;
		//Node *tmp = metro_node[rMN]->getNode(rNode);
		vector<Node*> ntmp = getPrimary(metro_node[rMN], metro_node, site_node);
		Node *tmp = ntmp[ Random(ntmp.size()) ];
		int mn1 = site_node[tmp->getId()].id_mn[0];
		int mn2 = site_node[tmp->getId()].id_mn[1];
		
		int t1 = TotalPrimary(metro_node[mn1], metro_node, site_node);
		int t2 = TotalPrimary(metro_node[mn2], metro_node, site_node);

		if(mn1 == rMN) {
			if(t1 > nbMinNodesMN) {
				delNode(tmp, metro_node, site_node);
				assert(!isIn(tmp, metro_node));
				delInstances--;				
			}
			else {
				metro_node_min.push_back(metro_node[mn1]);
				for(int i=0;i<(int)valid_mn.size();i++) {
					if(valid_mn[i] == mn1) {
						valid_mn.erase( valid_mn.begin() + i );
						break;
					}
				}				
			}
		}
 
		else if(mn2 == rMN) {
			if(t2 > nbMinNodesMN) {
				delNode(tmp, metro_node, site_node);
				assert(!isIn(tmp, metro_node));
				delInstances--;				
			}
			else {
				metro_node_min.push_back(metro_node[mn2]);
				for(int i=0;i<(int)valid_mn.size();i++) {
					if(valid_mn[i] == mn2) {
						valid_mn.erase( valid_mn.begin() + i );
						break;
					}
				}				
			}
		}
		else assert(false);
		/*if( t1 > nbMinNodesMN && t2 > nbMinNodesMN ) {
			delNode(tmp, metro_node, site_node);
			assert(!isIn(tmp, metro_node));
			delInstances--;
		}
		else {
			if(t1 <= nbMinNodesMN) {
				metro_node_min.push_back(metro_node[mn1]);
				for(int i=0;i<(int)valid_mn.size();i++) {
					if(valid_mn[i] == mn1) {
						valid_mn.erase( valid_mn.begin() + i );
						break;
					}
				}
			}
			if(t2 <= nbMinNodesMN) {
				metro_node_min.push_back(metro_node[mn2]);
				for(int i=0;i<(int)valid_mn.size();i++) {
					if(valid_mn[i] == mn2) {
						valid_mn.erase( valid_mn.begin() + i );
						break;
					}
				}

				//metro_node.erase( metro_node.begin() + mn2 );
			}

		}
		*/
		/*if(delInstances < 200 || delInstances % maxNumber == 10) {
			cout<<"delInstance: "<<delInstances<<" mn1 "<<mn1<<" mn2: "<<mn2<<endl;
			//cout<<"T1: "<<t1<<" T2: "<<t2<<endl;
			int tt=0;
			for(int i=0;i<(int)metro_node.size();i++) {
				int ft=TotalPrimary(metro_node[i], metro_node, site_node);
				tt+=ft;
				cout<<"MMNN["<<i<<"] "<<ft<<", ";
				assert(ft>=nbMinNodesMN);
			}
			cout<<endl;
			cout<<"FFFFF: "<<tt<<" ============ ";
			for(int i=0;i<(int)valid_mn.size();i++) {
				cout<<valid_mn[i]<<" , ";
			}
			cout<<endl;
		}*/

	}

	cout<<"nbMinNodesMN: "<<nbMinNodesMN<<endl;
	for(int i=0;i<(int)metro_node.size();i++) {
		int tPrimary = TotalPrimary(metro_node[i], metro_node, site_node);
		cout<<"MN["<<i<<"]: "<<tPrimary<<" nodes: "<<metro_node[i]->node.size()<<endl;
	}

	stringstream ss;
	ss<<metro_node.size();
	fname=fname+"_MetroNodes_"+ss.str()+"_ExchangeSites_";
	stringstream ss1;
	ss1<<maxNumber;
	fname=fname+ss1.str();
	//fname=fname+ss1.str()+"_primsAndSecs.txt";


	inst2file(metro_node, site_node, fname);
	
}


DEFINE_int64(seed, -1, "seed number");
DEFINE_int64(max_iter, numeric_limits<int>::max(), "max iter");
DEFINE_string(data_exchange_site,"../data/ire_data_100_cover.txt"," file exchange site nodes");
DEFINE_string(data_metro_node,"../data/HEAnet_metronodes.txt"," file metro nodes");
DEFINE_string(data_exchange_site_mn,"../data/em_decomp2_135_16.txt","file exchange site -> metro nodes");
DEFINE_int64(max_time, numeric_limits<int>::max(), "max time");
DEFINE_int64(input_format, 1, "input format");
DEFINE_string(first_improvement, "false", "first improvement");
DEFINE_string(move_operator, "node", "move operator");
DEFINE_int64(iter_no_imp, 50, "input format");
DEFINE_int64(perturbation_steps, 1, "input format");
DEFINE_string(sol_file,"console","solution output file");
DEFINE_int64(total_exchange_sites,100, "total exchange sites"); 

//Solution best_sol;
//timer mclock;


int main(int argc, char *argv[]) {
	//signal(SIGINT, signalHandler);  
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


    cout<<"SEED: "<<param.seed<<endl;
    Randomize_Seed(param.seed);

	string fname = "SEED_";
	stringstream ss;
	ss<<param.seed;
	fname+=ss.str();
	
	GenInstances(FLAGS_total_exchange_sites, fname);
    return 1;
}
