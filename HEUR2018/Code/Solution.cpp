
#ifndef SOLUTION_CPP
#define SOLUTION_CPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include <assert.h>
#include <math.h>
#include "Solution.h"

Solution::Solution(void) {
    cost = numeric_limits<double>::max();
}

Solution::~Solution(void) {
	/*for(int i=0;i<nbNodes;i++) {
		delete [] matrix[i];
		for(int j=0;j<nbNodes; j++) {
			delete [] sol[i][j];
		}
		delete [] sol[i];
	}
	delete [] matrix;
	delete [] sol;*/
	
	for(unsigned int i=0;i<metro.size(); i++) delete metro[i];
	metro.clear();
	
}

void Solution::init(int n, vector<MetroNode*> _mn, vector<SiteNode> sn)  {
    cost = numeric_limits<double>::max();
	nbNodes = n;
	//matrix = new int*[n];
	//sol = new int**[n];
	unsigned int MNusage = 0.0;
    if( metro.empty() ) {
        for(int i=0;i<n;i++) {
            if(i<(int)_mn.size()) {
                //cout<<"MN size: "<<sizeof(MetroNode)<<endl;
                //cout<<"Node Size: "<<sizeof(Node)<<endl;
                MNusage+=sizeof(MetroNode);
                MetroNode *nMN = new MetroNode();
                //nMN->tps = tps;
                nMN->tps = _mn[i]->tps;
                for(unsigned int j=0; j<_mn[i]->node.size(); j++)  {
                    MNusage+=sizeof(Node); 
                    Node *nNode = new Node();
                    nNode->setMetroNode(nMN);
                    nNode->setMatrix(matrix);
                    //nNode->setSiteNodeVector(sn);
                    nMN->node.push_back(nNode);
                }
                metro.push_back(nMN);
            }
            /*matrix[i] = new int[n];
             sol[i] = new int*[n];
             for(int j=0;j<n;j++) {
                //matrix[i][j] = 0;
                sol[i][j] = new int[2];
                sol[i][j][0] = -1; sol[i][j][1] = -1;
             }*/
        }
        site_node = sn;
	}
    else {
        //only need to reset information
        for(int i=0;i<n;i++) {
            if( i<(int)_mn.size() ) {
                for(unsigned int j=0; j<_mn[i]->node.size(); j++) {
                    _mn[i]->node[i]->reset(false);
                }
            }
        }
    }

    if(tps.empty()) {
        for(unsigned int i=0;i<sn.size(); i++) {
            MNusage+=sizeof(TwoPred);
            tps.push_back(new TwoPred());
        }
        cout<<"MMUsage-tps: "<<MNusage<<endl;
    }
    else {
        cout<<"restoring tps: "<<tps.size()<<endl;
        for(unsigned int i=0;i<tps.size(); i++) {
            tps[i]->reset();
        }
    }
}

void Solution::delCurrentSolution(void) {
	/*for(int i=0; i<nbNodes; i++)
		for(int j=0; j<nbNodes; j++) matrix[i][j] = 0;*/
	//for(vector<MetroNode*>::iterator it=metro.begin(); it!=metro.end(); it++) {
	for(unsigned int i=0; i<metro.size(); i++) {
		if(metro[i])
			delete metro[i];
		metro[i] = NULL;
	}
}

void Solution::setSolution(MetroNode *mn) {
    MetroNode *lMN = metro[ mn->getId() ];
    assert(lMN->getId() == mn->getId());
    lMN->copy(mn);
    int index;
    for(unsigned int i=0;i<lMN->node.size();i++) {
        /*index = lMN->node[i]->getId();
        tps[index]->copy(mn->tps[index]);*/
        Node *tmp = lMN->node[i];
        Node *tmp2 = mn->node[i];
        assert(tmp->isPrimary == tmp2->isPrimary);
        index = tmp->getId();
        
        //int p=tmp->isPrimary?0:1;
        //cout<<"id: "<<tmp->getId()<<" pred: ("<<tps[index]->pred[0]<<", "<<tps[index]->pred[1]<<") new pred["<<(p)<<"]: "<<mn->tps[index]->pred[p]<<endl;
        if(tmp->getPred()) {
            //cout<<"Actual pred: "<<tmp->getPred()->getId()<<endl;
            SiteNode sn = site_node[index];
            int idmn2 = sn.id_mn[0]==mn->getId()?sn.id_mn[1]:sn.id_mn[0];
            //cout<<" mn1: "<<mn->getId()<<" mn2: "<<idmn2<<endl;
            if(idmn2 != -1) {
                //MetroNode *mn2 = metro[idmn2];
                /*cout<<"mn2 size: "<<mn2->node.size()<<endl;
                for(int j=0;j<(int)mn2->node.size();j++) {
                    assert(mn2->node[i]);
                    if( mn2->node[i]->getId() == tmp->getId() ) {
                        if(mn2->node[i]->getPred()) {
                            cout<<"pred2: "<<mn2->node[i]->getPred()<<" in MN: "<<mn->getMetroNode()->getIdString();
                        } cout<<"No predecessor2: "<<endl;
                        break;
                    }
                }*/
            }
        }
        
        tps[index]->copy(mn->tps[index], tmp->isPrimary);
    }
}

void Solution::setSolution(vector<MetroNode*> _metro, vector<TwoPred*> _tps, int **_matrix, int nbNodes, double _cost, int _time) {
	//delCurrentSolution();
	/*if(_matrix) {
		for(int i=0; i<nbNodes; i++)
			for(int j=0; j<nbNodes; j++)  {
				//matrix[i][j] = _matrix[i][j];
				sol[i][j][0]=-1; sol[i][j][1]=-1;
			}
	}*/
	
	for(unsigned int i=0;i<tps.size(); i++) {
		tps[i]->copy(_tps[i]);
	}

	for(unsigned int i=0; i<metro.size(); i++) {
		metro[i]->copy(_metro[i]);
#ifdef DEBUG
		int mn = _metro[i]->getId();
		/*if(_matrix) {
			for(int j=0; j<(int)_metro[i]->node.size(); j++) {
				Node *tmp = _metro[i]->getNode(j);
				if(tmp == tmp->getMetroNode()) continue;
				int a = tmp->getPred()->getId();
				int b = tmp->getId();
				if(sol[a][b][0] == -1) {
					sol[a][b][0] = mn;
					matrix[a][b] = 1;
				}
				else {
					if(sol[a][b][1] != -1) {
						metro[i]->printNodes();
						cout<<"metro_node: "<<tmp->getMetroNode()->getId()<<endl;
						cout<<"sol[0]: "<<sol[a][b][0]<<" sol[1]: "<<sol[a][b][1]<<endl;
						cout<<"MATRIX["<<a<<" - > "<<b<<"]: "<<_matrix[a][b]<<endl;
					}
					assert(sol[a][b][1] == -1);
					sol[a][b][1] = mn;
					matrix[a][b] = 1;
				}
			}
		}*/
#endif
	}
	
	cost = _cost;
	time2sol = _time;
}

double Solution::copySolution(MetroNode *mn) {
    MetroNode *lMN = metro[ mn->getId() ];
    assert(lMN->getId() == mn->getId());
    int index;
    for(unsigned int i=0;i<lMN->node.size();i++) {
        Node *tmp = lMN->node[i];
        Node *tmp2 = mn->node[i];
        assert(tmp->isPrimary == tmp2->isPrimary);
        index = tmp->getId();
        //mn->tps[index]->copy(tps[index]);
        mn->tps[index]->copy(tps[index], tmp->isPrimary);
    }
    mn->copy(lMN);
    return mn->tCost;
}

double Solution::copySolution(vector<MetroNode*> _metro, vector<TwoPred*> _tps, int **_matrix) {
	assert(_tps.size() == tps.size() ); 
	/*for(int i=0; i<nbNodes; i++) {
		for(int j=0; j<nbNodes; j++) {
			_matrix[i][j] = matrix[i][j];
		}
	}*/
	
	for(unsigned int i=0;i<tps.size(); i++) {
		_tps[i]->copy(tps[i]);
	}
	
	for(unsigned int i=0; i<metro.size(); i++) {
		_metro[i]->copy(metro[i]);
	}
	return cost;
}


void Solution::printTree(void) {
	cout<<"digraph G {"<<endl
    	<<" size =\"4,4\";"<<endl;

    for(int i=0;i<nbNodes;i++) {
        cout<<"Node_"<<site_node[i].id<<" [pos=\""<<site_node[i].x<<","<<site_node[i].y<<"!\"]"<<endl;
    }

    //the label should be the distance between a given node and its predecessor
	for(unsigned int i=0; i<metro.size(); i++) {
        metro[i]->printTree();
    }
    
}


//#define DIST(N1, N2)  sqrt(pow(getX(N1->getId()) - getX(N2->getId()), 2) + pow(getY(N1->getId()) - getY(N2->getId() ), 2) )

double Solution::DIST(Node *N1, Node *N2) {
/*#ifdef SCALE2
	return sqrt(pow(getX(N1->getId())*FACTOR - getX(N2->getId())*FACTOR, 2) + pow(getY(N1->getId())*FACTOR - getY(N2->getId() )*FACTOR, 2) );
#else*/
	return sqrt(pow(getX(N1->getId()) - getX(N2->getId()), 2) + pow(getY(N1->getId()) - getY(N2->getId() ), 2) );
//#endif
}


double Solution::solCost() {
	double cost = 0.0;
	for(unsigned int i=0; i<metro.size(); i++) {
		for(unsigned int j=0;j<metro[i]->node.size(); j++) {
			Node *tmp = metro[i]->getNode(j);
			if(tmp==tmp->getMetroNode()) continue;
			cost+=DIST(tmp, tmp->getPred());
		}
	}
	return cost;
}

double Solution::aggregateCost(void) {
    double cost=0;
    for(unsigned int i=0;i<metro.size();i++) cost+=metro[i]->tCost;
    return cost;
}

void Solution::print(MetroNode *mn, string sol_file) {
    if(!mn) {
        print(sol_file);
        return;
    }
    
    string s;
    s+="Node Id \tPredecessor Id \tDistance root2NodeId\n";
    for(unsigned int i=0;i<mn->node.size();i++) {
        Node *tmp=mn->node[i];
        if(tmp->getPred()) {
            stringstream sd;
            //sd<<tmp->getDistance(tmp->getPred());
            sd<<tmp->getDistance();
            //s+=tmp->getIdString()+" \t"+tmp->getPred()->getIdString()+" \t"+tmp->getIdString()+" \t"+sd.str()+"\n";
            s+=tmp->getIdString()+" \t"+tmp->getPred()->getIdString()+" \t"+sd.str()+"\n";
        }
    }

	if(sol_file.compare("console") == 0) {
		cout<<"SOLUTION "<<endl;
		cout<<s;
		cout<<"END SOLUTION"<<endl;
	}
	else if(sol_file.compare("none") == 0) {}
	else{
		ofstream mfile;
		mfile.open(sol_file.c_str());
		mfile<<s;
		mfile.close();
	}

}

void Solution::print(string sol_file) {
	//vector<Pair> node;
	//int nbExchangeSites = nbNodes - (int)metro.size();
	vector< pair<int, pair<string, string> > > node;
	//for(int i=0;i<nbExchangeSites; i++) node.push_back( make_pair(-1, make_pair("none", "none") ) );
	
	string s;
    

    /*
    for(unsigned int i=0;i<metro.size();i++) {
        MetroNode *m = metro[i];
        for(unsigned int j=0;j<m->node.size();j++) {
            Node *n = m->node[j];
            if(n->getPred()) {
                s+=(n->getMetro()->getIdString()+"\t"+n->getPred()->getIdString()+"\t"+n->getIdString()+"\n");
            }
        }
    }
     */
    for(unsigned int i=0;i<site_node.size();i++) {
        if(site_node[i].id_mn[0] == -1 && site_node[i].id_mn[1] == -1) {
            continue;
        }
        Node *n1 = site_node[i].n1;
        Node *n2 = site_node[i].n2;
        assert(n1->getId() == n2->getId());
        stringstream sd1, sd2;
        sd1<<n1->getDistance(n1->getPred());
        sd2<<n2->getDistance(n2->getPred());

        //s+=(ss.str()+"\t"+n1->getIdString()+"\t"+n1->getMetro()->getIdString()+"\t"+n2->getIdString()+"\t"+n2->getMetro()->getIdString()+"\n")
        s+=(n1->getIdString()+  "\t"+n1->getPred()->getIdString()+"\t"+n1->getMetro()->getIdString()+"\t"+sd1.str()+
                                "\t"+n2->getPred()->getIdString()+"\t"+n2->getMetro()->getIdString()+"\t"+sd2.str()+
            "\n");
    }
    

	/*
	for(int i=0; i<(int)metro.size(); i++) {
        if(!mn || mn != metro[i]) continue;
		for(int j=0;j<(int)metro[i]->node.size();j++) {
			Node *tmp = metro[i]->getNode(j);
			if(tmp == tmp->getMetroNode()) {
				continue;
			}
			for(int p=0; p<(int)node.size(); p++) {
				if(node[p].first == tmp->getInputId()) {
					if(node[p].second.first.compare("none") == 0) {
						cout<<"id: "<<node[p].first<<" "<<node[p].second.first<<endl;
						assert(false);
					}
					else if(node[p].second.second.compare("none") == 0) {
						//node[p].second.second = tmp->getPred()->getIdString();
                        node[p].second.second = tmp->getPred()->getIdString() + "\t" + tmp->getMetroNode()->getIdString();
						break;
					}
					else {
						cout<<"HHHHH: "<<endl;
						cout<<"Node: "<<tmp->getInputId()<<" 1) "<<node[p].second.first<<" 2) "<<node[p].second.second<<endl;
						cout<<"trying "<<tmp->getPred()->getIdString()<<endl;
						assert(false);
					}
				}
				else if(node[p].first == -1) {
					node[p].first = tmp->getInputId();
					//node[p].second.first = tmp->getPred()->getIdString();
                    node[p].second.first = tmp->getPred()->getIdString() +"\t"+ tmp->getMetroNode()->getIdString();
					break;
				}
			}
		}
	}

	for(int i=0;i<(int)node.size();i++) {
		//cout<<node[i].first<<"\t"<<node[i].second.first<<"\t"<<node[i].second.second<<endl;
		stringstream ss;
		ss<<node[i].first;
		s+=ss.str()+"\t"+node[i].second.first+"\t"+node[i].second.second+"\n";
	}
     */
	if(sol_file.compare("console") == 0) {
		cout<<"SOLUTION "<<endl;
		cout<<s;
		cout<<"END SOLUTION"<<endl;
	}
	else if(sol_file.compare("none") == 0) {}
	else{
		ofstream mfile;
		mfile.open(sol_file.c_str());
		mfile<<s;
		mfile.close();
	}
	//printTree();
}



#endif