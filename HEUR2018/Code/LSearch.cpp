

#include <iostream>
#include <vector>

#include <math.h>
#include <assert.h>

#include <iomanip>

#include "BH.h"
#include "LSearch.h"
#include "ProbMatching.h"


LSearch::LSearch(Parameters _param) : g(_param.seed) {
	mclock.init();
	bestSol = numeric_limits<double>::max();
	lastChangeSol = numeric_limits<double>::max();
	totalPerturbations = 0;
	lastChangeSol = 0;
	iter = 0;
	param = _param;

    maxLocalTime=static_cast<double>(_param.local_time);

    targetSol = param.target_sol;

	maxNoImp = param.iter_no_imp;
	maxIter = param.max_iter;
	maxTime = param.max_time;
        
	nbMetroNodes = metro_node.size();
    
    initData = true;
    activeMN = NULL;
    
    g_conflict = false;
    
    randomSeed(_param.seed);
}

void LSearch::randomSeed(int s) {
    sgenrand(s, &mt);
}

int LSearch::random(int i) {
    return static_cast<unsigned int>(genrand(&mt)) % i;
}

void LSearch::setMetroNodeRandomSeed(void) {
    for(int i=0;i<(int)metro_node.size();i++) metro_node[i]->randomSeed( random(numeric_limits<int>::max()) );
}

double LSearch::costTrivialSolution(void) {
    double cost=0;
    for(int i=0;i<(int)metro_node.size();i++) {
        for(int j=0;j<(int)metro_node[i]->node.size();j++) {
            Node *tmp = metro_node[i]->node[j];
            if(tmp!=tmp->getMetroNode())
                cost+=tmp->getCost(tmp->getMetroNode());
        }
    }
    return cost;
}

void LSearch::InitMatrix(void) {
	/*cout<<"INITIAL MATRIX: "<<endl;
	matrix = new int*[nbNodes];
	for(int i=0; i<nbNodes ;i++) {
		matrix[i] = new int[nbNodes];
		for(int j=0; j<nbNodes; j++) matrix[i][j]=0;
	}*/
}

void LSearch::init(void) {
    //loading data
    readData();

	for(int i=0; i<(int)metro_node.size(); i++) {
		//metro_node[i]->setPosNodes();

		/*for(int j=0;j<(int)metro_node[i]->node.size(); j++) {
			Node *tmp = metro_node[i]->getNode(j);
			if(tmp!=tmp->getMetroNode()) 
				valid_nodes.push_back(tmp);
		}
		inv_nodes.push_back(vector<Node*>());*/
        metro_node[i]->activeNodes();
        activeMetroNode.push_back(metro_node[i]);
	} 
	setMetroNodeRandomSeed();
    
    if(g_conflict) {
        g.computeNodeConflict(metro_node);
    }
	g.init(metro_node, site_node);
	//g.print();
	//assert( (int) valid_nodes.size() == (nbExchangeSites*2) && invalid_nodes.size() == 0 );
	best_sol.init(nbNodes, metro_node, site_node);
}

void LSearch::initMST(void) {
    readData();
    g.init(metro_node, site_node);
    //g.initEmpty(metro_node, site_node);
    initData = false;

    setMetroNodeRandomSeed();
    /*
    for(int i=0; i<(int)metro_node.size(); i++) {
		//metro_node[i]->setPosNodes();
        inv_nodes.push_back(vector<Node*>());
	}
    */
	best_sol.init(nbNodes, metro_node, site_node);
}

void LSearch::initLocalInfoMST(MetroNode *mn) {
    for(int i=0;i<(int)tps.size();i++) tps[i]->reset();

    /*for(int i=0;i<(int)metro_node.size();i++) {
        metro_node[i]->resetActiveAndInactive();
        if(metro_node[i]!=mn) inactiveMetroNode.push_back(metro_node[i]);
        //if(metro_node[i]==mn) activeMetroNode.push_back(
    }
    mn->activeNodes();
    activeMetroNode.clear();*/
    activeMetroNode.clear();
    mn->activeNodes();
    activeMetroNode.push_back(mn);

    /*
    valid_nodes.clear();
    for(int i=0;i<(int)inv_nodes.size(); i++)
        inv_nodes[i].clear();
    inv_nodes.clear();
    
    for(int i=0;i<(int)metro_node.size(); i++) {
        inv_nodes.push_back(vector<Node*>());
        if(mn == metro_node[i]) {
            for(int j=0;j<(int)mn->node.size(); j++) {
                Node *tmp = metro_node[i]->getNode(j);
                if(tmp!=tmp->getMetroNode())
                    valid_nodes.push_back(tmp);
            }
        }
    }
    */
}

//bool NodeGreater (Node *i, Node *j) { return (i->getInputId() < j->getInputId()); }
//bool EdgeGreater( pair<Node*, Node*> i, pair<Node*,Node*> j) { return (i.first->getDistance(i.second) < j.first->getDistance(j.second)); }

void LSearch::createTreePrimAlg(MetroNode *metro) {
    assert(metro);
    //priority_queue<Node*, vector<Node*>, NodeGreater> tmp;
    //priority_queue< pair<Node*,Node*>, vector< pair<Node*, Node*> >, EdgeGreater > q;
    priority_queue< edgeGraph, vector< edgeGraph >, EdgeGreater> pq;

    //compute shortest paths
    metro->dijkstra();
    double dcost=0;
    for(unsigned int i=0;i<metro->node.size();i++) {
        Node *tmp = metro->node[i];
        if(!tmp->isMetroNode()) dcost+=tmp->getCost(tmp->spred);
    }
    cout<<"Dijkstra cost: "<<dcost<<endl;
    

    double minlambda=0.0;
    for(unsigned int i=0;i<metro->node.size();i++) {
        Node *tmp = metro->node[i];
        //if(!tmp->isMetroNode() && minlambda < tmp->getDistance(tmp->getMetroNode())) minlambda=tmp->getDistance(tmp->getMetroNode());
        if(!tmp->isMetroNode() && minlambda < tmp->getSDistance()) minlambda=tmp->getSDistance();
    }
    cout<<"Min lambda: "<<minlambda<<endl;
    if(LAMBDA==-1) {
        LAMBDA = minlambda;
    }

    int addedNodes=0;
    double c=0;
    //rooted minimum spanning tree with distance constraint (prim's algorithm)
    Node *s = metro->getMetroNode();
    Node *e;
    do {
        assert(s);
        e = NULL;
        for(unsigned int i=0;i<metro->node.size();i++) {
            e = metro->node[i];
            if(!e->isMetroNode() && !e->getPred()) {
                //cout<<"Opt: "<<s->getIdString()<<" -> "<<e->getIdString()<<" cost: "<<s->getDistance(e)<<endl;
                pq.push( make_pair(s, e) );
            }
            /*else {
                cout<<"Invalid opt: "<<s->getIdString()<<" -> "<<e->getIdString()<<" cost: "<<s->getDistance(e)<<endl;
            }*/
        }
        e = NULL;
        s = NULL;
        do {
            edgeGraph edge = pq.top();
            pq.pop();
            if(!endNode(edge)->getPred()) {
                s = startNode(edge);
                e = endNode(edge);
            }
            if(e && !e->validMove( startNode(edge) ) ){
                s=NULL;
                e=NULL;
            }
            //cout<<"Tested: "<<(s?s->getIdString():"NULL")<<" -> "<<(e?e->getIdString():"NULL")<<endl;
        }while(!e && !pq.empty());
        addedNodes++;
        //cout<<"Tested: "<<(s?s->getIdString():"NULL")<<" -> "<<(e?e->getIdString():"NULL")<<" Dist: "<<s->getDistance(e)<<endl;
        c+=s->getDistance(e);
        //cout<<"s: "<<(s?s->getIdString():"NULL")<<" e: "<<(e?e->getIdString():"NULL")<<" pq: "<<pq.empty()<<endl;
        if(e && s) {
            e->moveSubTree(s,NULL);
        }
        s = e;
    } while(s);

    //cout<<"Added nodes with Prim algorithm: "<<addedNodes<<" cost: "<<c<<endl;
    //adding remaining nodes using shorthest path..
    for(unsigned int i=0;i<metro->node.size();i++) {
        Node *tmp = metro->node[i];
        //not added yet
        if(!tmp->getPred() && !tmp->isMetroNode()) {
            vector< edgeGraph >npath;
            npath.push_back(make_pair( tmp->spred, tmp));
            double ndist = tmp->getDistance(tmp->spred);
            //cout<<"NODE: "<<tmp->getIdString()<<" spred: "<<tmp->spred->getIdString()<<" dist: "<<tmp->getDistance(tmp->spred)<<" lambda: "<<tmp->getLambda()<<endl;
            tmp=tmp->spred;
            while(tmp) {
                Node *spPred=tmp->spred;
                //cout<<"Node: "<<tmp->getIdString()<<endl;
                //if(tmp->isMetroNode()) cout<<"Metro Node"<<endl;
                if(tmp->isMetroNode()) {
                    if(ndist > tmp->getLambda(tmp,NULL)) cout<<"Error ndist: "<<ndist<<" lambda: "<<tmp->getLambda(tmp,NULL)<<endl;
                    assert(ndist <= tmp->getLambda(tmp,NULL));
                    break;
                }
                if(tmp->getPred()) {
                    //assert(tmp->getPred());
                    //pre-computed shortest path
                    double lambda=tmp->getLambda();
                    //if existing path is good enough, then use it...
                    if(ndist+tmp->getDistance() < lambda) {
                        break;
                    }
                }
                //cout<<"ndist: "<<(ndist+tmp->getDistance())<<": "<<lambda<<endl;
                ndist+=tmp->getDistance(spPred);
                npath.insert(npath.begin(), make_pair(spPred,tmp));
                tmp=tmp->spred;
            }
            
            for(unsigned int i=0;i<npath.size();i++) {
                Node *s=startNode(npath[i]);
                Node *e=endNode(npath[i]);
                e->changePred(s);
            }
        }
    }
    //metro->printTree();
    //assert(false);
}

void LSearch::createGreedyTree(MetroNode *metro) {
    if(metro) {
        cout<<"Creating Greedy Tree for "<<metro->getIdString()<<endl;
    }
    
    vector<Node*> nodes;
    //vector<Node*> notAdded;
    MetroNode *mn = NULL;
    Node::firstImprovement = true;
    //double maxDistance = -1;
    for(int i=0;i<(int)metro_node.size();i++) {
        if(!metro) mn = metro_node[i];
        else if(metro == metro_node[i]) mn=metro;
        else mn = NULL;
        if(mn) {
            vector<Node*> sameAsLE;
            Node *tmpSuc = NULL;
            for(int j=0;j<(int)mn->node.size();j++) {
                Node *tmp = mn->node[j];
                if(tmp != tmp->getMetroNode()) {
                    if(tmp->getDistance(tmp->getMetroNode()) == 0) {
                        sameAsLE.push_back(tmp);
                    }
                    else tmpSuc = tmp;
                    if(!tmp->validMoveNode2(tmp->getMetroNode(), NULL) ) {
                        cout<<"Error invalid Trivial Solution..."<<endl
                            <<"Distance ("<<tmp->getIdString()<<" , "<<tmp->getMetroNode()->getIdString()<<"): "<<tmp->getDistance(tmp->getMetroNode())<<endl
                            <<"Max Lambda:"<<tmp->getLambda(tmp->getMetroNode(), NULL)<<endl;
                        assert(false);
                    }
                    tmp->moveSubTree(tmp->getMetroNode(), NULL);
                    nodes.push_back(tmp);
                }
            }
            assert(tmpSuc);
            while(!sameAsLE.empty()) {
                Node *tmp = sameAsLE[0];
                assert(tmpSuc->validMoveNode2(tmp,NULL));
                tmpSuc->validMoveNode2(tmp,NULL);
                tmpSuc=tmp;
                sameAsLE.erase(sameAsLE.begin());
            }
            
            cout<<"First phase cost: "<<mn->tCost<<endl;
            int reallocated;
            int initIter = 0;
            do {
                initIter++;
                reallocated=0;
                nodes = mn->getMetroNode()->suc;
                //std::sort(nodes.begin(), nodes.end(), NodeGreater);
                for(int index=0;index<(int)nodes.size();index++) {
                    Node *bNode = nodes[index];
                    pair<Node**, double> m = bNode->bestMoveSubTree();
                    if(m.first) {
                        bNode->moveSubTree(m.first[0], m.first[1]);
                        reallocated++;
                    }
                    //bNode->moveSubTree(bNode->getMetroNode(), NULL);
                }
                cout<<"Init iter: "<<initIter<<" moved nodes: "<<reallocated<<" cost: "<<mn->tCost<<" First level: "<<mn->getMetroNode()->suc.size()<<endl;
            } while (reallocated > 0);
        }
    }
    Node::firstImprovement = false;
    cout<<"greedy tree done!!"<<endl;
}

void LSearch::createTrivialTree(MetroNode *metro) {
    if(metro) {
        cout<<"Create trivial tree for "<<metro->getIdString()<<endl;
    }
    vector<Node*> nodes;
    MetroNode *mn = NULL;
    double maxDistance=-1;
    for(int i=0;i<(int)metro_node.size();i++) {
        if(!metro) mn = metro_node[i];
        else if(metro == metro_node[i]) mn=metro;
        else mn = NULL;
        if(mn) {
            for(int j=0;j<(int)mn->node.size();j++) {
                Node *bNode = mn->node[j];
                if(bNode!=bNode->getMetroNode()) {
                    if(bNode->getDistance(bNode->getMetroNode()) > maxDistance) maxDistance = bNode->getDistance(bNode->getMetroNode());
                    bNode->moveSubTree(bNode->getMetroNode(), NULL);
                }
            }
        }
    }
    cout<<"Max Distance: "<<maxDistance<<endl;
}

void LSearch::createTree(MetroNode *metro) {
    vector<Node*> nodes;
    vector<Node*> nAdded;
    vector<Node*> addedNode;
    if(metro)
        cout<<"Create tree for:" <<metro->getId()<<endl;
    //if(idMN==-1) idMN=(int)metro_node.size();
    MetroNode *mn;
    for(int i=0;i<(int)metro_node.size();i++) {
        if(!metro) mn = metro_node[i];
        else if(metro == metro_node[i]) mn = metro;
        else mn = NULL;
        if(mn) {
            for(int j=0;j<(int)mn->node.size();j++) {
                if(mn->node[j]->getMetroNode() != mn->node[j])
                    nodes.push_back(mn->node[j]);
            }
        }
    }
    int nindex;
    int bNode = -1;
    while(!nodes.empty()) {
        bool added = false;
        while(!nodes.empty()) {
            bNode = random(nodes.size());
            //pair<Node**, double> best = nodes[bNode]->bestDistanceMoveNode();
            pair<Node**, double> best = nodes[bNode]->bestMoveST(numeric_limits<double>::max());
            if(best.first) {
                added=true;
                nodes[bNode]->moveSubTree(best.first[0], best.first[1]);
                addedNode.push_back(nodes[bNode]);
            }
            else
                nAdded.push_back(nodes[bNode]);
            nodes.erase( nodes.begin() + bNode );
        }
        for(int i=0;i<(int)nAdded.size();i++) nodes.push_back(nAdded[i]);
        nAdded.clear();
        if(!added) {
            //metro_node[0]->printTree();
            cout<<"Tree size: "<<addedNode.size()<<endl;
            //deleting 10% of nodes in the tree
            int tdiv=addedNode.size() * 0.5;
            if(tdiv<5) tdiv=addedNode.size();
            //cout<<"added nodes: "<<addedNode.size()<<" tdiv: "<<tdiv<<" ";//<<" max len: "<<metro_node[0]->getMetroNode()->dist2leaf<<endl;
            //for(int i=0;i<(int)metro_node[0]->getMetroNode()->suc.size();i++) cout<<metro_node[0]->getMetroNode()->suc[i]->dist2leaf<<" ";
            //cout<<endl;
            /*cout<<"Not added: ";
            for(int i=0;i<(int)nodes.size();i++) cout<<nodes[i]->getIdString()<<" ";
            cout<<endl;*/
            for(int i=0;i<tdiv;i++) {
                nindex = random(addedNode.size());
                Node *aux = addedNode[nindex];
                if(aux->delNode()) {
                    addedNode.erase( addedNode.begin() + nindex );
                    nodes.push_back( aux );
                }
            }
        }

        /*while(!nAdded.empty()) {
            bNode = Random(nAdded.size());
            pair<Node**, double> best = nAdded[bNode]->bestDistanceMoveNode();
            //pair<Node**, double> best = nodes[bNode]->bestMoveST(numeric_limits<double>::max());
            if(best.first) {
                nAdded[bNode]->moveSubTree(best.first[0], best.first[1]);
                addedNode.push_back(nAdded[bNode]);
            }
            else
                nodes.push_back(nAdded[bNode]);
            nAdded.erase( nAdded.begin() + bNode );
        }*/
    }
    cout<<"TREE SIZE: "<<addedNode.size()<<endl;
    /*cout<<"Initial solution: "<<endl;
    metro_node[0]->printTree();
    cout<<"------"<<endl;*/
    nodes.clear();
    addedNode.clear();
}

void LSearch::InitialSolution(void) {
	//cout<<"Initial Solution\n"<<endl;
	totalCost = 0;

	/*for(int i=0; i<nbNodes; i++)
		for(int j=0; j<nbNodes; j++) matrix[i][j] = 0;*/ 

	for(vector<MetroNode*>::iterator it=metro_node.begin(); it!=metro_node.end(); it++) {
        MetroNode *itMN = *it;
    //for(int j=0;j<(int)metro_node.size();j++) {
        
		//MetroNode *itMN = metro_node[j];
		for(int i=0; i<(int) itMN->node.size(); i++) { 
			Node *tmp = itMN->getNode(i);
			tmp->setMetroNode( itMN );
			//tmp->setSiteNodeVector(site_node);
			tmp->setMatrix(matrix);
		}

		Node *MN = itMN->getMetroNode();
		itMN->size = itMN->node.size()-1;
			//cout<<"Metro Node: "<<MN<<endl;
		MN->reset(false);
        if(method==2 && (param.sol_metro == -1 || param.sol_metro==itMN->getId() ) )
            //createGreedyTree(itMN);
            //createTrivialTree(itMN);
            //createTree(itMN);
            createTreePrimAlg(itMN);
			//dist2leaf root
        //itMN->InitialTree();
        //itMN->printTree();
        //assert(false);
        //itMN->BDBHeuristic();
        /*
		double distTmp = -1;
		for(int i=1; i<(int) itMN->node.size(); i++) {
			itMN->getNode(i)->initPred();
			if(distTmp < itMN->getNode(i)->getDistance() ) distTmp=itMN->getNode(i)->getDistance();
		}
		MN->dist2leaf = distTmp;
			//it->printNodes();
			//cout<<"MN["<<it->getId()<<" ]: "<<it->size<<" dist2leaf: "<<MN->dist2leaf<<endl;
         */
	}
    cout<<"Done reading data"<<endl;
    if(method==1)
        createTrivialTree(NULL);
        //createTree(NULL);

    for(int i=0;i<(int)metro_node.size();i++) {
        totalCost+=metro_node[i]->tCost;
        //metro_node[i]->printTree();
    }
    //metro_node[0]->printTree();
    //cout<<"cost: "<<totalCost<<endl;
    //assert(false);
}

void LSearch::initFromBestSol(void) {
	cout<<"metro_node size: "<<metro_node.size()<<" metro size: "<<best_sol.metro.size()<<endl;
	assert(metro_node.size() == best_sol.metro.size());
	for(int i=0; i<(int)metro_node.size(); i++) {
		metro_node[i]->copy(best_sol.metro[i]);
	}

	/*for(int i=0;i<nbNodes;i++) 
		for(int j=0;j<nbNodes;j++)
		matrix[i][j] = best_sol.matrix[i][j];*/
	totalCost = best_sol.cost;
}


//NEED TO RE-DO this to change the optimization function to cost-related values
void LSearch::testAllDistances(void) {
		//test distances (top - down)
	//int m2[nbNodes][nbNodes];
	int **m2 = new int*[nbNodes];
	for(int i=0;i<nbNodes;i++) {
		m2[i] = new int[nbNodes];
		for(int j=0;j<nbNodes;j++) m2[i][j] = 0;
	}

	double cost=0.0;
	for(int i=0; i<nbMetroNodes; i++) {
        if(param.sol_metro!=-1 || param.sol_metro != metro_node[i]->getId()) continue;
		//cout<<"Testing MetroNode["<<i<<"] with "<<metro_node[i].node.size()<<" nodes"<<endl;
		metro_node[i]->testDistance();

		for(int j=0;j<(int) metro_node[i]->node.size(); j++) {
			Node *tmp = metro_node[i]->getNode(j);
			//if(tmp->getPred()) {
			if(tmp!=tmp->getMetroNode()) {
					cost+= tmp->getCost( tmp->getPred() );
					//dist+= DIST2(tmp, tmp->getPred() );
                    if(method!=2) {
                        m2[tmp->getPred()->getId()][tmp->getId()]++;
                        assert(m2[tmp->getPred()->getId()][tmp->getId()] == 1);
                    }
					tmp->checkLoop();
				}
				//cout<<"dist: "<<tmp->getDistance()<<endl;
				if(tmp->getDistance() > LAMBDA) {
					cout<<tmp->getPred()->getId()<<"->"<<tmp->getId()<<endl;
					cout<<"("<<getX(tmp->getPred()->getId())<<","<<getY(tmp->getPred()->getId())<<") -> ("<<getX(tmp->getId())<<", "<<getY(tmp->getId())<<")"<<endl;
					cout<<"dist: "<<tmp->getDistance()<<" LAMBDA: "<<LAMBDA<<endl;
				}
				assert(tmp->getDistance() <= LAMBDA);
			}
		}


    if(param.sol_metro == 1) {
        if(fabs(cost - totalCost) >= EPSILON) cout<<"dist: "<<cost<<" totalCost: "<<totalCost<<endl;
        assert( fabs(cost - totalCost) < EPSILON );
    }

	//test distances (down - top) dist2leaf
	for(int i=0; i<nbMetroNodes; i++) {
        if(param.sol_metro != -1 && param.sol_metro!= metro_node[i]->getId()) {
		//cout<<"metroNode["<<i<<"]: "<<metro_node[i].getMetroNode()->dist2leaf<<endl;
             assert(metro_node[i]->getMetroNode()->dist2leaf <= LAMBDA);
            metro_node[i]->testDist2Leaf(metro_node[i]->getMetroNode());
        }
	}
	
	for(int i=0;i<nbNodes;i++) {
		delete [] m2[i];
	}
	delete [] m2;
	
}

void LSearch::localStart(void) {  }
bool LSearch::localStop(void) {
    return stop();
}
bool LSearch::stop(void) {
	if(iter < maxIter && maxTime > mclock.elapsed() && !forceStop && totalCost > targetSol && !param.initial_sol_only) return false;
	return true;
}

bool LSearch::isDiversification(void) {
    return activeMetroNode.empty();
}

void LSearch::readData(void) {
    if( param.input_format == 1) {
        vector<Pair> metro_node1;
        vector<Pair> site_node1;
        //read_sites_nodes_format2("../data/ire_data_100_cover.txt", site_node1);
        //read_metro_nodes_format2("../data/HEAnet_metronodes.txt", metro_node1);
        
        read_sites_nodes_format2(param.data_exchange_site.c_str(), site_node1);
        read_metro_nodes_format2(param.data_metro_node.c_str(), metro_node1);
        
        Set_Init_ExchangeSites_Format2(param.data_exchange_site_mn.c_str(), metro_node1, site_node1, metro_node
                                       , site_node);
    }
    
    else if(param.input_format == 2) {
        read_metro_nodes(param.data_metro_node.c_str(), metro_node, site_node);
        read_site_nodes(param.data_exchange_site.c_str(), metro_node.size(), site_node);
        Set_Init_ExchangeSites(param.data_exchange_site_mn.c_str(), site_node, metro_node);
    }
    else {
        Read_Customers(param.data_exchange_site.c_str(), metro_node, site_node);
    }
    nbMetroNodes = metro_node.size();
    nbExchangeSites = site_node.size() - metro_node.size();
    nbNodes = nbMetroNodes + nbExchangeSites;
    cout<<"nbMetroNodes: "<<nbMetroNodes<<endl<<"nbExchangeSites: "<<nbExchangeSites<<endl<<"nbNodes: "<<nbNodes<<endl;
    //printing global info...
    /*for(int i=0;i<(int)metro_node.size(); i++) {
        cout<<"Metro["<<metro_node[i]->getMetroNode()->getInputId()<<"] ";
        for(int j=0; j<(int)metro_node[i]->node.size(); j++)
            cout<<metro_node[i]->getNode(j)->getInputId()<<" ";
        cout<<endl;
    }*/
    //tps = new TwoPred[site_node.size()];
    for(int i=0;i<(int)site_node.size();i++) {
        TwoPred *tp = new TwoPred();
        tp->id_node =  site_node[i].id;
        tps.push_back(tp);
    }


    for(int i=0; i<(int)metro_node.size(); i++) {
        metro_node[i]->road_factor = param.road_factor;
        metro_node[i]->setPosNodes();
        //metro_node[i]->initDistMatrix(site_node);
        //metro_node[i]->initCostMatrix(site_node);
        metro_node[i]->tps = tps;
    }
    //read_cost_dist("d", "dmn", metro_node, site_node);
    read_cost_dist(param.cost_dist_es2es.c_str(), param.cost_dist_mn2es.c_str(), metro_node, site_node);
    //InitMatrix();
    InitialSolution();
    cout<<"Initial Solution OK"<<endl;
}

//cost of a metro node solution (no disjointness constraint)
double LSearch::MSTCost(MetroNode *mn) {
    double cost = 0.0;
    int tnodes=0;
    //cout<<"Nodes["<<mn->getId()<<"]: ";
    for(int i=0;i<(int)mn->node.size(); i++) {
        tnodes++;
        Node *tmp = mn->getNode(i);
        //cout<<tmp->getInputId()<<" ";
        if(tmp!=tmp->getMetroNode()) {
            cost+=tmp->getCost(tmp->getPred());
        }
    }
    //cout<<endl<<"tNodes: "<<tnodes<<endl;
    cout<<"tNodes MetroNode["<<mn->getId()<<"]: "<<tnodes<<endl;
    return cost;
}

void LSearch::ILS_MST(int mn_i) {
    initMST();
    g.clear();
    for(int i=0;i<(int)metro_node.size(); i++) {
        if( mn_i == metro_node[i]->getMetroNode()->getInputId() ) {
            MetroNode *mn = metro_node[i];
            activeMN = mn;
            initLocalInfoMST(mn);
            ILS();
            double lcost = MSTCost(mn);
            cout<<"Cost for MN["<<i<<"]: "<<lcost<<endl;
            for(int j=0;j< (int)mn->node.size(); j++) {
                Node *tmp = mn->getNode(j);
                if(tmp!=tmp->getMetroNode())
                    cout<<tmp->getPred()->getInputId()<<" -> "<<tmp->getInputId()
                        <<"[label=\""<<tmp->getDistance(tmp->getPred())<<"\""<<endl;
            }
        }
    }
}

//iterative local search method for the minimum spanning tree
void LSearch::ILS_MST(void) {
    initMST();
    g.clear();
    vector<double> localDistance;
    cout<<"SOL METRO : "<<param.sol_metro<<endl;
    for(int i=0;i<(int)metro_node.size(); i++) {
        //Node *mn = metro_node[i]->getMetroNode();
        if(param.sol_metro != -1 && param.sol_metro != metro_node[i]->getId() ) continue;
        activeMN = metro_node[i];
        cout<<"Solving metro node: ["<<metro_node[i]->getId()<<"]: "<<endl;
        initLocalInfoMST(metro_node[i]);
        double lcost = MSTCost(metro_node[i]);
        cout<<"initial local cost: "<<lcost<<endl;
        ILS();
        lcost = MSTCost(metro_node[i]);
        localDistance.push_back(lcost);
        //MetroNode *mn = metro_node[i];
        cout<<"best local cost: "<<lcost<<endl;
        /*cout<<"First level child: "<<mn->getMetroNode()->suc.size()<<" with sizes: "<<endl;
        for(int j=0;j<(int)mn->getMetroNode()->suc.size();j++) {
            cout<<"PON: "<<j<<" -> "<<mn->getMetroNode()->suc[j]->getSizeTree()<<" ("<<mn->getMetroNode()->suc[j]->dist2leaf<<") "<<endl;
        }
        cout<<"Users: ======"<<endl;
        for(int index=0;index<mn->node.size();index++) {
            Node *tmp = mn->node[index];
            if(!tmp->isMetroNode()) {
                cout<<tmp->getIdString()<<" PON id: "<<tmp->getFNode()->getIdString()<<" Dist2LocalExchange: "<<tmp->getDistance(tmp->getMetroNode())<<" Dist2FirstLevel: "<<tmp->getDistance(tmp->getFNode())<<endl;
            }
        }*/
    }
    if(param.sol_metro == -1) {
        cout<<"Results======="<<endl;
        for(int i=0;i<(int)metro_node.size(); i++) {
            cout<<"MetroNode["<<metro_node[i]->getMetroNode()->getInputId()<<"]: "<<localDistance[i]<<endl;
        }
        cout<<"=============="<<endl;
    }
}

void LSearch::ILSPar(void) {
	cout << "fixed:\n" <<fixed;
    Solution best_ils;
    g_conflict = true;
    if(initData)
        init();
    InitData();
    mclock.init();
    cout<<"Initial Solution: "<<totalCost<<endl;
    cout<<"Trivial Solution: "<<costTrivialSolution()<<endl;

    //g.printConflicts();
    //g.printGraph();
    best_ils.init(nbNodes, metro_node, site_node);
    global_best.init(nbNodes, metro_node, site_node);
    best_ils.name = "best_ils";
    global_best.name = "global_best";
    best_sol.name = "best_sol";

    testAllDistances();

    bestSol = numeric_limits<double>::max();
    LSearch::updateBest();

    lastChange = 0;
    iter = 0;
    
    //int nbThreads = 4;
    int idThread;
    omp_set_num_threads(nbThreads);
    vector<int> mns;
    
    ProbMatching pm;
    double p_min = 1.0;
    double alpha = 0.5;
    pm.init( (int)metro_node.size(), p_min, alpha);


    global_best.setSolution(best_sol.metro, best_sol.tps, NULL, nbNodes, best_sol.cost, best_sol.time2sol);
    best_ils.setSolution(best_sol.metro, best_sol.tps, best_sol.matrix, nbNodes, best_sol.cost, best_sol.time2sol);

    int tt=0;
    while(!stop()) {
        tt++;
        //global_best.setSolution(best_sol.metro, best_sol.tps, NULL, nbNodes, best_sol.cost, best_sol.time2sol);
        //best_ils.setSolution(best_sol.metro, best_sol.tps, best_sol.matrix, nbNodes, best_sol.cost, best_sol.time2sol);
        //mns = g.indepdentSet(nbThreads, &pm);

        /*mns = g.indepdentSet(nbThreads, NULL);
        metro_node_core.clear();
        assert(nbThreads <= (int) mns.size());
        //cout<<"ttIter: "<<tt<<" MetroNodes: ";

        for(int i=0;i<nbThreads;i++) {
            //cout<<"Thread: "<<i<<" Metro Node: "<<metro_node[ mns[i]]->getMetroNode()->getIdString()<<" tt: "<<tt<<endl;
            //cout<<metro_node[ mns[i]]->getMetroNode()->getIdString()<<" ";
            metro_node_core.push_back(metro_node[ mns[i] ]);
        }
         */
        selectMetroNodes();
        localStart();
        stopSearch = false;
        //assert(tt==1);
        //cout<<endl;

        /*
        for(int i=0;i<1000;i++) {
            mns = g.indepdentSet(nbThreads);
            metro_node_core.clear();
            for(int j=0;j<nbThreads;j++) {
                //MetroNode *mn = idThread<(int)metro_node_core.size()?metro_node_core[idThread]:NULL;
                MetroNode *mn = metro_node[ mns[j] ];
                if(mn) mn->usage++;
            }
        }
        cout<<"usage: "<<endl;
        for(int i=0;i<(int)metro_node.size();i++) {
            cout<<metro_node[i]->getId()<<", "<<metro_node[i]->usage<<endl;
        }
        cout<<"===="<<endl;*/
        //cout<<"Global tt iter: "<<tt<<endl;
        #pragma omp parallel private(idThread)
        {
            idThread = omp_get_thread_num();
            //MetroNode *mn = idThread<(int)metro_node_core.size()?metro_node_core[idThread]:NULL;
            MetroNode *mn = NULL;
            if(idThread < (int) metro_node_core.size())
                mn = metro_node_core[idThread];
            int indexMetro = -1;
            double initGlobalBest = numeric_limits<double>::max();
            if(mn) {
                indexMetro = mn->getId();
                initGlobalBest = global_best.metro[idThread]->tCost;
            }
            //bool updateBestIls = false;

            if(mn) {
                while( mn && !localStop() ) {
                    int metroIndex = mn->getId();

                    //best_sol.copySolution(mn);

                    if(isDiversification())
                        searchDiversification();

                    searchIntensification();

                    updateBest(global_best, best_sol, false);

                    bool forceUpdate = (mn->random(100) < 5);
                    if(best_ils.metro[metroIndex]->tCost > best_sol.metro[metroIndex]->tCost || forceUpdate) {
                        updateBest(best_ils, best_sol, forceUpdate);
                    }
                    else {
                        updateBest(best_sol, best_ils, false);
                    }
                    best_sol.copySolution(mn);
                }
                updateBest(global_best, best_sol, true);
                updateBest(best_ils, best_sol, true);
                if(global_best.metro[indexMetro]->tCost < initGlobalBest) {
                    //mn->usage++;
                    pm.updateQualityVector(mn->getId(), 1);
                }
                mn->usage++;
            }
        }
        clearSelectMetroNodes();
        //cout<<"Iter: "<<tt<<" global best: "<<global_best.aggregateCost()<<" time: "<<mclock.elapsed()<<" size set: "<<metro_node_core.size()<<endl;
        //for(int i=0;i<(int)metro_node.size();i++) cout<<metro_node[i]->getId()<<" ("<<metro_node[i]->usage<<"): ";//<<endl;
        //cout<<endl;

    }
    for(int i=0;i<(int)metro_node_core.size();i++) {
        cout<<" "<<metro_node_core[i]->getId()<<"("<<metro_node_core[i]->usage<<")";
    }
    cout<<endl;

    //}
    cout<<"Cost Parallel best_sol: "<<best_sol.aggregateCost()<<endl;
    //cout<<"Cost best_ils: "<<best_ils.aggregateCost()<<endl;
    cout<<"Time: "<<mclock.elapsed()<<endl;
    cout<<"Usage: "<<endl;
    for(int i=0;i<(int)metro_node.size();i++) {      
        best_sol.copySolution(metro_node[i]);
        cout<<metro_node[i]->usage<<" MN: "<<metro_node[i]->getIdString()<<endl;
    }
    /*for(int i=0;i<(int)metro_node_core.size();i++)
        metro_node_core[i]->printTree();*/
	testAllDistances();
}

void LSearch::solve(void) {
    ILS();
}

void LSearch::ILS(void) {
	cout << "fixed:\n" <<fixed;
	Solution best_ils;
	Solution global_best;

    if(initData)
        init();
#ifdef PCLOCK
    pclock.init();
#endif
  	mclock.init();
    cout<<"Initial Solution: "<<totalCost<<endl;
    cout<<"Trivial Solution: "<<costTrivialSolution()<<endl;
	best_ils.init(nbNodes, metro_node, site_node);
	global_best.init(nbNodes, metro_node, site_node);
	testAllDistances();

    bestSol = numeric_limits<double>::max();
    
    global_best.setSolution(metro_node, tps, matrix, nbNodes, totalCost, mclock.elapsed());
    updateBest();
	lastChange = 0;
	iter = 0;

	//checking initial solutions
	searchIntensification();

	if(global_best.cost > best_sol.cost && fabs(global_best.cost - best_sol.cost) >= EPSILON ) {
		global_best.setSolution(best_sol.metro, best_sol.tps, NULL, nbNodes, best_sol.cost, best_sol.time2sol);
//#ifdef DEBUG2
		if(iter>100 && method == 1)
			cout<<"Best sol: "<<global_best.cost<<" time2sol: "<<global_best.time2sol<<" iter: "<<iter<<endl;
//#endif
	}
	////best_ils.setSolution(best_sol.metro, best_sol.matrix, nbNodes, best_sol.cost, best_sol.time2sol);
	//best_sol.init(nbNodes, metro_node, site_node);
	while(!stop()) {
		//starting with best sol
		totalCost = best_sol.copySolution(metro_node, tps, matrix);
		bestSol = totalCost;
		//perturbing best sol
		searchDiversification();
		//greedy step
		searchIntensification();
		//updating global best
		if(global_best.cost > best_sol.cost && fabs(global_best.cost - best_sol.cost) >= EPSILON ) {
			global_best.setSolution(best_sol.metro, best_sol.tps, NULL, nbNodes, best_sol.cost, best_sol.time2sol);
#ifdef DEBUG2
			if(iter>100 && method == 1)
				cout<<"Best sol: "<<global_best.cost<<" time2sol: "<<global_best.time2sol<<" iter: "<<iter<<endl;
#endif
		}
		//acceptance criterion
		if( (best_ils.cost > best_sol.cost && fabs(best_ils.cost - best_sol.cost) >= EPSILON ) || random(100) < 5) {
			best_ils.setSolution(best_sol.metro, best_sol.tps, best_sol.matrix, nbNodes, best_sol.cost, best_sol.time2sol);
		}
		else best_sol.setSolution(best_ils.metro, best_ils.tps, best_ils.matrix, nbNodes, best_ils.cost, best_ils.time2sol);
		//totalDistance = best_sol.copySolution(metro_node, matrix);
	}
	cout<<"Time: "<<mclock.elapsed()<<endl
		<<"BEST SOL: "<<global_best.cost<<endl
		//<<"BEST SOL: "<<global_best.cost<<" actual cost: "<<global_best.solCost()<<endl
		<<"Time2Sol: "<<global_best.time2sol<<endl
		<<"Iterations: "<<iter<<endl;
	//testing final state.. not necessarly the best solution....
	totalCost = global_best.copySolution(metro_node, tps, matrix);
	//best_sol.printTree();
	testAllDistances();
	global_best.print(activeMN, param.sol_file);
    /*for(int i=0;i<metro_node.size();i++)
        metro_node[i]->printTree();*/
}

void LSearch::searchIntensification(void) {
    /*#pragma omp critical (SI)
    {
	cout<<"Intensification"<<endl;
    }*/
	lastChangeSol = totalCost;
	lastChange = iter;
    /*for(int i=0;i<metro_node.size();i++)  {
        metro_node[i]->printTree();
        metro_node[i]->checkSize();
    }*/
    Move m = divStep(iter);

    //while(m.nNode && !stop()) {
    while(m.nNode && !stopSearch) {
        /*#pragma omp critical (SI)
        {
            if(omp_get_thread_num() == 0)
                cout<<"thread["<<omp_get_thread_num()<<"] ->"<<metro_node_core[omp_get_thread_num()]->tCost<<endl;
        }*/
		//cout<<"INT: "<<iter<<endl;
#ifdef DEBUG
#endif
		iter++;
		SelElementIntensification(m);
#ifdef DEBUG
		cout<<mclock.elapsedMiliSeconds()<<" Iter: "<<iter<<" Total Distance: "<<totalCost<<" Best: "<<best_sol.cost<<" tvalid: "<<activeMetroNode.size()<<endl;
//		testAllDistances();
#endif
		if( totalCost < lastChangeSol ) {
			lastChange = iter;
			lastChangeSol = totalCost;
		}
        if(omp_get_thread_num() == 0 && stop()) {
            stopSearch=true;
        }
		updateBest();
        m = divStep(iter);
	}
    if(m.nNode) {
        //assert(stop());
        assert(stopSearch);
        unLockNode(m.nNode);
        unLockMetroNode(m.nNode->getMetro());
    }
	//cout<<"END INT"<<endl;
}

void LSearch::searchDiversification(void) {
    /*#pragma omp critical (SD)
    {
	cout<<"Diversification: "<<activeMetroNode.size()<<endl;
    }*/
    int divIter = 0;
	do {
		//int lIter = iter + param.perturbation_steps;
        int lIter = divIter + param.perturbation_steps;
		while( divIter<lIter ) {
			//cout<<"DIVVV: "<<activeMetroNode.size()<<" iter: "<<iter<<" lIter: "<<lIter<<endl;
			SelElementDiversication();
            //assert(false);
			updateBest();
			divIter++;
		}
    }while(isDiversification());
    iter+=divIter;
	//}while(valid_nodes.empty());
	//cout<<"After diversification: "<<valid_nodes.size()<<endl;
}

void LSearch::updateBest(void) {
	if(totalCost < bestSol) {
		best_sol.setSolution(metro_node, tps, matrix, nbNodes, totalCost, mclock.elapsed());
		bestSol = totalCost;
#ifdef PCLOCK
        if(pclock.elapsedMiliSeconds() > maxLocalTime ) {
            double bcost = global_best.cost;
            if(global_best.cost > best_sol.cost) {
                bcost = totalCost;
            }
            cout<<"Current Best Solution Cost: "<<bcost<<" time: "<<mclock.elapsed()<<" iter: "<<iter<<endl;
            pclock.init();
        }
#endif

#ifdef DEBUG
		cout<<"BEST SOL: "<<bestSol<<" "<<iter<<endl;
#endif
	}
}

Node *startNode(edgeGraph g) { return g.first; }
Node *endNode(edgeGraph g) { return g.second; }
double distanceEdge(edgeGraph g) {return startNode(g)->getDistance(endNode(g)); }
double costEdge(edgeGraph g) { return startNode(g)->getCost(endNode(g)); }
