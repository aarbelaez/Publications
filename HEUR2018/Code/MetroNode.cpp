#ifndef METRONODE__CPP
#define METRONODE__CPP

#include "BH.h"
#include "MetroNode.h"

#include "vector"
#include "stack"
#include "set"

#include <assert.h>
#include <math.h>

using namespace std;

MetroNode::MetroNode(void) { 
    tCost = 0;
    state = available;
    usage = 0;
    
    validEdges = NULL;
    
    road_factor = 1;
}
MetroNode::~MetroNode(void) {
    tCost=0;
	Node *tmp = getMetroNode();
	tmp->delTree();
    usage = 0;

    validEdges = NULL;
    road_factor = 1;
}

int MetroNode::random(int i) {
    return static_cast<unsigned int> (genrand(&mt)) % i;
}

void MetroNode::randomSeed(int s) {
    sgenrand(s, &mt);
}


FirstLevelNode* MetroNode::getNewFirstLevelNode(Node *n) {
    return fLevelList.newFirstLevelNode(n);
}

void MetroNode::decFirstLevelNode(FirstLevelNode *&fn) {
    if(fn) {
        fn->decSize();
        if(fn->isEmpty()) fLevelList.delNode(fn);
    }
}

void MetroNode::checkSize(void) {
    stack<Node*> s;

    s.push(getMetroNode());
    int tsize=0;
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        if(tmp->sizeTree==0) {cout<<"Error sizeTree = 0 for: "<<tmp->getIdString()<<endl; }
        assert(tmp->sizeTree>0);
        if(!tmp->isMetroNode()) {
            if(!tmp->pred->isMetroNode() && tmp->fnode!= tmp->pred->fnode) {
                cout<<"Error fnode error, pred != current "<<tmp->fnode->getIdString()<<" "<<tmp->pred->fnode->getIdString()<<endl;
            }
            if(tmp->pred->isMetroNode()) {
                if(tmp->fnode != tmp) {
                    cout<<"fnode: "<<tmp->fnode->getIdString()<<" expected: "<<tmp->getIdString()<<endl;
                    printTree();
                }
                //else cout<<"FNODE OK"<<endl;
                assert(tmp->fnode == tmp);
            }
            if(!tmp->fnode) {
                cout<<"Node failed.. no fnode: "<<tmp->getIdString()<<endl;
                //Node *t; t->getIdString();
            }
            if(tmp->sizeTree >= tmp->pred->sizeTree) {
                cout<<tmp->pred->getIdString()<<" ("<<tmp->pred->sizeTree<<") -> "<<tmp->getIdString()<<" ("<<tmp->sizeTree<<")"<<endl;
            }
            assert(tmp->sizeTree < tmp->pred->sizeTree);
            assert(tmp->fnode);
            if(!tmp->pred->isMetroNode())
                assert(tmp->fnode == tmp->pred->fnode);
        }
        tsize++;
        for(int i=0;i<(int)tmp->suc.size();i++) {
            s.push(tmp->suc[i]);
        }
    }
    //cout<<"METRO SIZE: "<<getMetroNode()->sizeTree<<" tsize: "<<tsize<<endl;
    if(getMetroNode()->sizeTree!=tsize) {
        cout<<"metro size: "<<getMetroNode()->sizeTree<<" tsize: "<<tsize<<endl;
        printTree();
        /*
        s.push(getMetroNode());
        int tsize=0;
        while(!s.empty()) {
            Node *tmp = s.top();
            s.pop();
            cout<<"Node_"<<tmp->getIdString()<<" size: "<<tmp->sizeTree<<endl;
            tsize++;
            for(int i=0;i<(int)tmp->suc.size();i++) {
                cout<<tmp->getIdString()<<" -> "<<tmp->suc[i]->getIdString()<<endl;
                s.push(tmp->suc[i]);
            }
        }*/

        Node *tmp = NULL;
        tmp->sizeTree=1;
    }
    assert(getMetroNode()->sizeTree == tsize);
    /*
    //s.push(getMetroNode());
    for(int i=0;i<(int)getMetroNode()->suc.size();i++) {
        s.push(getMetroNode()->suc[i]);
        //FirstLevelNode *fn=getMetroNode()->suc[i]->firstLevel;
        Node *fn = getMetroNode()->suc[i];
        int tsize=0;
        while(!s.empty()) {
            Node *tmp = s.top();
            s.pop();
            tsize++;
            if(fn!=tmp->fnode) {
                cout<<"fn: "<<fn->getIdString()<<" tmp->firstLevel: "<<tmp->fnode->getIdString()<<endl;
                printTree();
            }
            assert(fn==tmp->fnode);
            
            for(int i=0;i<(int)tmp->suc.size();i++)
                s.push(tmp->suc[i]);
        }

        if(tsize!=fn->sizeTree) {
            printTree();
            cout<<"Error with tsize: "<<tsize<<" getSize: "<<fn->sizeTree<<endl;
        }
        assert(tsize==fn->sizeTree);
    }
     */
    /*if(fn)
        assert(tsize==fn->getSize());*/

}

bool MetroNode::isActive(void) {
    return (!active.empty());
}

Node *MetroNode::getActiveNode(int i) {
    return active[i];
}

int MetroNode::getTotalActiveNodes(void) {
    return active.size();
}

void MetroNode::resetActiveAndInactive(void) {
    active.clear();
    inactive.clear();
}

void MetroNode::activeNodes(void) {
    //cout<<"Activating nodes for: "<<getMetroNode()->getIdString()<<endl;
    active.clear();
    for(int i=0;i<(int)node.size();i++) {
        if(node[i] != getMetroNode()) active.push_back(node[i]);
    }
    inactive.clear();
}
void MetroNode::setActiveNodes(void) {
    //cout<<"MMMM: "<<getMetroNode()->getIdString()<<" active: "<<active.size()<<" inactive: "<<inactive.size()<<endl;
    for(int i=0;i<(int)inactive.size();i++) {
        active.push_back(inactive[i]);
    }
    inactive.clear();
}

//return true when the metro node must be also deactivated
bool MetroNode::setInactiveNode(int pos_node) {
    Node *n1 = active[pos_node];
    inactive.push_back(n1);
    active.erase( active.begin() + pos_node );
    return active.empty();
}

void MetroNode::printActiveAndInactiveNodes(void) {
    cout<<"MetroNode: "<<getMetroNode()->getIdString()<<endl;
    cout<<"Active: ";
    for(int i=0;i<(int)active.size();i++) cout<<active[i]->getIdString()<<" ";
    cout<<endl<<"Inactive: ";
    for(int i=0;i<(int)inactive.size();i++) cout<<inactive[i]->getIdString()<<" ";
    cout<<endl;
}

void MetroNode::checkDistance(void) {
    Node *root = getMetroNode();
    for(unsigned int i=0;i<root->suc.size();i++) {
        Node *tmp = root->suc[i];
        double lambda = tmp->getLambda(tmp->pred,NULL);
        if(lambda==0) cout<<"Error lambda for: "<<tmp->getIdString()<<" sizeTree: "<<tmp->sizeTree<<endl;
        assert(lambda>0);
    }
}

void MetroNode::checkTree(void) {
    //checking tree
    for(int i=0;i<(int)node.size();i++)
        if(node[i]!=getMetroNode()) {
            //if(!node[i]->fnode) cout<<"Node: "<<node[i]->getIdString()<<" fnode: "<<node[i]->fnode<<endl;
            assert(node[i]->fnode);
            Node *tmp=node[i];
            //cout<<tmp->pred->getIdString()<<"->"<<tmp->getIdString()<<endl;
            for(tmp=node[i];tmp->pred!=NULL;tmp=tmp->pred) {
                bool valid=false;
                for(int j=0;j<(int)tmp->pred->suc.size();j++) 
                    if(tmp->pred->suc[j]==tmp) valid=true;
                assert(valid);
            }
            if(tmp!=getMetroNode()) {
                cout<<"Error with node...: "<<tmp->getIdString()<<" pointer: "<<tmp<<endl;
                for(int j=0;j<(int)node.size();j++) {
                    if(node[j]->pred)
                        cout<<node[j]->pred->getIdString()<<"->"<<node[j]->getIdString()<<endl;
                }
            }
            assert(tmp==getMetroNode());
        }
    int tn=0;
    stack<Node*>s;
    s.push(getMetroNode());
    while(!s.empty()) {
        Node *tmp = s.top(); s.pop();
        tn++;
        for(int i=0;i<(int)tmp->suc.size();i++) s.push(tmp->suc[i]);
    }
    assert(tn==(int)node.size());
}

void MetroNode::BDBHeuristic(void) {
    cout<<"BDBHeuristic: "<<endl;
    vector<Node*> unConnected;
    for(int i=0;i<(int)node.size();i++)
        if(node[i]!=getMetroNode()) unConnected.push_back(node[i]);

    Node *next;
    while(!unConnected.empty()) {
        next=getNextBDBNode(getMetroNode(), unConnected);
        if(!next) {
            //cout<<"Not candidate node has been found.. current cost: "<<totalCost<<endl;
            Node *aux = NULL;
            pair<Node*, double> brelax = make_pair(aux, numeric_limits<double>::max());
            Node *candidate = NULL;
            for(int i=0;i<(int)node.size();i++) {
                if(node[i]!=getMetroNode() && node[i]->pred) {
                    pair<Node*, double> tmp = getNewEdge(getMetroNode(), node[i]);
                    if(tmp.second < brelax.second) {
                        brelax = tmp;
                        candidate = node[i];
                    }
                }
            }
            assert(candidate && brelax.second < 0 && !unConnected.empty());
            candidate->delSubTree();
            candidate->moveSubTree(brelax.first, NULL);
        }
        else {
            for(int i=0;i<(int)unConnected.size();i++) {
                if(unConnected[i]==next) {
                    unConnected.erase( unConnected.begin() + i );
                    break;
                }
            }
            Node *pred = next->pred;
            next->pred=NULL;
            next->moveSubTree(pred, NULL);
            //cout<<"Candidate node found: "<<next->getIdString()<<" max len: "<<getMetroNode()->dist2leaf<<endl;
        }
    }
    //checking tree
    checkTree();
}

//finding a new possition for n in the tree (with maximal dist2leaf reduction)
//pair<Node*, double> MetroNode::getNewEdge(Node *root, Node *n) {
pair<Node*, double> MetroNode::getNewEdge(Node *root, Node *n) {
    assert(n->pred);
    stack<Node*> s;
    s.push(root);
    Node *aux = NULL;
    pair<Node*, double> best = make_pair(aux, numeric_limits<double>::max());
    double relaxation;

    while(!s.empty()) {
        Node *top = s.top();
        assert(top!=n);
        s.pop();
        if( n->pred != top && top->validMove(n) ) {
            relaxation = n->getDistance() - (top->getDistance() + top->getDistance(n));
            if(best.second > relaxation && relaxation < 0) {
                best.second = relaxation;
                best.first = top;
            }
        }
        for(int i=0;i<(int)top->suc.size();i++) {
            if(n!=top->suc[i])
                s.push(top->suc[i]);
        }
    }
    return best;
}

//building up the tree.....
//adding an unconnected node in the tree (the one with minimal cost, i.e., edge)
Node *MetroNode::getNextBDBNode(Node *root, vector<Node*> &unConnected) {
    stack<Node*> s;
    s.push(root);
    
    Node *best=NULL;
    Node *predBest=NULL;
    double bestCost=numeric_limits<double>::max();
    while(!s.empty()) {
        Node *top = s.top();
        s.pop();

        //checking a candidate
        for(int i=0;i<(int)unConnected.size();i++) {
            Node *tmp = unConnected[i];
            assert(!tmp->pred);
            if(bestCost > top->getCost(tmp) && tmp->validMove(top)) {
                predBest=top;
                best=tmp;
                //best->pred = top;
                bestCost=top->getCost(tmp);
            }
        }

        for(int i=0;i<(int)top->suc.size();i++) {
            s.push(top->suc[i]);
        }
    }
    if(best) best->pred=predBest;
    return best;
}

void MetroNode::dijkstra(void) {
//    priority_queue<Node*, vector <Node*>, NodeGreater> Q;
    vector<Node*> Q;
    //set<Node*, NodeSDistance> Q;

    for(int i=0;i<(int)node.size();i++) {
        node[i]->setMaxSDistance();
        node[i]->unMarkNode();
        node[i]->spred = NULL;
        //fully connected graph...
        Q.push_back(node[i]);
        //Q.insert(node[i]);
    }
    //make_heap<Node*>(Q.begin(), Q.end(), NodeSDistance);

    getMetroNode()->setSDistance(0);
    while(!Q.empty()) {
        Node *tmp = Q[0];
        int delIndex=0;
        for(int i=1;i<(int)Q.size();i++) {
            if(tmp->getSDistance() > Q[i]->getSDistance()) {
                tmp=Q[i];
                delIndex=i;
            }
        }
        Q.erase( Q.begin()+delIndex );
        /*Node *tmp = (*Q.begin());
        Q.erase(Q.begin());*/
        //Node *tmp = pop_heap<Node*>(Q.begin(), Q.end(), NodeSDistance);

        //cout<<"Vertex: "<<tmp->getIdString()<<" sdist: "<<tmp->getSDistance()<<endl;
        //= Q.top();
        //Q.pop();
        assert(!tmp->isMarked());

        //adjacent list.. fully connected graph
        for(int i=0;i<(int)node.size();i++) {
            if(tmp!=node[i]) {
                //assert(tmp->getDistance(node[i]) < 1000);
                //cout<<", ("<<tmp->getIdString()<<"): "<<tmp->getSDistance()<<", ("<<node[i]->getIdString()<<"): "<<tmp->getDistance(node[i])<<" === "<<node[i]->getSDistance()<<endl;
                if(tmp->getSDistance() + tmp->getDistance(node[i]) < node[i]->getSDistance()) {
                    node[i]->setSDistance(tmp->getSDistance() + tmp->getDistance(node[i]));
                    node[i]->spred = tmp;
                    //make_heap<Node*>(Q.begin(), Q.end(), NodeSDistance());
                    //cout<<node[i]->getIdString()<<": "<<node[i]->getSDistance()<<endl;
                }
            }
        }
        tmp->markNode();
    }
}


void MetroNode::InitialTree(void) {
    cout<<"Before dijkstra"<<endl;
    dijkstra();
    cout<<"After dijkstra"<<endl;    
    vector<Node*> leaves;

    stack<Node*> s;
    s.push(getMetroNode());
    getMetroNode()->distance = 0;
    
    for(int i=0;i<(int)node.size();i++) {
        //node[i]->distance = 0;
        //node[i]->unMarkNode();
        node[i]->unMarkNode();
        if(node[i]==getMetroNode()) continue;
        assert(node[i]->pred);
        node[i]->pred->addSuccessor(node[i]);
        node[i]->dist2leaf=0;

        /*
        TwoPred *tp = node[i]->getTP();
        tp->addPred(node[i]->pred, node[i]->isPrimary);
         */
        node[i]->addPredForest(node[i]->pred);

        //cout<<node[i]->pred->getIdString()<<" -> "<<node[i]->getIdString()<<endl;
    }


    while(!s.empty()) {
        Node *top = s.top();
        s.pop();
        if(top->suc.empty()) leaves.push_back(top);
        for(int i=0;i<(int)top->suc.size();i++) {
            top->suc[i]->distance = top->getDistance() + top->getDistance(top->suc[i]);
            tCost+=top->getCost(top->suc[i]);
            
            s.push(top->suc[i]);
        }
    }
    //cout<<"FFFFFFFFFF"<<endl;
    //adding dist2leaf
    for(int i=0;i<(int)leaves.size();i++) {
        for(Node *tmp=leaves[i]; tmp->pred!=NULL; tmp=tmp->pred)  {
            if(tmp->pred->dist2leaf < tmp->dist2leaf + tmp->getDistance(tmp->pred)) {
                tmp->pred->dist2leaf = tmp->dist2leaf + tmp->getDistance(tmp->pred);
            }
            else break;
        }
    }
    cout<<"min valid lambda: "<<getMetroNode()->dist2leaf<<endl;
    leaves.clear();
}

//distance val.metro node to local exchange
void MetroNode::setDistValueMN2LE(int idLE, double dist) {
    Node *n1=NULL;
    for(int i=0;i<(int)node.size();i++) {
        if(idLE == node[i]->getInputId() && node[i]!=getMetroNode()) {
            n1=node[i];
            break;
        }
    }
    if(!n1) cout<<"idLE: "<<idLE<<" not found in this metro node: "<<getMetroNode()->getInputId()<<endl;
    assert(n1);
    //cout<<"COST: "<<getMetroNode()->pos_mn<<" -- ";
    //cout<<n1->getInputId()<<" ==  "<<n1->pos_mn<<": "<<dist<<endl;
    distMatrix[ getMetroNode()->pos_mn ][ n1->pos_mn ] = dist * road_factor;
    distMatrix[ n1->pos_mn ][ getMetroNode()->pos_mn ] = dist * road_factor;
}

void MetroNode::setDistMatrixValue(int id1, int id2, double dist) {
    Node *n1=NULL;
    Node *n2=NULL;
    for(int i=0;i<(int)node.size();i++) {
        if(id1 == node[i]->getInputId() && node[i]!= getMetroNode())
            n1=node[i];
        if(id2 == node[i]->getInputId() && node[i]!= getMetroNode())
            n2=node[i];
        if(n1!=NULL && n2!=NULL) break;
    }
    
    if(!(n1!=NULL && n2!=NULL) ) {
        cout<<"MetroNode: "<<getInputId()<<" n1: "<<n1<<" n2: "<<n2<<" id1: "<<id1<<" id2: "<<id2<<endl;
        cout<<"Nodes: ";
        for(int i=0;i<(int)node.size();i++) cout<<node[i]->getInputId()<<", ";
        cout<<endl;
    }
    assert(n1!=NULL && n2!=NULL);
    distMatrix[n1->pos_mn][n2->pos_mn] = dist * road_factor;
    distMatrix[n2->pos_mn][n1->pos_mn] = dist * road_factor;
}

//cost value metro node to local exchange
void MetroNode::setCostValueMN2LE(int idLE, double cost) {
    Node *n1=NULL;
    for(int i=0;i<(int)node.size();i++) {
        if(idLE == node[i]->getInputId() && node[i]!=getMetroNode()) {
            n1=node[i];
            break;
        }
    }
    assert(n1!=NULL);
    costMatrix[ getMetroNode()->pos_mn ][ n1->pos_mn ] = cost;
    costMatrix[ n1->pos_mn ][ getMetroNode()->pos_mn ] = cost;
}

void MetroNode::setCostMatrixValue(int id1, int id2, double cost) {
    Node *n1=NULL;
    Node *n2=NULL;
    for(int i=0;i<(int)node.size();i++) {
        if(id1 == node[i]->getInputId() && node[i]!=getMetroNode())
            n1=node[i];
        if(id2 == node[i]->getInputId() && node[i]!=getMetroNode())
            n2=node[i];
        if(n1!=NULL && n2!=NULL) break;
    }
    assert(n1!=NULL && n2!=NULL);
    costMatrix[n1->pos_mn][n2->pos_mn] = cost;
    costMatrix[n2->pos_mn][n1->pos_mn] = cost;
}

void MetroNode::setDistAndCostMatrixValue(int id1, int id2, double dist, double cost) {
    Node *n1=NULL;
    Node *n2=NULL;
    for(int i=0;i<(int)node.size();i++) {
        if(id1 == node[i]->getInputId() && node[i]!= getMetroNode())
            n1=node[i];
        if(id2 == node[i]->getInputId() && node[i]!= getMetroNode())
            n2=node[i];
        if(n1!=NULL && n2!=NULL) break;
    }
    
    if(!(n1!=NULL && n2!=NULL) ) {
        cout<<"MetroNode: "<<getInputId()<<" n1: "<<n1<<" n2: "<<n2<<" id1: "<<id1<<" id2: "<<id2<<endl;
    }
    assert(n1!=NULL && n2!=NULL);
    distMatrix[n1->pos_mn][n2->pos_mn] = dist * road_factor;
    distMatrix[n2->pos_mn][n1->pos_mn] = dist * road_factor;
    costMatrix[n1->pos_mn][n2->pos_mn] = cost;
    costMatrix[n2->pos_mn][n1->pos_mn] = cost;
}


void MetroNode::setDistAndCostMatrixValue(Node *n1, Node *n2, double dist, double cost) {
    /*if( !(n1->getMetro() == this && n2->getMetro() == this) ) {
        cout<<"n1->getMetro: "<<n1->getMetro()<<" n2->getMetro: "<<n2->getMetro()<<endl;
    }
    assert(n1->getMetro() == this && n2->getMetro() == this);*/
    assert(n1!=NULL && n2!=NULL);
    distMatrix[n1->pos_mn][n2->pos_mn] = dist * road_factor;
    distMatrix[n2->pos_mn][n1->pos_mn] = dist * road_factor;
    costMatrix[n1->pos_mn][n2->pos_mn] = cost;
    costMatrix[n2->pos_mn][n1->pos_mn] = cost;
}


void MetroNode::setDistAndCostValueMN2LE(int idLE, double dist, double cost) {
    Node *n1=NULL;
    for(int i=0;i<(int)node.size();i++) {
        if(idLE == node[i]->getInputId() && node[i]!=getMetroNode()) {
            n1=node[i];
            break;
        }
    }
    if(!n1) cout<<"idLE: "<<idLE<<" not found in this metro node: "<<getMetroNode()->getInputId()<<endl;
    assert(n1);
    //cout<<"COST: "<<getMetroNode()->pos_mn<<" -- ";
    //cout<<n1->getInputId()<<" ==  "<<n1->pos_mn<<": "<<dist<<endl;
    distMatrix[ getMetroNode()->pos_mn ][ n1->pos_mn ] = dist * road_factor;
    distMatrix[ n1->pos_mn ][ getMetroNode()->pos_mn ] = dist * road_factor;
    costMatrix[ getMetroNode()->pos_mn ][ n1->pos_mn ] = cost;
    costMatrix[ n1->pos_mn ][ getMetroNode()->pos_mn ] = cost;
}


//checking whether the edge n1->n2 is valid
bool MetroNode::validEdge(Node *n1, Node *n2) {
    if(!validEdges) return true;
    assert(n1->pos_mn < (int)node.size() && n2->pos_mn < (int)node.size());
/*#pragma omp critical (ValidEdge)
    {
    if(!validEdges[n1->pos_mn][n2->pos_mn]) cout<<"invalid "<<n1->getIdString()<<" -> "<<n2->getIdString()<<endl;
    }*/
    return validEdges[n1->pos_mn][n2->pos_mn];
}

//set invalid link n1->n2
void MetroNode::setInValidEdge(Node *n1, Node *n2) {
    assert(n1->pos_mn < (int)node.size() && n2->pos_mn < (int)node.size());
    validEdges[n1->pos_mn][n2->pos_mn] = false;
}

//set valid all nodes
void MetroNode::setValidAllEdges(void) {
    for(unsigned int i=0;i<node.size(); i++) {
        Node *n1 = node[i];
        for(unsigned int j=0;j<node.size(); j++) {
            Node *n2 = node[j];
            setValidEdge(n1, n2);
        }
    }
}

void MetroNode::setValidEdge(Node *n1, Node *n2) {
    assert(n1->pos_mn < (int)node.size() && n2->pos_mn < (int)node.size());
    validEdges[n1->pos_mn][n2->pos_mn] = true;
}

void MetroNode::initValidEdges(void) {
    validEdges = new bool*[node.size()];
    for(unsigned int i=0;i<node.size();i++) {
        validEdges[i] = new bool[node.size()];
        for(unsigned int j=0;j<node.size();j++) {
            validEdges[i][j]=true;
        }
    }
}

void MetroNode::initDistMatrix(vector<SiteNode> site_node) {
	distMatrix = new double*[node.size()];
	for(int i=0;i<(int)node.size(); i++) {
		distMatrix[i] = new double[node.size()];
		Node *n_i = node[i];
		for(int j=0; j<(int)node.size(); j++) {
			Node *n_j = node[j];
			distMatrix[i][j] = 
		    	sqrt( pow(getX(n_i->getId()) - getX(n_j->getId()), 2) + pow(getY(n_i->getId()) - getY(n_j->getId() ), 2) );
		}
	}
}

void MetroNode::initCostMatrix(vector<SiteNode> site_node) {
    costMatrix = new double*[node.size()];
    for(int i=0; i<(int)node.size(); i++) {
        costMatrix[i] = new double[node.size()];
        for(int j=0; j<(int)node.size(); j++) {
            costMatrix[i][j] = distMatrix[i][j];
        }
    }
}

void MetroNode::setId(int _id) {
    id = _id;
}
int MetroNode::getSize(void) {
    return size;
}


void MetroNode::setPosNodes(void) {
	for(int i=0;i<(int)node.size();i++) node[i]->pos_mn= i;
}

void MetroNode::copy(MetroNode *mn) {
	assert(mn->node.size() == node.size() && mn != this);
	id = mn->id;
	input_id = mn->input_id;
	size = mn->size;
	distMatrix = mn->distMatrix;
    tCost = mn->tCost;
    state = mn->state;
    fLevelList.copy(mn->fLevelList);
    
    road_factor = mn->road_factor;

	for(int i=0; i<(int) node.size(); i++) {
		Node *tmp = getNode(i);
		tmp->copy(mn->getNode(i));
	}
}

void MetroNode::setNodes(void) {
    //node = new Node[size+1];
    for(int i=0;i<=size;i++) {
        //node[i] = Node(id,i);
        node.push_back( new Node(id, i) );
    }
    node[0]->setMetroNode();
}
void MetroNode::addNode(int i, Node *n) {
    node[i] = n;
}

Node * MetroNode::getNode(int i) {
    return node[i];
}


void MetroNode::printTree(void) {
    //cout<<"============="<<endl;
    //cout<<"MetroNode: "<<id<<" exchange sites: "<<size<<endl;
    int treeSize=0;
    /*for(FirstLevelNode* tmp=fLevelList.start;tmp;tmp=tmp->next)*/
    Node *aux=getMetroNode();
    for(int i=0;i<(int)aux->suc.size();i++) {
        Node *tmp=aux->suc[i];
        cout<<"Node_"<<tmp->getIdString()<<" [style=filled] "<<endl;
    }
    for(int i=0; i<=size; i++) {
        //cout<<"Node_"<<node[i].getId()<<" -> Node_"<<node[i].getPred()->getId()<<";"<<endl;
        //cout<<"Node_"<<node[i]->getPred()->getId()<<" -> Node_"<<node[i]->getId()<<endl;
		//cout<<"Node_"<<node[i]->getPred()->getId()<<" -> Node_"<<node[i]->getId()<<" [label=\""<<node[i]->getDistance()<<"\"]"<<endl;
		//cout<<"Node_"<<node[i]->getPred()->getId()<<" -> Node_"<<node[i]->getId()<<" [label=\""<<node[i]->id_metro_node<<"\"]"<<endl;
        if(node[i]->getPred())  {
            //cout<<"Node_"<<node[i]->getPred()->getIdString()<<" -> Node_"<<node[i]->getIdString()<<" [label=\""<<node[i]->getDistance()<<","<<getMetroNode()->getDistance(node[i])<<"\"]"<<endl;
            cout<<"Node_"<<node[i]->getPred()->getIdString()<<" -> Node_"<<node[i]->getIdString();
            /*if(node[i]->pred->isMetroNode() && node[i]->fnode)
                cout<<" [label=\""<<node[i]->fnode->sizeTree<<"\"]";
            else if(!node[i]->isMetroNode())
                cout<<" [label=\""<<node[i]->fnode->getIdString()<<"\"]";
            cout<<endl;*/
            cout<<" [label=\"level: "<<node[i]->treeLevel<<", size:"<<node[i]->sizeTree<<" (dist2: "<<node[i]->getDistance(node[i]->getPred())<<", dist: "<<node[i]->getDistance()<<")\"]"<<endl;
            treeSize++;
        }
    }
    cout<<"Tree size: "<<treeSize<<endl;
}

void MetroNode::checkLoop(void) {
    for(int i=0; i<(int)node.size(); i++) {
        if( node[i]->isMarked() ) {
            cout<<"Error, Marked node..."<<endl;
            assert(false);
        }
    }
}

void MetroNode::testDistance(void) {
    for(int i=1; i<(int) node.size(); i++) {
        node[i]->testDistance();
    }
}

double MetroNode::testDist2Leaf(Node *r) {
    assert(r);
    double dist=0.0;

    for(vector<Node*>::iterator it=r->suc.begin(); it!=r->suc.end(); it++) {
        Node *tmp = *it;
        double tdist = testDist2Leaf(tmp) + r->getDistance(tmp);
        if( dist < tdist ) dist=tdist;
    }
    //debug
    /*if(!( fabs(dist - r->dist2leaf) < EPSILON )) {
        cout<<"r: "<<r->getId()<<endl;
        cout<<"dist: "<<dist<<" r->dist: "<<r->dist2leaf<<endl;
        this->printTree();
    }*/

    assert( fabs(dist - r->dist2leaf) < EPSILON );
    //if( !(dist<LAMBDA) ) cout<<"dist: "<<dist<<" LAMBDA: "<<LAMBDA<<endl;
    assert( dist <= LAMBDA );

    return dist;
}


#endif