#ifndef NODE_CPP
#define NODE_CPP

#include <iostream>
#include <vector>
#include <string>
#include <stack>

#include <assert.h>
#include <math.h>

#include "BH.h"
#include "Node.h"
#include "MetroNode.h"

//static bool Node::firstImprovement = false;
bool Node::firstImprovement = false;

Node::Node() {
    fnode = NULL;
    firstLevel = NULL;
    sizeTree = 1;
    treeLevel = 1;
	pos_mn = -1;
	pred = NULL;
	input_id = -1;
    junction = false;
    state=available;
    isPrimary=2;
}

Node::Node(int id_mn, int id) {
    fnode = NULL;
    firstLevel = NULL;
    sizeTree = 1;
    treeLevel = 1;
    id_metro_node = id_mn;
    id_node = id;
	input_id = -1;
    pred = NULL;
    mark = 0;
    distance = 0.0;
    dist2leaf = 0.0;
    usage = 0;
    junction = false;
    state=available;
    isPrimary=2;
}

Node::Node(int id_mn) {
    fnode = NULL;
    firstLevel = NULL;
    sizeTree = 1;
    treeLevel = 1;
    id_metro_node = id_mn;
    id_node = -1;
	input_id = -1;
    pred = NULL;
    mark = 0;
    distance = 0.0;
    dist2leaf = 0.0;
    usage = 0;
    junction = false;
    
    state=available;
    isPrimary=2;
}

double Node::getRoadFactor(void) {
    return getMetro()->road_factor;
}

bool Node::isMetroNode(void) {
    return (getMetroNode() == this);
}

int Node::random(int i) {
    return metro_node->random(i);
}

void Node::setJunction(void) { 
    junction = true;
}

bool Node::isJunction(void) {
    return junction;
}

TwoPred *Node::getTP(void) {
	return metro_node->tps[getId()]; 
}

void Node::copyAtr(Node *n) {
    id_metro_node = n->id_metro_node;
    id_node = n->id_node;
	pos_mn = n->pos_mn;
	input_id = n->input_id;
    //pred = NULL;
    mark = 0;
    usage = 0;
    distance = n->distance;
    dist2leaf = n->dist2leaf;
    state = n->state;
    junction = n->junction;
    isPrimary = n->isPrimary;

    sizeTree = n->sizeTree;
    treeLevel = n->treeLevel;
    firstLevel = n->firstLevel;
    fnode = n->fnode;
}

void Node::copy(Node *n) {
	copyAtr(n);
	suc.clear();
	for(unsigned int i=0; i<n->suc.size(); i++) {
		Node *tmp = n->suc[i];
		assert(tmp->pos_mn>=0);
		Node *s = metro_node->getNode(tmp->pos_mn);
		suc.push_back(s);
		s->pred = this;
	}
}

Node * Node::getMetroNode(void) {
    return metro_node->getNode(0);
}

double Node::getCost(Node *_n) {
    if(!_n) return 0;
    return metro_node->costMatrix[pos_mn][_n->pos_mn];
}

double Node::getDistance(Node *_n) {
    if(!_n) return 0;
#ifdef DISTMATRIX
	return metro_node->distMatrix[pos_mn][_n->pos_mn];
#else 
    return sqrt( pow(getX(getId()) - getX(_n->getId()), 2) + pow(getY(getId()) - getY(_n->getId() ), 2) );
#endif
}

double Node::getEuclideanDistance(Node *_n) {
    return sqrt( pow(getX(getId()) - getX(_n->getId()), 2) + pow(getY(getId()) - getY(_n->getId() ), 2) );
}

double Node::getDistance(void) {
    return distance;
}

void Node::setDistance(double _d) {
    distance = _d;
}

void Node::setPred(Node *_pred) {
    pred = _pred;
}

void Node::addPredForest(Node *nPred) {
    assert(getMetro() == nPred->getMetro());
    TwoPred *tp = getTP();
    tp->addPred(nPred, isPrimary);
}

void Node::delPredForest(Node *nPred) {
    assert(getMetro() == nPred->getMetro());
    TwoPred *tp = getTP();
    tp->delPred(nPred, isPrimary);
}

bool Node::validPredForest(Node *nPred) {
    if( !(getMetro() == nPred->getMetro() ) ) cout<<"Error invalid metro nodes mn1: "<<getMetro()->getIdString()<<" nPred: "<<nPred->getMetro()->getIdString()<<endl;
    assert(getMetro() == nPred->getMetro());
    //if( !metro_node->validEdge(this, nPred) ) return false;
    //checking whether is valid nPred->this
    if( !metro_node->validEdge(nPred, this) ) return false;
    TwoPred *tp = getTP();
    return tp->validPred(nPred, isPrimary);
}

void Node::initPred(Node *_pred) {
    assert( _pred == getMetroNode() && distance == 0);
    distance = getDistance(_pred);
    metro_node->tCost += getCost(_pred);

    pred = _pred;

    /*
	TwoPred *tp = getTP();
	tp->delPred( _pred, isPrimary );
	tp->addPred( pred, isPrimary );
    */
    
    delPredForest(_pred);
    addPredForest(pred);

    /*matrix[_pred->getId()][getId()]--;
    matrix[pred->getId()][getId()]++;*/
}

void Node::initPred(void) {
    reset(false);
    pred = getMetroNode();
    pred->suc.push_back(this);
    distance = getDistance(pred);
    metro_node->tCost += getCost(pred);

	TwoPred *tp = getTP();
	tp->addPred( pred, isPrimary );

	/*assert(matrix[pred->getId()][getId()] == 0);
    matrix[pred->getId()][getId()]++;*/
}

void Node::addSuccessor(Node *s) {
    suc.push_back(s);
}

void Node::delSuccessor(Node *s) {
    for(vector<Node*>::iterator it=suc.begin(); it!= suc.end(); it++) {
        Node *tmp = *it;
        if( tmp == s ) {
            suc.erase(it);
            break;
        }
    }
}


double Node::getDist2leaf(void) {
    double maxDist = 0.0;
    double tmpDist;
    for(vector<Node*>::iterator it=suc.begin(); it!=suc.end(); it++) {
        Node *tmp = *it;
        tmpDist = tmp->dist2leaf + getDistance(tmp);
        if(maxDist < tmpDist) maxDist = tmpDist;
    }
    return maxDist;
}

void Node::updateDistance(Node *oldPred) {
    //cout<<"UPDATE DISTANCE"<<endl;
    //updating distance
    double oldDistance = getDistance();
    double newDistance = getPred()->getDistance() + getDistance(getPred());
    //totalDistance+=( getDistance(getPred()) - getDistance(oldPred) );
    metro_node->tCost+=( getCost(getPred()) - getCost(oldPred) );
    setDistance(newDistance);

    stack<Node*> s;
    s.push(this);
    while(!s.empty()) {
        Node *top = s.top();
        s.pop();
        if( top != this) {
            double nDist = top->getDistance() + newDistance - oldDistance;
            top->setDistance(nDist);
        }
        for(vector<Node*>::iterator it=top->suc.begin(); it!=top->suc.end(); it++) {
            s.push(*it);
        }
    }

    //update dist2leaf
    //oldPred -> root
    double ndist = 0;
    Node *tmp;

    for(tmp=oldPred; tmp!=NULL; tmp=tmp->pred) {
        ndist = tmp->getDist2leaf();
        if( fabs(ndist - tmp->dist2leaf) < EPSILON )
            break;
        tmp->dist2leaf = ndist;
    }

    //update pred -> root
    ndist = dist2leaf;
    for(tmp=this; tmp->pred!=NULL; tmp=tmp->pred) {
        ndist = ndist + tmp->getDistance( tmp->pred );
        if(ndist > tmp->pred->dist2leaf) tmp->pred->dist2leaf = ndist;
        else break;
    }
}

//Moving n to a new valid predecessor (nPred)
void Node::move(Node *nPred) {
    assert( !isMetroNode() );
    assert( nPred != pred );
    Node *oldPred = pred;
    //pred = nPred;
    setPred(nPred);

    if(oldPred)
        oldPred->delSuccessor(this);

    nPred->addSuccessor(this);

    updateDistance(oldPred);

    /*
	TwoPred *tp = getTP();
    if(oldPred)
        tp->delPred(oldPred, isPrimary);
	tp->addPred(nPred, isPrimary);
    */
    if(oldPred)
        delPredForest(oldPred);
    addPredForest(nPred);
}

Node *Node::randomMove(void) {
    Node *root = getMetroNode();
    stack<Node*> s;
    s.push(root);

    vector<Node*> candidate;
    while(!s.empty()) {
        Node *top = s.top();

        s.pop();
        if(top != this) {
            //not testing the current pred
            if( top != pred && validMove(top) )
                candidate.push_back(top);
            for(vector<Node*>::iterator it=top->suc.begin(); it!= top->suc.end(); it++) {
                s.push(*it);
            }
        }
    }
    //cout<<"Candidates: "<<candidate.size()<<endl;
    if( candidate.size() == 0 ) return NULL;
    return candidate[ random(candidate.size()) ];
}

//would it be valid to add a link a->this
bool Node::validMove(Node *a) {
    double lambda = getLambda(a, NULL);
    if( a->getDistance() + getDistance(a) + dist2leaf > lambda ) {
        return false;
    }

	//if( getTP()->validPred(a, isPrimary) ) return true;
    if( validPredForest(a) ) return true;

    //assert(false); 
    return false;
}

double Node::getLambda(void) {
    return getLambda(fnode,NULL);
}

double Node::getLambda(Node *a, Node *b) {
    if(LAMBDA != -1) return LAMBDA;
    //int currentSize = a->treeLevel+a->sizeTree + sizeTree;
    int currentSize = sizeTree;
    //if(!a->isMetroNode()) currentSize+=((a->treeLevel - 1) + a->sizeTree );
    if(!a->isMetroNode()) {
        if(!a->fnode)
            cout<<"currentSize: this: "<<getIdString()<<" a: "<<a->getIdString()<<endl;
        currentSize+=(a->fnode->sizeTree);
    }
    if(a->isMetroNode() && b) currentSize+=b->sizeTree;
    if(currentSize >= 512) return 0.0;
    if(currentSize >= 256 && currentSize < 512) return 10.0;
    if(currentSize >= 128 && currentSize < 256) return 20.0;
    if(currentSize >= 64 && currentSize < 128) return 30.0;
    if(currentSize >= 32 && currentSize < 64) return 40.0;
    if(currentSize >= 16 && currentSize < 32) return 50.0;
    if(currentSize >= 8 && currentSize < 16) return 60.0;
    if(currentSize >= 4 && currentSize < 8) return 70.0;
    if(currentSize >= 2 && currentSize < 4) return 80.0;
    if(currentSize == 1 ) return 90;
    metro_node->printTree();
    cout<<"Error current size: "<<currentSize<<" for "<<getIdString()<<endl;
    assert(false);
    return -1;
    //return LABMDA;
}

bool Node::validMoveNode2(Node *a, Node *b) {
    //a->this disjointness check
    //if( !getTP()->validPred(a, isPrimary) )
    if( !validPredForest(a) )
        return false;

    double lambda = getLambda(a,b);
    int currentSize = sizeTree;
    if(!a->isMetroNode()) {
        currentSize+=(a->fnode->sizeTree);
    }

    //lambda check A -> this (and subtree)
    if( a->getDistance() + getDistance(a) + dist2leaf > lambda) {
            return false;
    }
    if(b) {
        //lambda check A->this->b
        if( b->dist2leaf + getDistance(b) + getDistance(a) + a->getDistance() > lambda ) {
            return false;
        }
        //this->b disjointness check
        //if( !b->getTP()->validPred( this, b->isPrimary ) ) {
        if( !b->validPredForest(this) ) {
            return false;
        }
    }
    return true;
}

bool Node::validMoveNode(Node *a, Node *b) {
	//if( !getTP()->validPred(a, isPrimary) ) {
    if( !validPredForest(a) ) {
		return false;
	}

    double lambda = getLambda(a, b);
    if(!b) {
        if( a->getDistance() + getDistance(a) > lambda ) {
			return false;
		}
    }
    else {
        if( b->dist2leaf + getDistance(b) + getDistance(a) + a->getDistance() > lambda ) {
			return false;
		}
		//if( !b->getTP()->validPred( this, b->isPrimary ) ) {
        if( !b->validPredForest(this) ) {
			return false;
		}
    }

    //checking new connections
    for(vector<Node*>::iterator it=suc.begin(); it!=suc.end(); it++) {
        Node *tmp = *it;
		//if( !tmp->getTP()->validPred(pred, tmp->isPrimary) ) {
        if( !tmp->validPredForest(pred) ) {
            return false;
        }
    }

    return true;
}


Node *Node::bestMove(void) {
    Node *root = getMetroNode();
    stack<Node*> s;
    s.push(root);

    double oldCost = getCost(pred);
    double newCost;
    double moveVal;
    //double best = numeric_limits<double>::max();

    double best = metro_node->tCost;
	bool imp=false;

    vector<Node*> candidate;
    while(!s.empty()) {
        Node *top = s.top();

        s.pop();
        if(top != this) {
            //not testing the current pred
            if( top != pred && validMove(top)) {
                newCost = getCost(top);
                moveVal = metro_node->tCost + (newCost - oldCost );
              	if( fabs(moveVal - best) < EPSILON && imp)
                   	candidate.push_back(top);
               	else if( moveVal < best ) {
					imp = true;
                   	if(firstImprovement) return top;
                   	candidate.clear();
                   	best = moveVal;
                  	candidate.push_back(top);
				}
            }
            for(vector<Node*>::iterator it=top->suc.begin(); it!= top->suc.end(); it++) {
                s.push(*it);
            }
        }
    }
    //cout<<"Candidates: "<<candidate.size()<<endl;
    if( candidate.size() == 0 ) return NULL;
    return candidate[ random(candidate.size()) ];
}


void Node::reset(bool weak) {
    distance = 0;
    dist2leaf = 0;

    if(!weak)
        suc.clear();

    if(pred == NULL) return;

    //remove (this) from pred suc list
    for(unsigned int i=0; i<pred->suc.size(); i++) {
        if( pred->suc[i] == this ) {
            pred->suc.erase( pred->suc.begin()+i );
            break;
        }
    }
    pred = NULL;
}

//changing the existing pred with nPred...
void Node::changePred(Node *nPred) {
    if(pred)
        delSubTree();
    assert(validMove(nPred));
    moveSubTree(nPred,NULL);
}

//Moving n to a new valid predecessor (nPred)
void Node::restoreSubTree(Node *nPred) {
    //cout<<"Restore subtree: "<<nPred->getIdString()<<" ("<<nPred->sizeTree<<") -> "<<getIdString()<<" ("<<sizeTree<<")"<<endl;
    assert( !isMetroNode() );
    assert( !pred );
    setPred(nPred);

    nPred->addSuccessor(this);

    distance = nPred->distance + getDistance(nPred);
    metro_node->tCost+=getCost(nPred);

    //restore distances in subtree
    stack<Node*>s;
    s.push(this);
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        if(tmp!=this)
            tmp->distance=tmp->pred->distance + tmp->getDistance(tmp->pred);
        for(unsigned int i=0;i<tmp->suc.size();i++)
            s.push(tmp->suc[i]);
    }

	TwoPred *tp = getTP();
	tp->addPred(nPred, isPrimary);

    
    //update pred -> root
    /*double ndist = dist2leaf;
    for(Node *tmp=this; tmp->pred!=NULL; tmp=tmp->pred) {
        ndist = ndist + tmp->getDistance( tmp->pred );
        if(ndist > tmp->pred->dist2leaf) tmp->pred->dist2leaf = ndist;
        else break;
    }*/
    //old pred ----> root
    dist2leaf=-1;
    for(Node *tmp=this; tmp!=NULL; tmp=tmp->pred) {
        double distTmp = tmp->getDist2leaf();
        //if( fabs(distTmp - tmp->dist2leaf) < EPSILON ) break;
        if(tmp!=this) tmp->sizeTree+=sizeTree;
        tmp->dist2leaf = distTmp;
    }
    
    //cout<<"after Restore subtree: "<<nPred->getIdString()<<" ("<<nPred->sizeTree<<") -> "<<getIdString()<<" ("<<sizeTree<<")"<<endl;

}


void Node::delSubTree(void) {
    //cout<<getMetroNode()->getIdString()<<" Del: "<<pred->getIdString()<<" ("<<pred->sizeTree<<") -> "<<getIdString()<<" ("<<sizeTree<<")"<<endl;
    Node *oldPred = pred;
    double maxDist = 0;

    int delIndex = -1;
    //removing from pred's sucessor list
    for(unsigned int i=0;i<oldPred->suc.size(); i++) {
        Node *tmp = oldPred->suc[i];
        if(this == tmp) {
            assert(delIndex==-1);
            delIndex = i;
        }
        else {
            double tmpDist = tmp->dist2leaf + tmp->getDistance(oldPred);
            if(maxDist < tmpDist) maxDist = tmpDist;
        }
    }
    pred=NULL;
    distance=0;

    assert(delIndex!=-1);
    oldPred->suc.erase( oldPred->suc.begin() + delIndex );
    //oldPred->dist2leaf = maxDist;
    oldPred->dist2leaf = -1;

    double costOld = -getCost(oldPred);
    metro_node->tCost+=costOld;

    /*
    TwoPred *tp = getTP();
	tp->delPred(oldPred, isPrimary);
     */
    delPredForest(oldPred);

    //old pred ----> root
    //if(oldPred!=getMetroNode()) {
        for(Node *tmp=oldPred; tmp!=NULL; tmp=tmp->pred) {
            double distTmp = tmp->getDist2leaf();
            //if( fabs(distTmp - tmp->dist2leaf) < EPSILON ) break;
            tmp->dist2leaf = distTmp;
            tmp->sizeTree-=sizeTree;
        }
    //}
    
    //metro_node->checkSize();
    //cout<<"After "<<getMetroNode()->getIdString()<<" Del: "<<oldPred->getIdString()<<" ("<<oldPred->sizeTree<<") -> "<<getIdString()<<" ("<<sizeTree<<")"<<endl;

}

bool Node::validMoveSubTree(Node *a, Node *b) {
    return ( (getDistance(a) + dist2leaf > LAMBDA ) && (getDistance(a) + getDistance(b) + b->dist2leaf > LAMBDA) );
}

double Node::moveSubTree(Node *a, Node *b) {
    assert(!pred);
    double cBefore=metro_node->tCost;
    double bDistanceLeaf = 0;
    //updating new location
    //assert(b && b->pred && b->pred == a);

    if(b && b->pred) assert(b->pred == a);

    pred = a;
    a->suc.push_back(this);

    int sizeSubTree=sizeTree;
    //cout<<"moveSubTree: "<<a->getIdString()<<" ("<<(a->fnode?a->fnode->getIdString():"NULL")<<") -> "<<getIdString()<<" ("<<(fnode?fnode->getIdString():"NULL")<<") -> ";
    if(b) {
        //cout<<b->getIdString()<<" ("<<(b->fnode?b->fnode->getIdString():"NULL")<<")";
        b->pred = this;
        suc.push_back(b);
        bDistanceLeaf = b->dist2leaf;
        //remove b from suc in a
        unsigned int i;
        for(i=0;i<a->suc.size(); i++) {
            if(a->suc[i] == b) {
                a->suc.erase(a->suc.begin()+i);
                break;
            }
        }
        sizeTree+=b->sizeTree;
    }
    //cout<<endl;
    if(a->isMetroNode()) {
        fnode = this;
    }
    else {
        fnode = a->fnode;
    }
    //cout<<"moveSubTree "<<getIdString()<<" fnode: "<<fnode<<endl;

    //update distance after reallocating
    double dist2leaf2 = bDistanceLeaf + getDistance(b);
    if(dist2leaf < dist2leaf2) dist2leaf = dist2leaf2;

    //double distDiff = getDistance(a) + getDistance(b) - a->getDistance(b);
    double costDiff = getCost(a) + getCost(b) - a->getCost(b);

    distance = pred->distance + getDistance(pred);
    metro_node->tCost+=costDiff;

    //metro_node->checkSize();
    //pred ----> root
    for(Node *tmp=pred; tmp!=NULL; tmp=tmp->pred) {
        double distTmp = tmp->getDist2leaf();
        tmp->sizeTree+=sizeSubTree;
        //if( fabs(distTmp - tmp->dist2leaf) < EPSILON ) break;
        tmp->dist2leaf = distTmp;
        
        if(!tmp->isMetroNode()) {
            assert(tmp->fnode == fnode);
            if(tmp->sizeTree >= maxTreeSize) {
                cout<<"SIZE ERROR "<<maxTreeSize<<": "<<tmp->getIdString()<<endl;
                for(Node *tmp1=pred;tmp1->pred;tmp1=tmp1->pred) {
                    cout<<"NODE: "<<tmp1->getIdString()<<" fnode: "<<tmp1->fnode->getIdString()<<endl;
                    assert(tmp1->fnode == tmp1->pred->fnode);
                }
                //metro_node->printTree();
            }
            assert(tmp->sizeTree < maxTreeSize);
        }
    }

    stack<Node*> s;
    for(unsigned int i=0;i<suc.size();i++) s.push(suc[i]);

    treeLevel = pred->treeLevel+1;
    /*if(b)
        s.push(b);*/
    //suc[this] ---> leafs
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();

        //tmp->distance+=distDiff;
        tmp->distance = tmp->pred->distance + tmp->getDistance(tmp->pred);
        tmp->treeLevel = tmp->pred->treeLevel+1;
        //updating first level node
        tmp->fnode=fnode;

        //tmp->sizeTree+=sizeTree;
        for(vector<Node*>::iterator it=tmp->suc.begin(); it!=tmp->suc.end(); it++) {
            s.push(*it);
        }
    }

    if(b) {
        /*
		TwoPred *btp = b->getTP();
		btp->delPred(a, b->isPrimary);
		btp->addPred(this, b->isPrimary);
         */
        b->delPredForest(a);
        b->addPredForest(this);
    }

    /*
    TwoPred *tp = getTP();
	tp->addPred(a, isPrimary);
     */
    addPredForest(a);

    /*cout<<"After Moving: "<<a->getIdString()<<" ("<<a->sizeTree<<") -> "<<getIdString()<<" ("<<sizeTree<<")";
    if(b) cout<<" -> "<<b->getIdString()<<" ("<<b->sizeTree<<")";
    cout<<endl;*/

    
    //getMetro()->checkSize();
    if(sizeTree>=maxTreeSize) {
        cout<<"Node: "<<getIdString()<<" SIZE TREE: "<<sizeTree<<" level: "<<treeLevel<<" fnode: "<<fnode->getIdString()<<" a: "<<a->getIdString()<<endl;
        Node *tmp=NULL;
        tmp->getSizeTree();
    }
    assert(sizeTree < maxTreeSize && fnode->sizeTree < maxTreeSize);
    /*cout<<"fnNew size: "<<fnNew->getSize()<<" node: "<<fnNew->getNode()->getIdString()<<endl;*/
    //getMetro()->printTree();

#ifdef DEBUG
    s.push(getMetroNode());
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        tmp->testDistance();
        for(unsigned int i=0;i<tmp->suc.size();i++) {
            s.push(tmp->suc[i]);
        }
    }
#endif

    return (metro_node->tCost-cBefore);
}

//ATTENTION: Have to double check this
void Node::restoreNode(Node *nPred) {
    double costDiff=0;
    double maxDist=0;

    if(nPred->isMetroNode()) fnode=this;
    else fnode=nPred->fnode;
    //nPred->sizeTree++;

    stack<Node*>s;

    //restore suc list
    for(unsigned int i=0;i<suc.size();i++) {
        Node *tmp = suc[i];
        bool del=false;
        for(unsigned int j=0;j<nPred->suc.size(); j++) {
            if(nPred->suc[j] == tmp) {
                s.push(nPred->suc[j]);
                nPred->suc.erase(nPred->suc.begin()+j);
                del=true;
                break;
            }
        }
        assert(del);
        //max leaf
        double distTmp = tmp->dist2leaf+ tmp->getDistance(this);
        if(distTmp > maxDist) 
            maxDist=distTmp;
        //update distance
        costDiff+=(tmp->getCost(this) - tmp->getCost(nPred));

        assert(nPred==tmp->pred);
        /*
        TwoPred *tmpTP = tmp->getTP();
        tmpTP->delPred( nPred, tmp->isPrimary );
        tmpTP->addPred( this, tmp->isPrimary );
         */
        tmp->delPredForest(nPred);
        tmp->addPredForest(this);
        tmp->pred = this;
    }
    dist2leaf=maxDist;
    metro_node->tCost+=costDiff;

    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        
        sizeTree++;
        tmp->fnode = tmp->pred->fnode;
        for(unsigned int i=0;i<tmp->suc.size();i++) {
            s.push(tmp->suc[i]);
        }
    }
    
    //restoreSubTree would update all nodes in the path from this -> rootnode with the size of the subtree, but we're only restoring one node
    int sizeTreeTmp = sizeTree;
    sizeTree = 1;
    
    /*for(Node *tmp=nPred;tmp;tmp=tmp->pred) {
        cout<<"updating sizetree for: "<<tmp->getIdString()<<" --- "<<" current: "<<tmp->sizeTree<<" ==== "<<sizeTree<<endl;
        tmp->sizeTree=-sizeTree;
    }*/

    restoreSubTree(nPred);

    sizeTree = sizeTreeTmp;
}

//removing a node
bool Node::delNode(void) {
    //cout<<"Deleting Node: "<<getIdString()<<" fnode: "<<fnode->getIdString()<<" level: "<<treeLevel<<endl;
    Node *oldPred = pred;
    //double maxDist = 0;
    //double distSuc = -getDistance(oldPred);
    double costSuc = 0;

    stack<Node*> s;

    //checking sucs list in pred
    for(unsigned int i=0;i<suc.size();i++) {
        Node *tmp = suc[i];
        if(!tmp->validMove(oldPred)) {
            return false;
        }
    }
    
    //adding sucs to pred
    for(unsigned int i=0;i<suc.size();i++) {
        Node *tmp = suc[i];
        costSuc+=(oldPred->getCost(tmp) - getCost(tmp));
        /*double distTmp = tmp->dist2leaf + tmp->getDistance(oldPred);
        if(maxDist < distTmp) maxDist = distTmp;
        tmp->distance = oldPred->distance + oldPred->getDistance(tmp);*/
        /*
        TwoPred *tmpTP = tmp->getTP();
        tmpTP->delPred(this, tmp->isPrimary);
        tmpTP->addPred(oldPred, tmp->isPrimary);
         */
        tmp->delPredForest(this);
        tmp->addPredForest(oldPred);

        oldPred->suc.push_back(tmp);
        tmp->pred = oldPred;
        s.push(tmp);
    }

    //if(oldPred->dist2leaf < maxDist)
    //    oldPred->dist2leaf = maxDist;

    //double distOld = getDistance(oldPred);
    metro_node->tCost+=(costSuc);
    //distance=0;
    dist2leaf=0;
    suc.clear();

    //isolated node
    //oldPred->sizeTree--;
    sizeTree = 1;
    fnode = NULL;
    //updating distances in reallocated nodes
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        tmp->distance=tmp->pred->getDistance() + tmp->getDistance(tmp->pred);
        if(tmp->pred->isMetroNode()) {
            tmp->fnode = tmp;
        }
        else {
            tmp->fnode = tmp->pred->fnode;
        }

        for(unsigned int i=0;i<tmp->suc.size();i++) {
            s.push(tmp->suc[i]);
        }
    }


    delSubTree();

    //cout<<"CHECK DEL: "<<endl;
    metro_node->checkSize();
    //cout<<"END deleting node: "<<getIdString()<<endl;
    //metro_node->printTree();

    return true;
}

//adding the node into the tree, this node must not be an existing node in the tree
void Node::moveNode2(Node *a, Node *b) {
    assert(!pred && suc.empty());
    

    double bDistanceLeaf = 0;
    //updating new location
    //assert(b && b->pred && b->pred == a);
    
    if(b && b->pred) assert(b->pred == a);
    
    pred = a;
    a->suc.push_back(this);
    
    if(b) {
        b->pred = this;
        suc.push_back(b);
        bDistanceLeaf = b->dist2leaf;
        //remove b from suc in a
        unsigned int i;
        for(i=0;i<a->suc.size(); i++) {
            if(a->suc[i] == b) {
                a->suc.erase(a->suc.begin()+i);
                break;
            }
        }
    }
        
    //update distance after reallocating
    dist2leaf = bDistanceLeaf + getDistance(b);
    double distDiff = getDistance(a) + getDistance(b) - a->getDistance(b);
    
    double costDiff = getCost(a) + getCost(b) - a->getCost(b);

    distance = pred->distance + getDistance(pred);
    metro_node->tCost+=costDiff;
    //pred ----> root
    for(Node *tmp=pred; tmp!=NULL; tmp=tmp->pred) {
        double distTmp = tmp->getDist2leaf();
        if( fabs(distTmp - tmp->dist2leaf) < EPSILON ) break;
        tmp->dist2leaf = distTmp;
    }

    stack<Node*> s;

    if(b)
        s.push(b);
    //this ---> leafs
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        tmp->distance+=distDiff;
        for(vector<Node*>::iterator it=tmp->suc.begin(); it!=tmp->suc.end(); it++) {
            s.push(*it);
        }
    }

    if(b) {
        /*
		TwoPred *btp = b->getTP();
		btp->delPred(a, b->isPrimary);
		btp->addPred(this, b->isPrimary);
         */
        b->delPredForest(a);
        b->addPredForest(this);
    }
    
    /*
    TwoPred *tp = getTP();
	tp->addPred(a, isPrimary);
     */
    addPredForest(a);

}


double Node::moveNode(Node *a, Node *b) {
    //assert( a!=this && a!=pred);
    assert( a!= this && !pred);
    if(b) assert(b!=this);
    
    double cBefore=metro_node->tCost;
    
    //assert( a!=this && b!=this && a!=pred);
    Node *oldPred = pred;
    double maxDist = 0;
    stack<Node*> s;
    double costSuc = -getCost(oldPred);

    
    //adding sucs to pred
    for(vector<Node*>::iterator it=suc.begin(); it!=suc.end();it++) {
        Node *tmp = *it;
        costSuc+=(oldPred->getCost(tmp) - getCost(tmp));
        double distTmp = tmp->dist2leaf + tmp->getDistance(oldPred);
        if( maxDist < distTmp ) maxDist = distTmp;

        /*
		TwoPred *tmpTP = tmp->getTP();
		tmpTP->delPred(this, tmp->isPrimary);
		tmpTP->addPred(oldPred, tmp->isPrimary);
        */
        tmp->delPredForest(this);
        tmp->addPredForest(oldPred);

        oldPred->suc.push_back(tmp);
        tmp->pred=oldPred;
        s.push(tmp);
    }

    double distOld = getDistance(oldPred);
    double distDiff = maxDist - distOld;
    double costDiff;
    

    reset(false);

    //updating pred to root
    for(Node *tmp=oldPred; tmp!=NULL; tmp=tmp->pred) {
        double distTmp = tmp->getDist2leaf();
        if( fabs(distTmp - tmp->dist2leaf) < EPSILON ) break;
        tmp->dist2leaf = distTmp;
    }

    double bDistanceLeaf = 0;
    //updating new location
    //assert(b && b->pred && b->pred == a);

    if(b && b->pred) assert(b->pred == a);

    pred = a;
    a->suc.push_back(this);

    if(b) {
        b->pred = this;
        suc.push_back(b);
        bDistanceLeaf = b->dist2leaf;
        //remove b from suc in a
        unsigned int i;
        for(i=0;i<a->suc.size(); i++) {
            if(a->suc[i] == b) {
                a->suc.erase(a->suc.begin()+i);
                break;
            }
        }
    }

    //updating subtree
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        tmp->distance=tmp->pred->getDistance() + tmp->getDistance(tmp->pred);
        for(vector<Node*>::iterator it=tmp->suc.begin(); it!=tmp->suc.end(); it++) {
            s.push(*it);
        }
    }

    //update distance after reallocating
    dist2leaf = bDistanceLeaf + getDistance(b);
    distDiff = getDistance(a) + getDistance(b) - a->getDistance(b);
    
    costDiff = getCost(a) + getCost(b) - a->getCost(b);

    distance = pred->distance + getDistance(pred);
    metro_node->tCost+=(costDiff+costSuc);
    //pred ----> root
    for(Node *tmp=pred; tmp!=NULL; tmp=tmp->pred) {
        /*double distTmp = tmp->dist2leaf + distDiff;
        if(distTmp > tmp->dist2leaf) tmp->dist2leaf = distTmp;
        else break;*/
        double distTmp = tmp->getDist2leaf();
        if( fabs(distTmp - tmp->dist2leaf) < EPSILON ) break;
        tmp->dist2leaf = distTmp;
    }

    if(b)
        s.push(b);
    //this ---> leafs
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        tmp->distance+=distDiff;
        for(vector<Node*>::iterator it=tmp->suc.begin(); it!=tmp->suc.end(); it++) {
            s.push(*it);
        }
    }

    if(b) {
        /*
		TwoPred *btp = b->getTP();
		btp->delPred(a, b->isPrimary);
		btp->addPred(this, b->isPrimary);
         */
        b->delPredForest(a);
        b->addPredForest(this);
    }

    /*
	TwoPred *tp = getTP();
	tp->delPred(oldPred, isPrimary);
	tp->addPred(a, isPrimary); 
    */
    delPredForest(oldPred);
    addPredForest(a);
    
    return (metro_node->tCost-cBefore);

}

double Node::costMove2(Node *a, Node *b) {
    //if(!validMoveNode2(a,b)) assert(false); 
    assert(validMoveNode2(a,b));
    return getCost(a) + getCost(b) - a->getCost(b);
    /*double costUpdate = getDistance(a) + getDistance(b) - a->getDistance(b);
    return costUpdate;*/
}

//Not being flexible with the move... sometimes it over estimate the cost
double Node::costMove(Node *a, Node *b) {
    //cost of desconnecting and cost of  reallocating

    //cost of deleting *this*
    double costDel = -getCost(pred);
    for(vector<Node*>::iterator it = suc.begin(); it!=suc.end(); it++) {
        Node *tmp = *it;
        costDel += (pred->getCost(tmp) - getCost(tmp) );
    }

    //if(!validMoveNode(a, b)) return numeric_limits<double>::max();
    if(!validMoveNode(a, b)) assert(false);

    //cost of removing
    double costUpdate = getCost(a) + getCost(b) + costDel - a->getCost(b);

    return costUpdate;
}

pair<Node **, double> Node::bestMoveSubTree(void) {
    //getMetro()->printTree();
    //cout<<"bestMoveSubTree: "<<getId()<<endl;
    //cout<<"tCost bb: "<<metro_node->tCost<<endl;
    double cbefore=metro_node->tCost;
    Node *oldPred = pred;
    delSubTree();
    double tCost = getCost(oldPred) + metro_node->tCost;
    //Node **bMove = bestMoveST(tCost);
    pair<Node**, double> bMove = bestMoveST(tCost);
    //restoring tree...
    if(!bMove.first) {
        //cout<<"Invalid move"<<endl;
        restoreSubTree(oldPred);
    }
    //return bMove.first;
    return make_pair(bMove.first, metro_node->tCost-cbefore);
}


//Node **Node::bestMoveST(double tCost) {
pair<Node **, double> Node::bestMoveST(double tCost) {
    vector< Node** > node;
    Node **n;
    
    stack<Node*> s;
    s.push(getMetroNode());

    Node **aux = NULL;
    pair<Node**, double> bNode = make_pair(aux, -1);
    
    //double best = numeric_limits<double>::max();
    double best = tCost;
    //need to add solutions that improve the objective
    bool added=false;

    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        assert(tmp!=this);
        if(validMoveNode2(tmp, NULL)) {
            Node **nTmp = new Node*[2];
            nTmp[0]=tmp;
            nTmp[1]=NULL;
            //node.push_back(n);

            double costTmp = metro_node->tCost + costMove2(nTmp[0], nTmp[1]);
            if (fabs(costTmp - best) < EPSILON && added) {
                node.push_back(nTmp);
            }
            //else if( distTmp < best ) {
            else if( best - costTmp > EPSILON ) {
                added=true;
                for(unsigned int i=0;i<node.size();i++) {
                    delete [] node[i];
                }
                node.clear();
                node.push_back(nTmp);
                best=costTmp;
                if(firstImprovement) break;
            }
            else delete[] nTmp;
        }
        
        //if(validMove(tmp) && tmp!=this) {
            for(vector<Node*>::iterator it=tmp->suc.begin(); it!=tmp->suc.end(); it++) {
                s.push(*it);
                if( validMoveNode2(tmp, *it) && (*it) != this ) {
                    Node **nTmp = new Node*[2];
                    nTmp[0] = tmp;
                    nTmp[1] = *it;
                    
                    double costTmp = metro_node->tCost + costMove2(nTmp[0], nTmp[1]);
                    if (fabs(costTmp - best) < EPSILON && added) {
                        node.push_back(nTmp);
                    }
                    //else if( distTmp < best ) {
                    else if( best - costTmp > EPSILON ) {
                        added=true;
                        for(unsigned int i=0;i<node.size();i++) {
                            delete [] node[i];
                        }
                        node.clear();
                        node.push_back(nTmp);
                        best=costTmp;
                        if(firstImprovement) break;
                    }
                    else delete[] nTmp;

                    //node.push_back(n);
                }
            }
        //}
    }

    if(node.size() == 0) {
        //return NULL;
        return bNode;
	}
	else {
    	n = node[ random( node.size() ) ];
    	assert(n[0]!=this && n[1]!=this);
	}

    for(unsigned int i=0; i<node.size(); i++) {
        if(node[i] != n)
            delete [] node[i];
    }
    node.clear();
    //return n;

    bNode.first = n;
    bNode.second = best;
    return bNode;
}

Node **Node::randomMoveST(void) {
    vector< Node** > node;
    Node **n;
    
    stack<Node*> s;
    s.push(getMetroNode());
       
    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        assert(tmp != this);
        if(validMoveNode2(tmp, NULL)) {
            Node **nTmp = new Node*[2];
            nTmp[0]=tmp;
            nTmp[1]=NULL;
            node.push_back(nTmp);
        }

        for(unsigned int i=0;i<tmp->suc.size();i++) {
            Node *it = tmp->suc[i];
            s.push(it);
            if( validMoveNode2(tmp, it) ) {
                Node **nTmp = new Node*[2];
                nTmp[0] = tmp;
                nTmp[1] = it;
                node.push_back(nTmp);
            }
        }
    } 

    if(node.size() == 0) {
        return NULL;
	}
	else {
    	n = node[ random( node.size() ) ];
    	assert(n[0]!=this && n[1]!=this);
	}
    
    for(unsigned int i=0; i<node.size(); i++) {
        if(node[i] != n)
            delete [] node[i];
    }
    node.clear();
    
    return n;
}

//NOT WORKING PROPERLY... SHOULDN'T BE USING THIS FUNCTION
pair<Node**, double> Node::bestDistanceMoveNode(void) {
    assert(!pred);
    stack <Node*> s;
    s.push(getMetroNode());
    vector<Node**> node;
    
    Node **aux = NULL;
    pair<Node**, double> best = make_pair(aux, numeric_limits<double>::max());
    
    double bestMove=numeric_limits<double>::max();
    bool added=false;

    while(!s.empty()) {
        Node *tmp = s.top();
        s.pop();
        
        assert(tmp!=this);
        
        if(validMoveNode2(tmp, NULL)) {
            Node **nTmp = new Node*[2];
            nTmp[0] = tmp;
            nTmp[1] = NULL;
        
            double distTmp = tmp->dist2leaf + getDistance(tmp);
            if(fabs(distTmp - bestMove) < EPSILON && added) {
                node.push_back(nTmp);
            }
            else if( bestMove - distTmp > EPSILON ) {
                added=true;
                for(unsigned int i=0;i<node.size();i++) {
                    delete [] node[i];
                }
                node.clear();
                node.push_back(nTmp);
                bestMove = distTmp;
            }
            else delete [] nTmp;
        }
        
        //if(validMove(tmp)) {
            for(int i=0;i<(int)tmp->suc.size();i++) {
                Node *t2 = tmp->suc[i];
                s.push(t2);
                if( validMoveNode2(tmp, t2) ) {
                    Node **nTmp = new Node*[2];
                    nTmp[0] = tmp;
                    nTmp[1] = t2;
                    //breaking a link tmp->t2
                    //new dist2leaf = tmp->distance + t2->dist2leaf + distance(tmp) + distance(t2)
                    double distTmp = t2->dist2leaf + tmp->distance + getDistance(tmp) + getDistance(t2);
                    if(fabs(distTmp-bestMove) < EPSILON && added) {
                        node.push_back(nTmp);
                    }
                    else if(bestMove - distTmp > EPSILON) {
                        added=true;
                        for(i=0;i<(int)node.size();i++)
                            delete [] node[i];
                        node.clear();
                        node.push_back(nTmp);
                        bestMove = distTmp;
                    }
                    else delete [] nTmp;
                }
            }
        //}
    }
    if(node.size() == 0) {
        return best;
    }
    else {
        best.first = node[ random(node.size()) ];
        best.second = bestMove;
        assert(best.first[0]!=this && best.first[1]!=this);
    }
    for(int i=0;i<(int)node.size();i++) {
        if(node[i] != best.first)
            delete [] node[i];
    }
    node.clear();
    return best;
}

//A junction node can be deleted (or added in the tree)
pair<Node **, double> Node::bestMoveNode(void) {
    //assert(false);
    //cout<<"bestMoveNode: "<<getIdString()<<" fnode: "<<fnode->getIdString()<<" level: "<<treeLevel<<endl;
    assert(pred || !isJunction());
    Node *oldPred = pred;
    vector<Node*> tsuc;

    double cbefore = metro_node->tCost;

    for(int i=0;i<(int)suc.size();i++)
        tsuc.push_back(suc[i]);
    double tCostBefore = metro_node->tCost;

    /*metro_node->checkSize();
    metro_node->checkDistance();*/

    if(!delNode()) return make_pair( static_cast<Node**> (NULL), -1);
    //metro_node->checkSize();
    //double tCostAfter = metro_node->tCost;
    Node **bMove = bestMoveST(tCostBefore).first;

    if(!bMove) {
        for(int i=0;i<(int)tsuc.size();i++)
            suc.push_back(tsuc[i]);
        //cout<<"restore.."<<endl;
        restoreNode(oldPred);
    }
    /*else {
        cout<<"bMove:"<<endl;
        if(!bMove[0]->isMetroNode())
            cout<<"MOV1: "<<bMove[0]->fnode->getIdString()<<" ("<<bMove[0]->fnode->sizeTree<<") "<<getIdString()<<endl;
        else if(bMove[1]) cout<<"MOV2: "<<bMove[1]->fnode->getIdString()<<" ("<<bMove[1]->fnode->sizeTree<<") "<<getIdString()<<endl;
    }
    metro_node->checkSize();
    cout<<"FFF"<<endl;
    metro_node->checkDistance();
    cout<<"AAA: "<<endl;*/
    return make_pair(bMove, metro_node->tCost-cbefore);
}

pair<Node **, double> Node::randomMoveSubTree(void) {
    //getMetro()->printTree();
    Node *oldPred = pred;
    double cbefore=metro_node->tCost;
    //metro_node->checkTree();
    delSubTree();
    //Node **bMove = bestMoveST(numeric_limits<double>::max());
    Node **bMove = randomMoveST();
    if(!bMove) {
        restoreSubTree(oldPred);
    }
    return make_pair(bMove, metro_node->tCost- cbefore);
}

pair<Node **, double> Node::randomMoveNode(void) {
    //assert(false);
    //cout<<"randomMoveNode"<<endl;
    assert(pred || !isJunction());
    Node *oldPred = pred;
    vector<Node*> tsuc;
    double cbefore=metro_node->tCost;
    
    for(int i=0;i<(int)suc.size();i++) 
        tsuc.push_back(suc[i]);

    //metro_node->printTree();
    if(!delNode()) return make_pair( static_cast<Node**> (NULL), -1);
    //selecing a random move
    //Node **bMove = bestMoveST(numeric_limits<double>::max());
    Node **bMove = randomMoveST();
    if(!bMove) {
        for(int i=0;i<(int)tsuc.size(); i++) {
            suc.push_back(tsuc[i]);
        }
        //cout<<"restoring node: "<<getIdString()<<endl;
        restoreNode(oldPred);
    }
    //metro_node->checkSize();
    return make_pair(bMove, metro_node->tCost-cbefore);
}

int Node::totalNeighbours(void) {
    return metro_node->size;
}

void Node::printValidNeighbours(void) {
    vector<Node*> node = metro_node->node;
    //priority_queue<Node*> tmp;
    priority_queue<Node*, vector<Node*>, NodeGreater> tmp;

    cout<<"Node: "<<getId()<<" Invalid Size: "<<invalid.size()<<" Neighbours: "<<endl;

    Node *first = NULL;
    if( !invalid.empty() )  {
        first = invalid.top();
        invalid.pop();
        tmp.push(first);
    }

    for(int i=0; i<(int)node.size(); i++ ){
        for(; (i<(int)node.size() && first == node[i] ); i++) {
            cout<<"INVALID: "<<first->getId()<<endl;
            if(invalid.empty()) {
                first = NULL;
                break;
            }
            first = invalid.top();
            invalid.pop();
            tmp.push(first);
        }
        cout<<"VALID: "<<node[i]->getId()<<", "<<endl;
    }
    invalid = tmp;
}

void Node::checkLoop(void) {
    Node *tmp = NULL;
    for(int i=0;i< (int) metro_node->node.size(); i++) {
        if( metro_node->node[i]->isMarked() ){
            cout<<"MARKED: "<<metro_node->node[i]->getId()<<endl;
            assert(false);
        }
    }
	bool mn = false;
    for(tmp=this; tmp!=NULL; tmp=tmp->pred) {
		//root must be the metro node
		if(tmp == getMetroNode()) mn = true;
		if(!tmp->pred) {
			assert(tmp == getMetroNode());
		}
        if( tmp->isMarked() ) {
            cout<<"Error... Loop line, checkLoop NODE: "<<getId()<<" NODE2: "<<tmp->getId()<<endl;
            metro_node->printTree();
            assert(false);
        }
        tmp->markNode();
    }
	assert(mn);
    for(tmp=this; tmp!=NULL; tmp=tmp->pred) {
        tmp->unMarkNode();
    }
}


//only for debuging...
void Node::testDistance(void) {
    double dist=0.0;
    //cout<<"TEST NODE: "<<getId()<<endl;
    for(Node *tmp = this; tmp->pred != NULL; tmp=tmp->pred) {
        //cout<<"tmp: "<<tmp->getPred()->getId()<<" -> "<<tmp->getId()<<" === "<<tmp->getDistance(tmp->getPred())<<endl;
        dist+= tmp->getDistance(tmp->getPred());
    }
    //cout<<"dist: "<<dist<<" :::: "<<getDistance()<<endl;
    //if(dist!=getDistance())
    if( fabs(dist - getDistance()) >= EPSILON ) {
        cout<<"MN: "<<metro_node->getId()<<" Error node: "<<getId()<<" real distance: "<<dist<<" computed distance: "<<getDistance()<<endl;
        assert(false);
        //metro_node->printTree();
    }
    //assert(dist==getDistance());
    assert( fabs(dist - getDistance()) < EPSILON );
    
    //test max length constraint
    metro_node->testDist2Leaf(this);
}


void Node::delTree(void) {
	stack<Node*> s;
	s.push(this);
	while(!s.empty()) {
		Node *tmp = s.top();
		s.pop();
		for(vector<Node*>::iterator it = tmp->suc.begin(); it!=tmp->suc.end(); it++) {
			s.push(*it);
		}
		tmp->suc.clear();
		delete tmp;
	}
}


pair<Node**,double> Node::add2Tree(void) {
    assert(!pred && suc.size() == 0);
    pair<Node**, double> best;
    if(random(100) < 50)
        best = bestDistanceMoveNode();
    else 
        //best = make_pair( randomMoveST(), -1); 
        best = bestMoveST(numeric_limits<double>::max());
    //can't add this node
    /*if(!best.first) {
        return false;
    }*/
    return best;
    //moveSubTree(best.first[0], best.first[1]);
    //return true;
}


bool NodeGreater (Node *i, Node *j) { return (i->getInputId() < j->getInputId()); }
bool NodeGreaterEdge (Node *i, Node *j) { return (i->getDistance(i->getPred()) < j->getDistance(i->getPred())); }


#endif
