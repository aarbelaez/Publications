#ifndef SEARCH__H
#define SEARCH__H

#include "BH.h"
#include "TabuList.h"
#include "LSearch.h"

#include "Move.h"

#include "read_files.h"


class Search : public LSearch {
private:
	virtual MetroNode* selectRandomMetroNode(void) {
        if(activeMN) return activeMN;
		int randMN = random(nbMetroNodes);
		return metro_node[randMN];
	}
	
	Node *selectRandomNode(MetroNode *mn) {
		int randNode = random(mn->size) + 1;
		return mn->getNode( randNode );
	}

	//Node *nNode;
	//Node **n;
    //pair<Node **, double> n;
public:

    virtual pair<Node**, double> bestMove(Node *) = 0;
    virtual pair<Node**, double> randomMove(Node *) = 0;
    virtual double moveElement(Node *, Node *, Node *) = 0;

    virtual bool divStep2(int i) { return (i-lastChange > maxNoImp); }

    //WARNING....
    //assuming that for diversification the number of metro nodes is higher than the number of metro nodes
	virtual Node *selectNode2(void) {
        MetroNode *smn = NULL;
        Node *nTmp = NULL;
        while(true) {
            //locking the selected MN
            #pragma omp critical(MetroNode)
            {
                smn = selectRandomMetroNode();
                if(smn->state == available) {
                    smn->state = busy;
                } else smn = NULL;
            }
            if(smn) {
                #pragma omp critical(Node)
                {
                    nTmp = selectRandomNode(smn);
                    if(nTmp->state == available) {
                        nTmp->state = busy;
                    } else nTmp = NULL;
                }
                //return selectRandomNode(smn);
                if(nTmp) return nTmp;
            }
        }
        assert(false);
        return NULL;
	}
	
	/*virtual Node *selectNode(void) {
        assert(false);
        Node *nTmp = NULL;
        #pragma omp critical(MetroNode)
        {
            //if(valid_nodes.empty() ) {
            if(isDiversification()) {
                massert(!nNode);
                //return selectNode2();
                nTmp = selectNode2();
            }
            //massert(nNode);
        }
        if(!nTmp) assert(nNode);
        return nNode;
	}*/

    void setActiveMetroNode(MetroNode *mn) {
        if(!mn->isActive()) activeMetroNode.push_back(mn);
        assert(activeMetroNode.size()>0);
#ifdef DEBUG
        bool isIn=false;
        for(int i=0;i<(int)activeMetroNode.size();i++) if(activeMetroNode[i]==mn) isIn=true;
        assert(isIn);
#endif
        mn->setActiveNodes();
        //cout<<"ACTIVATING node "<<mn->active.size()<<" metro node: "<<mn->getMetroNode()->getIdString()<<endl;
    }

    virtual void resAllInvalidNodeGraph(MetroNode *mn) {
        //cout<<"resAllInvalidNodeGraph"<<endl;
        #pragma omp critical (MetroNode)
        {
            //cout<<"resAllInvalidNodeGraphL: "<<mn->getMetroNode()->getIdString()<<endl;
            setActiveMetroNode(mn);
            vector<MetroNode*> neighbours = g.getNeighbours(mn);
            for(int i=0;i<(int)neighbours.size();i++) {
                //cout<<"iter: "<<iter<<" Activating metro node: "<<neighbours[i]->getMetroNode()->getIdString()<<" Active nodes: "<<neighbours[i]->active.size()<<" active metro nodes: "<<activeMetroNode.size()<<endl;
                setActiveMetroNode(neighbours[i]);
            }
        }
    }

    virtual void setUnActiveMetroNode(int i) {
        //cout<<"Iter: "<<iter<<" De-activating metro node: "<<(*mn)->getMetroNode()->getIdString()<<" --- "<<(*mn)->active.size()<<endl;
        #pragma omp critical (MetroNode)
        {
            assert(!activeMetroNode[i]->isActive());
            activeMetroNode.erase(activeMetroNode.begin() + i);
        }
    }

    virtual MetroNode *getActiveMetroNode(int i) {
        //cout<<"Search::getActiveMetroNode"<<endl;
        MetroNode *MN = NULL;
        #pragma omp critical (MetroNode)
        {
            MN = activeMetroNode[i];
            if(MN->state == busy) MN=NULL;
            else {
                MN->state = busy;
            }
        }
        return MN;
    }
    
    Node *getActiveNode(MetroNode *mn, int i) {
        Node *nTmp = NULL;
        #pragma omp critical (MetroNode)
        {
            nTmp = mn->getActiveNode(i);
            if(nTmp->state == busy) nTmp = NULL;
            //cout<<"Locking Node: "<<nTmp->getIdString()<<endl;
            else nTmp->state = busy;
        }
        return nTmp;
    }

    //releasing a locked metro node mn
    void unLockMetroNode(MetroNode *mn) {
        #pragma omp critical (MetroNode)
        {
            //cout<<"UnLockMetro: "<<mn->getMetroNode()->getIdString()<<" --- thread: "<<omp_get_thread_num()<<endl;
            if(mn->state == available) cout<<"mn: "<<mn->getMetroNode()->getIdString()<<" thread: "<<omp_get_thread_num()<<endl;
            assert(mn->state == busy);
            mn->state = available;
        }
    }
    //releasing a locked node m
    void unLockNode(Node *m) {
        assert(m->state == busy);
        m->state = available;
        //cout<<"Unlocking Node: "<<m->getIdString()<<endl;
    }
    
    virtual pair<MetroNode *, int> getRandomActiveMetroNode(void) {
        int randMetroNode = random(activeMetroNode.size());
        MetroNode *mn = getActiveMetroNode(randMetroNode);
        return make_pair(mn, randMetroNode);
    }

	virtual Move divStep(int i) {
        //cout<<"divStep"<<endl;
		Node *nNode = NULL;
        pair<Node **, double> n = make_pair(static_cast<Node**> (NULL), -1);
		//n.first = NULL;
		assert(!isDiversification());
        MetroNode *MN = NULL;
		do {
            int randMetroNode, randNode;

            if(nNode) { 
                assert(MN);
                unLockNode(nNode);
                nNode=NULL;
            }
            
            if(MN) {
                unLockMetroNode(MN);
            }

            /*
            randMetroNode = random(activeMetroNode.size());
            MN = getActiveMetroNode(randMetroNode);
             */
            pair<MetroNode*, int> rmetro = getRandomActiveMetroNode();
            MN = rmetro.first;
            randMetroNode = rmetro.second;

            if(!MN) continue;

            randNode = MN->random(MN->getTotalActiveNodes());

            nNode = getActiveNode(MN, randNode);

            if(!nNode) continue;
            assert(nNode!=nNode->getPred());

            n = bestMove(nNode);
            if(!n.first) {
                if(MN->setInactiveNode(randNode)) setUnActiveMetroNode( randMetroNode );
            }
        }while( !n.first && !isDiversification() );

        if(isDiversification()) {
            if(nNode) {
                unLockNode(nNode);
                unLockMetroNode(MN);
            }
            nNode = NULL;
        }
		//if(valid_nodes.empty()) nNode = NULL;
		else if(!nNode) {
            //cout<<"Before unlock -thread: "<<omp_get_thread_num()<<endl;
            unLockMetroNode(MN);
			//return true;
		}
  		if(nNode) { assert(nNode!=nNode->getMetroNode()); }
		//return false;
        //return nNode;
        return Move(nNode, n);
	}

    //virtual void SelElement(int move) {
    virtual void SelElement(Move m) {
        //cout<<"SelElement"<<endl;
		Node *sNode = NULL;
        if(m.nNode) {
            /*#pragma omp critical
            {
            cout<<"FFFFF: thread -> "<<(omp_get_thread_num())<<" ---- "<<m.nNode<<endl;
            }*/
            /*
            assert(nNode);
			sNode = selectNode();
	        assert(sNode != sNode->getMetroNode() );
			if(!n.first) {
            	n = bestMove(sNode);
            }
            */
            sNode = m.nNode;
            assert(sNode != sNode->getMetroNode());
        }
        else {
			//assert(!nNode);
            sNode = selectNode2();
  	        assert(sNode != sNode->getMetroNode() );
			m.n = randomMove(sNode);
        }
        totalCost+=m.n.second;
/*#pragma omp barrier
#pragma omp critical
        {
        cout<<"AFTER Barrier tmp --- thread: "<<omp_get_thread_num()<<endl;
        }
#pragma omp barrier*/
        if(!m.n.first) {
            //cout<<"NO MOVE Thread: ["<<omp_get_thread_num()<<"] unlock metro: "<<(metro_node[ sNode->getIdMetroNode()])->getMetroNode()<<endl;
            //assert(sNode->isJunction());
            unLockMetroNode( metro_node[ sNode->getIdMetroNode()] );
            unLockNode(sNode);
            m.nNode=NULL;
            return;
        }
        //updating metro node connectivity graph
		MetroNode *mnIter = metro_node[m.n.first[0]->getIdMetroNode()];
        resAllInvalidNodeGraph(mnIter);
		//resAllInvalidNodeGraph(mnIter);
        //m.n.first[0]->usage++;
        //cout<<"MOVE::: "<<mnIter->tCost<<endl;
        totalCost+=moveElement(sNode, m.n.first[0], m.n.first[1]);
        //cout<<"MOVE AFTER::: "<<mnIter->tCost<<endl;
/*#pragma omp barrier
#pragma omp critical (MMM)
        {
        cout<<"Before unlock: "<<mnIter->getMetroNode()->getIdString()<<" thread: "<<omp_get_thread_num()<<" n: "<<m.n.first[0]<<" MNID: "<<m.n.first[0]->getIdMetroNode()<<endl;
        }*/
        unLockMetroNode(mnIter);
        unLockNode(sNode);
        
        delete [] m.n.first;
		m.n.first=NULL;
    }

/*	
    virtual void SelElement(int move) {

		Node *sNode = selectNode();
        assert(sNode != sNode->getMetroNode() );

        Node **n;
        if(move == 1) {
            n = bestMove(sNode);
        }
        else {
            n = randomMove(sNode);
        }
        if(n==NULL) {
            return;
        }
        n[0]->usage++;

        moveElement(sNode, n[0], n[1]);
        delete n;
    }
*/    
    void printTree(void) {
        cout<<"digraph G {"<<endl
        <<" size =\"4,4\";"<<endl;
        
        for(int i=0;i<(nbExchangeSites+nbMetroNodes);i++) {
            cout<<"Node_"<<site_node[i].id<<" [pos=\""<<site_node[i].x<<","<<site_node[i].y<<"!\"]"<<endl;
        }
        //the label should be the distance between a given node and its predecessor
        for(int i=0; i<nbMetroNodes; i++) {
            metro_node[i]->printTree();
        }
        cout<<"}"<<endl;
    }

    virtual void updateBest(Solution &, Solution &, bool) {
        assert(false);
    }
    
public:
	Search(Parameters _param) :LSearch(_param) { }
    
    void SelElementDiversication() { 
        SelElement(Move());
    }
    void SelElementIntensification(Move m) {
        assert(m.nNode);
        SelElement(m);
    }
};


class NodeSearch : public Search { 
    int opt;
public:
    NodeSearch(Parameters _param) : Search(_param) { }
    pair<Node **, double> bestMove(Node *s) {
        opt=1;
        return s->bestMoveNode();
    }
    pair<Node **, double> randomMove(Node *s) {
        opt=2;
        return s->randomMoveNode(); 
    }
    //void moveElement(Node *s, Node *n1, Node *n2) { s->moveNode(n1, n2); }
    double moveElement(Node *s, Node *n1, Node *n2) {
        n1->getMetro()->checkSize();
        double cDiff;
        if(opt==1) 
            cDiff = s->moveSubTree(n1, n2);
        else if(opt==2)
            cDiff= s->moveSubTree(n1, n2);
            //s->move(n1);
        else assert(false);
        s->getMetro()->checkSize();
        s->getMetro()->checkTree();
        s->getMetro()->checkDistance();
        return cDiff;
    }
        
};

class SubTreeSearch : public Search {
    int opt; 
public:
    SubTreeSearch(Parameters _param) : Search(_param) { 
        cout<<"SubTreeSearch"<<endl;
    }
    pair<Node **, double> bestMove(Node *s) {
        opt=1;
        //Node **n = s->bestMoveSubTree();
        return s->bestMoveSubTree();
    }
    pair<Node **, double> randomMove(Node *s) {
        /*opt=2;
        Node **n = new Node*[2];
        n[0] = s->randomMove();
        n[1] = NULL;
        if(!n[0]) {
            delete []n;
            return NULL;
        }
        return n;*/
        return s->randomMoveSubTree();
    }

    double moveElement(Node *s, Node *n1, Node* n2) {
        //if(s->getInputId() == 682) cout<<"SIZE TREE: "<<s->sizeTree<<" iter: "<<iter<<endl;
        return s->moveSubTree(n1, n2);
        /*if(opt==1) { 
            s->moveSubTree(n1, n2); 
        }
        else if(opt==2) {
            s->move(n1);
        }
        else assert(false);*/
    }

};

class SubTreeSearchPar : public SubTreeSearch {
protected:
    timer localClock;
public:
    SubTreeSearchPar(Parameters _param) : SubTreeSearch(_param) {
        cout<<"SubTreeSearchPar"<<endl;
        nbThreads = _param.nbCores;
    }
    
    virtual void solve(void) {
        ILSPar();
    }
    bool isDiversification(void) {
        assert(metro_node_core.size() > 0);
        int idThread = omp_get_thread_num();
        return !metro_node_core[idThread]->isActive();
    }

    virtual void selectMetroNodes(void) {
        vector<int> mns = g.indepdentSet(nbThreads, NULL);
        metro_node_core.clear();
        //assert(nbThreads <= (int) mns.size());
        
        for(unsigned int i=0;i<mns.size();i++) {
            metro_node_core.push_back(metro_node[ mns[i] ]);
        }
    }

    virtual void updateBest(void) {
        int idThread = omp_get_thread_num();
        assert(idThread < (int)metro_node_core.size());
        MetroNode *mn = metro_node_core[idThread];
        MetroNode *mnBest = best_sol.metro[ mn->getId() ];
        assert(mn->getId() == mnBest->getId());
        if(mnBest->tCost > mn->tCost) {
            /*
            #pragma omp critical (UpdateBest)
            {
                cout<<"update cost thread["<<idThread<<"]: best: "<<mnBest->tCost<<" current: "<<mn->tCost<<endl;
            }
             */
            best_sol.setSolution(mn);
        }
    }
    
    virtual MetroNode* selectRandomMetroNode(void) {
        int idThread = omp_get_thread_num();
        assert(idThread < (int) metro_node_core.size());
        return metro_node_core[idThread];
    }

    virtual void updateBest(Solution &s1, Solution &s2, bool force) {
        int idThread = omp_get_thread_num();
        assert(idThread < (int) metro_node_core.size());
        int indexMetro = metro_node_core[idThread]->getId();
        MetroNode *mn1 = s1.metro[indexMetro];
        MetroNode *mn2 = s2.metro[indexMetro];
        if(!(mn1->getId() == indexMetro && mn2->getId() == indexMetro)) {
            cout<<"mn1["<<s1.name<<"]: "<<mn1->getId()<<" mn2["<<s2.name<<"]: "<<mn2->getId()<<" indexMetro: "<<indexMetro<<endl;
        }
        assert(mn1->getId() == indexMetro && mn2->getId() == indexMetro);
        if(mn1->tCost > mn2->tCost || force)
            s1.setSolution(mn2);
        
    }
    
    virtual bool stop(void) {
        //double cost = best_sol.aggregateCost();
        if( maxTime > mclock.elapsed() && !forceStop && !param.initial_sol_only) return false;
        return true;
    }
    
    virtual bool localStop(void) {
        if(maxLocalTime > localClock.elapsedMiliSeconds() && !stop()) return false;
        //if ( maxLocalTime > localClock.elapsed() && ! stop()) return false;
        return true;
    }
    virtual void localStart(void) {
        localClock.init();
    }

    virtual void setUnActiveMetroNode(int i) {
        int idThread = omp_get_thread_num();
        assert(omp_get_thread_num() < (int)metro_node_core.size());
        MetroNode *mn = metro_node_core[idThread];
        assert(!mn->isActive());
    }
        
    /*virtual MetroNode *getActiveMetroNode(int) {
        int idThread = omp_get_thread_num();
        assert(idThread < (int) metro_node.size());
        MetroNode *mn = metro_node_core[idThread];
        assert(mn->state == available);
        mn->state = busy;
        return mn;
    }*/
    
    virtual pair<MetroNode*, int> getRandomActiveMetroNode(void) {
        int idThread = omp_get_thread_num();
        assert(idThread < (int) metro_node.size());
        MetroNode *mn = metro_node_core[idThread];
        assert(mn->state == available);
        mn->state = busy;
        return make_pair(mn, -1);
    }

    virtual void resAllInvalidNodeGraph(MetroNode *mn) {
        //cout<<"resAllInvalidNodeGraph"<<endl;
#pragma omp critical (MetroNode)
        {
            //cout<<"resAllInvalidNodeGraphL: "<<mn->getMetroNode()->getIdString()<<endl;
            setActiveMetroNode(mn);
        }
    }

    /*
    virtual void setUnActiveMetroNode(int) {
        //cout<<"Iter: "<<iter<<" De-activating metro node: "<<(*mn)->getMetroNode()->getIdString()<<" --- "<<(*mn)->active.size()<<endl;
        assert(!activeMetroNode[i]->isActive());
        activeMetroNode.erase(activeMetroNode.begin() + i);
    }
    
    virtual void resAllInvalidNodeGraph(MetroNode *mn) {
        //cout<<"resAllInvalidNodeGraph"<<endl;
        setActiveMetroNode(mn);
        vector<MetroNode*> neighbours = g.getNeighbours(mn);
        for(int i=0;i<(int)neighbours.size();i++) {
            //cout<<"iter: "<<iter<<" Activating metro node: "<<neighbours[i]->getMetroNode()->getIdString()<<" Active nodes: "<<neighbours[i]->active.size()<<" active metro nodes: "<<activeMetroNode.size()<<endl;
            setActiveMetroNode(neighbours[i]);
        }
    }
     */

};
bool MetroNodeGreater (MetroNode *i,MetroNode *j) { return (i->getId()<j->getId()); }

class SubTreeSearchPar2 : public SubTreeSearchPar {
public:
    SubTreeSearchPar2(Parameters _param) : SubTreeSearchPar(_param) {
        cout<<"SubTreeSearchPar2"<<endl;
    }

    virtual void selectMetroNodes(void) {
        assert(metro_node_core.empty());
        vector<int> validSet;
        //g.printConflicts();
        for(unsigned int i=0;i<metro_node.size();i++) validSet.push_back(metro_node[i]->getId());
        for(int i=0;i<nbThreads;i++) {
            int rgn = random(validSet.size());
            metro_node_core.push_back( metro_node[ validSet[rgn] ] );
            //cout<<"thread: "<<i<<" MN: "<<metro_node[ validSet[rgn] ]->getIdString()<<endl;
            validSet.erase( validSet.begin() + rgn );
        }
        
        sort(metro_node_core.begin(), metro_node_core.end(), MetroNodeGreater);
        
        for(unsigned int i=0;i<metro_node_core.size();i++) {
            for(unsigned int j=i+1;j<metro_node_core.size();j++) {
                /*cout<<"Conflicts between: "<<metro_node_core[i]->getIdString()
                <<" and "<<metro_node_core[j]->getIdString()<<"  -> "
                <<g.g_conflict[ metro_node_core[i]->getId() ][ metro_node_core[j]->getId()].size()
                <<endl;*/
                //vector< pair<Node *, Node*> > conflicts = g.g_conflict[i][j];
                vector< pair<Node*, Node*> > conflicts = g.g_conflict[ metro_node_core[i]->getId() ][ metro_node_core[j]->getId()];
                for(unsigned int c=0;c<conflicts.size();c++) {
                    Node *n11 = conflicts[c].first;
                    Node *n12 = conflicts[c].second;

                    assert(n11!=n12);
                    int indexN1 = n11->getId();
                    int indexN2 = n12->getId();
                    SiteNode sn1 = site_node[indexN1];
                    SiteNode sn2 = site_node[indexN2];
                    
                    Node *n21 = (sn1.n1==n11)?sn1.n2:sn1.n1;
                    Node *n22 = (sn2.n1==n12)?sn2.n2:sn2.n1;

                    assert(n11->getId() == n21->getId());
                    assert(n12->getId() == n22->getId());
                    assert(n21!=n22);
                    assert(n11!=n21 && n11!=n22 && n12!= n21 && n12!=n22);

                    //check whether link n1->n2 or n2->n1 is already included in mn1 or mn2
                    TwoPred *tp1 = tps[indexN1];
                    TwoPred *tp2 = tps[indexN2];
                    
                    MetroNode *metro1 = n11->getMetro();
                    MetroNode *metro2 = n21->getMetro();
                    if(!(metro1 == n11->getMetro() && metro1 == n12->getMetro() &&
                         metro2 == n21->getMetro() && metro2 == n22->getMetro() ) ) {
                        cout<<"metro1: "<<metro1->getIdString()<<" metro2: "<<metro2->getIdString()
                        <<" n11 metro: "<<n11->getMetro()->getIdString()<<" n12 metro: "
                        <<n12->getMetro()->getIdString()<<" n21 metro: "<<n21->getMetro()->getIdString()
                        <<" n22 metro :"<<n22->getMetro()->getIdString()
                        <<endl;
                        cout<<"n11: "<<n11<<" n12: "<<n12<<" n21: "<<n21<<" n22: "<<n22<<endl;
                    }
                    //assert(metro1 == n22->getMetro() && metro2 == n12->getMetro());
                    assert(metro1 == n11->getMetro() && metro1 == n12->getMetro() &&
                           metro2 == n21->getMetro() && metro2 == n22->getMetro() );

                    MetroNode *mdel1 = NULL; // removing sn1->sn2 in this metro
                    MetroNode *mdel2 = NULL; // removing sn2->sn1 in this metro

                    //link n2->n1 in metro1...
                    if(n11->getPred() == n12) {
                        mdel2 = metro2;
                        if( !(tp1->pred[(n11->isPrimary?0:1)] == n12->getId()) ) cout<<tp1->toString()<<endl;
                        assert( tp1->pred[(n11->isPrimary?0:1)] == n12->getId() );
                    }
                    //link n1->n2 in metro1
                    if(n12->getPred() == n11) {
                        mdel1 = metro2;
                        if( !(tp2->pred[(n12->isPrimary?0:1)] == n11->getId()) ) cout<<tp2->toString()<<endl;
                        assert( tp2->pred[(n12->isPrimary?0:1)] == n11->getId() );
                    }
                    //link n2->n1 in metro2
                    if(n21->getPred() == n22) {
                        assert(!mdel2);
                        mdel2 = metro1;
                        if( !(tp1->pred[(n21->isPrimary?0:1)] == n22->getId()) ) {
                            cout<<"metro1: "<<metro1->getIdString()<<" metro2: "<<metro2->getIdString()
                            <<" n11 metro: "<<n11->getMetro()->getIdString()<<" n12 metro: "
                            <<n12->getMetro()->getIdString()<<" n21 metro: "<<n21->getMetro()->getIdString()
                            <<" n22 metro :"<<n22->getMetro()->getIdString()
                            <<endl;

                            cout<<"FOUND: "<<n22->getId()<<" -> "<<n21->getId()<<" metro: "<<n22->getMetro()->getIdString()<<endl;
                            cout<<"tps2: "<<tp2->toString()<<endl;
                            cout<<"tps1: "<<tp1->toString()<<endl;
                        }
                        assert( tp1->pred[(n21->isPrimary?0:1)] == n22->getId() );
                    }
                    //link n1->n2 in metro2
                    if(n22->getPred() == n21) {
                        assert(!mdel1);
                        mdel1 = metro1;
                        if( !(tp2->pred[(n22->isPrimary?0:1)] == n21->getId()) ) cout<<tp2->toString()<<endl;
                        assert( tp2->pred[(n22->isPrimary?0:1)] == n21->getId() );
                    }
                    
                    if(!mdel1) {
                        if(random(2) == 0) mdel1 = metro1;
                        else mdel1 = metro2;
                    }
                    if(!mdel2) {
                        if(random(2) == 0) mdel2 = metro1;
                        else mdel2 = metro2;
                    }

                    addInvalidEdge(mdel1, sn1, sn2);
                    addInvalidEdge(mdel2, sn2, sn1);
                    //addInvalidEdge(mnRemove, site_node[n1->getId()], site_node[n2->getId()]);
                }
            }
        }
    }
    
    void clearSelectMetroNodes(void) {
        restoreEdges();
        metro_node_core.clear();
    }
    //need to remove link s1 -> s2
    void addInvalidEdge(MetroNode *mn, SiteNode &s1, SiteNode &s2) {
        Node *n1 = s1.n1;
        Node *n2 = s2.n1;

        if(n1->getMetro() != mn) n1 = s1.n2;
        if(n2->getMetro() != mn) n2 = s2.n2;

        assert(n1->getMetro() == mn && n2->getMetro() == mn);
        //cout<<"Invalid edge: "<<n1->getId()<<" -> "<<n2->getId()<<" in metro: "<<mn->getIdString()<<endl;
        assert(n1!=n2);
        mn->setInValidEdge(n1, n2);
    }
    
    void restoreEdges(void) {
        for(unsigned int i=0;i<metro_node_core.size();i++) {
            metro_node_core[i]->setValidAllEdges();
        }
    }

    void InitData(void) {
        for(unsigned int i=0;i<metro_node.size();i++) metro_node[i]->initValidEdges();
    }

    virtual bool localStop(void) {
        if(SubTreeSearchPar::localStop()) {
            //restore invalid edges
            return true;
        }
        return false;
    }
    virtual void localStart(void) {
        SubTreeSearchPar::localClock.init();   
    }


};

/*
//ATENTION ReWireSearch is no longer supported
class ReWireSearch : public Search {
public:
	ReWireSearch(Parameters _param) : Search (_param) { cout<<"ReWireSearch "<<endl;}
    Node **bestMove(Node *s) { 
		Node **n = new Node*[2];
        n[0] = s->bestMove();
        n[1]=NULL;
        if(!n[0]) {
			delete [] n;
            return NULL;
        }
        return n;
    }
    Node **randomMove(Node *s) {
		Node **n = new Node*[2];
        n[0] = s->randomMove();
        n[1] = NULL;
        if(!n[0]) {
			delete[] n;
            return NULL;
        }
        return n;
    }
    void moveElement(Node *s, Node *n1, Node*) { s->move(n1); }
    //Node **bestMove(Node *s) { return s->bestMove(); }
    //Node **randomMove(Node *s) { return s->randomMove(); }
};


class ReWireExhaustive : public ReWireSearch {
public:
	TabuList t;
	ReWireExhaustive(Parameters _param) : ReWireSearch(_param) { cout<<"ReWireExhaustive: "<<endl; }
	
	Node *selectNode() {
		double maxCost = -1.0;
		Node *bestNode = NULL;
		for(vector<MetroNode*>::iterator it=metro_node.begin(); it!=metro_node.end(); it++) {
			MetroNode *itMN = *it;
			for(int i=1;i<(int)itMN->node.size(); i++) {
				Node *tmp = itMN->getNode(i);
				double tmpCost = tmp->getDistance(tmp->getPred());
				if(tmpCost > maxCost && !t.isTabu(tmp)) {
					bestNode = tmp;
					maxCost = tmpCost;
				}
			}
		}
		t.addElement(bestNode);
		assert(bestNode);
		return bestNode;
	}
};


class ReWire_Node : public Search {
public:
    int opt;
	ReWire_Node(Parameters _param) : Search(_param) {  cout<<"ReWire_Node: "<<endl; }
    Node **bestMove(Node *s) { 
        opt=1;
        Node **n = new Node*[2];
        n[0] = s->bestMove();
        n[1] = NULL;
        if(!n[0]) {
            delete [] n;
            return NULL;
        }
        return n;
    }
    Node **randomMove(Node *s) { 
        opt=2;
        return s->randomMoveNode(); 
    }

    void moveElement(Node *s, Node *n1, Node* n2) { 
        if(opt==1) { 
            s->move(n1); 
        }
        else if(opt==2) {
            s->moveNode(n1, n2);
        }
        else assert(false);
    }
};
*/

#endif
