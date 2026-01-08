
#include <iostream>

#include "BH.h"
#include "Node.h"
#include "MetroNode.h"
#include "read_files.h"
#include "timer.h"

#include <algorithm>

void read_sites_nodes_format2(const char *fname, vector<Pair> &snode) {
    string fline;
    ifstream file;
    file.open(fname);
    getline(file, fline);
    
    double id, x, y, aux;
    while (file>>x>>y>>aux>>id) {
        Pair p; p.id=id; p.x=x; p.y=y;
#ifdef SCALE2
		p.x=p.x/FACTOR;
		p.y=p.y/FACTOR;
#endif
        snode.push_back(p);
        if(file.eof()) break;
    }
    cout<<"sites_node: "<<snode.size()<<endl;
}

void read_metro_nodes_format2(const char *fname, vector<Pair> &mnode) {
    ifstream file;
    file.open(fname);
    
    double x, y;
    int index=0;
    
    while(file>>x>>y) {
        Pair p; 
        p.id=index; index++;
        p.x=x; p.y=y;
#ifdef SCALE2
		p.x=p.x/FACTOR;
		p.y=p.y/FACTOR;
#endif
        mnode.push_back(p);
        if(file.eof()) break;
    }
    cout<<"metro_nodes: "<<mnode.size()<<endl;
}

void Set_Init_ExchangeSites_Format2(const char *fname, vector<Pair> mnode1, vector<Pair> snode1, vector<MetroNode*> &mnodes, vector<SiteNode> &snode) {
    ifstream file;
    file.open(fname);

    int id, mn1, mn2;
    double aux;
    vector<int> lsites;
    vector<int> metro1;
    vector<int> metro2;
    vector<int> validMetroNode;

    while(file>>id>>mn1>>mn2>>aux>>aux) {
        lsites.push_back(id);
        bool addVal1=true, addVal2=true;
        metro1.push_back(mn1);
        metro2.push_back(mn2);
        for(vector<int>::iterator it=validMetroNode.begin(); it!=validMetroNode.end(); it++) {
            if( *it == mn1 ) addVal1=false;
            if( *it == mn2 ) addVal2=false;
        }
        if(addVal1)
            validMetroNode.push_back(mn1);
        if(addVal2)
            validMetroNode.push_back(mn2);
        
        if(file.eof()) break;
    }

    for(int i=0; i<(int)validMetroNode.size(); i++) {
        MetroNode *n = new MetroNode();
        n->setId(i);
        
        //adding the node into nodes **as metro node or root**
        Node *nNode = new Node(i, 0);
        n->node.push_back(nNode);
        nNode->setMetroNode();
        
        mnodes.push_back(n);
        
        int idFormat = validMetroNode[i];
		nNode->setInputId(idFormat);
        double x, y;
        for(vector<Pair>::iterator it=mnode1.begin(); it!=mnode1.end();  it++) {
            if(it->id == idFormat) {
                x=it->x; y=it->y;
            }
        }

        SiteNode n1(x, y);
        n1.id = i;
        snode.push_back(n1);
        
    }
    cout<<"Valid metro nodes: "<<mnodes.size()<<" _-- "<<validMetroNode.size()<<endl;

    int index = snode.size();
    for(int i=0;i<(int)snode1.size(); i++) {
        bool valid=false;
        int iLnode=-1;
        for(int j=0; j<(int) lsites.size();j++) {
            if(lsites[j] == snode1[i].id ) {
                valid=true;
                iLnode=j;
                break;
            }
        }
        if(!valid) continue;


        int mn1=-1, mn2=-1;
        for(int j=0;j<(int) validMetroNode.size(); j++) {
            if(validMetroNode[j] == metro1[iLnode]) mn1=j;
            if(validMetroNode[j] == metro2[iLnode]) mn2=j;
        }
        assert(mn1!=-1 && mn2!=-1);

        SiteNode n(snode1[i].x, snode1[i].y);
        n.id = index;
		n.id_mn[0] = mn1; n.id_mn[1] = mn2;
		//n.id_mn[0] = metro1[iLnode]; n.id_mn[1] = metro2[iLnode];
        snode.push_back(n);

		Node *n1 = new Node(mn1, index);
		Node *n2 = new Node(mn2, index);
        n1->isPrimary=true;
        n2->isPrimary=false;
		n1->setInputId(snode1[i].id);
		n2->setInputId(snode1[i].id);
        
        n.n1 = n1;
        n.n2 = n2;
        
		assert(mn1 == mnodes[mn1]->getId() && mn2==mnodes[mn2]->getId());
		mnodes[mn1]->node.push_back(n1);
		mnodes[mn2]->node.push_back(n2);
        /*mnodes[mn1]->node.push_back( new Node(mn1, index) );
        mnodes[mn2]->node.push_back( new Node(mn2, index) );*/
        index++;
    }
    
    cout<<"site_nodes: "<<snode.size()<<" === "<<snode1.size()<<" lsites: "<<lsites.size()<<" mnodes: "<<mnodes.size()<<endl;
}


void read_site_nodes(const char *fname, int nbMetroNodes, vector<SiteNode> &node) {
	string fline;
    ifstream file;
    file.open(fname);
    getline(file, fline);
    int id, aux;
	double x, y;
    int index = nbMetroNodes;
    while(file>>x>>y>>aux>>id) {
#ifdef SCALE2
				x=x/FACTOR; y=y/FACTOR;
#endif
        SiteNode n(x, y);
        n.id = index; index++;
        node.push_back(n);
        if(file.eof()) break;
    }

    cout<<"site nodes: "<<node.size()<<endl;
}

void read_metro_nodes(const char *fname, vector<MetroNode*> &mnode, vector<SiteNode> &snode) {
    ifstream file;
    file.open(fname);

    int id, aux;
	double x, y;
    int index=snode.size();
    while(file>>id>>x>>y>>aux) {
		//cout<<"ID["<<id<<"] : "<<x<<","<<y<<endl;
#ifdef SCALE2
				x=x/FACTOR; y=y/FACTOR;
#endif

        MetroNode *n = new MetroNode();
        n->setId(index);
        //adding the node into nodes **as metro node or root**
        Node *nNode = new Node(index, 0);
        n->node.push_back(nNode);
        nNode->setMetroNode();
		nNode->setInputId(id);
		//cout<<"nNode: "<<nNode->getInputId()<<endl;

        mnode.push_back(n);
        SiteNode n1(x, y);
        n1.id = index; index++;
        snode.push_back(n1);

        if(file.eof()) break;
    }
    
    cout<<"metro nodes: "<<mnode.size()<<endl;
}

void Set_Init_ExchangeSites(const char *fname, vector<SiteNode> &snode, vector<MetroNode*> &mnode) {
    ifstream file;
    file.open(fname);

    int id1, id2, aux;

    int index = (int)mnode.size();
	int inputId=0;
    while(file>>id1>>id2>>aux) {
        /*
        id1 = id1 + (int) mnode.size();
        id2 = id2 + (int) mnode.size();
        int i1 = -1, i2 = -1;
        //searching the metro node with this id
        for(int i=0; i<(int) mnode.size(); i++) {
            if( snode[id1].x == snode[ mnode[i]->getId() ].x && snode[id1].y == snode[ mnode[i]->getId() ].y )
                i1 = i;
            if( snode[id2].x == snode[ mnode[i]->getId() ].x && snode[id2].y == snode[ mnode[i]->getId() ].y )
                i2 = i;
        }
        */
        int i1 = -1, i2 = -1;
        for(int i=0; i<(int) mnode.size(); i++) {
            Node *nMN = mnode[i]->getMetroNode();
            assert(nMN);
            if( nMN->getInputId() == id1 ) i1 = i;
            if( nMN->getInputId() == id2 ) i2 = i;
        }
        assert(i1 != -1 && i2 != -1);

        snode[index].id_mn[0] = i1; snode[index].id_mn[1] = i2;
		Node *n1 = new Node(i1, index);
		Node *n2 = new Node(i2, index);
		//cout<<inputId<<" "<<i1<<" - "<<i2<<endl;
		n1->setInputId(inputId);
		n2->setInputId(inputId);
        n1->isPrimary=true;
        n2->isPrimary=false;

        snode[index].n1 = n1;
        snode[index].n2 = n2;
        
		inputId++;

        mnode[i1]->node.push_back( n1 );
        mnode[i2]->node.push_back( n2 );
        index++;
        if(file.eof()) break;
    }

	/*
    for(vector<MetroNode>::iterator it=mnode.begin(); it!=mnode.end(); it++) {
        it->size = it->node.size()-1;
        cout<<"Metro Node: "<<it->getNode(1)->getMetroNode()<<endl;
        for(int i=1; i<(int) it->node.size(); i++) {
            it->getNode(i)->initPred();
        }
        it->sortNodes();
        //it->printNodes();
        cout<<"MN["<<it->getId()<<" ]: "<<it->size<<endl;
    }
	*/
}

//will be using this to load data in the ODN part... here the association is local exchanges are metro nodes and customers are local exchanges
//there must be only one metro node... and all exchange sites are associated to this metro
//of course this is for building the distance-constrainted minimum bounded spanning tree (DCMBST)
void Read_Customers(const char *fname, vector<MetroNode*> &mnode, vector<SiteNode> &snode) {
    cout<<"Reading Customers: "<<fname<<endl;
    ifstream file;
    file.open(fname);

    double x, y;

    int index = 0;
	int indexId=0;

   
    file>>x>>y;
#ifdef SCALE2
    x=x/FACTOR; y=y/FACTOR;
#endif
    
    MetroNode *mn = new MetroNode();
    mn->setId(index);
    //adding the node into nodes **as metro node or root**
    Node *nNode = new Node(index, 0);
    mn->node.push_back(nNode);
    nNode->setMetroNode();
    nNode->setInputId(0);
    //cout<<"nNode: "<<nNode->getInputId()<<endl;
    
    mnode.push_back(mn);
    SiteNode n1(x, y);
    n1.id = index; index++;
    snode.push_back(n1);

    while(file>>x>>y) {
#ifdef SCALE2
        x=x/FACTOR; y=y/FACTOR;
#endif
        SiteNode sn(x, y);
        sn.id = index;
        snode.push_back(sn);
        //cout<<"Adding node: "<<indexId<<" mn: "<<mn->getInputId()<<endl;
        Node *node = new Node(0, index);
        node->setInputId(indexId);
        node->isPrimary=true;
        sn.n1 = node;
        sn.n2 = NULL;

        mn->node.push_back( node );

        indexId++;
        index++;
        if(file.eof()) break;
    }
    file.close();
}


//bool MetroNodeGreater (MetroNode *i,MetroNode *j) { return (i->getId()<j->getId()); }

//bool NodeGreater (Node *i, Node *j) { return (i->getInputId() < j->getInputId()); }

Node *BinarySearch(vector<Node*> array, int key) {
    // Keep halving the search space until we reach the end of the vector
    int begin = 0;
    int end = array.size()-1;
    
    while(begin <= end) {
        // Find the median value between the iterators
        //Iterator Middle = begin + (std::distance(begin, end) / 2);
        int Middle = begin + ((end - begin)/2);
        
        // Re-adjust the iterators based on the median value
        if( array[Middle]->getInputId() == key ) {
            return array[Middle];
        }
        else if(array[Middle]->getInputId() > key) {
            end = Middle -1;
        }
        else {
            begin = Middle + 1;
        }
    }
    
    return NULL;
}

double getEuclideanDistance(Node *n1, Node *n2, vector<SiteNode> &site_node) {
    return sqrt( pow(getX(n1->getId()) - getX(n2->getId()), 2) + pow(getY(n1->getId()) - getY(n2->getId() ), 2) );
}


void read_cost_euclidean_distance(vector<MetroNode*> &mnode, vector<SiteNode> &snode) {
    for(int i=0;i<(int)mnode.size();i++) {
        MetroNode *mn = mnode[i];
        int sTmp = mn->node.size();
        mn->distMatrix = new double*[sTmp];
        mn->costMatrix = new double*[sTmp];

        for(int node_i=0;node_i<(int)mn->node.size();node_i++) {
            mn->distMatrix[node_i] = new double[sTmp];
            mn->costMatrix[node_i] = new double[sTmp];
        }
        for(int node_i=0;node_i<(int)mn->node.size();node_i++) {
            Node *ni = mn->node[node_i];
            for(int node_j=node_i;node_j<(int)mn->node.size();node_j++) {
                Node *nj = mn->node[node_j];
                //if(nj->isMetroNode()) continue;
                double dist = getEuclideanDistance(ni, nj, snode);
                mn->setDistAndCostMatrixValue(ni, nj, dist, dist);
            }
        }
    }
}

void read_cost_dist(const char *fname, const char *fnameMN, vector<MetroNode*> &mnode, vector<SiteNode> &snode) {
    ifstream file;
    file.open(fname);
    
    int id1, id2;
    double cost, dist;
    
    timer t;
    t.init();
    
    string sfile(fname);
    string sfileMN(fnameMN);
    if( sfile.compare("none") == 0 && sfileMN.compare("none") == 0  ) {
        cout<<"computing euclidean distance matrix"<<endl;
        read_cost_euclidean_distance(mnode, snode);
        return;
    }
    else {
        cout<<"sfile: "<<sfile<<" sfileMN: "<<sfileMN<<endl;
        //assert(false);
    }

    //distance and cost between local exchanges
    cout<<"READ_COST_DIST_LE2LE: "<<fname<<" -- "<<mnode.size()<<endl;
    MetroNode *MN = NULL;
    vector< vector< Node* > > node_metro;
    for(int i=0;i<(int)mnode.size();i++) {
        MN = mnode[i];
        //cout<<"Init metro node: "<<i<<" size: "<<MN->node.size()<<endl;
        int sTmp = MN->node.size();
        MN->distMatrix = new double*[sTmp];
        MN->costMatrix = new double*[sTmp];

        vector<Node *> tmp;
        for(int j=0;j<sTmp; j++) {
            //if( !MN->node[j]->isMetroNode() ) tmp.push_back(MN->node[j]);
            if( MN->node[j] != MN->getMetroNode() ) tmp.push_back(MN->node[j]);
            MN->distMatrix[j] = new double[sTmp];
            MN->costMatrix[j] = new double[sTmp];
            for(int p=0;p<sTmp;p++) {
                if(j==p) MN->distMatrix[j][p] = 0;
                else MN->distMatrix[j][p] = numeric_limits<double>::max();
                
            }
        }
        std::sort(tmp.begin(), tmp.end(), NodeGreater);
        /*for(int i=0;i<(int)tmp.size()-1;i++) {
            if(tmp[i]->getInputId() < tmp[i+1]->getInputId()) {
                for(int j=0;j<(int)tmp.size();j++) cout<<tmp[j]->getInputId()<<" ";
                cout<<endl;
                assert(false);
            }
        }*/
        node_metro.push_back(tmp);
    }
    cout<<"Elapsed read before building distance matrix: "<<t.elapsed()<<endl;
#ifndef NNNN
    //distance between local exchanges
    while(file>>id1>>id2>>cost>>dist) {
        for(int i=0;i<(int)mnode.size();i++) {
            MN = mnode[i];
            Node *n1 = BinarySearch( node_metro[i], id1 );
            Node *n2 = NULL;
             if(n1) n2 = BinarySearch( node_metro[i], id2 );
            //Node *n2 = BinarySearch( node_metro[i], id2 );
            if( n1 && n2 ) {
                //MN->setDistMatrixValue(id1, id2, dist);
                //MN->setCostMatrixValue(id1, id2, dist);
                //MN->setDistAndCostMatrixValue(id1, id2, dist, cost);
                MN->setDistAndCostMatrixValue(n1, n2, dist, cost);
            }
            
            /*
            MN = mnode[i];
            int nodesMN=0;
            //if(BinarySearch(node_metro[i], id1)) nodesMN++;
            //if(BinarySearch(node_metro[i], id2)) nodesMN++;
            //if(nodesMN==2) {
                nodesMN=0;
                for(int j=0;j<(int)node_metro[i].size();j++) {
                    Node *tmp = node_metro[i][j];
                    if(id1 == tmp->getInputId() && tmp!=MN->getMetroNode()) {
                        nodesMN++;
                    }
                    if(id2 == tmp->getInputId() && tmp!=MN->getMetroNode()) {
                        nodesMN++;
                    }
                    if(nodesMN==2) break;

                }
                //if(nodesMN!=2)
                //    cout<<"Error con metro["<<i<<"] node: "<<id1<<" not here"<<endl;
                //assert(nodesMN==2);
            //}
            //if(binary_search(node_metro[i].begin(), node_metro[i].end(), id1, NodeGreater)) nodesMN++;
            //if(binary_search(node_metro[i].begin(), node_metro[i].end(), id2, NodeGreater)) nodesMN++;
            if(nodesMN==2) {
                assert(BinarySearch(node_metro[i], id1) && BinarySearch(node_metro[i], id2));
                //cout<<"Inserting id1: "<<id1<<" id2: "<<id2<<" dist: "<<dist<<endl;
                //MN->setDistAndCostMatrixValue(id1, id2, dist, cost);
                MN->setDistMatrixValue(id1, id2, dist);
                MN->setCostMatrixValue(id1, id2, dist);
            }*/
            /*
            
            int nodesMN=0;
            MN = mnode[i];
            if(! (MN->node.size() == node_metro[i].size()+1) ) {cout<<"MN: "<<MN->node.size()<< " M2: "<<node_metro[i].size()<<endl;}
            assert(MN->node.size() == node_metro[i].size()+1);
            for(int j=0;j<(int)node_metro[i].size();j++) {
                Node *tmp = node_metro[i][j];
                if(id1 == tmp->getInputId() && tmp!=MN->getMetroNode()) nodesMN++;
                if(id2 == tmp->getInputId() && tmp!=MN->getMetroNode()) nodesMN++;
                if(nodesMN==2) {
                    MN->setDistMatrixValue(id1, id2, dist);
                    MN->setCostMatrixValue(id1, id2, dist);
                    break;
                }
            }*/
        }
        if(file.eof()) break;
    }

//#endif
#else
    
    //distance between local exchanges
    while(file>>id1>>id2>>cost>>dist) {
        for(int i=0;i<(int)mnode.size();i++) {
            int nodesMN=0;
            MN = mnode[i];
            for(int j=0;j<(int)MN->node.size();j++) {
                Node *tmp = MN->node[j];
                if(id1 == tmp->getInputId() && tmp!=MN->getMetroNode()) nodesMN++;
                if(id2 == tmp->getInputId() && tmp!=MN->getMetroNode()) nodesMN++;
                if(nodesMN==2) {
                    MN->setDistMatrixValue(id1, id2, dist);
                    MN->setCostMatrixValue(id1, id2, dist);
                    break;
                }
            }
        }
        if(file.eof()) break;
    }
#endif
    file.close();
    cout<<"Elapsed read after building distance matrix: "<<t.elapsed()<<endl;
    cout<<"READ_COST_DIST_MN2ES: "<<fnameMN<<endl;
    //distance between a metro node and a local exchange
    file.open(fnameMN);
    MN=NULL;
    while(file>>id1>>id2>>cost>>dist) {
        if(!MN || MN->getMetroNode()->getInputId() != id1) {
            for(int i=0;i<(int)mnode.size();i++) {
                if(mnode[i]->getMetroNode()->getInputId() == id1) {
                    MN=mnode[i];
                    break;
                }
            }
        }
        if(!MN) cout<<"MN: "<<id1<<" -> "<<id2<<" not found in: "<<fnameMN<<endl;
        assert(MN);
        //cout<<"id1: "<<id1<<" id2: "<<id2<<" MN: "<<MN->getMetroNode()->getMetroNode()<<endl;
        /*MN->setDistValueMN2LE(id2, dist);
        MN->setCostValueMN2LE(id2, cost);*/
        MN->setDistAndCostValueMN2LE(id2, dist, cost);
        if(file.eof()) break;
    }
    file.close();
    
    /*
    while(file>>id1>>id2>>cost>>dist) {
        assert(id1 != id2);
        for(int i=0;i<(int)mnode.size();i++) {
            MN = mnode[i];
            //cout<<"MM["<<MN->getId()<<"]: "<<id1<< ", "<<id2<<endl;
            MN->setDistMatrixValue(id1, id2, dist);
            MN->setCostMatrixValue(id1, id2, cost);
        }

        if( file.eof() ) break;
    }
    file.close();

    //distance and cost between metro nodes to local exchanges
    file.open(fnameMN);
    MN=NULL;
    while(file>>id1>>id2>>cost>>dist) {
        assert(id1 < (int) mnode.size());
        mnode[id1]->setDistValueMN2LE(id2, dist);
        mnode[id1]->setCostValueMN2LE(id2, cost);
        if( file.eof() ) break;
    }
    file.close();
     */
}
