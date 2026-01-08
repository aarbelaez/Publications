#ifndef MN_GRAPH_HH
#define MN_GRAPH_HH

#include "BH.h"
#include "MetroNode.h"
#include "Node.h"
#include "ProbMatching.h"

#include <iostream>
#include <vector>
#include <algorithm>

//using namespace std;


class MNGraph {
public:
    vector< vector < vector< pair<Node*, Node*> > > > g_conflict;
private:
	vector< vector<MetroNode*> > g;
    vector<MetroNode*> metro_node;
    struct mt19937p mt;
    int seed;
    
    //assuming that n1 and n2 are sharing their primary and secundary metro nodes
    void addConflict(Node *n1, Node *n2, vector<SiteNode> sn) {
        assert(n1 != n2);
        if(g_conflict.empty()) return;
        int mn1 = sn[n1->getId()].id_mn[0];
        int mn2 = sn[n1->getId()].id_mn[1];
        if(mn1>mn2) swap(mn1, mn2);
        
        vector< pair<Node*, Node*> > conflicts = g_conflict[mn1][mn2];
        for(unsigned int i=0;i<conflicts.size();i++) {
            Node *t1 = conflicts[i].first;
            Node *t2 = conflicts[i].second;
            /*if( (t1 == n1 || t1 == n2) && (t2 == n1 || t2 == n2) ) {
                return;
            }*/
            if( (t1->getId() == n1->getId() || t1->getId() == n2->getId()) &&
                (t2->getId() == n1->getId() || t2->getId() == n2->getId()) ) {
                return;
            }
        }
        g_conflict[mn1][mn2].push_back( make_pair(n1, n2) );
    }

    int random(int i) {
        //return (static_cast<int>(genrand(&mt)) % i);
        int rval = static_cast<unsigned int>(genrand(&mt)) % i;
        //cout<<"i : "<<rval<<" "<<i<<endl;
        assert(rval<i && rval>=0);
        return rval;
    }
    double randomDouble(void) {
        double y = (double) genrand(&mt);
        return (y/(unsigned long)0xffffffff);
    }
public:
    MNGraph(int _seed) {
        seed =_seed;
        sgenrand(seed, &mt);
    }
    
    void clear(void) {
        for(int i=0;i<(int)g.size();i++) {
            g[i].clear();
        }
    }
    
    void printConflicts(void) {
        if(!(g_conflict.size() == g.size())) cout<<"g_conflict: "<<g_conflict.size()<<" g: "<<g.size()<<endl;
        assert(g_conflict.size() == g.size());
        int size = (int)g.size();
        for(int i=0;i<size;i++) {
            MetroNode *mn1 = metro_node[i];
            //cout<<"Conflicts for MN: "<<mn1->getMetroNode()->getIdString()<<endl;
            for(int j=i+1;j<size;j++) {
                MetroNode * mn2 = metro_node[j];
                cout<<"Conflicts for MNs "<<mn1->getMetroNode()->getIdString()<<" -> "<<mn2->getMetroNode()->getIdString()<<" size: "<<g_conflict[i][j].size()<<endl;
                for(int c=0;c<(int)g_conflict[i][j].size();c++) {
                    Node *n1 = g_conflict[i][j][c].first;
                    Node *n2 = g_conflict[i][j][c].second;
                    cout<<"("<<n1->getIdString()<<", "<<n2->getIdString()<<") ";
                }
                cout<<endl<<"=========="<<endl;
            }
        }
    }

    void computeNodeConflict(vector<MetroNode *> metro) {
        int size = (int)metro.size();
        for(int i=0;i<size;i++) {
            vector< vector< pair<Node*, Node*> > > tmp;
            for(int j=0;j<size;j++) tmp.push_back( vector< pair<Node*, Node*> >() );
            g_conflict.push_back( tmp );
        }
    }

    void initEmpty(vector<MetroNode*> metro, vector<SiteNode> sn) {
        int size = (int)metro.size();
		for(int i=0;i<size; i++) {
            g.push_back( vector<MetroNode*>() );
            metro_node.push_back(metro[i]);
        }
    }

	void init2(vector<MetroNode*> metro, vector<SiteNode> sn) {
        initEmpty(metro, sn);
		for(int i=0;i<(int)sn.size();i++) {
			//metro node
			if(sn[i].id_mn[0] ==-1 && sn[i].id_mn[1] == -1) continue;
			
			if(!(sn[i].id_mn[0] < (int)metro.size() && sn[i].id_mn[1] <(int)metro.size())) {
				cout<<"MN1: "<<sn[i].id_mn[0]<<" MN2: "<<sn[i].id_mn[1]<<" MNS: "<<metro.size()<<endl;
			}
			assert(sn[i].id_mn[0] < (int)metro.size() && sn[i].id_mn[1] <(int)metro.size());
			/*if(!(metro[sn[i].id_mn[0]]->getId() == sn[i].id_mn[0] && metro[sn[i].id_mn[1]]->getId() == sn[i].id_mn[1])) {
				cout<<metro[sn[i].id_mn[0]]->getId()<<" : "<<sn[i].id_mn[0]<<" --- "<<metro[sn[i].id_mn[1]]->getId()<<" : "<<sn[i].id_mn[1]<<endl;
			}*/
			assert(metro[sn[i].id_mn[0]]->getId() == sn[i].id_mn[0] && metro[sn[i].id_mn[1]]->getId() == sn[i].id_mn[1]);
			MetroNode *mn1 = metro[sn[i].id_mn[0]];
			MetroNode *mn2 = metro[sn[i].id_mn[1]];
			assert(mn1 && mn2);
			//cout<<"edige: "<<mn1->getId()<<" -> "<<mn2->getId()<<endl;
			addEdge(mn1, mn2);
		}
	}
    
    void init(vector<MetroNode*> metro, vector<SiteNode> sn) {
        initEmpty(metro, sn);
        if(metro.size() < 2 ) return;
        cout<<"Computing connectivity graph ------------"<<endl;
        for(int i=0;i<(int)metro.size();i++) {
            MetroNode *mn = metro[i];
            for(int j=0;j<(int)mn->node.size();j++) {
                Node *n1 = mn->node[j];
                if( !n1->isMetroNode() ) {
                    int mn11 = sn[n1->getId()].id_mn[0];
                    int mn12 = sn[n1->getId()].id_mn[1];
                    assert(mn11!=mn12 && (mn11 == mn->getId() || mn12 == mn->getId()) );

                    for(int p=j+1;p<(int)mn->node.size();p++) {
                        Node *n2 = mn->node[p];
                        int mn21, mn22;
                        if( !n2->isMetroNode() ) {
                            mn21 = sn[n2->getId()].id_mn[0];
                            mn22 = sn[n2->getId()].id_mn[1];
                            assert(mn21!=mn22 && (mn21 == mn->getId() || mn22 == mn->getId()) );
                            if( (mn11 == mn21 || mn11 == mn22) && (mn12 == mn21 || mn12 == mn22 ) ) {
                                assert(metro[mn11]->getId() == mn11 && metro[mn12]->getId() == mn12);
                                addEdge(metro[mn11], metro[mn12]);
                                addConflict(n1, n2, sn);
                            }
                        }
                    }
                }
            }
        }
        //print();
    }

    void printGraph(void) {
        cout<<"Graph G {"<<endl<<"size=\"4,4\";"<<endl;
        for(unsigned int i=0;i<g_conflict.size();i++) {
            cout<<metro_node[i]->getMetroNode()->getIdString()<<" [label=\""<<metro_node[i]->getMetroNode()->getIdString()<<": "<<metro_node[i]->node.size()<<"\"];"<<endl;
        }
        for(unsigned int i=0;i<g_conflict.size();i++) {
            for(unsigned int j=i+1;j<g_conflict[i].size();j++) {
                if(g_conflict[i][j].size()>0) {
                    cout<<metro_node[i]->getMetroNode()->getIdString()<<" -- "<<metro_node[j]->getMetroNode()->getIdString()<<"[label=\""<<g_conflict[i][j].size()<<"\"];"<<endl;
                }
            }
        }
    }

	void print(void) {
        vector<int> iset;// = independentSet();
        //vector<int> iset = indepdentSet(8);
		vector< pair<int,int> > edges;
		cout<<"graph g {"<<endl<<"size=\"4,4\";"<<endl;
        for(int i=0;i<(int)g.size();i++) {
            string s;
            for(int j=0;j<(int)iset.size();j++) {
                if(iset[j] == i) s+="[style=filled]";
            }
            //cout<<"Node_"<<i<<s<<";"<<endl;
            cout<<metro_node[i]->getMetroNode()->getIdString()<<s<<";"<<endl;
        }
		for(int i=0;i<(int)g.size();i++) {
			//cout<<"i["<<i<<"]: ";
			for(int j=0;j<(int)g[i].size(); j++) {
				bool valid=true;
				for(int p=0;p<(int)edges.size();p++) {
					if( (edges[p].first == i && edges[p].second == g[i][j]->getId()) || 
						(edges[p].second == i && edges[p].first == g[i][j]->getId()) ) {
						valid=false;
						break;
					}
				}
				if(valid) {
					edges.push_back( make_pair(i, g[i][j]->getId()) );
					/*cout<<"Node_"<<i<<" -- Node_"
						<<g[i][j]->getId()<<endl;*/
                    Node *N1 = metro_node[i]->getMetroNode();
                    Node *N2 = g[i][j]->getMetroNode();
                    cout<<N1->getIdString()<<" -- "<<N2->getIdString()<<endl;
				}
			}
			//cout<<endl;
		}
		cout<<"}"<<endl;
	}
	vector< MetroNode* > getNeighbours(MetroNode *mn) {
		return g[mn->getId()];
	}

	void addEdge(MetroNode *a, MetroNode *b) {
		addNeighbour(a,b);
		addNeighbour(b,a);
	}

	void addNeighbour(MetroNode *a, MetroNode *b) {
		assert(a && b && a->getId() < (int)g.size() && b->getId() < (int)g.size());
		vector< MetroNode* > g_n = getNeighbours(a);
		for(int i=0;i<(int)g_n.size(); i++) {
			if(g_n[i] == b) return;
		}
		//cout<<"ADDING"<<endl;
		//g_n.push_back(b);
		g[a->getId()].push_back(b);
	}
    
    vector<int> indepdentSet(int size, ProbMatching *pm) {
        vector< vector<MetroNode*> > G;
        vector<int> iSet;
        vector<int> validSet;
        vector<int> delSet;
        for(int i=0;i<(int)g.size();i++) {
            vector<MetroNode *> tmp;
            validSet.push_back(i);
            for(int j=0;j<(int)g[i].size();j++) {
                tmp.push_back(g[i][j]);
            }
            G.push_back(tmp);
        }
        while(size > (int)iSet.size() && delSet.size() < G.size() && !validSet.empty() ) {
            //int vertex = validSet[ random(validSet.size()) ];
            int vertex;
            if(pm)
                vertex = pm->selectOption( validSet, randomDouble() );
            else vertex = validSet[ random(validSet.size()) ];
            
            iSet.push_back(vertex);

            while(!G[vertex].empty()) {
                MetroNode *mn = G[vertex][0];
                for(int i=0;i<(int)G[mn->getId()].size();i++) {
                    MetroNode *mn2 = G[mn->getId()][i];
                    for(int j=0;j<(int)G[mn2->getId()].size(); j++) {
                        if(G[mn2->getId()][j] == mn) {
                            G[mn2->getId()].erase( G[mn2->getId()].begin() + j );
                            break;
                        }
                    }
                }
                //G.erase( G.begin() + mn->getId() );
                G[mn->getId()].clear();
                delSet.push_back(mn->getId());
                validSet.erase( find(validSet.begin(), validSet.end(), mn->getId()) );
            }
            G[vertex].clear();
            delSet.push_back(vertex);
            validSet.erase( find(validSet.begin(), validSet.end(), vertex) );
            //G.erase( G.begin() + vertex );
        }
        return iSet;
    }
    
    vector<int> independentSet(void) {
        vector< vector<MetroNode*> > G;
        for(int i=0;i<(int)g.size(); i++) {
            vector<MetroNode *> tmp;
            for(int j=0;j<(int)g[i].size();j++) {
                tmp.push_back(g[i][j]);
            }
            G.push_back(tmp);
        }
        //independent set
        vector<int> iset;
        vector<int> delset;
        while(true) {
            int mDegree = G.size() + 1;
            int vertex = -1;
            for(int i=0;i<(int)G.size(); i++) {
                if( mDegree > (int)G[i].size() && find(delset.begin(), delset.end(), i)==delset.end() )  {
                    vertex = i;
                    mDegree = (int) G[i].size();
                }
            }
            if(vertex == -1) break;
            cout<<"Vertex: "<<vertex<<" size: "<<mDegree<<endl;
            iset.push_back(vertex);
            while(!G[vertex].empty()) {
                MetroNode *mn = G[vertex][0];
                for(int i=0;i<(int)G[mn->getId()].size();i++) {
                    MetroNode *mn2 = G[mn->getId()][i];
                    for(int j=0;j<(int)G[mn2->getId()].size(); j++) {
                        if(G[mn2->getId()][j] == mn) {
                            G[mn2->getId()].erase( G[mn2->getId()].begin() + j );
                            break;
                        }
                    }
                }
                //G.erase( G.begin() + mn->getId() );
                G[mn->getId()].clear();
                delset.push_back(mn->getId());
            }
            G[vertex].clear();
            delset.push_back(vertex);
            //G.erase( G.begin() + vertex );
        }
        cout<<"Independent set"<<endl;
        for(int i=0;i<(int)iset.size();i++) {
            cout<<iset[i]<<" ";
        }
        cout<<endl<<"==============="<<endl;
        return iset;
    }
};


#endif