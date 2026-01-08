#ifndef READ_FILES__H
#define READ_FILES__H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <assert.h>

#include "BH.h"
#include "Node.h"
#include "MetroNode.h"


using namespace std;


Node *BinarySearch(vector<Node*>, int);

void read_sites_nodes_format2(const char *fname, vector<Pair> &snode);

void read_metro_nodes_format2(const char *fname, vector<Pair> &mnode);

void Set_Init_ExchangeSites_Format2(const char *fname, vector<Pair> mnode1, vector<Pair> snode1, vector<MetroNode*> &mnodes, vector<SiteNode> &snode);

void read_site_nodes(const char *fname, int nbMetroNodes, vector<SiteNode> &node);

void read_metro_nodes(const char *fname, vector<MetroNode*> &mnode, vector<SiteNode> &snode);

void Set_Init_ExchangeSites(const char *fname, vector<SiteNode> &snode, vector<MetroNode*> &mnode);

void read_cost_dist(const char *fname, const char *fnameMN, vector<MetroNode*> & mnode, vector<SiteNode> &snode);

void read_cost_euclidean_distance(vector<MetroNode*> &mnode);

void Read_Customers(const char *fname, vector<MetroNode*> &mnode, vector<SiteNode> &snode);
#endif