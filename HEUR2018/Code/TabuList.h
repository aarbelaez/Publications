#ifndef TABU_LIST_H
#define TABU_LIST_H

#include "BH.h"

class TabuList {
public:
	int tabuTenure;
	vector<Node*> tabu_list;
	TabuList(void) { 
		tabuTenure = 30;
		for(int i=0;i<tabuTenure;i++) tabu_list.push_back(NULL);
	}
	
	bool isTabu(Node *n) {
		for(vector<Node*>::iterator it=tabu_list.begin(); it!=tabu_list.end(); it++) {
			Node *tmp = *it;
			if(tmp == n) return true;
		}
		return false;
	}
	void addElement(Node *n) {
		//tabu_list.insert(tabu_list.begin(), n);
		tabu_list.push_back(n);
		tabu_list.erase(tabu_list.begin());
	}
};

#endif