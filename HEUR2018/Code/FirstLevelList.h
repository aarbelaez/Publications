#ifndef FIRST_LEVE_LIST__H
#define FIRST_LEVE_LIST__H

#include "Node.h"
#include "iostream"
#include "vector"

using namespace std;

class FirstlevelList;

class FirstLevelNode {
    friend class FirstLevelList;
private:
    Node *n;
    int size;
public:
    FirstLevelNode *next;
    FirstLevelNode *prev;
    FirstLevelNode(Node *_n);
    int getId(void);
    void setNode(Node *_n);
    Node *getNode(void);
    void incSize(void) {
        size++;
    }
    void decSize(void) {
        assert(size>0);
        size--;
    }
    bool isEmpty(void) {
        assert(size>=0);
        return size==0;
    }
    int getSize(void) {
        return size;
    }
};

class FirstLevelList {
public:
    FirstLevelNode *start;
    FirstLevelList(void);
    FirstLevelNode *newFirstLevelNode(Node *_n);
    
    void addNode(FirstLevelNode *fn);
    void delNode(FirstLevelNode *&fn);
    
    void print(void);
    void copy(FirstLevelList &f) {
        start = f.start;
    }
};

#endif