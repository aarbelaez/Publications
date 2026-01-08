
#include "FirstLevelList.h"

FirstLevelNode::FirstLevelNode(Node *_n) {
    size=0;
    n=_n;
    next=NULL;
    prev=NULL;
}

int FirstLevelNode::getId(void) {
    return n->getId();
}
void FirstLevelNode::setNode(Node *_n) {
    n=_n;
}
Node *FirstLevelNode::getNode(void) {
    return n;
}


FirstLevelList::FirstLevelList(void) {
    start=NULL;
}
FirstLevelNode *FirstLevelList::newFirstLevelNode(Node *_n) {
    FirstLevelNode *n = new FirstLevelNode(_n);
    addNode(n);
    return n;
}

void FirstLevelList::addNode(FirstLevelNode *fn) {
    assert(fn);
    if(!start) {
        start=fn;
        return;
    }
    FirstLevelNode *t1 = start;
    fn->next=start;
    t1->prev=fn;
    start=fn;
}
void FirstLevelList::delNode(FirstLevelNode *&fn) {
    assert(fn);
    if(fn==start) {
        start = fn->next;
        if(start)
            start->prev = NULL;
        delete fn;
        fn=NULL;
        return;
    }
    fn->prev->next = fn->next;
    if(fn->next)
        fn->next->prev = fn->prev;
    delete fn;
    fn=NULL;
}

void FirstLevelList::print(void) {
    cout<<"PRINTING....";
    for(FirstLevelNode *tmp=start; tmp;tmp=tmp->next)
        cout<<tmp->getId()<<" ("<<tmp->prev<<", "<<tmp->next<<")"<<endl;
    cout<<endl;
}


#ifdef MAIN_TEST
int main() {
    Node *n1 = new Node();
    Node *n2 = new Node();
    Node *n3 = new Node();
    n1->setId(1);
    n2->setId(2);
    n3->setId(3);
    
    
    FirstLevelList fl;
    FirstLevelNode *fn1 = fl.newFirstLevelNode(n1);
    //fl.delNode(fn1);
    FirstLevelNode *fn2 = fl.newFirstLevelNode(n2);
    FirstLevelNode *fn3 = fl.newFirstLevelNode(n3);
    fl.delNode(fn3);
    fn3 = fl.newFirstLevelNode(n3);
    
    fl.print();
    fl.delNode(fn2);
    fl.print();
    fl.delNode(fn1);
    fl.print();
    cout<<"FFFF:"<<endl;
    fl.delNode(fn3);
    fl.print();
    cout<<"fn2: "<<fn2<<endl;
    fn2=fl.newFirstLevelNode(n2);
    //fl.addNode(fn2);
    fl.print();
    
    return 1;
}
#endif

