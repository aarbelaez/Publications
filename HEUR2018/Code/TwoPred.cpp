

#include "TwoPred.h"


TwoPred::TwoPred(void) {
    id_node = -1;
    pred[0] = -1;
    pred[1] = -1;
}

TwoPred::~TwoPred(void) { }

bool TwoPred::checkDisjointNess(void) {
    if(method==2) return false;
    return true;
}

void TwoPred::addPred(Node *n, bool isPrimary) {
    if(!checkDisjointNess()) return;
    /*
     assert(pred[0]!=id1 && pred[1]!=id1);
     if(pred[0] == -1) pred[0] = id1;
     else if (pred[1] == -1) pred[1] = id1;
     else assert(false);
     */
    int id1 = n->getId();
    int index = isPrimary?0:1;
    if(!(pred[index]==-1 && pred[1-index]!=id1)) {
        cout<<"add pred["<<index<<"]: "<<pred[index]<<" pred["<<(1-index)<<"]: "<<pred[1-index]<<" ID: "<<id1<<"("<<n->getIdString()<<") isPrimary: "<<isPrimary<<" es: "<<id_node<<" MN ID: "<<n->getMetroNode()->getIdString()<<endl;
    }
    assert(pred[index]==-1 && pred[1-index]!=id1);
    pred[index]=id1;
}

void TwoPred::delPred(Node *n, bool isPrimary) {
    if(!checkDisjointNess()) return;
    /*
     assert(pred[0]==id1 || pred[1]==id1);
     if(pred[0] == id1) pred[0]=-1;
     else if(pred[1] == id1) pred[1]=-1;
     */
    int id1 = n->getId();
    int index = isPrimary?0:1;
    if(!(pred[index]==id1 && pred[1-index]!=id1)) {
        
        cout<<"ID: "<<id_node<<"del pred["<<index<<"]: "<<pred[index]<<" pred["<<(1-index)<<"]: "<<pred[1-index]<<" ID: "<<id1<<" isPrimary: "<<isPrimary<<" es: "<<id_node<<" MN ID: "<<n->getMetroNode()->getIdString()<<endl;
    }
    
    assert(pred[index]==id1 && pred[1-index]!=id1);
    pred[index]=-1;
}

bool TwoPred::validPred(Node *n, bool isPrimary) {
    if(!checkDisjointNess()) return true;
    int id1 = n->getId();
    int index = isPrimary?0:1;
    if(!(pred[index]!=id1)) {cout<<"HHHHHPPPP: "<<pred[index]<<" -- "<<id1<<endl; };
    assert(pred[index]!=id1);
    if(pred[1-index] == id1) return false;
    return true;
    /*if(pred[0] == id1 || pred[1] == id1) return false;
     return true;*/
    /*if(pred1 == -1) return true;
     if(pred2 == -1) return true;
     return false;*/
}

void TwoPred::copy(TwoPred *t) {
    id_node = t->id_node;
    pred[0] = t->pred[0];
    pred[1] = t->pred[1];
}

void TwoPred::copy(TwoPred *t, bool isPrimary) {
    id_node = t->id_node;
    int index=isPrimary?0:1;
    int valbefore = pred[index];
    pred[index]=t->pred[index];
    if(!(pred[1-index]!=pred[index] || pred[index] ==-1)) {
        cout<<"ID: "<<id_node<<"pred["<<(index)<<"]: "<<pred[index]<<" pred["<<(1-index)<<"]: "<<pred[1-index]<<" valbef: "<<valbefore<<" thread: "<<omp_get_thread_num()<<endl;
    }
    assert(pred[1-index]!=pred[index] || pred[index] ==-1);
}

void TwoPred::reset(void) {
    id_node = -1;
    pred[0] = -1;
    pred[1] = -1;
}

string TwoPred::toString(void) {
    string s;
    stringstream sd, sd1, sd2;
    sd<<id_node;
    sd1<<pred[0];
    sd2<<pred[1];
    //s+=(sd.str()+" pred[0]: "+sd1.str()+" pred[1]: "+sd2.str());
    s+=(sd1.str()+" -> "+sd.str()+" .... "+sd2.str()+" -> "+sd.str());
    return s;
}

