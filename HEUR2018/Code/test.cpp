#include <iostream>

using namespace std;

class foo {
    int a;
};

void swap(int *& n1, int *& n2) {
    int *n3 = n1;
    n1 = n2;
    n2 = n3;
}

foo f(int i) {
    foo var;
    var.a = i;
    return var;
}


int main() {
    int *n1 = new int(1);
    int *n2 = new int(2);
    swap(n1, n2);
    cout<<(*n1)<<" "<<(*n2)<<endl;
    
    foo t = 
    return 1;
}