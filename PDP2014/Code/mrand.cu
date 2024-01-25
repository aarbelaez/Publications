/*
 *  Adaptive search - GPU Version
 *
 *
 * Please visit the https://pauillac.inria.fr/~diaz/adaptive/manual/index.html for a complete version of the original Adaptive Search code
 */

#ifndef MRAND_CC
#define MRAND_CC

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include<curand_kernel.h>


__shared__ curandState cudaRand;

__device__ double Random_Double(void) {
    return ((double) curand_uniform(&cudaRand));
}

__device__ int Random1(int n) {
	int r;
	//do {
		r=(int) (Random_Double() * (n));
		if(r<n) return r;			
		printf("KKK r: %d n: %d\n", r, n);
		return r-1;
	//}while(r==n);
	//return r;
}
//curand_uniform returns number between (0..1]..
__device__ int Random(int n) {
    int r = (int) (Random_Double() * (n));
    return ((r==n) ? (r-1) : r);
    //return ( (r==n)? Random(n): n);
    //return (int) (Random_Double() * (n));
}

__device__ void Random_Permut(int *vec, int total, int base_value) {
    int i,j;

    vec[0] = base_value;
    for(i = 1; i < total; i++) {
        j = Random(i + 1);
        vec[i] = vec[j];
        vec[j] = base_value + i;
    }
}


#ifdef RMAIN
__global__ void kernel(int seed) {
    int max=10000000;
    curand_init(seed, 0, 0, &cudaRand);
    printf("seed: %d\n",seed);
    printf("Max: %d\n",max);
    int c=0; int r=0;
    for(int i=0;i<1000;i++) {
	int n=Random(max);
	if(n==0) c++;
	if(n>=0 && n<max) r++;
	if(n<0 || n>=max) 
	  printf("%d\n ",n);
    }
    printf("c: %d --- r: %d\n",c,r);
}

int main() {
    kernel<<<1,1>>>(time(NULL));
    cudaDeviceSynchronize();
}
#endif

#endif
