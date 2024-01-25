/*
 *  Adaptive search - GPU Version
 *
 *
 * Please visit the https://pauillac.inria.fr/~diaz/adaptive/manual/index.html for a complete version of the original Adaptive Search code
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>


__global__ void kernel(int k) {
  int t=0;
  int b=0;
  while(t<k) {
    b++;
    t++;
  }
  printf("b: %d\n",b);
}

#ifndef THREADS
#define THREADS 2
#endif

int main() {
  kernel<<<1, THREADS>>>(1000);
  return 0;
}
