/*
 *  Adaptive search - GPU Version
 *
 *
 * Please visit the https://pauillac.inria.fr/~diaz/adaptive/manual/index.html for a complete version of the original Adaptive Search code
 */

#include <cuda.h>
#include <stdio.h>

int main(int argc, char** argv) {

      size_t limit = 0;

      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
      printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
      printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

      limit = 9999;
      
      cudaDeviceSetLimit(cudaLimitStackSize, limit);
      cudaDeviceSetLimit(cudaLimitPrintfFifoSize, limit);
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);

      limit = 0;

      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("New cudaLimitStackSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
      printf("New cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
      printf("New cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

      return 0;
}
