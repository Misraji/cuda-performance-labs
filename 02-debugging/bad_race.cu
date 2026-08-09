#include <cuda_runtime.h>

#include <cstdlib>

__global__ void BadRace(int* output) {
  __shared__ int value;
  value = threadIdx.x;
  if (threadIdx.x == 0) {
    *output = value;
  }
}

int main() {
  int* output = nullptr;
  cudaMalloc(&output, sizeof(int));

  BadRace<<<1, 256>>>(output);
  cudaDeviceSynchronize();

  cudaFree(output);
  return EXIT_SUCCESS;
}