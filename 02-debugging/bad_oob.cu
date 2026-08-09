#include <cuda_runtime.h>

#include <cstdlib>

__global__ void BadKernel(float* data, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Intentionally missing the if(i < n) check.
  data[i] = static_cast<float>(i);
}

int main() {
  constexpr int kElements = 1000;

  float* data = nullptr;
  cudaMalloc(&data, kElements * sizeof(float));

  BadKernel<<<4, 256>>>(data, kElements);
  cudaDeviceSynchronize();

  cudaFree(data);
  return EXIT_SUCCESS;
}