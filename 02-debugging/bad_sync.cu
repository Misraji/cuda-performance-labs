#include <cuda_runtime.h>

#include <cstdlib>

__global__ void BadSync(float* data) {
  const int tid = threadIdx.x;
  if (tid & 1) {
    // Only half the block reaches this barrier
    __syncthreads();
  }
  data[tid] = static_cast<float>(tid);
}

int main() {
  float* data = nullptr;
  cudaMalloc(&data, 256 * sizeof(float));

  BadSync<<<1, 256>>>(data);
  cudaDeviceSynchronize();

  cudaFree(data);
  return EXIT_SUCCESS;
}