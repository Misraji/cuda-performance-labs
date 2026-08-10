#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    const cudaError_t error = (call);                                     \
    if (error != cudaSuccess) {                                           \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ":" \
                << cudaGetErrorString(error) << std::endl;                \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

__global__ void Conventional(const float* input, float* output, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = input[i] * 2.0f + 1.0f;
  }
}

__global__ void PersistentWorker(const float* input, float* output, int n,
                                 int* work_counter) {
  __shared__ int base;
  while (true) {
    if (threadIdx.x == 0) {
      base = atomicAdd(work_counter, blockDim.x);
    }
    __syncthreads();

    // base is shared, so every thread takes the same branch here. The block
    // therefore exits uniformly and does not strand threads at a later barrier.
    if (base >= n) {
      return;
    }

    const int index = base + threadIdx.x;
    if (index < n) {
      output[index] = input[index] * 2.0f + 1.0f;
    }
  }
}

int main() {
  const int kElements = 1 << 24;
  constexpr int kThreads = 256;
  const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);

  std::vector<float> host(kElements, 3.0f);
  float* input = nullptr;
  float* output = nullptr;
  int* counter = nullptr;

  CUDA_CHECK(cudaMalloc(&input, bytes));
  CUDA_CHECK(cudaMalloc(&output, bytes));
  CUDA_CHECK(cudaMalloc(&counter, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(input, host.data(), bytes, cudaMemcpyHostToDevice));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  const int conventional_blocks = (kElements + kThreads - 1) / kThreads;
  const int persistent_blocks = properties.multiProcessorCount * 2;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  Conventional<<<conventional_blocks, kThreads>>>(input, output, kElements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float conventional_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&conventional_ms, start, stop));

  CUDA_CHECK(cudaMemset(counter, 0, sizeof(int)));
  CUDA_CHECK(cudaEventRecord(start));
  PersistentWorker<<<persistent_blocks, kThreads>>>(input, output, kElements,
                                                    counter);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float persistent_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&persistent_ms, start, stop));

  std::cout << "SM count: " << properties.multiProcessorCount << std::endl;
  std::cout << "conventional blocks: " << conventional_blocks << std::endl;
  std::cout << "persistent blocks: " << persistent_blocks << std::endl;
  std::cout << "conventional: " << conventional_ms << " ms" << std::endl;
  std::cout << "persistent: " << persistent_ms << " ms" << std::endl;

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  CUDA_CHECK(cudaFree(counter));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  return EXIT_SUCCESS;
}