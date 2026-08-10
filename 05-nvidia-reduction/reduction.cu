#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                        \
      std::cerr << "CUDA error at " << __FILE__ ":" << __LINE__ << ":" \
                << cudaGetErrorString(error) << std::endl;             \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

__global__ void ReduceShared(const float* input, float* partial, int n) {
  __shared__ float shared[kBlockSize];

  const int tid = threadIdx.x;
  const int i = blockIdx.x * (kBlockSize * 2) + tid;

  float value = 0.0f;
  if (i < n) {
    value += input[i];
  }
  if (i + kBlockSize < n) {
    value += input[i + kBlockSize];
  }

  shared[tid] = value;
  __syncthreads();

  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = shared[0];
  }
}

__device__ float WarpSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void ReduceShuffle(const float* input, float* partial, int n) {
  __shared__ float warp_sums[32];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * (kBlockSize * 2) + tid;

  float value = 0.0f;
  if (i < n) {
    value += input[i];
  }

  if (i + kBlockSize < n) {
    value += input[i + kBlockSize];
  }

  value = WarpSum(value);
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;

  if (lane == 0) {
    warp_sums[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    value = lane < (kBlockSize / kWarpSize) ? warp_sums[lane] : 0.0f;
    value = WarpSum(value);
    if (lane == 0) {
      partial[blockIdx.x] = value;
    }
  }
}

using Kernel = void (*)(const float*, float*, int);

float RunAndTime(Kernel kernel, const float* input, float* partial, int blocks,
                 int n, int iterations) {
  for (int i = 0; i < 5; ++i) {
    kernel<<<blocks, kBlockSize>>>(input, partial, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    kernel<<<blocks, kBlockSize>>>(input, partial, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  return elapsed_ms / iterations;
}

int main() {
  constexpr int kElements = 1 << 24;
  constexpr int kIterations = 50;

  const int blocks = (kBlockSize + kBlockSize * 2 - 1) / (kBlockSize * 2);

  std::vector<float> host_input(kElements, 1.0f);
  std::vector<float> host_partial(blocks);

  float* input = nullptr;
  float* partial = nullptr;
  CUDA_CHECK(cudaMalloc(&input, kElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&partial, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(input, host_input.data(), kElements * sizeof(float),
                        cudaMemcpyHostToDevice));

  const float shared_ms =
      RunAndTime(ReduceShared, input, partial, blocks, kElements, kIterations);
  CUDA_CHECK(cudaMemcpy(host_partial.data(), partial, blocks * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float shared_result =
      std::accumulate(host_partial.begin(), host_partial.end(), 0.0f);

  const float shuffle_ms =
      RunAndTime(ReduceShuffle, input, partial, blocks, kElements, kIterations);
  CUDA_CHECK(cudaMemcpy(host_partial.data(), partial, blocks * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float shuffle_result =
      std::accumulate(host_partial.begin(), host_partial.end(), 0.0f);

  std::cout << "shared result: " << shared_result << " kernel: " << shared_ms
            << " ms" << std::endl;
  std::cout << "shuffle result: " << shuffle_result << " kernel: " << shuffle_ms
            << " ms" << std::endl;

  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(partial));

  return EXIT_SUCCESS;
}