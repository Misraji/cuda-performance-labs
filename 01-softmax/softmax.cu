#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(error) << std::endl;                 \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

constexpr int BLOCK_SIZE = 256;

// V0: Essentially uses no parallelization. Essentially, we are
// - Using 1 block per row.
// - ONLY 1 GPU thread (thread 0) per Block.
// to calculate the values for every row.
// Essentially CPU code executed in the GPU.
__global__ void softmax_v0(const float* input, float* output, int rows,
                           int cols) {
  int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  const float* x = input + rows * cols;
  float* y = output + rows * cols;

  float max_value = -CUDART_INF_F;

  for (int col = 0; col < cols; ++col) {
    max_value = fmaxf(max_value, x[col]);
  }

  float sum = 0.0f;
  for (int col = 0; col < cols; ++col) {
    sum += expf(x[col] - max_value);
  }

  for (int col = 0; col < cols; ++col) {
    y[col] = expf(x[col] - max_value) / sum;
  }
}

// V1: Essentially, we are still using 1 block per Row. However, this
// version now uses multiple threads, per block, to calculate the
// softmax value for the column.
__global__ void softmax_v1(const float* input, float* output, int rows,
                           int cols) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* x = input + row * cols;
  float* y = output + row * cols;

  extern __shared__ float shared[];
  float local_max = -CUDART_INF_F;
  __syncthreads();

  // All threads in the block, iterate and find the local max
  // for their "series".
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, x[col]);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();

  // Here we always have initial "stride" number of threads,
  // accumulating from next stride threads.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_value = shared[0];

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_sum += expf(x[col] - max_value);
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float sum = shared[0];

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    y[col] = expf(x[col] - max_value) / sum;
  }
}

__device__ float warp_reduce_max(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// v2: 1 Block per row. Every block has multiple threads.
// This version uses Warp shuffle primitives to speed up processing.
__global__ void softmax_v2(const float* input, float* output, int rows,
                           int cols) {
  int row = blockIdx.x;
  if (row > rows) {
    return;
  }

  const float* x = input + row * cols;
  float* y = output + row * cols;

  constexpr int kMaxWarps = 32;
  __shared__ float warp_values[kMaxWarps];
  int lane = threadIdx.x & (warpSize - 1);
  int warp_id = threadIdx.x >> 5;
  int num_warps = blockDim.x / warpSize;

  // Warp Level computation of max.
  float local_max = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, x[col]);
  }
  float warp_max = warp_reduce_max(local_max);
  if (lane == 0) {
    warp_values[warp_id] = warp_max;
  }
  __syncthreads();
  float block_max = -CUDART_INF_F;
  if (warp_id == 0) {
    block_max = lane < num_warps ? warp_values[lane] : -CUDART_INF_F;
    block_max = warp_reduce_max(block_max);
    if (lane == 0) {
      warp_values[0] = block_max;
    }
  }
  __syncthreads();
  block_max = warp_values[0];

  // Similar warp level computation of sum.
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_sum += expf(x[col] - block_max);
  }
  float warp_sum = warp_reduce_sum(local_sum);
  if (lane == 0) {
    warp_values[warp_id] = warp_sum;
  }
  __syncthreads();
  float block_sum = 0.0f;
  if (warp_id == 0) {
    block_sum = lane < num_warps ? warp_values[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      warp_values[0] = block_sum;
    }
  }
  __syncthreads();

  block_sum = warp_values[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    y[col] = expf(x[col] - block_max) / block_sum;
  }
}

__global__ void softmax_v3(const float* input, float* output, int rows,
                           int cols) {
  int row = blockIdx.x;
  if (row >= rows || (cols % 4) != 0) {
    return;
  }

  const float4* x = reinterpret_cast<const float4*>(input + row * cols);
  float4* y = reinterpret_cast<float4*>(output + row * cols);

  int cols4 = cols / 4;
  __shared__ float
      warp_values[32];  // Maximum number of warps in a block is 32.

  int lane = threadIdx.x & (warpSize - 1);
  int warp_id = threadIdx.x >> 5;
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;

  float local_max = -CUDART_INF_F;
  for (int col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
    float4 v = x[col4];
    local_max = fmaxf(local_max, v.x);
    local_max = fmaxf(local_max, v.y);
    local_max = fmaxf(local_max, v.z);
    local_max = fmaxf(local_max, v.w);
  }

  float warp_max = warp_reduce_max(local_max);
  if (lane == 0) {
    warp_values[warp_id] = warp_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_max = lane < num_warps ? warp_values[lane] : -CUDART_INF_F;
    block_max = warp_reduce_max(block_max);
    if (lane == 0) {
      warp_values[0] = block_max;
    }
  }
  __syncthreads();
  float max_value = warp_values[0];

  float local_sum = 0.0f;
  for (int col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
    float4 v = x[col4];
    local_sum += expf(v.x - max_value) + expf(v.y - max_value) +
                 expf(v.z - max_value) + expf(v.w - max_value);
  }
  float warp_sum = warp_reduce_sum(local_sum);
  if (lane == 0) {
    warp_values[warp_id] = warp_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < num_warps ? warp_values[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      warp_values[0] = block_sum;
    }
  }
  __syncthreads();
  float sum = warp_values[0];

  for (int col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
    float4 v = x[col4];
    float4 result;
    result.x = expf(v.x - max_value) / sum;
    result.y = expf(v.y - max_value) / sum;
    result.z = expf(v.z - max_value) / sum;
    result.w = expf(v.w - max_value) / sum;
    y[col4] = result;
  }
}

// V4: This version uses cooperative groups and tiles.
__global__ void softmax_v4(const float* input, float* output, int rows,
                           int cols) {
  int row = blockIdx.x;
  if (row > rows) {
    return;
  }

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  // Allocated shared memory to hold intermediate warp results.
  __shared__ float warp_values[32];
  const int thread_id = block.thread_rank();
  const int lane = tile.thread_rank();
  const int warp_id = block.thread_rank() >> 5;
  const int num_warps = (block.num_threads() + warpSize - 1) / warpSize;

  const float* x = input + row * cols;
  float* y = output + row * cols;

  float local_max = -CUDART_INF_F;
  for (int col = block.thread_rank(); col < cols; col += block.num_threads()) {
    local_max = fmaxf(local_max, x[col]);
  }
  float warp_max = cg::reduce(tile, local_max, cg::greater<float>());
  if (lane == 0) {
    warp_values[warp_id] = warp_max;
  }
  block.sync();
  if (warp_id == 0) {
    float value = lane < num_warps ? warp_values[lane] : 0.0f;
    float block_sum = cg::reduce(tile, value, cg::plus<float>());

    if (lane == 0) {
      warp_values[0] = block_sum;
    }
  }
  block.sync();
  const float max_value = warp_values[0];

  float local_sum = 0.0f;
  for (int col = block.thread_rank(); col < cols; col += block.num_threads()) {
    local_sum = expf(x[col] - max_value);
  }
  float warp_sum = cg::reduce(tile, local_sum, cg::plus<float>());
  if (lane == 0) {
    warp_values[warp_id] = warp_sum;
  }
  block.sync();
  if (warp_id == 0) {
    float value = lane < num_warps ? warp_values[lane] : 0.0f;

    float block_sum = cg::reduce(tile, value, cg::plus<float>());

    if (lane == 0) {
      warp_values[0] = block_sum;
    }
  }
  block.sync();
  const float sum = warp_values[0];

  for (int col = block.thread_rank(); col < cols; col += block.num_threads()) {
    y[col] = expf(x[col] - max_value) / sum;
  }
}

void launch(const std::string& version, const float* input, float* output,
            int rows, int cols) {
  if (version == "v0") {
    softmax_v0<<<rows, 1>>>(input, output, rows, cols);

  } else if (version == "v1") {
    softmax_v1<<<rows, BLOCK_SIZE>>>(input, output, rows, cols);

  } else if (version == "v2") {
    softmax_v2<<<rows, BLOCK_SIZE>>>(input, output, rows, cols);

  } else if (version == "v3") {
    softmax_v3<<<rows, BLOCK_SIZE>>>(input, output, rows, cols);

  } else if (version == "v4") {
    softmax_v4<<<rows, BLOCK_SIZE>>>(input, output, rows, cols);

  } else {
    std::cerr << "Version must be v0, v1, v2, v3 or v4.\n";
    std::exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}

void cpu_softmax(const std::vector<float>& input, std::vector<float>& output,
                 int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    const float* x = input.data() + row * cols;
    float* y = output.data() + row * cols;

    float m = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < cols; ++c) {
      m = std::max(m, x[c]);
    }

    double s = 0.0;
    for (int c = 0; c < cols; ++c) {
      s += std::exp((double)x[c] - m);
    }

    for (int c = 0; c < cols; ++c) {
      y[c] = (float)(std::exp((double)x[c] - m) / s);
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: ./softmax v0|v1|v2|v3|v3 \n";
    return 1;
  }
  const std::string version = argv[1];

  constexpr int rows = 4096;
  constexpr int cols = 1024;
  constexpr int warmups = 10;
  constexpr int iterations = 100;

  size_t elements = (size_t)rows * cols;
  size_t bytes = elements * sizeof(float);

  // Create input array.
  std::vector<float> h_input(elements);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-4.0f, 4.0f);
  for (float& value : h_input) {
    value = distribution(rng);
  }

  std::cout << "Computing CPU Reference ... \n";
  std::vector<float> h_reference(elements);
  cpu_softmax(h_input, h_reference, rows, cols);

  float *d_input = nullptr, *d_output = nullptr;

  // Prepare for Device invocation and computation.
  // Allocate and prefill memory on device.
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
  for (int i = 0; i < warmups; ++i) {
    launch(version, d_input, d_output, rows, cols);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Initiate actual computation on the Device.
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch(version, d_input, d_output, rows, cols);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  // Copy output from Device to Host, to the vector, h_output.
  std::vector<float> h_output(elements);
  CUDA_CHECK(
      cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  float max_error = 0.0f;
  for (size_t i = 0; i < elements; ++i) {
    max_error = std::max(max_error, std::abs(h_output[i] - h_reference[i]));
  }
  std::cout << "version: " << version << "\n"
            << "average kernel time: " << milliseconds / iterations << " ms\n"
            << "max absolute error: " << max_error << "\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return 0;
}