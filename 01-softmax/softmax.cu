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

// V0: Essentially uses no parallelization. ONLY 1 GPU thread (thread 0)
// to calculate the values. Essentially CPU code executed in the GPU.
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

__global__ void softmax_v1(const float* input, float* output, int rows,
                           int cols) {}
__global__ void softmax_v2(const float* input, float* output, int rows,
                           int cols) {}
__global__ void softmax_v3(const float* input, float* output, int rows,
                           int cols) {}
__global__ void softmax_v4(const float* input, float* output, int rows,
                           int cols) {}

void launch(const std::string& version, const float* input, float* output,
            int rows, int cols) {
  if (version == "v0") {
    softmax_v0<<<rows, 1>>>(input, output, rows, cols);
  } else if (version == "1") {
    softmax_v1<<<rows, BLOCK_SIZE>>>(input, output, rows, cols);
  } else if (version == "2") {
    softmax_v2<<<rows, BLOCK_SIZE>>>(input, output, rows, cols);
  } else if (version == "3") {
    softmax_v3<<<rows, BLOCK_SIZE>>>(input, output, rows, cols);
  } else if (version == "4") {
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