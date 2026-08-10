#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    const cudaError_t error = (call);                                    \
    if (error != cudaSuccess) {                                          \
      std::cerr << "CUDA error at" << __FILE__ << ":" << __LINE__ << ":" \
                << cudaGetErrorString(error) << std::endl;               \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

constexpr int kElements = 1024;
constexpr int kSteps = 20;
constexpr int kIterations = 10000;

__global__ void KernelA(float* data) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kElements) {
    data[i] += 1.0f;
  }
}
__global__ void KernelB(float* data) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kElements) {
    data[i] *= 1.000001f;
  }
}

__global__ void KernelC(float* data) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kElements) {
    data[i] -= 0.5f;
  }
}

void EnqueueOneSequence(float* data, dim3 grid, dim3 block,
                        cudaStream_t stream) {
  for (int step = 0; step < kSteps; ++step) {
    KernelA<<<grid, block, 0, stream>>>(data);
    KernelB<<<grid, block, 0, stream>>>(data);
    KernelC<<<grid, block, 0, stream>>>(data);
  }
}

int main() {
  float* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, kElements * sizeof(float)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const dim3 block(256);
  const dim3 grid((kElements + block.x - 1) / block.x);

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaMemsetAsync(data, 0, kElements * sizeof(float), stream));
  CUDA_CHECK(cudaEventRecord(start, stream));

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    EnqueueOneSequence(data, grid, block, stream);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float normal_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&normal_ms, start, stop));

  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  EnqueueOneSequence(data, grid, block, stream);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  CUDA_CHECK(cudaMemsetAsync(data, 0, kElements * sizeof(float), stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float graph_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&graph_ms, start, stop));

  std::cout << "Ordinary Launches: " << normal_ms << " ms." << std::endl;
  std::cout << "Graph Replay: " << graph_ms << " ms." << std::endl;
  std::cout << "Ordinary/Graph-Replay: " << normal_ms / graph_ms << "x"
            << std::endl;

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(data));
  return EXIT_SUCCESS;
}