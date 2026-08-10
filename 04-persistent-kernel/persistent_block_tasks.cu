#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                        \
      std::cerr << "CUDA error at " << __FILE__ ":" << __LINE__ << ":" \
                << cudaGetErrorString(error) << std::endl;             \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

constexpr int kThreads = 256;
constexpr int kTasks = 4096;
constexpr int kElementsPerTask = 1024;
constexpr int kElements = kElementsPerTask * kTasks;

struct Task {
  int begin;
  int length;
  int rounds;
};

__global__ void PersistentBlockScheduler(const Task* tasks, int task_count,
                                         const float* input, float* output,
                                         int* next_task) {
  __shared__ int task_id;
  while (true) {
    if (threadIdx.x == 0) {
      task_id = atomicAdd(next_task, 1);
    }
    __syncthreads();

    if (task_id >= task_count) {
      return;
    }

    const Task task = tasks[task_id];

    // Every thread in the resident block participates in the claimed task.
    for (int offset = threadIdx.x; offset < task.length; offset += blockDim.x) {
      const int index = task.begin + offset;
      float value = input[index];

      // Different tasks intentionally have different costs.
      for (int round = 0; round < task.rounds; ++round) {
        value = value * 1.000001f + 0.000001f;
      }

      output[index] = value;
    }

    // The block must finish the current task, before it reuses its shared
    // task_id for another claim.
    __syncthreads();
  }
}

int main() {
  Task* tasks = nullptr;
  float* input = nullptr;
  float* output = nullptr;
  int* next_task = nullptr;

  CUDA_CHECK(cudaMallocManaged(&tasks, kTasks * sizeof(Task)));
  CUDA_CHECK(
      cudaMalloc(&input, static_cast<size_t>(kElements) * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&output, static_cast<size_t>(kElements) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&next_task, sizeof(int)));

  for (int i = 0; i < kTasks; ++i) {
    Task& task = tasks[i];
    task.begin = i * kElementsPerTask;
    task.length = kElementsPerTask;
    task.rounds = (i % 17) ? 256 : 8;
  }

  CUDA_CHECK(
      cudaMemset(input, 0, static_cast<size_t>(kElements) * sizeof(float)));
  CUDA_CHECK(
      cudaMemset(output, 0, static_cast<size_t>(kElements) * sizeof(float)));
  CUDA_CHECK(cudaMemset(next_task, 0, sizeof(int)));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  int active_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, PersistentBlockScheduler, kThreads, 0));

  // Start with one active worker block per SM. Vary this experimentally.
  const int worker_blocks_per_sm = 1;
  const int worker_blocks =
      properties.multiProcessorCount * worker_blocks_per_sm;

  PersistentBlockScheduler<<<worker_blocks, kThreads>>>(tasks, kTasks, input,
                                                        output, next_task);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int claimed = 0;
  CUDA_CHECK(
      cudaMemcpy(&claimed, next_task, sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "SM count: " << properties.multiProcessorCount << std::endl;
  std::cout << "possible blocks/SM: " << active_blocks_per_sm << std::endl;
  std::cout << "persistent worker blocks: " << worker_blocks << std::endl;
  std::cout << "logical tasks: " << kTasks << std::endl;
  std::cout << "queue claims (includes terminal claims): " << claimed
            << std::endl;

  CUDA_CHECK(cudaFree(next_task));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(tasks));
}
