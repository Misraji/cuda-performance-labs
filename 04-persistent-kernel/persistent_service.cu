#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    const cudaError_t error = (call);                                     \
    if (error != cudaSuccess) {                                           \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ":" \
                << cudaGetErrorString(error) << std::endl;                \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

constexpr int kThreads = 256;
constexpr int kTasks = 128;
constexpr int kElementsPerTask = 4096;
constexpr int kElements = kTasks * kElementsPerTask;

struct Queue {
  int head;
  int tail;
  int shutdown;
  int tasks[kTasks];
};

__device__ bool TryClaimTask(Queue* queue, int* task_id) {
  while (true) {
    const int head = atomicAdd(&queue->head, 0);
    const int tail = atomicAdd(&queue->tail, 0);

    if (head >= tail) {
      return false;
    }

    if (atomicCAS(&queue->head, head, head + 1) == head) {
      *task_id = queue->tasks[head];
      return true;
    }
  }
}

__device__ void PollBackoff() {
#if __CUDA_ARCH__ >= 700
  // A tiny backoff vaoids the tighest polling loop on Volta+
  __nanosleep(1000);
#endif
}

__global__ void PersistentService(const float* input, float* output,
                                  Queue* queue) {
  __shared__ int block_task;

  while (true) {
    if (threadIdx.x == 0) {
      int task_id = -1;

      if (TryClaimTask(queue, &task_id)) {
        block_task = task_id;

      } else {
        const int shutdown = atomicAdd(&queue->shutdown, 0);
        const int head = atomicAdd(&queue->head, 0);
        const int tail = atomicAdd(&queue->tail, 0);

        // -2 means: producer promised that no future tasks will arrive and
        // the currently published queue is fully drained.
        block_task = (shutdown != 0 && head >= tail) ? -2 : -1;
      }
    }

    __syncthreads();

    if (block_task == -2) {
      return;
    }

    if (block_task == -1) {
      // The queue is temporarily empty. The kernel stays alive rather than
      // returning to the host. This is the clearest persistence
      // demonstration.
      if (threadIdx.x == 0) {
        PollBackoff();
      }
      __syncthreads();
      continue;
    }

    // One block cooperatively processes one task-sized chunk.
    const int begin = block_task * kElementsPerTask;
    const int end = begin + kElementsPerTask;

    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
      output[i] = input[i] * 2.0f + 1.0f;
    }
    __syncthreads();
  }
}

__global__ void Producer(Queue* queue) {
  if (blockIdx.x || threadIdx.x) {
    return;
  }

  // Populate descriptors first.
  for (int task = 0; task < kTasks; ++task) {
    queue->tasks[task] = task;
  }

  // Publish descriptor writes before publishing the new tail value.
  __threadfence();
  atomicExch(&queue->tail, kTasks);
}

__global__ void RequestShutdown(Queue* queue) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // This does not force an immediate exit. Consumers first drain tail.
    atomicExch(&queue->shutdown, 1);
  }
}

int main() {
  const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);
  float* input = nullptr;
  float* output = nullptr;
  Queue* queue = nullptr;

  CUDA_CHECK(cudaMalloc(&input, bytes));
  CUDA_CHECK(cudaMalloc(&output, bytes));
  CUDA_CHECK(cudaMalloc(&queue, sizeof(Queue)));

  CUDA_CHECK(cudaMemset(input, 0, bytes));
  CUDA_CHECK(cudaMemset(output, 0, bytes));
  CUDA_CHECK(cudaMemset(queue, 0, sizeof(Queue)));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm, PersistentService, kThreads, 0));

  // Deliberately use about half the theoretical block residency. The producer
  // needs execution resources while the persistent consumer is still running.
  const int max_resident_blocks =
      properties.multiProcessorCount * max_blocks_per_sm;
  const int worker_blocks = std::max(1, max_resident_blocks / 2);

  cudaStream_t consumer_stream;
  cudaStream_t producer_stream;
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking));
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking));

  std::cout << "Launching persistent service with empty queue ..." << std::endl;
  PersistentService<<<worker_blocks, kThreads, 0, consumer_stream>>>(
      input, output, queue);
  CUDA_CHECK(cudaGetLastError());

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  std::cout << "Publishing " << kTasks << " tasks after 100 ms ..."
            << std::endl;
  Producer<<<1, 1, 0, producer_stream>>>(queue);
  CUDA_CHECK(cudaGetLastError());

  // Same stream guarantees Producer finishes before shutdown is requested.
  RequestShutdown<<<1, 1, 0, producer_stream>>>(queue);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(producer_stream));
  CUDA_CHECK(cudaStreamSynchronize(consumer_stream));

  Queue final_queue{};
  CUDA_CHECK(
      cudaMemcpy(&final_queue, queue, sizeof(Queue), cudaMemcpyDeviceToHost));

  std::cout << "consumer blocks: " << worker_blocks << std::endl;
  std::cout << "final head: " << final_queue.head << std::endl;
  std::cout << "final tail: " << final_queue.tail << std::endl;
  std::cout << "service exited after explicit shutdown and drain" << std::endl;

  CUDA_CHECK(cudaStreamDestroy(consumer_stream));
  CUDA_CHECK(cudaStreamDestroy(producer_stream));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(queue));

  return EXIT_SUCCESS;
}