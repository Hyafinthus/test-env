// Cross-rank staged GPU dependency bandwidth test.
//
// HEFT mapping:
//   - This approximates a GPU-resident dependency moved across ranks through
//     host staging: D2H on sender + MPI host send/recv + H2D on receiver.
//   - It is closer to the current daemon/handler path than raw MPI bandwidth.
//
// Example:
//   nvcc -O2 test_mpi_cuda_staged_bandwidth.cu -ccbin mpicxx \
//     -o test_mpi_cuda_staged_bandwidth
//   mpirun -n 2 ./test_mpi_cuda_staged_bandwidth

#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

static void *alloc_host(size_t bytes, int pinned) {
  void *ptr = nullptr;
  if (pinned) {
    CHECK_CUDA(cudaMallocHost(&ptr, bytes));
  } else {
    ptr = std::malloc(bytes);
    if (!ptr) {
      std::fprintf(stderr, "malloc failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  return ptr;
}

static void free_host(void *ptr, int pinned) {
  if (pinned) {
    CHECK_CUDA(cudaFreeHost(ptr));
  } else {
    std::free(ptr);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    if (rank == 0) {
      std::fprintf(stderr, "Need at least 2 MPI ranks.\n");
    }
    MPI_Finalize();
    return 1;
  }
  if (rank > 1) {
    MPI_Finalize();
    return 0;
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  int device = 0;
  if (argc > 1) {
    device = std::atoi(argv[1]);
  }
  device %= device_count;
  CHECK_CUDA(cudaSetDevice(device));

  int max_mb = 256;
  int iters = 100;
  int warmup = 10;
  int pinned = 1;
  if (argc > 2) {
    max_mb = std::atoi(argv[2]);
  }
  if (argc > 3) {
    iters = std::atoi(argv[3]);
  }
  if (argc > 4) {
    pinned = std::atoi(argv[4]) != 0;
  }

  const size_t max_size = (size_t)max_mb * 1024 * 1024;
  void *device_buf = nullptr;
  void *host_buf = alloc_host(max_size, pinned);
  CHECK_CUDA(cudaMalloc(&device_buf, max_size));
  CHECK_CUDA(cudaMemset(device_buf, rank == 0 ? 0x6b : 0, max_size));
  std::memset(host_buf, 0, max_size);

  if (rank == 0) {
    std::printf("# MPI staged GPU dependency one-way bandwidth (0 -> 1)\n");
    std::printf("# device=%d pinned_host=%d\n", device, pinned);
    std::printf("# size_bytes end_to_end_GiB_s\n");
  }

  char ack = 0;
  for (size_t bytes = 1024; bytes <= max_size; bytes *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup; ++i) {
      if (rank == 0) {
        CHECK_CUDA(cudaMemcpy(host_buf, device_buf, bytes,
                              cudaMemcpyDeviceToHost));
        MPI_Send(host_buf, (int)bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&ack, 1, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(host_buf, (int)bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        CHECK_CUDA(cudaMemcpy(device_buf, host_buf, bytes,
                              cudaMemcpyHostToDevice));
        MPI_Send(&ack, 1, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
      if (rank == 0) {
        CHECK_CUDA(cudaMemcpy(host_buf, device_buf, bytes,
                              cudaMemcpyDeviceToHost));
        MPI_Send(host_buf, (int)bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&ack, 1, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(host_buf, (int)bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        CHECK_CUDA(cudaMemcpy(device_buf, host_buf, bytes,
                              cudaMemcpyHostToDevice));
        MPI_Send(&ack, 1, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
      }
    }
    double end = MPI_Wtime();

    if (rank == 0) {
      double gib_s = ((double)bytes * (double)iters) / (end - start) /
                     (1024.0 * 1024.0 * 1024.0);
      std::printf("%zu %.6f\n", bytes, gib_s);
      std::fflush(stdout);
    }
  }

  CHECK_CUDA(cudaFree(device_buf));
  free_host(host_buf, pinned);
  MPI_Finalize();
  return 0;
}
