// CUDA same-node GPU-to-GPU bandwidth test.
//
// HEFT mapping:
//   - Use this for same-rank cross-device communication and SNMD split D2D copy.
//   - Reports each src GPU -> dst GPU pair separately.
//
// Example:
//   nvcc -O2 test_cuda_p2p_bandwidth.cu -o test_cuda_p2p_bandwidth
//   ./test_cuda_p2p_bandwidth

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static double measure_pair(int src_dev, int dst_dev, size_t bytes, int iters,
                           int warmup) {
  void *src = nullptr;
  void *dst = nullptr;
  cudaStream_t stream;
  cudaEvent_t start, stop;

  CHECK_CUDA(cudaSetDevice(src_dev));
  CHECK_CUDA(cudaMalloc(&src, bytes));
  CHECK_CUDA(cudaMemset(src, 0x5a, bytes));

  CHECK_CUDA(cudaSetDevice(dst_dev));
  CHECK_CUDA(cudaMalloc(&dst, bytes));
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaSetDevice(src_dev));
  CHECK_CUDA(cudaFree(src));
  CHECK_CUDA(cudaSetDevice(dst_dev));
  CHECK_CUDA(cudaFree(dst));

  const double seconds = static_cast<double>(ms) / 1000.0;
  const double total_bytes = static_cast<double>(bytes) * iters;
  return total_bytes / seconds / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char **argv) {
  int max_mb = 256;
  int iters = 100;
  int warmup = 10;
  if (argc > 1) {
    max_mb = std::atoi(argv[1]);
  }
  if (argc > 2) {
    iters = std::atoi(argv[2]);
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::fprintf(stderr, "Need at least 2 CUDA devices.\n");
    return 1;
  }

  for (int dev = 0; dev < device_count; ++dev) {
    CHECK_CUDA(cudaSetDevice(dev));
    for (int peer = 0; peer < device_count; ++peer) {
      if (peer == dev) {
        continue;
      }
      int can_access = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, dev, peer));
      if (can_access) {
        cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
          CHECK_CUDA(err);
        }
        cudaGetLastError();
      }
    }
  }

  std::printf("# CUDA P2P/cudaMemcpyPeerAsync bandwidth\n");
  std::printf("# src_dev dst_dev peer_access size_bytes bandwidth_GiB_s\n");

  const size_t min_size = 1024;
  const size_t max_size = static_cast<size_t>(max_mb) * 1024 * 1024;
  for (int src = 0; src < device_count; ++src) {
    for (int dst = 0; dst < device_count; ++dst) {
      if (src == dst) {
        continue;
      }
      int peer_access = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&peer_access, dst, src));
      for (size_t bytes = min_size; bytes <= max_size; bytes *= 2) {
        double gib_s = measure_pair(src, dst, bytes, iters, warmup);
        std::printf("%d %d %d %zu %.6f\n", src, dst, peer_access, bytes,
                    gib_s);
        std::fflush(stdout);
      }
    }
  }

  return 0;
}
