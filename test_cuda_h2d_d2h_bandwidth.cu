// CUDA host<->device bandwidth test.
//
// HEFT mapping:
//   - Use D2H/H2D numbers for cross-rank staging around MPI.
//   - Pinned memory is closest to runtime staging buffers when you can control
//     allocation. Pageable memory shows the slower fallback path.
//
// Example:
//   nvcc -O2 test_cuda_h2d_d2h_bandwidth.cu -o test_cuda_h2d_d2h_bandwidth
//   ./test_cuda_h2d_d2h_bandwidth

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

enum CopyKind {
  H2D,
  D2H,
};

static double measure_copy(void *host, void *device, size_t bytes, int iters,
                           int warmup, CopyKind kind) {
  cudaMemcpyKind cuda_kind =
      kind == H2D ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  void *dst = kind == H2D ? device : host;
  void *src = kind == H2D ? host : device;

  for (int i = 0; i < warmup; ++i) {
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cuda_kind));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cuda_kind));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  auto stop = std::chrono::steady_clock::now();

  const double seconds =
      std::chrono::duration<double>(stop - start).count();
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

  const size_t max_size = static_cast<size_t>(max_mb) * 1024 * 1024;
  const size_t min_size = 1024;

  std::printf("# CUDA host<->device bandwidth\n");
  std::printf("# device size_bytes pinned_H2D_GiB_s pinned_D2H_GiB_s "
              "pageable_H2D_GiB_s pageable_D2H_GiB_s\n");

  for (int dev = 0; dev < device_count; ++dev) {
    CHECK_CUDA(cudaSetDevice(dev));

    void *device = nullptr;
    void *pinned = nullptr;
    void *pageable = nullptr;
    CHECK_CUDA(cudaMalloc(&device, max_size));
    CHECK_CUDA(cudaMallocHost(&pinned, max_size));
    pageable = std::malloc(max_size);
    if (!pageable) {
      std::fprintf(stderr, "malloc failed\n");
      return 1;
    }
    std::memset(pinned, 0x3c, max_size);
    std::memset(pageable, 0x7e, max_size);
    CHECK_CUDA(cudaMemset(device, 0, max_size));

    for (size_t bytes = min_size; bytes <= max_size; bytes *= 2) {
      double pinned_h2d = measure_copy(pinned, device, bytes, iters, warmup, H2D);
      double pinned_d2h = measure_copy(pinned, device, bytes, iters, warmup, D2H);
      double pageable_h2d =
          measure_copy(pageable, device, bytes, iters, warmup, H2D);
      double pageable_d2h =
          measure_copy(pageable, device, bytes, iters, warmup, D2H);
      std::printf("%d %zu %.6f %.6f %.6f %.6f\n", dev, bytes, pinned_h2d,
                  pinned_d2h, pageable_h2d, pageable_d2h);
      std::fflush(stdout);
    }

    std::free(pageable);
    CHECK_CUDA(cudaFreeHost(pinned));
    CHECK_CUDA(cudaFree(device));
  }

  return 0;
}
