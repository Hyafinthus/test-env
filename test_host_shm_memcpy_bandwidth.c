// POSIX shared-memory mmap memcpy bandwidth test.
//
// HEFT mapping:
//   - Use this for handler<->daemon shared-memory staging cost.
//   - The runtime path usually does more than one host copy, so account for
//     the number of copies separately in the model.
//
// Example:
//   cc -O2 test_host_shm_memcpy_bandwidth.c -o test_host_shm_memcpy_bandwidth -lrt
//   ./test_host_shm_memcpy_bandwidth

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static double measure_copy(void *dst, const void *src, size_t bytes, int iters) {
  volatile unsigned char sink = 0;
  double start = now_seconds();
  for (int i = 0; i < iters; ++i) {
    memcpy(dst, src, bytes);
    sink ^= ((volatile unsigned char *)dst)[i & (bytes - 1)];
  }
  double end = now_seconds();
  (void)sink;
  return ((double)bytes * (double)iters) / (end - start) /
         (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char **argv) {
  size_t max_size = 256u * 1024u * 1024u;
  int iters = 100;
  if (argc > 1) {
    max_size = (size_t)atoll(argv[1]);
  }
  if (argc > 2) {
    iters = atoi(argv[2]);
  }

  const char *name = "/sycl_heft_shm_bw_test";
  int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    perror("shm_open");
    return 1;
  }
  shm_unlink(name);
  if (ftruncate(fd, (off_t)max_size) != 0) {
    perror("ftruncate");
    return 1;
  }

  void *shm = mmap(NULL, max_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (shm == MAP_FAILED) {
    perror("mmap");
    return 1;
  }

  void *host = NULL;
  if (posix_memalign(&host, 4096, max_size) != 0) {
    fprintf(stderr, "posix_memalign failed\n");
    return 1;
  }
  memset(host, 0x2a, max_size);
  memset(shm, 0, max_size);

  printf("# POSIX shm mmap memcpy bandwidth\n");
  printf("# size_bytes host_to_shm_GiB_s shm_to_host_GiB_s\n");

  for (size_t bytes = 1024; bytes <= max_size; bytes *= 2) {
    double h2s = measure_copy(shm, host, bytes, iters);
    double s2h = measure_copy(host, shm, bytes, iters);
    printf("%zu %.6f %.6f\n", bytes, h2s, s2h);
    fflush(stdout);
  }

  munmap(shm, max_size);
  close(fd);
  free(host);
  return 0;
}
