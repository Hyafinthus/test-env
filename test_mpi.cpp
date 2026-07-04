#include <mpi.h>

#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>

#define MPI_CHECK(call)                                                       \
  do {                                                                        \
    int err = (call);                                                         \
    if (err != MPI_SUCCESS) {                                                 \
      char errstr[MPI_MAX_ERROR_STRING];                                      \
      int errlen = 0;                                                         \
      MPI_Error_string(err, errstr, &errlen);                                 \
      std::cerr << "[MPI ERROR] rank " << rank << " at " << __FILE__ << ":"  \
                << __LINE__ << " : " << std::string(errstr, errlen)           \
                << std::endl;                                                 \
      MPI_Abort(MPI_COMM_WORLD, err);                                         \
    }                                                                         \
  } while (0)

int main(int argc, char **argv) {
  int rank = -1;
  int size = -1;

  MPI_CHECK(MPI_Init(&argc, &argv));

  MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len = 0;
  MPI_CHECK(MPI_Get_processor_name(hostname, &hostname_len));

  // --------------------------------------------------------------------------
  // 1. Gather hostnames
  // --------------------------------------------------------------------------
  std::vector<char> all_hostnames(size * MPI_MAX_PROCESSOR_NAME, '\0');

  MPI_CHECK(MPI_Allgather(
      hostname,
      MPI_MAX_PROCESSOR_NAME,
      MPI_CHAR,
      all_hostnames.data(),
      MPI_MAX_PROCESSOR_NAME,
      MPI_CHAR,
      MPI_COMM_WORLD));

  if (rank == 0) {
    std::cout << "========== MPI basic info ==========" << std::endl;
    std::cout << "Total ranks: " << size << std::endl;

    for (int r = 0; r < size; ++r) {
      const char *h = all_hostnames.data() + r * MPI_MAX_PROCESSOR_NAME;
      std::cout << "Rank " << r << " on host " << h << std::endl;
    }

    std::cout << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // --------------------------------------------------------------------------
  // 2. Point-to-point ring connectivity test
  // --------------------------------------------------------------------------
  {
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    int send_value = rank;
    int recv_value = -1;

    MPI_CHECK(MPI_Sendrecv(
        &send_value,
        1,
        MPI_INT,
        right,
        100,
        &recv_value,
        1,
        MPI_INT,
        left,
        100,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    if (recv_value != left) {
      std::cerr << "[FAIL] rank " << rank
                << " ring recv expected " << left
                << ", got " << recv_value << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  if (rank == 0) {
    std::cout << "[PASS] Ring MPI_Sendrecv connectivity test" << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // --------------------------------------------------------------------------
  // 3. MPI_Bcast test
  // --------------------------------------------------------------------------
  {
    int value = 0;

    if (rank == 0) {
      value = 123456789;
    }

    MPI_CHECK(MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (value != 123456789) {
      std::cerr << "[FAIL] rank " << rank
                << " MPI_Bcast got wrong value: " << value << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }

  if (rank == 0) {
    std::cout << "[PASS] MPI_Bcast test" << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // --------------------------------------------------------------------------
  // 4. MPI_Allreduce test
  // --------------------------------------------------------------------------
  {
    long long local = rank;
    long long global_sum = -1;

    MPI_CHECK(MPI_Allreduce(
        &local,
        &global_sum,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD));

    long long expected = static_cast<long long>(size) * (size - 1) / 2;

    if (global_sum != expected) {
      std::cerr << "[FAIL] rank " << rank
                << " MPI_Allreduce expected " << expected
                << ", got " << global_sum << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
  }

  if (rank == 0) {
    std::cout << "[PASS] MPI_Allreduce SUM test" << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // --------------------------------------------------------------------------
  // 5. MPI_Alltoall test
  //    This checks every rank can exchange data with every other rank.
  // --------------------------------------------------------------------------
  {
    std::vector<int> sendbuf(size);
    std::vector<int> recvbuf(size);

    for (int dst = 0; dst < size; ++dst) {
      sendbuf[dst] = rank * 100000 + dst;
    }

    MPI_CHECK(MPI_Alltoall(
        sendbuf.data(),
        1,
        MPI_INT,
        recvbuf.data(),
        1,
        MPI_INT,
        MPI_COMM_WORLD));

    for (int src = 0; src < size; ++src) {
      int expected = src * 100000 + rank;
      if (recvbuf[src] != expected) {
        std::cerr << "[FAIL] rank " << rank
                  << " MPI_Alltoall from src " << src
                  << " expected " << expected
                  << ", got " << recvbuf[src] << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 4);
      }
    }
  }

  if (rank == 0) {
    std::cout << "[PASS] MPI_Alltoall full-rank exchange test" << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // --------------------------------------------------------------------------
  // 6. MPI_Scatter test
  // --------------------------------------------------------------------------
  {
    std::vector<int> scatter_sendbuf;

    if (rank == 0) {
      scatter_sendbuf.resize(size);
      for (int i = 0; i < size; ++i) {
        scatter_sendbuf[i] = i * 10;
      }
    }

    int scatter_recv = -1;

    MPI_CHECK(MPI_Scatter(
        rank == 0 ? scatter_sendbuf.data() : nullptr,
        1,
        MPI_INT,
        &scatter_recv,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD));

    int expected = rank * 10;

    if (scatter_recv != expected) {
      std::cerr << "[FAIL] rank " << rank
                << " MPI_Scatter expected " << expected
                << ", got " << scatter_recv << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 5);
    }
  }

  if (rank == 0) {
    std::cout << "[PASS] MPI_Scatter test" << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // --------------------------------------------------------------------------
  // 7. MPI_Gather test
  // --------------------------------------------------------------------------
  {
    int local_value = rank + 1000;
    std::vector<int> gather_recvbuf;

    if (rank == 0) {
      gather_recvbuf.resize(size);
    }

    MPI_CHECK(MPI_Gather(
        &local_value,
        1,
        MPI_INT,
        rank == 0 ? gather_recvbuf.data() : nullptr,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD));

    if (rank == 0) {
      for (int r = 0; r < size; ++r) {
        int expected = r + 1000;
        if (gather_recvbuf[r] != expected) {
          std::cerr << "[FAIL] MPI_Gather from rank " << r
                    << " expected " << expected
                    << ", got " << gather_recvbuf[r] << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 6);
        }
      }
    }
  }

  if (rank == 0) {
    std::cout << "[PASS] MPI_Gather test" << std::endl;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // --------------------------------------------------------------------------
  // 8. Barrier latency rough test
  // --------------------------------------------------------------------------
  {
    constexpr int n_iter = 100;

    double t0 = MPI_Wtime();

    for (int i = 0; i < n_iter; ++i) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time = 0.0;

    MPI_CHECK(MPI_Reduce(
        &local_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD));

    if (rank == 0) {
      std::cout << "[INFO] Average MPI_Barrier time: "
                << max_time / n_iter * 1e6
                << " us" << std::endl;
    }
  }

  if (rank == 0) {
    std::cout << std::endl;
    std::cout << "========== ALL MPI TESTS PASSED ==========" << std::endl;
  }

  MPI_CHECK(MPI_Finalize());
  return 0;
}