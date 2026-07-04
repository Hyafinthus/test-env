#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char *argv[]) {
    int rank, size;

    // 配置参数
    const int max_msg_size = 128 * 1024 * 1024; // 最大消息 128 MB
    const int min_msg_size = 1 * 1024;          // 最小消息 1 KB
    const int factor       = 2;                 // 每次消息大小翻倍
    const int warmup_iters = 10;                // 预热次数
    const int iters        = 100;               // 正式测量次数

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "需要至少 2 个 MPI 进程（建议每个节点 1 个）。\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank > 1) {
        // 只使用 rank 0 和 1，其它进程直接退出
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("# MPI one-way bandwidth test (0 -> 1)\n");
        printf("# message_size(bytes)\tbandwidth_GB/s\n");
        fflush(stdout);
    }

    // 分配最大缓冲区
    char *buf = (char *)malloc(max_msg_size);
    if (!buf) {
        fprintf(stderr, "Rank %d: malloc failed for %d bytes\n", rank, max_msg_size);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // 简单填充一下，避免某些系统做极端优化
    for (int i = 0; i < max_msg_size; ++i) {
        buf[i] = (char)(i & 0xFF);
    }

    for (int msg_size = min_msg_size; msg_size <= max_msg_size; msg_size *= factor) {
        MPI_Barrier(MPI_COMM_WORLD);

        // 预热，不计时
        for (int i = 0; i < warmup_iters; ++i) {
            if (rank == 0) {
                MPI_Send(buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Recv(buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        // 正式测试
        for (int i = 0; i < iters; ++i) {
            if (rank == 0) {
                MPI_Send(buf, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Recv(buf, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        double end = MPI_Wtime();

        if (rank == 0) {
            double total_time = end - start; // 秒
            int64_t total_bytes = (int64_t)msg_size * iters; // 单向，总共发送字节数
            double bandwidth = (double)total_bytes / total_time; // B/s
            double bandwidth_GBs = bandwidth / (1024.0 * 1024.0 * 1024.0);

            printf("%d\t\t\t%.6f\n", msg_size, bandwidth_GBs);
            fflush(stdout);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}