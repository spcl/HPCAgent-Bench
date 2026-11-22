/* Two-node MPI probe (mpi_two_node_check.sbatch): one COMM_WORLD across the nodes, a host allreduce, and a
 * GPU-aware allreduce of a device buffer. Rank 0 prints the hosts it gathered and MPI2N PASS or MPI2N FAIL. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#define GPU(x) cuda##x
#else
#include <hip/hip_runtime.h>
#define GPU(x) hip##x
#endif

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0, one = 1, sum = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char host[MPI_MAX_PROCESSOR_NAME] = {0};
    int len = 0;
    MPI_Get_processor_name(host, &len);
    MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int ndev = 0;
    if (GPU(GetDeviceCount)(&ndev) != GPU(Success) || ndev < 1) {
        fprintf(stderr, "rank %d on %s: no GPU\n", rank, host);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    (void)GPU(SetDevice)(rank % ndev);
    int *dsend = NULL, *drecv = NULL, mine = rank, got = -1;
    (void)GPU(Malloc)((void **)&dsend, sizeof(int));
    (void)GPU(Malloc)((void **)&drecv, sizeof(int));
    (void)GPU(Memcpy)(dsend, &mine, sizeof(int), GPU(MemcpyHostToDevice));
    MPI_Allreduce(dsend, drecv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    (void)GPU(Memcpy)(&got, drecv, sizeof(int), GPU(MemcpyDeviceToHost));
    static char hosts[1024][MPI_MAX_PROCESSOR_NAME];
    if (size > 1024) MPI_Abort(MPI_COMM_WORLD, 3);
    MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int distinct = 1;
        for (int i = 1; i < size; i++) distinct += strcmp(hosts[i], hosts[0]) != 0 && strcmp(hosts[i], hosts[i - 1]) != 0;
        int want = size * (size - 1) / 2;
        printf("size=%d hosts=%d host_sum=%d device_sum=%d want=%d first=%s last=%s\n", size, distinct, sum, got, want,
               hosts[0], hosts[size - 1]);
        printf(sum == size && got == want && distinct > 1 ? "MPI2N PASS\n" : "MPI2N FAIL\n");
    }
    MPI_Finalize();
    return 0;
}
