#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TAG_D1_REQ  0
#define TAG_D2_REQ  1
#define TAG_D1_RES  2
#define TAG_D2_RES  3

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M    = atoi(argv[1]);
    int D1   = atoi(argv[2]);
    int D2   = atoi(argv[3]);
    int T    = atoi(argv[4]);
    int seed = atoi(argv[5]);

    /* -------------------- Buffers -------------------- */
    double *buffer_D1 = malloc(M * sizeof(double));
    double *buffer_D2 = malloc(M * sizeof(double));

    double *recv_D1   = malloc(M * sizeof(double));
    double *recv_D2   = malloc(M * sizeof(double));

    double *res_D1    = malloc(M * sizeof(double));
    double *res_D2    = malloc(M * sizeof(double));

    srand(seed);

    for (int i = 0; i < M; i++) {
        buffer_D1[i] = (double)rand() * (rank + 1) / 10000.0;
        buffer_D2[i] = buffer_D1[i];
    }

    /* ------------ Precompute neighbors ------------ */
    int upD1   = (rank + D1 < size) ? rank + D1 : -1;
    int downD1 = (rank - D1 >= 0)   ? rank - D1 : -1;

    int upD2   = (rank + D2 < size) ? rank + D2 : -1;
    int downD2 = (rank - D2 >= 0)   ? rank - D2 : -1;

    int evenD1 = ((rank / D1) % 2 == 0);
    int evenD2 = ((rank / D2) % 2 == 0);

    double start = MPI_Wtime();

    /* ================= MAIN LOOP ================= */
    for (int iter = 0; iter < T; iter++) {

        /* -------- PHASE 1: Send requests (D1 + D2) -------- */

        /* D1 */
        if (evenD1) {
            if (upD1 != -1)
                MPI_Send(buffer_D1, M, MPI_DOUBLE, upD1,
                         TAG_D1_REQ, MPI_COMM_WORLD);
        } else {
            if (downD1 != -1)
                MPI_Recv(recv_D1, M, MPI_DOUBLE, downD1,
                         TAG_D1_REQ, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }

        /* D2 */
        if (evenD2) {
            if (upD2 != -1)
                MPI_Send(buffer_D2, M, MPI_DOUBLE, upD2,
                         TAG_D2_REQ, MPI_COMM_WORLD);
        } else {
            if (downD2 != -1)
                MPI_Recv(recv_D2, M, MPI_DOUBLE, downD2,
                         TAG_D2_REQ, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }

        /* Opposite parity completes requests */

        if (!evenD1) {
            if (upD1 != -1)
                MPI_Send(buffer_D1, M, MPI_DOUBLE, upD1,
                         TAG_D1_REQ, MPI_COMM_WORLD);
        } else {
            if (downD1 != -1)
                MPI_Recv(recv_D1, M, MPI_DOUBLE, downD1,
                         TAG_D1_REQ, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }

        if (!evenD2) {
            if (upD2 != -1)
                MPI_Send(buffer_D2, M, MPI_DOUBLE, upD2,
                         TAG_D2_REQ, MPI_COMM_WORLD);
        } else {
            if (downD2 != -1)
                MPI_Recv(recv_D2, M, MPI_DOUBLE, downD2,
                         TAG_D2_REQ, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }

        /* -------- PHASE 2: Compute both transforms -------- */

        if (downD1 != -1 || downD2 != -1) {
            for (int i = 0; i < M; i++) {
                if (downD1 != -1)
                    res_D1[i] = recv_D1[i] * recv_D1[i];

                if (downD2 != -1)
                    res_D2[i] = log(recv_D2[i]);
            }
        }

        /* -------- PHASE 3: Send results back -------- */

        if (downD1 != -1)
            MPI_Send(res_D1, M, MPI_DOUBLE, downD1,
                     TAG_D1_RES, MPI_COMM_WORLD);

        if (downD2 != -1)
            MPI_Send(res_D2, M, MPI_DOUBLE, downD2,
                     TAG_D2_RES, MPI_COMM_WORLD);

        /* -------- PHASE 4: Receive processed results -------- */

        if (upD1 != -1)
            MPI_Recv(recv_D1, M, MPI_DOUBLE, upD1,
                     TAG_D1_RES, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        if (upD2 != -1)
            MPI_Recv(recv_D2, M, MPI_DOUBLE, upD2,
                     TAG_D2_RES, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        /* -------- PHASE 5: Update local buffers -------- */

        if (upD1 != -1) {
            for (int i = 0; i < M; i++)
                buffer_D1[i] =
                    (double)((unsigned long long)recv_D1[i] % 100000ULL);
        }

        if (upD2 != -1) {
            for (int i = 0; i < M; i++)
                buffer_D2[i] = recv_D2[i] * 100000.0;
        }
    }

    /* ================= REDUCTIONS ================= */

    double local_max_D1 = -1.0, local_max_D2 = -1.0;

    if (upD1 != -1)
        for (int i = 0; i < M; i++)
            if (buffer_D1[i] > local_max_D1)
                local_max_D1 = buffer_D1[i];

    if (upD2 != -1)
        for (int i = 0; i < M; i++)
            if (buffer_D2[i] > local_max_D2)
                local_max_D2 = buffer_D2[i];

    double global_max_D1, global_max_D2;

    MPI_Reduce(&local_max_D1, &global_max_D1, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_max_D2, &global_max_D2, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_time = MPI_Wtime() - start;
    double max_time;

    MPI_Reduce(&local_time, &max_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("%lf %lf %lf\n",
               global_max_D1, global_max_D2, max_time);

    free(buffer_D1);
    free(buffer_D2);
    free(recv_D1);
    free(recv_D2);
    free(res_D1);
    free(res_D2);

    MPI_Finalize();
    return 0;
}

