# Distance-Based MPI Communication & Scaling Study

A deadlock-safe MPI program that has each process exchange data with two neighbors at fixed rank distances (D1, D2), applies a transform, and iterates — plus a performance study of how it scales across process counts and data sizes on a cluster (PARAM Rudra). Course project for CS633 (Parallel Programming) at IIT Kanpur.

## What it does

Each of the P processes:
1. Allocates two local buffers of size M, seeded with rank-scaled random values.
2. Computes its communication partners at distance D1 and D2 (`rank ± D`), treating out-of-range ranks as "no neighbor."
3. Runs T iterations where, each round, it exchanges its buffers with those neighbors, applies a transform to what it receives (square for the D1 buffer, log for the D2 buffer), sends the result back, and folds the response into its own buffer for the next round.
4. After T iterations, reduces to the global max of each buffer and the max wall-clock time across all ranks (via `MPI_Reduce`), which rank 0 prints.

The interesting part is the communication pattern: since it's built entirely on blocking `MPI_Send`/`MPI_Recv`, a naive "everyone sends first" would deadlock. It's avoided with a parity-based two-phase scheme — ranks are split into even/odd groups by `(rank / D) % 2`, and each phase pairs a send on one group with a matching receive on the other, so there's never a send without a receive waiting on the other end.

## Performance results

Benchmarked on PARAM Rudra with P = 8, 16, 32 processes and M = 262144, 1048576 elements per buffer, 5 runs each (`timing_data.txt` / `boxplot.png`):

| P  | M = 262144 | M = 1048576 |
|----|-----------|-------------|
| 8  | 0.095 s   | 0.360 s     |
| 16 | 0.437 s   | 0.821 s     |
| 32 | 0.805 s   | 1.214 s     |

Two takeaways:
- **Runtime grows with P, not shrinks** — since each process still handles a fixed M elements, adding processes only adds more total communication volume and synchronization overhead rather than splitting a fixed workload. The blocking sends/receives mean processes spend more time waiting as P grows, so this is a communication-bound program, not a compute-bound one.
- **Runtime scales close to linearly with M** — going from M = 262144 to M = 1048576 (4x) increases runtime by roughly 3–4x at every process count, which is the expected behavior for a buffer-size-driven workload.

## Files

- `src.c` — the MPI program (compile with `mpicc -o src src.c -lm`)
- `src1_<P>_M_<M>.sh` — Slurm batch scripts, one per (P, M) configuration, each running 5 repetitions
- `assign1_*.out` — raw Slurm job outputs
- `timing_data.txt` — the aggregated (P, M, run, time) table used for plotting
- `plot.py` — generates `boxplot.png` from `timing_data.txt`
- `Group51.pdf` — full write-up with the code walkthrough and results analysis

## Running it

```bash
module load compiler/oneapi-2024/mpi
mpicc -o src src.c -lm

# interactive run
mpirun -np 16 ./src 262144 2 4 10 1000
# args: M D1 D2 T SEED

# or submit the batch jobs
sbatch src1_8_M_262144.sh
sbatch src1_16_M_262144.sh
sbatch src1_32_M_262144.sh
sbatch src1_8_M_1048576.sh
sbatch src1_16_M_1048576.sh
sbatch src1_32_M_1048576.sh

# once timing_data.txt is populated
python3 plot.py
```

## Team

Group project for CS633, IIT Kanpur — Khusal Nikam, Prafull Joshi, Pranjal, Sarthak Dumbre, Vishal Junjare.
