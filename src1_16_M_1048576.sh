#!/bin/bash
#SBATCH --job-name=mpi_assign1
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --output=assign1_16_M_1048576_%j.out
#SBATCH --error=assign1_16_M_1048576%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:05:00

module load compiler/oneapi-2024/mpi

D1=2
D2=4
T=10
SEED=1000

for run in {1..5}
    do
        mpirun -np 16 ./src 1048576 $D1 $D2 $T $SEED
    done

