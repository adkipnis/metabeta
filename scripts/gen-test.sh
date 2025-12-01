#!/bin/bash

#SBATCH --job-name=gen-test-3-1
#SBATCH --output=logs/gen-test/%j.out
#SBATCH --error=logs/gen-test/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

source $HOME/.bashrc

# prevent BLAS from oversubscribing
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# generate data
conda activate mb
cd ../metabeta/data
python generate.py -d 3 -q 1 -b -1 --semi

