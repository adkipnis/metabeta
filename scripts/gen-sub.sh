#!/bin/bash

DVAL=$1
QVAL=$2

#SBATCH --job-name=gen-sub-${DVAL}-${QVAL}
#SBATCH --output=logs/gen-sub-${DVAL}-${QVAL}/%j.out
#SBATCH --error=logs/gen-sub-${DVAL}-${QVAL}/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

source $HOME/.bashrc
conda activate mb
cd ../metabeta/data

python generate.py -d ${DVAL} -q ${QVAL} -b -1 --sub --slurm
