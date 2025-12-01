#!/bin/bash

DVAL=$1
QVAL=$2

#SBATCH --job-name=gen-train-${DVAL}-${QVAL}
#SBATCH --output=logs/gen-train-${DVAL}-${QVAL}/%j.out
#SBATCH --error=logs/gen-train-${DVAL}-${QVAL}/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

source $HOME/.bashrc
conda activate mb
cd ../metabeta/data

python generate.py -d ${DVAL} -q ${QVAL} -b 0 -i 100 --semi
