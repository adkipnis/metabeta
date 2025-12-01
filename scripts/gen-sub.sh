#!/bin/bash

#SBATCH --job-name=gen-sub-3-1
#SBATCH --output=logs/gen-sub/%j.out
#SBATCH --error=logs/gen-sub/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

source $HOME/.bashrc
conda activate mb
cd ../metabeta/data
python generate.py -d 3 -q 1 -b -1 --sub --slurm

