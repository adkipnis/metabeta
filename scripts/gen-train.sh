#!/bin/bash

#SBATCH --job-name=gen-train-3-1
#SBATCH --output=logs/gen-train-3-1/%j.out
#SBATCH --error=logs/gen-train-3-1/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

source $HOME/.bashrc
conda activate mb
cd ../metabeta/data
python generate.py -d 3 -q 1 -b 0 -i 100 --semi

