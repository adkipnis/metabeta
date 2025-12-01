#!/bin/bash

#SBATCH --job-name=gen-test-3-1
#SBATCH --output=logs/gen-test/%j.out
#SBATCH --error=logs/gen-test/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=6:00:00

source $HOME/.bashrc
conda activate mb
cd ../metabeta/data
python generate.py -d 3 -q 1 -b -1 --semi

