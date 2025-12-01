#!/bin/bash

#SBATCH --job-name=generate-test-3-1
#SBATCH --output=logs/%j/generate-test.out
#SBATCH --error=logs/%j/generate-test.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --nice=1000

source $HOME/.bashrc
conda activate mb
cd ../metabeta/data
python generate.py -d 3 -q 1 -b -1

