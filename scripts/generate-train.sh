#!/bin/bash

#SBATCH --job-name=generate-train-3-1
#SBATCH --output=logs/generate-train/%j.out
#SBATCH --error=logs/generate-train/%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --nice=1000

source $HOME/.bashrc
conda activate mb
cd ../metabeta/data
python generate.py -d 3 -q 1 -b 0 -i 50 --semi

