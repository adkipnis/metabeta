#!/bin/bash

#SBATCH --job-name=eval-3-1
#SBATCH --output=logs/eval/%j.out
#SBATCH --error=logs/eval/%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --nice=1000

source $HOME/.bashrc
conda activate mb
cd ../metabeta
python evaluate.py -d 3 -q 1 -l 0 -i 50 --c_tag default

