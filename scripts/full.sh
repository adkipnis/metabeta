#!/bin/bash

#SBATCH --job-name=full
#SBATCH --output=logs/full/%j.out
#SBATCH --error=logs/full/%j.err

#SBATCH --gres=gpu:1
#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ---------------------------
# Define variables
# ---------------------------
D=5
Q=2
I=50

# ---------------------------
# Load environment
# ---------------------------
source $HOME/.bashrc
conda activate mb

# ---------------------------
# Run scripts with variables
# ---------------------------
cd ../metabeta/data
python generate.py -d $D -q $Q -b 0 -i $I --semi
cd ..
python train.py -d $D -q $Q -l 0 -i $I
