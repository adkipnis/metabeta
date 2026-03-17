#!/bin/bash

#SBATCH --job-name=train-toy
#SBATCH --output=logs/train-toy/%j.out
#SBATCH --error=logs/train-toy/%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/trianing
python train.py --name cluster --device cuda
