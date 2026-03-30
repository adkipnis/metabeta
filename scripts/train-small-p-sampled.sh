#!/bin/bash

#SBATCH --job-name=train-small-p-sampled
#SBATCH --output=logs/train/small-p-sampled_%j.out
#SBATCH --error=logs/train/small-p-sampled_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name small-p-sampled -e 500 --wandb --device cuda
