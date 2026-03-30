#!/bin/bash

#SBATCH --job-name=train2-large-n-mixed
#SBATCH --output=logs/train/large-n-mixed_2_%j.out
#SBATCH --error=logs/train/large-n-mixed_2_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name large-n-mixed -e 500 --wandb --device cuda --load_best
