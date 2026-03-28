#!/bin/bash

#SBATCH --job-name=train-huge-b-mixed
#SBATCH --output=logs/train/huge-b-mixed_%j.out
#SBATCH --error=logs/train/huge-b-mixed_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name huge-b-mixed -e 250 --wandb --device cuda
