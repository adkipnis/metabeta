#!/bin/bash

#SBATCH --job-name=train-mid-b-mixed
#SBATCH --output=logs/train/mid-b-mixed_%j.out
#SBATCH --error=logs/train/mid-b-mixed_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name mid-b-mixed -e 250 --wandb --device cuda
