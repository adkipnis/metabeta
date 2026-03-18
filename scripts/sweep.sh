#!/bin/bash

#SBATCH --job-name=train-toy
#SBATCH --output=logs/train-toy/%A_%a.out
#SBATCH --error=logs/train-toy/%A_%a.err
#SBATCH --array=0-71

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name cluster --m_tag sweep_1_${SLURM_ARRAY_TASK_ID}
