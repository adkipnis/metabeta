#!/bin/bash

#SBATCH --job-name=sweep-2
#SBATCH --output=logs/sweep-2/%A_%a.out
#SBATCH --error=logs/sweep-2/%A_%a.err
#SBATCH --array=0-2

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name cluster --m_tag sweep_2_${SLURM_ARRAY_TASK_ID}
