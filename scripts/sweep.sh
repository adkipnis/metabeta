#!/bin/bash

#SBATCH --job-name=sweep-5
#SBATCH --output=logs/sweep-5/%A_%a.out
#SBATCH --error=logs/sweep-5/%A_%a.err
#SBATCH --array=0-1

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name cluster_${SLURM_ARRAY_TASK_ID} --m_tag affine
