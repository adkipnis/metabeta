#!/bin/bash

#SBATCH --job-name=replenish
#SBATCH --output=logs/replenish/%A_%a.out
#SBATCH --error=logs/replenish/%A_%a.err
#SBATCH --array=0-7

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

START_EPOCH=51
CHUNK_SIZE=25
BEGIN=$((START_EPOCH + SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((BEGIN + CHUNK_SIZE - 1))

cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag medium-n --partition train --begin ${BEGIN} --epochs ${END}
