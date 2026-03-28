#!/bin/bash

#SBATCH --job-name=replenish-small-p-sampled
#SBATCH --output=logs/replenish/small-p-sampled_%A_%a.out
#SBATCH --error=logs/replenish/small-p-sampled_%A_%a.err
#SBATCH --array=0-23

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

START_EPOCH=11
CHUNK_SIZE=10
BEGIN=$((START_EPOCH + SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((BEGIN + CHUNK_SIZE - 1))

cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag small-p-sampled --partition train --begin ${BEGIN} --epochs ${END}
