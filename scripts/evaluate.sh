#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate/evaluate_%A_%a.out
#SBATCH --error=logs/evaluate/evaluate_%A_%a.err
#SBATCH --array=0-35

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/evaluation

CONFIGS=(
    small-b-mixed
    small-b-sampled
    small-n-mixed
    small-n-sampled
    small-p-mixed
    small-p-sampled
    mid-b-mixed
    mid-b-sampled
    mid-n-mixed
    mid-n-sampled
    mid-p-mixed
    mid-p-sampled
    medium-b-mixed
    medium-b-sampled
    medium-n-mixed
    medium-n-sampled
    medium-p-mixed
    medium-p-sampled
    large-b-mixed
    large-b-sampled
    large-n-mixed
    large-n-sampled
    large-p-mixed
    large-p-sampled
    big-b-mixed
    big-b-sampled
    big-n-mixed
    big-n-sampled
    big-p-mixed
    big-p-sampled
    huge-b-mixed
    huge-b-sampled
    huge-n-mixed
    huge-n-sampled
    huge-p-mixed
    huge-p-sampled
)

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Evaluating: $CFG (task $SLURM_ARRAY_TASK_ID)"
python evaluate.py --name "$CFG" --device cuda --save_tables
