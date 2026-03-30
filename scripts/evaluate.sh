#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate/evaluate_%j.out
#SBATCH --error=logs/evaluate/evaluate_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

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

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $cfg"
    echo "=========================================="
    python evaluate.py --name "$cfg" --device cuda --save_tables
done
