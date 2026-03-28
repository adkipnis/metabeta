#!/bin/bash

#SBATCH --job-name=fit-medium-n-sampled-missing
#SBATCH --output=logs/fit/medium-n-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/medium-n-sampled-missing_%A_%a.err
#SBATCH --array=0-3

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

JOB_TMPDIR="$HOME/tmp/pytensor_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export PYTENSOR_FLAGS="base_compiledir=$JOB_TMPDIR"

# Missing indices from check output:
MISSING_IDXS=(28 29 64 89)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag medium-n-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag medium-n-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
