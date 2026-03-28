#!/bin/bash

#SBATCH --job-name=fit-medium-p-sampled-missing
#SBATCH --output=logs/fit/medium-p-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/medium-p-sampled-missing_%A_%a.err
#SBATCH --array=0-16

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

# Missing indices from check output for medium-p-sampled:
# 013-029
MISSING_IDXS=(13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag medium-p-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag medium-p-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
