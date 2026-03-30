#!/bin/bash

#SBATCH --job-name=fit-large-p-sampled-missing
#SBATCH --output=logs/fit/large-p-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/large-p-sampled-missing_%A_%a.err
#SBATCH --array=0-19

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
# 000, 004, 012, 018, 023, 029, 036, 043, 053, 056,
# 062, 071, 076, 082, 087, 096, 104, 108, 112, 117
MISSING_IDXS=(0 4 12 18 23 29 36 43 53 56 62 71 76 82 87 96 104 108 112 117)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag large-p-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag large-p-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
