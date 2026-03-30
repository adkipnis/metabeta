#!/bin/bash

#SBATCH --job-name=fit-small-p-sampled-missing
#SBATCH --output=logs/fit/small-p-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/small-p-sampled-missing_%A_%a.err
#SBATCH --array=0-29

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
# 002, 005, 011, 013, 017, 023, 029, 033, 036, 041,
# 045, 047, 051, 057, 063, 065, 073, 077, 082, 083,
# 088, 091, 095, 102, 108, 112, 115, 118, 123, 126
MISSING_IDXS=(2 5 11 13 17 23 29 33 36 41 45 47 51 57 63 65 73 77 82 83 88 91 95 102 108 112 115 118 123 126)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag small-p-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag small-p-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
