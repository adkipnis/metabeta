#!/bin/bash

#SBATCH --job-name=fit-small-n-sampled-missing
#SBATCH --output=logs/fit/small-n-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/small-n-sampled-missing_%A_%a.err
#SBATCH --array=0-27

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
# 003, 004, 035, 036, 048, 051, 059, 061, 063, 064, 065, 066,
# 069, 071, 077, 080, 088, 090, 095, 096, 098, 099,
# 105, 106, 110, 111, 119, 120
MISSING_IDXS=(3 4 35 36 48 51 59 61 63 64 65 66 69 71 77 80 88 90 95 96 98 99 105 106 110 111 119 120)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag small-n-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag small-n-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
