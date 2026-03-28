#!/bin/bash

#SBATCH --job-name=fit-small-n-sampled-missing
#SBATCH --output=logs/fit/small-n-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/small-n-sampled-missing_%A_%a.err
#SBATCH --array=0-9

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

source "$HOME/.bashrc"
source "$HOME/metabeta/.venv/bin/activate"

JOB_TMPDIR="$HOME/tmp/pytensor_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export PYTENSOR_FLAGS="base_compiledir=$JOB_TMPDIR"

# Missing indices from check output:
# 012, 032, 033, 035, 059, 062, 066, 074, 085, 091
MISSING_IDXS=(12 32 33 35 59 62 66 74 85 91)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag small-n-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag small-n-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
