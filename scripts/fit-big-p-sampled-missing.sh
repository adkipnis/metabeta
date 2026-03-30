#!/bin/bash

#SBATCH --job-name=fit-big-p-sampled-missing
#SBATCH --output=logs/fit/big-p-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/big-p-sampled-missing_%A_%a.err
#SBATCH --array=0-23

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
# 013, 018, 023, 030, 043, 046, 053, 056, 065, 076,
# 080, 085, 092, 099, 108, 113, 114, 118, 119, 120,
# 124, 125, 126, 127
MISSING_IDXS=(13 18 23 30 43 46 53 56 65 76 80 85 92 99 108 113 114 118 119 120 124 125 126 127)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag big-p-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag big-p-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
