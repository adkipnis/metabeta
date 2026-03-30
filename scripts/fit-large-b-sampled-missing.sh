#!/bin/bash

#SBATCH --job-name=fit-large-b-sampled-missing
#SBATCH --output=logs/fit/large-b-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/large-b-sampled-missing_%A_%a.err
#SBATCH --array=0-36

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

# Missing indices from check output (nuts/advi):
# 001, 002, 003, 008, 009, 010, 012, 013, 015, 017,
# 018, 019, 020, 023, 024, 026, 031, 032, 033, 034,
# 035, 039, 044, 053, 056, 060, 066, 071, 078, 084,
# 089, 092, 101, 107, 111, 115, 125
MISSING_IDXS=(1 2 3 8 9 10 12 13 15 17 18 19 20 23 24 26 31 32 33 34 35 39 44 53 56 60 66 71 78 84 89 92 101 107 111 115 125)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag large-b-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag large-b-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
