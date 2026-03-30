#!/bin/bash

#SBATCH --job-name=fit-mid-b-sampled-missing
#SBATCH --output=logs/fit/mid-b-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/mid-b-sampled-missing_%A_%a.err
#SBATCH --array=0-31

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
# 001, 002, 005, 006, 011, 012, 018, 019, 024, 025,
# 046, 048, 052, 055, 059, 062, 066, 069, 074, 076,
# 080, 083, 086, 092, 100, 103, 105, 108, 110, 114,
# 119, 121
MISSING_IDXS=(1 2 5 6 11 12 18 19 24 25 46 48 52 55 59 62 66 69 74 76 80 83 86 92 100 103 105 108 110 114 119 121)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag mid-b-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag mid-b-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
