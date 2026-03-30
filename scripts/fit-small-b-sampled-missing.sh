#!/bin/bash

#SBATCH --job-name=fit-small-b-sampled-missing
#SBATCH --output=logs/fit/small-b-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/small-b-sampled-missing_%A_%a.err
#SBATCH --array=0-28

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
# 003, 004, 008, 009, 024, 027, 034, 037, 042, 044,
# 052, 053, 058, 061, 067, 069, 076, 077, 084, 086,
# 090, 091, 098, 102, 107, 111, 118, 122, 124
MISSING_IDXS=(3 4 8 9 24 27 34 37 42 44 52 53 58 61 67 69 76 77 84 86 90 91 98 102 107 111 118 122 124)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag small-b-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag small-b-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
