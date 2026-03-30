#!/bin/bash

#SBATCH --job-name=fit-medium-n-sampled-missing
#SBATCH --output=logs/fit/medium-n-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/medium-n-sampled-missing_%A_%a.err
#SBATCH --array=0-37

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
# 001, 003, 011, 016, 021, 025, 027, 031, 034, 035,
# 039, 040, 044, 048, 053, 058, 064, 066, 068, 070,
# 075, 078, 080, 084, 087, 091, 095, 098, 102, 103,
# 107, 110, 114, 115, 117, 121, 124, 125
MISSING_IDXS=(1 3 11 16 21 25 27 31 34 35 39 40 44 48 53 58 64 66 68 70 75 78 80 84 87 91 95 98 102 103 107 110 114 115 117 121 124 125)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag medium-n-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag medium-n-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
