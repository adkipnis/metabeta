#!/bin/bash

#SBATCH --job-name=fit-medium-p-sampled-missing
#SBATCH --output=logs/fit/medium-p-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/medium-p-sampled-missing_%A_%a.err
#SBATCH --array=0-30

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
# 003, 006, 013, 015, 019, 023, 027, 032, 034, 040,
# 043, 047, 051, 055, 062, 065, 067, 070, 077, 079,
# 083, 086, 090, 093, 103, 105, 110, 112, 117, 121, 126
MISSING_IDXS=(3 6 13 15 19 23 27 32 34 40 43 47 51 55 62 65 67 70 77 79 83 86 90 93 103 105 110 112 117 121 126)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag medium-p-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag medium-p-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
