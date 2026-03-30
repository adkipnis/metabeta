#!/bin/bash

#SBATCH --job-name=fit-mid-n-sampled-missing
#SBATCH --output=logs/fit/mid-n-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/mid-n-sampled-missing_%A_%a.err
#SBATCH --array=0-39

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
# 002, 010, 014, 021, 023, 026, 029, 031, 033, 036,
# 041, 043, 047, 052, 054, 059, 061, 065, 068, 070,
# 072, 075, 076, 081, 084, 088, 094, 096, 100, 101,
# 104, 105, 109, 110, 113, 114, 119, 120, 123, 124
MISSING_IDXS=(2 10 14 21 23 26 29 31 33 36 41 43 47 52 54 59 61 65 68 70 72 75 76 81 84 88 94 96 100 101 104 105 109 110 113 114 119 120 123 124)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag mid-n-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag mid-n-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
