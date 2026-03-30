#!/bin/bash

#SBATCH --job-name=fit-big-n-sampled-missing
#SBATCH --output=logs/fit/big-n-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/big-n-sampled-missing_%A_%a.err
#SBATCH --array=0-41

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
# 001, 002, 006, 011, 018, 022, 029, 031, 034, 035,
# 038, 040, 044, 045, 049, 051, 057, 058, 062, 064,
# 067, 070, 075, 076, 079, 081, 086, 088, 091, 093,
# 096, 099, 102, 106, 109, 111, 114, 116, 118, 121,
# 123, 127
MISSING_IDXS=(1 2 6 11 18 22 29 31 34 35 38 40 44 45 49 51 57 58 62 64 67 70 75 76 79 81 86 88 91 93 96 99 102 106 109 111 114 116 118 121 123 127)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag big-n-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag big-n-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
