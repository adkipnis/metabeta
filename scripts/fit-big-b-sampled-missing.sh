#!/bin/bash

#SBATCH --job-name=fit-big-b-sampled-missing
#SBATCH --output=logs/fit/big-b-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/big-b-sampled-missing_%A_%a.err
#SBATCH --array=0-16

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
# 000, 003, 006, 012, 013, 015, 016, 019, 020,
# 032, 033, 035, 036, 039, 041, 048, 050
MISSING_IDXS=(0 3 6 12 13 15 16 19 20 32 33 35 36 39 41 48 50)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag big-b-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag big-b-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
