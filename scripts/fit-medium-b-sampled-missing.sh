#!/bin/bash

#SBATCH --job-name=fit-medium-b-sampled-missing
#SBATCH --output=logs/fit/medium-b-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/medium-b-sampled-missing_%A_%a.err
#SBATCH --array=0-33

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
# 002, 008, 011, 015, 016, 017, 020, 022, 027, 032,
# 037, 040, 043, 046, 049, 052, 056, 061, 064, 068,
# 072, 076, 079, 084, 087, 092, 097, 102, 108, 111,
# 113, 117, 120, 126
MISSING_IDXS=(2 8 11 15 16 17 20 22 27 32 37 40 43 46 49 52 56 61 64 68 72 76 79 84 87 92 97 102 108 111 113 117 120 126)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag medium-b-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag medium-b-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
