#!/bin/bash

#SBATCH --job-name=fit-mid-p-sampled-missing
#SBATCH --output=logs/fit/mid-p-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/mid-p-sampled-missing_%A_%a.err
#SBATCH --array=0-26

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
# 002, 007, 013, 015, 021, 022, 028, 029, 036, 037,
# 046, 047, 053, 054, 063, 066, 069, 070, 080, 082,
# 086, 087, 096, 097, 104, 105, 116
MISSING_IDXS=(2 7 13 15 21 22 28 29 36 37 46 47 53 54 63 66 69 70 80 82 86 87 96 97 104 105 116)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag mid-p-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag mid-p-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
