#!/bin/bash

#SBATCH --job-name=fit
#SBATCH --output=logs/fit/%A_%a.out
#SBATCH --error=logs/fit/%A_%a.err
#SBATCH --array=0-127

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

# setup tmp for job to avoid pytensor collision
JOB_TMPDIR="$HOME/tmp/pytensor_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export PYTENSOR_FLAGS="base_compiledir=$JOB_TMPDIR"

# Cleanup function (runs on exit, cancel, error, etc.)
cleanup() {
    rm -rf "$JOB_TMPDIR"
}
trap cleanup EXIT

# fit NUTS and ADVI
cd $HOME/metabeta/metabeta/simulation
python fit.py --d_tag medium-n --idx ${SLURM_ARRAY_TASK_ID} --method nuts --loop
python fit.py --d_tag medium-n --idx ${SLURM_ARRAY_TASK_ID} --method advi
