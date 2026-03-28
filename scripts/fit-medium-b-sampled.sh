#!/bin/bash

#SBATCH --job-name=fit-medium-b-sampled
#SBATCH --output=logs/fit/medium-b-sampled_%A_%a.out
#SBATCH --error=logs/fit/medium-b-sampled_%A_%a.err
#SBATCH --array=0-127

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

cd $HOME/metabeta/metabeta/simulation
python fit.py --d_tag medium-b-sampled --idx ${SLURM_ARRAY_TASK_ID} --method nuts --loop
python fit.py --d_tag medium-b-sampled --idx ${SLURM_ARRAY_TASK_ID} --method advi
rm -rf "$JOB_TMPDIR"
