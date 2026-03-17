#!/bin/bash

#SBATCH --job-name=fit-toy
#SBATCH --output=logs/fit-toy/%j.out
#SBATCH --error=logs/fit-toy/%j.err

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

# fit NUTS and ADVI
cd $HOME/metabeta/metabeta/simulation
python fit.py --d_tag toy --idx 0 --method nuts
python fit.py --d_tag toy --idx 0 --method advi
