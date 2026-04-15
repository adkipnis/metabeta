#!/bin/bash

#SBATCH --job-name=fit
#SBATCH --output=logs/fit/fit_%A_%a.out
#SBATCH --error=logs/fit/fit_%A_%a.err
#SBATCH --array=0-511

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

while [[ $# -gt 0 ]]; do
    case $1 in
        --d_tag) D_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$D_TAG" ]] && { echo "Usage: $0 --d_tag <tag>"; exit 1; }

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

JOB_TMPDIR="$HOME/tmp/pytensor_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export PYTENSOR_FLAGS="base_compiledir=$JOB_TMPDIR"

cd $HOME/metabeta/metabeta/simulation
python fit.py --d_tag "${D_TAG}" --idx ${SLURM_ARRAY_TASK_ID} --method nuts --loop
python fit.py --d_tag "${D_TAG}" --idx ${SLURM_ARRAY_TASK_ID} --method advi
rm -rf "$JOB_TMPDIR"
