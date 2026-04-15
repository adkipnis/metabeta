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
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$TAG" ]] && { echo "Usage: $0 --tag <size-family-ds_type>"; exit 1; }

IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"
case $FAM_NAME in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

JOB_TMPDIR="$HOME/tmp/pytensor_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export PYTENSOR_FLAGS="base_compiledir=$JOB_TMPDIR"

cd $HOME/metabeta/metabeta/simulation
python fit.py --size "${SIZE}" --family ${FAMILY} --ds_type "${DS_TYPE}" --idx ${SLURM_ARRAY_TASK_ID} --method nuts --loop
python fit.py --size "${SIZE}" --family ${FAMILY} --ds_type "${DS_TYPE}" --idx ${SLURM_ARRAY_TASK_ID} --method advi
rm -rf "$JOB_TMPDIR"
