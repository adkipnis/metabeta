#!/bin/bash

#SBATCH --job-name=nuts
#SBATCH --output=logs/nuts/%A_%a.out
#SBATCH --error=logs/nuts/%A_%a.err
#SBATCH --array=0-511

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        --srcdir)  SRCDIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${TAG:-}" ]] && {
    echo "Usage: $0 --data_id <size-family-ds_type|size-family-source> [--srcdir <path>]"
    exit 1
}

if [[ -z "${SRCDIR:-}" ]]; then
    IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"
    case "$FAM_NAME" in
        n) FAMILY=0 ;;
        b) FAMILY=1 ;;
        p) FAMILY=2 ;;
        *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
    esac
fi

SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"

mkdir -p logs/nuts
JOB_TMPDIR="$HOME/tmp/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMPDIR"

apptainer exec \
  --bind "$HOME:$HOME" \
  "$SIF" \
  bash -lc "
    set -euo pipefail

    cd '$HOME/metabeta'
    source '$VENV/bin/activate'

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    export PYTENSOR_FLAGS='base_compiledir=$JOB_TMPDIR,cxx=/usr/bin/g++'

    echo 'hostname:' \$(hostname)
    echo 'python:' \$(command -v python)
    python --version
    echo 'g++:' \$(command -v g++)
    g++ --version
    echo 'PYTENSOR_FLAGS:' \$PYTENSOR_FLAGS
    echo 'SLURM_CPUS_PER_TASK:' ${SLURM_CPUS_PER_TASK}

    cd '$HOME/metabeta/metabeta/simulation'

    if [[ -n "${SRCDIR:-}" ]]; then
      python fit.py \
        --config '${SRCDIR}/${TAG}/config.yaml' \
        --idx '${SLURM_ARRAY_TASK_ID}' \
        --method nuts \
        --srcdir '${SRCDIR}'
    else
      python fit.py \
        --size '$SIZE' \
        --family '$FAMILY' \
        --ds_type '$DS_TYPE' \
        --idx '${SLURM_ARRAY_TASK_ID}' \
        --method nuts
    fi
  "

rm -rf "$JOB_TMPDIR"
