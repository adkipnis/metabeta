#!/bin/bash

# E2 prior sensitivity: fit one prior-grid point per array task with fit.py (NUTS by default).
# The grid batch is written by
#   python experiments/evaluation/prior_sensitivity.py --rfx $rfx --stages export
# to metabeta/outputs/data/e2-{dataset}-{rfx}/. The grid is capped per dataset, so pass the array:
# its first n points (grid.csv rows ending in True) are the NUTS sub-grid,
#   n=$(grep -c ',True$' metabeta/outputs/data/e2-${ds}-${rfx}/grid.csv)
#   sbatch --array=0-$((n - 1)) scripts/fit-nuts-prior-grid.sh --dataset $ds --rfx $rfx
# and the rest of the grid follows (rebuttal: --array=$n-<last point>).

#SBATCH --job-name=nuts-e2
#SBATCH --output=logs/nuts-e2/%A_%a.out
#SBATCH --error=logs/nuts-e2/%A_%a.err

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

METHOD="nuts"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --rfx) RFX="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${DATASET:-}" || -z "${RFX:-}" ]] && {
    echo "Usage: $0 --dataset <sleep|cbpp|salamanders> --rfx <slope|intercept> [--method nuts|advi]"
    exit 1
}

SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"
# outputs/data is symlinked to workspace storage; bind it so the link resolves in-container
DATA_ROOT="/lustre/groups/hcai/workspace/alexander.kipnis/datasets"
CONFIG="$HOME/metabeta/metabeta/outputs/data/e2-${DATASET}-${RFX}/config.yaml"

mkdir -p logs/nuts-e2
JOB_TMPDIR="$HOME/tmp/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMPDIR"
trap 'rm -rf "$JOB_TMPDIR"' EXIT

apptainer exec \
  --bind "$HOME:$HOME" \
  --bind "$DATA_ROOT:$DATA_ROOT" \
  "$SIF" \
  bash -lc "
    set -euo pipefail

    cd '$HOME/metabeta'
    source '$VENV/bin/activate'

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    export TMPDIR="$JOB_TMPDIR"
    export PYTENSOR_FLAGS='base_compiledir=$JOB_TMPDIR,cxx=/usr/bin/g++'

    echo 'hostname:' \$(hostname)
    echo 'SLURM_CPUS_PER_TASK:' ${SLURM_CPUS_PER_TASK}

    cd '$HOME/metabeta/metabeta/simulation'

    python fit.py \
      --config '$CONFIG' \
      --idx '${SLURM_ARRAY_TASK_ID}' \
      --method '$METHOD' \
      --partition test
  "
