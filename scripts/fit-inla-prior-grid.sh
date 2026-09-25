#!/bin/bash

# E2: R-INLA on the Normal-slope prior grid of one dataset
# (experiments/evaluation/prior_sensitivity.py --stages inla). Four shards run in parallel, one
# INLA thread each, on the 4 cores one NUTS fit used, pinned to the CPU model every NUTS fit ran
# on (intel_xeon_6248r). Runs natively like fit-inla.sh: R-INLA lives in the host R library.
# Submit from ~/metabeta with the e2-prior-sensitivity branch checked out:
#   mkdir -p logs/inla-e2
#   sbatch scripts/fit-inla-prior-grid.sh --dataset sleep --rfx slope [--re-correlation diagonal] [--first 5]
# (--re-correlation diagonal only matters with a random slope; with one rfx the stage skips it)
# Then pull fits/test_inla_*.npz and fits/inla_*_timing_shard*.json and run the inla_report stage.

#SBATCH --job-name=inla-e2
#SBATCH --output=logs/inla-e2/%j.out
#SBATCH --error=logs/inla-e2/%j.err

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --constraint=intel_xeon_6248r
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

RE_CORRELATION="auto"
FIRST=()
N_SHARDS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --rfx) RFX="$2"; shift 2 ;;
        --re-correlation) RE_CORRELATION="$2"; shift 2 ;;
        --first) FIRST=(--inla_first "$2"); shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${DATASET:-}" || -z "${RFX:-}" ]] && {
    echo "Usage: $0 --dataset <sleep|cbpp|salamanders> --rfx <slope|intercept> [--re-correlation auto|diagonal] [--first N]"
    exit 1
}

REPO="$HOME/metabeta"
# /etc/bashrc and activate reference unset variables; relax nounset while sourcing them
set +u
source "$HOME/.bashrc"
source "$HOME/metabeta/.venv/bin/activate"
set -u
cd "$REPO"
export PYTHONPATH="$REPO"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo 'hostname:' "$(hostname)"
Rscript -e 'cat("INLA", as.character(packageVersion("INLA")), "\n")'

pids=()
for k in $(seq 0 $((N_SHARDS - 1))); do
    python experiments/evaluation/prior_sensitivity.py \
      --rfx "$RFX" \
      --stages inla \
      --datasets "$DATASET" \
      --inla_re_correlation "$RE_CORRELATION" \
      --inla_shard "$k" "$N_SHARDS" \
      "${FIRST[@]}" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
