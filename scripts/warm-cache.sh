#!/bin/bash
# Build the pre-warmed PyTensor cache of one likelihood family for scripts/fit-ref.sh --warm.
#
# Usage (submit node):  sbatch scripts/warm-cache.sh --family n
# Output:               ~/pytensor_cache_{family}.tar  (pass as --warm to fit-ref.sh)
#
# Runs scripts/warm_cache.py on small-{family}-sampled inside the campaign container with an
# empty node-local compile directory and tars the result.

#SBATCH --job-name=warm-cache
#SBATCH --output=logs/warm/%j.out
#SBATCH --error=logs/warm/%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_preemptible
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

FAMILY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --family) FAMILY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ "$FAMILY" =~ ^[nbp]$ ]] || { echo "Usage: $0 --family <n|b|p>" >&2; exit 1; }

SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"
DATA_ROOT="/lustre/groups/hcai/workspace/alexander.kipnis/datasets"
CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pytensor_warm_${SLURM_JOB_ID}"
OUT="$HOME/pytensor_cache_${FAMILY}.tar"
rm -rf "$CACHE_DIR"; mkdir -p "$CACHE_DIR"
trap 'rm -rf "$CACHE_DIR"' EXIT

apptainer exec \
  --bind "$HOME:$HOME" \
  --bind "$DATA_ROOT:$DATA_ROOT" \
  --bind /tmp:/tmp \
  "$SIF" \
  bash -lc "
    set -euo pipefail
    cd '$HOME/metabeta'
    source '$VENV/bin/activate'
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export PYTENSOR_FLAGS='base_compiledir=$CACHE_DIR,cxx=/usr/bin/g++'
    echo 'hostname:' \$(hostname)
    python scripts/warm_cache.py --data_id small-$FAMILY-sampled
  "
tar cf "$OUT" -C "$CACHE_DIR" .
echo "wrote $OUT ($(du -sh "$OUT" | cut -f1))"
