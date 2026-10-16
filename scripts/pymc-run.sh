#!/bin/bash
# Sourced by the cluster scripts that fit with PyMC. Creates $PYTENSOR_DIR, an empty (cold)
# node-local PyTensor compile directory that is purged when the sourcing script exits, and
# runPymc, which runs a command from the repo root inside the campaign container with that
# directory as compile cache and TMPDIR. Source it by absolute path: sbatch runs a copy of the
# job script from its spool directory, so paths relative to $0 do not resolve.

SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"
# outputs/data is symlinked to workspace storage; bind it so the link resolves in-container
DATA_ROOT="/lustre/groups/hcai/workspace/alexander.kipnis/datasets"

PYTENSOR_DIR=$(mktemp -d /tmp/pytensor_XXXXXX)
trap 'rm -rf "$PYTENSOR_DIR"' EXIT

runPymc() {
    echo "$(hostname): $*"
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
        export TMPDIR='$PYTENSOR_DIR'
        export PYTENSOR_FLAGS='base_compiledir=$PYTENSOR_DIR,cxx=/usr/bin/g++'
        $(printf '%q ' "$@")
      "
}
