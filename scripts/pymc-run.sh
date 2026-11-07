#!/bin/bash
# Source by absolute path (sbatch runs a spooled copy of the job script). runPymc runs a command
# from the repo root in the container, compiling into $PYTENSOR_DIR, a fresh /tmp directory
# removed when the sourcing script exits. A job that exports METABETA_REPO (or METABETA_ROOT, as
# fit-ref.sh does) runs that checkout (a worktree) instead of ~/metabeta, with ~/metabeta's
# container venv.

REPO="${METABETA_REPO:-${METABETA_ROOT:-$HOME/metabeta}}"
SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"
# outputs/data is symlinked to workspace storage; bind it so the link resolves in-container
DATA_ROOT="/lustre/groups/hcai/workspace/alexander.kipnis/metabeta-cache/datasets"

PYTENSOR_DIR=$(mktemp -d /tmp/pytensor_XXXXXX)
trap 'rm -rf "$PYTENSOR_DIR"' EXIT

runPymc() {
    echo "$(hostname): $*"
    apptainer exec \
      --bind "$HOME:$HOME" \
      --bind "$REPO:$REPO" \
      --bind "$DATA_ROOT:$DATA_ROOT" \
      --bind /tmp:/tmp \
      "$SIF" \
      bash -lc "
        set -euo pipefail
        cd '$REPO'
        source '$VENV/bin/activate'
        export PYTHONPATH='$REPO'  # the venv's editable install imports ~/metabeta
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
        export TMPDIR='$PYTENSOR_DIR'
        export PYTENSOR_FLAGS='base_compiledir=$PYTENSOR_DIR,cxx=/usr/bin/g++'
        $(printf '%q ' "$@")
      "
}
