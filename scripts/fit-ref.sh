#!/bin/bash
# Reference fits on the cluster: one SLURM array task per dataset, one method and budget
# level per submission. Submits itself, then runs metabeta/simulation/fit.py per task.
#
# Usage (from the repo root on the submit node):
#   scripts/fit-ref.sh --method nuts --level 2 --data_id small-n-sampled
#   scripts/fit-ref.sh --method advi --data_id small-n-sampled --partition valid
#   scripts/fit-ref.sh --method pathfinder --level 1 --data_id small-n-sampled --idx 3 17 42
#
# Methods: nuts (levels 0|1|2, 4 cores), advi (both snapshots in one run), pathfinder
# (levels 0|1), laplace; the last three use one core. --idx refits selected datasets only,
# the default array covers datasets 0-511. --qos cpu_preemptible runs under the second user
# cap (200 jobs) next to cpu_normal (100). --warm ~/pytensor_cache_{n|b|p}.tar (from
# scripts/warm-cache.sh) starts every task from that PyTensor cache instead of an empty one,
# so the recorded wall time is that of a user who has fitted one such GLMM before.
# Afterwards: metabeta/simulation/check.py.

set -euo pipefail

METHOD=""
LEVEL=0
TAG=""
PARTITION="test"
QOS="cpu_normal"
WARM=""
N_DATASETS=512
IDX_VALUES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --level) LEVEL="$2"; shift 2 ;;
        --data_id) TAG="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --warm) WARM="$2"; shift 2 ;;
        --n_datasets) N_DATASETS="$2"; shift 2 ;;
        --idx)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                IFS=',' read -r -a parts <<< "$1"
                IDX_VALUES+=("${parts[@]}")
                shift
            done
            ;;
        --idx_map) IFS=',' read -r -a IDX_VALUES <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$METHOD" && -n "$TAG" ]] || {
    echo "Usage: $0 --method <nuts|advi|pathfinder|laplace> [--level L] --data_id <size-family-ds_type> [--partition test|valid] [--idx i ...]" >&2
    exit 1
}
case "$PARTITION" in
    test|valid) ;;
    *) echo "Unknown partition: $PARTITION (use test or valid)" >&2; exit 1 ;;
esac
case "$METHOD" in
    nuts) CPUS=4; [[ "$LEVEL" == 2 ]] && TIME="12:00:00" || TIME="06:00:00" ;;
    advi|pathfinder|laplace) CPUS=1; TIME="06:00:00" ;;
    *) echo "Unknown method: $METHOD" >&2; exit 1 ;;
esac
for value in "${IDX_VALUES[@]}"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Invalid idx value: $value" >&2; exit 1; }
done
[[ -z "$WARM" || -f "$WARM" ]] || { echo "Warm cache not found: $WARM" >&2; exit 1; }

JOB_NAME="${METHOD}${LEVEL}"
LOG_DIR="logs/$JOB_NAME"

# ---------------------------------------------------------------------------
# Submit node: dispatch the array and stop.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$LOG_DIR"
    if [[ ${#IDX_VALUES[@]} -gt 0 ]]; then
        ARRAY="0-$(( ${#IDX_VALUES[@]} - 1 ))"
        IDX_MAP=$(IFS=,; echo "${IDX_VALUES[*]}")
        MAP_ARGS=(--idx_map "$IDX_MAP")
        echo "Submitting $JOB_NAME for $TAG/$PARTITION on indices $IDX_MAP"
    else
        ARRAY="0-$(( N_DATASETS - 1 ))"
        MAP_ARGS=()
        echo "Submitting $JOB_NAME for $TAG/$PARTITION on $N_DATASETS datasets"
    fi
    sbatch \
        --job-name="$JOB_NAME" \
        --output="$LOG_DIR/%A_%a.out" \
        --error="$LOG_DIR/%A_%a.err" \
        --array="$ARRAY" \
        --partition=cpu_p \
        --qos="$QOS" \
        --nodes=1 \
        --cpus-per-task="$CPUS" \
        --mem=16G \
        --time="$TIME" \
        "$0" --method "$METHOD" --level "$LEVEL" --data_id "$TAG" --partition "$PARTITION" \
            ${WARM:+--warm "$WARM"} "${MAP_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Array task: fit one dataset.
if [[ ${#IDX_VALUES[@]} -gt 0 ]]; then
    FIT_IDX="${IDX_VALUES[$SLURM_ARRAY_TASK_ID]}"
else
    FIT_IDX="$SLURM_ARRAY_TASK_ID"
fi

IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"
case "$FAM_NAME" in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)" >&2; exit 1 ;;
esac

SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"
# outputs/data is symlinked to workspace storage; bind it so the link resolves in-container
DATA_ROOT="/lustre/groups/hcai/workspace/alexander.kipnis/datasets"

# node-local compile directory; empty (cold) unless a warm cache is unpacked into it
JOB_TMPDIR="${SLURM_TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMPDIR"
trap 'rm -rf "$JOB_TMPDIR"' EXIT
[[ -n "$WARM" ]] && tar xf "$WARM" -C "$JOB_TMPDIR"

apptainer exec \
  --bind "$HOME:$HOME" \
  --bind "$DATA_ROOT:$DATA_ROOT" \
  --bind /tmp:/tmp \
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
    echo 'python:' \$(command -v python)
    python --version
    echo 'PYTENSOR_FLAGS:' \$PYTENSOR_FLAGS
    echo 'SLURM_CPUS_PER_TASK:' ${SLURM_CPUS_PER_TASK}
    echo 'Fitting dataset idx:' '$FIT_IDX'
    echo 'warm cache:' '${WARM:-none}'

    python metabeta/simulation/fit.py \
      --size '$SIZE' \
      --family '$FAMILY' \
      --ds_type '$DS_TYPE' \
      --partition '$PARTITION' \
      --method '$METHOD' \
      --level '$LEVEL' \
      --idx '$FIT_IDX'
  "
