#!/bin/bash
# Reference fits: one SLURM array task per dataset, one method and level per submission.
# Run from the repo root on the submit node, then metabeta/simulation/check.py.
#   scripts/fit-ref.sh --method nuts --level 2 --data_id small-n-sampled
#   scripts/fit-ref.sh --method pathfinder --level 1 --data_id small-n-sampled --idx 3 17 42
# Levels: nuts 0|1|2, pathfinder 0|1. --qos cpu_preemptible has its own job cap (200, vs 100).

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

source "$HOME/metabeta/scripts/pymc-run.sh"
[[ -n "$WARM" ]] && tar xf "$WARM" -C "$PYTENSOR_DIR"
echo "warm cache: ${WARM:-none}"
runPymc python metabeta/simulation/fit.py --size "$SIZE" --family "$FAMILY" --ds_type "$DS_TYPE" \
    --partition "$PARTITION" --method "$METHOD" --level "$LEVEL" --idx "$FIT_IDX"
