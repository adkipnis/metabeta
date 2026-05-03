#!/bin/bash

#SBATCH --job-name=fit-selected
#SBATCH --output=logs/fit-selected/%A_%a.out
#SBATCH --error=logs/fit-selected/%A_%a.err
#SBATCH --array=0-0

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/fit-selected.sh --method <nuts|advi> --data_id <size-family-ds_type> --idx <i1> [i2 ...]
  scripts/fit-selected.sh --method <nuts|advi> --data_id <size-family-ds_type> --idx <i1,i2,...>
  scripts/fit-selected.sh --method <nuts|advi> --data_id <size-family-ds_type> --idx <i1> --idx <i2>

Examples:
  scripts/fit-selected.sh --method nuts --data_id small-n-sampled --idx 0 7 13
  scripts/fit-selected.sh --method advi --data_id small-b-flat --idx 1,4,9
EOF
}

METHOD=""
TAG=""
IDX_MAP=""
IDX_VALUES=()

append_idx_values() {
    local raw="$1"
    local part
    IFS=',' read -r -a parts <<< "$raw"
    for part in "${parts[@]}"; do
        [[ -n "$part" ]] || continue
        IDX_VALUES+=("$part")
    done
}

validate_int_list() {
    local value
    for value in "$@"; do
        [[ "$value" =~ ^[0-9]+$ ]] || {
            echo "Invalid idx value: $value" >&2
            exit 1
        }
    done
}

dedupe_preserve_order() {
    local value
    local seen=" "
    local unique=()
    for value in "$@"; do
        [[ "$seen" == *" $value "* ]] && continue
        seen+="$value "
        unique+=("$value")
    done
    printf '%s\n' "${unique[@]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --data_id)
            TAG="$2"
            shift 2
            ;;
        --idx)
            shift
            [[ $# -gt 0 ]] || {
                echo "Missing value after --idx" >&2
                usage
                exit 1
            }
            while [[ $# -gt 0 && "$1" != --* ]]; do
                append_idx_values "$1"
                shift
            done
            ;;
        --idx_map)
            IDX_MAP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

[[ -n "$METHOD" && -n "$TAG" ]] || {
    usage
    exit 1
}

case "$METHOD" in
    advi)
        JOB_NAME="advi"
        LOG_DIR="logs/advi"
        CPUS_PER_TASK=1
        ;;
    nuts)
        JOB_NAME="nuts"
        LOG_DIR="logs/nuts"
        CPUS_PER_TASK=4
        ;;
    *)
        echo "Unknown method: $METHOD (use nuts or advi)" >&2
        exit 1
        ;;
esac

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ ${#IDX_VALUES[@]} -gt 0 ]] || {
        echo "At least one --idx value is required" >&2
        usage
        exit 1
    }

    validate_int_list "${IDX_VALUES[@]}"
    mapfile -t UNIQUE_IDX_VALUES < <(dedupe_preserve_order "${IDX_VALUES[@]}")
    IDX_MAP=$(IFS=,; echo "${UNIQUE_IDX_VALUES[*]}")
    ARRAY_MAX=$(( ${#UNIQUE_IDX_VALUES[@]} - 1 ))

    mkdir -p "$LOG_DIR"

    echo "Submitting $METHOD fit job for data_id=$TAG on indices: $IDX_MAP"
    sbatch \
        --job-name="$JOB_NAME" \
        --output="$LOG_DIR/%A_%a.out" \
        --error="$LOG_DIR/%A_%a.err" \
        --array="0-$ARRAY_MAX" \
        --partition=cpu_p \
        --qos=cpu_normal \
        --nodes=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem=16G \
        --time=24:00:00 \
        "$0" \
        --method "$METHOD" \
        --data_id "$TAG" \
        --idx_map "$IDX_MAP"
    exit 0
fi

[[ -n "$IDX_MAP" ]] || {
    echo "--idx_map is required inside the array job" >&2
    exit 1
}

IFS=',' read -r -a IDX_ARRAY <<< "$IDX_MAP"
TASK_SLOT="${SLURM_ARRAY_TASK_ID}"
[[ "$TASK_SLOT" =~ ^[0-9]+$ ]] || {
    echo "Invalid SLURM_ARRAY_TASK_ID: $TASK_SLOT" >&2
    exit 1
}
[[ "$TASK_SLOT" -lt "${#IDX_ARRAY[@]}" ]] || {
    echo "SLURM_ARRAY_TASK_ID=$TASK_SLOT out of bounds for idx map: $IDX_MAP" >&2
    exit 1
}

FIT_IDX="${IDX_ARRAY[$TASK_SLOT]}"
validate_int_list "$FIT_IDX"

IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"

case "$FAM_NAME" in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)" >&2; exit 1 ;;
esac

SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"

mkdir -p "$LOG_DIR"
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
    echo 'Requested idx map:' '$IDX_MAP'
    echo 'Array slot:' '${SLURM_ARRAY_TASK_ID}'
    echo 'Fitting dataset idx:' '$FIT_IDX'

    cd '$HOME/metabeta/metabeta/simulation'

    python fit.py \
      --size '$SIZE' \
      --family '$FAMILY' \
      --ds_type '$DS_TYPE' \
      --idx '$FIT_IDX' \
      --method '$METHOD'
  "

rm -rf "$JOB_TMPDIR"
