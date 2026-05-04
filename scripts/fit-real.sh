#!/bin/bash
# Submit NUTS or ADVI fits for all real-data source batches.
# Submits one SLURM array job (indices 0–31) per source without waiting.
#
# Usage: bash fit-real.sh --data_id <size-family> --method <nuts|advi>
# Example: bash fit-real.sh --data_id small-n --method nuts

set -euo pipefail

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        --method)  METHOD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${TAG:-}" ]]    && { echo "Usage: $0 --data_id <size-family> --method <nuts|advi>"; exit 1; }
[[ -z "${METHOD:-}" ]] && { echo "Usage: $0 --data_id <size-family> --method <nuts|advi>"; exit 1; }

case "$METHOD" in
    nuts) FIT_SCRIPT="fit-nuts.sh" ;;
    advi) FIT_SCRIPT="fit-advi.sh" ;;
    *) echo "Unknown method: $METHOD (use nuts or advi)"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd "$SCRIPT_DIR/../metabeta/outputs/real" 2>/dev/null && pwd || echo "$HOME/metabeta/metabeta/outputs/real")"

SOURCES=(
    "557_analcatdata_apnea1__grp_subject"
    "sleep__grp_group"
    "gcse__grp_group"
    "london__grp_group"
    "math__grp_group"
    "orthodont__grp_group"
    "oxboys__grp_group"
    "ergostool__grp_group"
    "machines__grp_group"
    "oats__grp_group"
    "pixel__grp_group"
    "hsb82__grp_group"
    "chem97__grp_group"
    "theoph__grp_group"
    "orange__grp_group"
    "indometh__grp_group"
)

IFS='-' read -r SIZE FAM_NAME <<< "$TAG"

submitted=0
for SOURCE in "${SOURCES[@]}"; do
    SOURCE_SHORT="${SOURCE%%__*}"
    DATA_ID="${SIZE}-${FAM_NAME}-${SOURCE_SHORT}"
    SRCDIR="${REAL_DIR}"

    if [[ ! -f "${SRCDIR}/${DATA_ID}/test.npz" ]]; then
        echo "Skipping ${DATA_ID}: test.npz not found (run generate-sub.sh first)"
        continue
    fi

    echo "Submitting ${METHOD} array job for ${DATA_ID}"
    sbatch \
        --array=0-31 \
        "${SCRIPT_DIR}/${FIT_SCRIPT}" \
        --data_id "${DATA_ID}" \
        --srcdir "${SRCDIR}"
    (( submitted++ ))
done

echo "Submitted ${submitted} array jobs (32 tasks each) for ${TAG} / ${METHOD}."
