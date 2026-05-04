#!/bin/bash
# Generate 32 real-data subsets per source dataset for a given regime.
#
# Usage: bash generate-sub.sh --data_id <size-family>
# Example: bash generate-sub.sh --data_id small-n
#
# Outputs: metabeta/outputs/real/<size>-<fam>-<source>/test.npz
# Run from the metabeta/simulation directory or via srun.

set -eo pipefail

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${TAG:-}" ]] && {
    echo "Usage: $0 --data_id <size-family>  (e.g. small-n, medium-n)"
    exit 1
}

SIZE="${TAG%%-*}"
FAM_NAME="${TAG##*-}"

case "$FAM_NAME" in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../metabeta/simulation"

for SOURCE in "${SOURCES[@]}"; do
    echo "Generating subsets for source: $SOURCE"
    python generate.py \
        --size "${SIZE}" \
        --family "${FAMILY}" \
        --ds_type real \
        --source "${SOURCE}" \
        --bs_test 32 \
        --partition test
done

echo "Done. Generated ${#SOURCES[@]} source batches for ${TAG}."
