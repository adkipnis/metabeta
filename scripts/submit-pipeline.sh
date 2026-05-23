#!/bin/bash

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --family) FAMILY="$2"; shift 2 ;;
        --size)   SIZE="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${FAMILY:-}" || -z "${SIZE:-}" ]] && {
    echo "Usage: $0 --family <n|b|p> --size <size>"
    exit 1
}

case "$FAMILY" in
    n|b|p) ;;
    *) echo "Unknown family: $FAMILY (use n, b, or p)"; exit 1 ;;
esac

SAMPLED_ID="${SIZE}-${FAMILY}-sampled"
MIXED_ID="${SIZE}-${FAMILY}-mixed"

cd "$SCRIPTS_DIR"

# Steps 1 & 2: independent
JOB1=$(sbatch --parsable generate-test.sh  --data_id "$SAMPLED_ID")
JOB2=$(sbatch --parsable generate-quick.sh --data_id "$MIXED_ID")
echo "Submitted generate-test  (sampled): $JOB1"
echo "Submitted generate-quick (mixed):   $JOB2"

# Steps 3, 5, 6: all wait on step 1 (sampled data); step 4 is independent
JOB3=$(sbatch --parsable --dependency=afterok:"$JOB1" \
    fit-nuts.sh --data_id "$SAMPLED_ID" --partition valid)
echo "Submitted fit-nuts valid (after $JOB1): $JOB3"

JOB4=$(sbatch --parsable generate-bulk.sh --data_id "$MIXED_ID")
echo "Submitted generate-bulk  (mixed):       $JOB4"

JOB5=$(sbatch --parsable --dependency=afterok:"$JOB1" \
    fit-nuts.sh --data_id "$SAMPLED_ID" --partition test)
echo "Submitted fit-nuts test  (after $JOB1): $JOB5"

JOB6=$(sbatch --parsable --dependency=afterok:"$JOB1" \
    fit-advi.sh --data_id "$SAMPLED_ID" --partition test)
echo "Submitted fit-advi test  (after $JOB1): $JOB6"

# Curate valid: after fit-nuts valid
JOB7=$(sbatch --parsable --dependency=afterok:"$JOB3" \
    curate-fits.sh --data_id "$SAMPLED_ID" --partition valid)
echo "Submitted curate valid   (after $JOB3): $JOB7"

# Curate test: after both test fits
JOB8=$(sbatch --parsable --dependency=afterok:"$JOB5":"$JOB6" \
    curate-fits.sh --data_id "$SAMPLED_ID" --partition test)
echo "Submitted curate test    (after $JOB5,$JOB6): $JOB8"

# Precompute sampled valid+test: after both curation jobs
JOB9=$(sbatch --parsable --array=0-1 --dependency=afterok:"$JOB7":"$JOB8" \
    precompute.sh --family "$FAMILY" --size "$SIZE")
echo "Submitted precompute sampled valid+test (after $JOB7,$JOB8): $JOB9"

# Precompute mixed train: after both generation jobs (valid/test not generated here)
JOB10=$(sbatch --parsable --array=2-8001 --dependency=afterany:"$JOB2":"$JOB4" \
    precompute.sh --family "$FAMILY" --size "$SIZE")
echo "Submitted precompute mixed train        (after $JOB2,$JOB4): $JOB10"

echo ""
echo "Pipeline submitted for family=${FAMILY}, size=${SIZE}"
echo "  $JOB1   generate-test  (sampled)"
echo "  $JOB2   generate-quick (mixed)"
echo "  $JOB3   fit-nuts valid (after $JOB1)"
echo "  $JOB4   generate-bulk  (mixed, independent)"
echo "  $JOB5   fit-nuts test  (after $JOB1)"
echo "  $JOB6   fit-advi test  (after $JOB1)"
echo "  $JOB7   curate valid   (after $JOB3)"
echo "  $JOB8   curate test    (after $JOB5,$JOB6)"
echo "  $JOB9   precompute sampled valid+test (after $JOB7,$JOB8)"
echo "  $JOB10  precompute mixed train        (after $JOB2,$JOB4)"
