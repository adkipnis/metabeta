#!/bin/bash

#SBATCH --job-name=sub
#SBATCH --output=logs/sub/%A_%a.out
#SBATCH --error=logs/sub/%A_%a.err
#SBATCH --array=0-15

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -eo pipefail

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${TAG:-}" ]] && {
    echo "Usage: sbatch $0 --data_id <size-family>  (e.g. small-n)"
    exit 1
}

IFS='-' read -r SIZE FAM_NAME <<< "$TAG"

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

SOURCE="${SOURCES[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p logs/sub

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/simulation

echo "Generating subsets for source: $SOURCE"
python generate.py \
    --size "${SIZE}" \
    --family "${FAMILY}" \
    --ds_type real \
    --source "${SOURCE}" \
    --bs_test 32 \
    --partition test
