#!/bin/bash

# E2 prior sensitivity: fit one prior-grid point per array task. Export the grid with
#   python experiments/evaluation/prior_sensitivity.py --rfx $rfx --stages export
# then submit its Normal slope-prior points at each NUTS level:
#   pts=$(python3 -c "import csv; print(','.join(r['point'] for r in csv.DictReader(open(
#       'metabeta/outputs/data/e2-${ds}-${rfx}/grid.csv')) if r['ffx_family'] == 'normal'))")
#   for level in 0 1 2; do
#       sbatch --array=$pts scripts/fit-nuts-prior-grid.sh --dataset $ds --rfx $rfx --level $level
#   done

#SBATCH --job-name=nuts-e2
#SBATCH --output=logs/nuts-e2/%A_%a.out
#SBATCH --error=logs/nuts-e2/%A_%a.err

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

METHOD="nuts"
LEVEL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --rfx) RFX="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --level) LEVEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${DATASET:-}" || -z "${RFX:-}" ]] && {
    echo "Usage: $0 --dataset <sleep|cbpp|salamanders> --rfx <slope|intercept> [--method nuts|advi] [--level 0|1|2]"
    exit 1
}

CONFIG="$HOME/metabeta/metabeta/outputs/data/e2-${DATASET}-${RFX}/config.yaml"

mkdir -p logs/nuts-e2
source "$HOME/metabeta/scripts/pymc-run.sh"
runPymc python metabeta/simulation/fit.py --config "$CONFIG" --idx "$SLURM_ARRAY_TASK_ID" \
    --method "$METHOD" --level "$LEVEL" --partition test
