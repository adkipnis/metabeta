#!/bin/bash
# Power analysis (experiments/evaluation/power_analysis.py): fit one dataset of its NUTS batch per
# array task, at one level of fit.py's ladder. Submit from the root of the checkout that exported
# the batch (a worktree, so ~/metabeta stays untouched), one array per level:
#   mkdir -p logs/nuts-power
#   n=$(( $(wc -l < metabeta/outputs/data/power-binary/units.csv) - 2 ))
#   for level in 0 1 2; do sbatch --array=0-$n scripts/power-nuts.sh --name binary --level $level; done
# then check the fits and reintegrate them, which also composes the nuts tag:
#   uv run python metabeta/simulation/check.py --partition test --data_id power-binary

#SBATCH --job-name=nuts-power
#SBATCH --output=logs/nuts-power/%A_%a.out
#SBATCH --error=logs/nuts-power/%A_%a.err

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

while [[ $# -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift 2 ;;
        --level) LEVEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${NAME:-}" || -z "${LEVEL:-}" ]] && {
    echo "Usage: sbatch --array=0-<n> $0 --name <run name> --level <0|1|2>"
    exit 1
}

# sbatch runs a spooled copy of this script, so the checkout is the submit directory
export METABETA_REPO="$SLURM_SUBMIT_DIR"
source "$METABETA_REPO/scripts/pymc-run.sh"
runPymc python metabeta/simulation/fit.py --config "metabeta/outputs/data/power-${NAME}/config.yaml" \
    --idx "$SLURM_ARRAY_TASK_ID" --method nuts --level "$LEVEL" --partition test
