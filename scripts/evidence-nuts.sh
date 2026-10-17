#!/bin/bash
# NUTS on the reduced (random intercept only) model of evidence.py's nested comparison: one array
# task per line "family size idx" of the task file (experiments/posthoc/evidence_nuts.py --tasks).
# Finished fits are skipped, so a resubmission only runs what is missing. Run from the repo root:
#   mkdir -p logs/evidence-nuts
#   TASKS=experiments/results/evidence/nuts_red/tasks.txt
#   sbatch --array=0-$(( $(wc -l < $TASKS) - 1 )) scripts/evidence-nuts.sh $TASKS
#SBATCH --job-name=evidence-nuts
#SBATCH --output=logs/evidence-nuts/%A_%a.out
#SBATCH --error=logs/evidence-nuts/%A_%a.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

read -r FAMILY SIZE IDX < <(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$1")
source "$HOME/metabeta/scripts/pymc-run.sh"
runPymc python experiments/posthoc/evidence_nuts.py --family "$FAMILY" --size "$SIZE" --idx "$IDX"
