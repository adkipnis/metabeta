#!/bin/bash
# Failure rate of Api.nuts at its defaults (experiments/posthoc/nuts_failures.py): one array task
# per cell, eight in all, each writing experiments/results/nuts-failures/{family}_{size}.csv.
# Submit from the root of the checkout to run (a worktree, so ~/metabeta stays untouched); a task
# that hits its time limit resumes when resubmitted with --array=<its index>:
#   mkdir -p logs/nuts-failures
#   sbatch scripts/nuts-failures.sh
# Extra arguments go to the driver (e.g. --n-datasets 2).

#SBATCH --job-name=nuts-failures
#SBATCH --output=logs/nuts-failures/%A_%a.out
#SBATCH --error=logs/nuts-failures/%A_%a.err
#SBATCH --array=0-7

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

# sbatch runs a spooled copy of this script, so the checkout is the submit directory
export METABETA_REPO="$SLURM_SUBMIT_DIR"
source "$METABETA_REPO/scripts/pymc-run.sh"
runPymc python experiments/posthoc/nuts_failures.py --cell "$SLURM_ARRAY_TASK_ID" "$@"
