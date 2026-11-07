#!/bin/bash
# Group budget on simulated data (experiments/evaluation/group_budget.py): one (family, cell)
# per job on a CPU node, then the grids. Needs the reintegrated fits of the cell
# (metabeta/simulation/check.py --partition test). Partition and QOS go on the command line:
#   sbatch -p cpu_p -q cpu_normal -c 16 --mem=64G -t 06:00:00 scripts/group-budget.sh run n gb200x3000
#   sbatch -p cpu_p -q cpu_normal -c 2 --mem=8G -t 00:20:00 scripts/group-budget.sh collect
# The code of METABETA_ROOT runs (default ~/metabeta), ahead of the venv's editable install.

#SBATCH --job-name=group-budget
#SBATCH --output=logs/group-budget/%j.out
#SBATCH --error=logs/group-budget/%j.err
#SBATCH --nodes=1

set -euo pipefail
METABETA_ROOT="${METABETA_ROOT:-$HOME/metabeta}"
cd "$METABETA_ROOT"
source "$HOME/metabeta/.venv/bin/activate"
export PYTHONPATH="$METABETA_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python -c 'import metabeta; print("metabeta:", metabeta.__file__)'

case "$1" in
    run) python experiments/evaluation/group_budget.py run --family "$2" --cell "$3" ;;
    collect) python experiments/evaluation/group_budget.py collect ;;
    *) echo "Unknown stage: $1 (use run or collect)" >&2; exit 1 ;;
esac
