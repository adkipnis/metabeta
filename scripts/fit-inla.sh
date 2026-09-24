#!/bin/bash

#SBATCH --job-name=inla
#SBATCH --output=logs/inla/%A_%a.out
#SBATCH --error=logs/inla/%A_%a.err
#SBATCH --array=0-511

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# One R-INLA fit per array task (fits/<partition>_inla_<idx>.npz), 4000 joint draws, one
# INLA thread. Runs natively, not in the apptainer image: R-INLA lives in the host R library
# (~/R/x86_64-redhat-linux-gnu-library/4.6), its GDAL/GEOS/PROJ in ~/usr (set by ~/.bashrc).
# Afterwards aggregate into <partition>.inla.npz on an interactive node (~4x the rfx block
# in RAM, 64G is ample):
#   python -m metabeta.simulation.inla --size S --family F --ds_type sampled --reintegrate

set -euo pipefail

PARTITION="test"
RE_CORRELATION="auto"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --re-correlation) RE_CORRELATION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${TAG:-}" ]] && {
    echo "Usage: $0 --data_id <size-family-ds_type> [--partition test|valid] [--re-correlation auto|diagonal]"
    exit 1
}

IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"

case "$FAM_NAME" in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac

mkdir -p logs/inla
# /etc/bashrc and activate reference unset variables; relax nounset while sourcing them
set +u
source "$HOME/.bashrc"
source "$HOME/metabeta/.venv/bin/activate"
set -u
cd "$HOME/metabeta"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo 'hostname:' "$(hostname)"
Rscript -e 'cat("INLA", as.character(packageVersion("INLA")), "\n")'

python -m metabeta.simulation.inla \
  --size "$SIZE" \
  --family "$FAMILY" \
  --ds_type "$DS_TYPE" \
  --partition "$PARTITION" \
  --idx "${SLURM_ARRAY_TASK_ID}" \
  --draws 4000 \
  --re-correlation "$RE_CORRELATION"
