#!/bin/bash
# Array layout: each task ID is the training epoch to precompute.

#SBATCH --job-name=precompute
#SBATCH --output=logs/precompute/%A_%a.out
#SBATCH --error=logs/precompute/%A_%a.err
#SBATCH --array=8001-12000

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00

OVERWRITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --family)    FAM_NAME="$2";    shift 2 ;;
        --size)      SIZE="$2";        shift 2 ;;
        --overwrite) OVERWRITE="--overwrite"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "${FAM_NAME:-}" || -z "${SIZE:-}" ]] && {
    echo "Usage: $0 --family <n|b|p> --size <size> [--overwrite]"; exit 1
}

case $FAM_NAME in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/analytical

python precompute.py --size "${SIZE}" --family ${FAMILY} --ds_type mixed --partition train --epoch ${SLURM_ARRAY_TASK_ID} ${OVERWRITE}
