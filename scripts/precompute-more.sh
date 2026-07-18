#!/bin/bash
# Array layout: task ID 0 maps to training epoch 8001.

#SBATCH --job-name=precompute
#SBATCH --output=logs/precompute/%A_%a.out
#SBATCH --error=logs/precompute/%A_%a.err
#SBATCH --array=0-3999

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00

CHUNK_SIZE=1
START_EPOCH=8001
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

EPOCH=$(( START_EPOCH + SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

python precompute.py --size "${SIZE}" --family ${FAMILY} --ds_type mixed --partition train --epoch ${EPOCH} ${OVERWRITE}
