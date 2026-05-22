#!/bin/bash
# Array layout:  0 → valid,  1 → test,  2+ → train epochs
# Total tasks:   2 + N_TRAIN_EPOCHS  (adjust --array accordingly)

#SBATCH --job-name=precompute
#SBATCH --output=logs/precompute/%A_%a.out
#SBATCH --error=logs/precompute/%A_%a.err
#SBATCH --array=0-8001

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=24:00:00

CHUNK_SIZE=1
START_EPOCH=1
OVERWRITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --family)    FAM_NAME="$2";    shift 2 ;;
        --size)      SIZE="$2";        shift 2 ;;
        --start)     START_EPOCH="$2"; shift 2 ;;
        --overwrite) OVERWRITE="--overwrite"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "${FAM_NAME:-}" || -z "${SIZE:-}" ]] && {
    echo "Usage: $0 --family <n|b|p> --size <size> [--start <epoch>] [--overwrite]"; exit 1
}

case $FAM_NAME in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac

NON_TRAIN=("valid" "test")
N_NON_TRAIN=${#NON_TRAIN[@]}

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/analytical

if [[ $SLURM_ARRAY_TASK_ID -lt $N_NON_TRAIN ]]; then
    PARTITION="${NON_TRAIN[$SLURM_ARRAY_TASK_ID]}"
    python precompute.py --size "${SIZE}" --family ${FAMILY} --ds_type sampled --partition "${PARTITION}" ${OVERWRITE}
else
    EPOCH=$(( START_EPOCH + (SLURM_ARRAY_TASK_ID - N_NON_TRAIN) * CHUNK_SIZE ))
    python precompute.py --size "${SIZE}" --family ${FAMILY} --ds_type mixed --partition train --epoch ${EPOCH} ${OVERWRITE}
fi
