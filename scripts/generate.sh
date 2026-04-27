#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --output=logs/generate/generate_%A_%a.out
#SBATCH --error=logs/generate/generate_%A_%a.err
#SBATCH --array=0-1999

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00

N_EPOCHS=2000
CHUNK_SIZE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$TAG" ]] && { echo "Usage: $0 --tag <size-family-ds_type>"; exit 1; }

IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"
case $FAM_NAME in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

START_EPOCH=1
BEGIN=$(( START_EPOCH + SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( BEGIN + CHUNK_SIZE - 1 ))

cd $HOME/metabeta/metabeta/simulation
python generate.py --size "${SIZE}" --family ${FAMILY} --ds_type "${DS_TYPE}" --partition train --begin ${BEGIN} --epochs ${END}
