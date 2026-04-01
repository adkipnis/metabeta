#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --output=logs/generate/generate_%A_%a.out
#SBATCH --error=logs/generate/generate_%A_%a.err
#SBATCH --array=0-499

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

N_EPOCHS=500
CHUNK_SIZE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --d_tag) D_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$D_TAG" ]] && { echo "Usage: $0 --d_tag <tag>"; exit 1; }

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

START_EPOCH=1
BEGIN=$(( START_EPOCH + SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( BEGIN + CHUNK_SIZE - 1 ))

cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag "${D_TAG}" --partition train --begin ${BEGIN} --epochs ${END}
