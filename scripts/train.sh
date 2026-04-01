#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

EPOCHS=500

while [[ $# -gt 0 ]]; do
    case $1 in
        --d_tag) D_TAG="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$D_TAG" ]] && { echo "Usage: $0 --d_tag <tag> [--epochs N] [--seed N]"; exit 1; }

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training
python train.py --name "${D_TAG}" -e ${EPOCHS} ${SEED:+--seed ${SEED}} --wandb --device cuda --no-plot
