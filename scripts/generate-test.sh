#!/bin/bash

#SBATCH --job-name=generate-test
#SBATCH --output=logs/generate-test/generate_%j.out
#SBATCH --error=logs/generate-test/generate_%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

while [[ $# -gt 0 ]]; do
    case $1 in
        --d_tag) D_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$D_TAG" ]] && { echo "Usage: $0 --d_tag <tag>"; exit 1; }


source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag "${D_TAG}" --partition valid
python generate.py --d_tag "${D_TAG}" --partition test
