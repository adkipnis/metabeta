#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --output=logs/generate/%j.out
#SBATCH --error=logs/generate/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag medium-n --partition all --epochs 50
