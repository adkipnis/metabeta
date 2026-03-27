#!/bin/bash

#SBATCH --job-name=gen-simple
#SBATCH --output=logs/gen-simple/%j.out
#SBATCH --error=logs/gen-simple/%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag simple-n --partition all --epochs 50
