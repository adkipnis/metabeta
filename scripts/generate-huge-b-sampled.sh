#!/bin/bash

#SBATCH --job-name=generate-huge-b-sampled
#SBATCH --output=logs/generate/huge-b-sampled_%j.out
#SBATCH --error=logs/generate/huge-b-sampled_%j.err

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag huge-b-sampled --partition all --epochs 50
