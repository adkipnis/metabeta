#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=gen-toy
#SBATCH --output=logs/gen-toy/%j.out
#SBATCH --error=logs/gen-toy/%j.err

source $HOME/.bashrc
cd $HOME/metabeta/metabeta/simulation
python generate.py --d_tag toy --partition all --epochs 100
