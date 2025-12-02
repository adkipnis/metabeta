gpu_boilerplate = '''
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_p
#SBATCH --qos gpu_priority
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
'''

def write(path: str, content: str):
    with open(path, 'w') as f:
        f.write(content)

def train(d: int, q: int, i: int = 100) -> str:
    out = '#!/bin/bash'
    out += f'''
#SBATCH --job-name=train-{d}-{q}
#SBATCH --output=logs/train-{d}-{q}/%j.out
#SBATCH --error=logs/train-{d}-{q}/%j.err'''
    out += gpu_boilerplate
    out += f'''
source $HOME/.bashrc
conda activate mb
cd %HOME/metabeta/metabeta
python train.py -d {d} -q {q} -l 0 -i {i} --c_tag default'''
    return out

if __name__ == '__main__':
    pairs = [(2,2), (3,1), (4,1), (5,2), (8,3), (12, 4)]
    for d,q in pairs:
        write(f'train-{d}-{q}.sh', train(d,q))

