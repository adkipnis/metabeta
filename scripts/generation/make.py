cpu_boilerplate = '''
#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
'''


def write(path: str, content: str):
    with open(path, 'w') as f:
        f.write(content)

def gen_train(d: int, q: int, i: int = 100) -> str:
    out = '#!/bin/bash'
    out += f'''
#SBATCH --job-name=gen-train-{d}-{q}
#SBATCH --output=logs/gen-train-{d}-{q}/%j.out
#SBATCH --error=logs/gen-train-{d}-{q}/%j.err'''
    out += cpu_boilerplate
    out += f'''
source $HOME/.bashrc
conda activate mb
cd $HOME/metabeta/metabeta/data
python generate.py -d {d} -q {q} -b 0 -i {i} --semi'''
    return out


def gen_test(d: int, q: int) -> str:
    out = '#!/bin/bash'
    out += f'''
#SBATCH --job-name=gen-test-{d}-{q}
#SBATCH --output=logs/gen-test-{d}-{q}/%j.out
#SBATCH --error=logs/gen-test-{d}-{q}/%j.err'''
    out += cpu_boilerplate
    out += f'''
source $HOME/.bashrc
conda activate mb
cd $HOME/metabeta/metabeta/data
python generate.py -d {d} -q {q} -b -1 --semi --slurm'''
    return out


def gen_sub(d: int, q: int) -> str:
    out = '#!/bin/bash'
    out += f'''
#SBATCH --job-name=gen-sub-{d}-{q}
#SBATCH --output=logs/gen-sub-{d}-{q}/%j.out
#SBATCH --error=logs/gen-sub-{d}-{q}/%j.err'''
    out += cpu_boilerplate
    out += f'''
source $HOME/.bashrc
conda activate mb
cd $HOME/metabeta/metabeta/data
python generate.py -d {d} -q {q} -b -1 --sub --slurm'''
    return out


if __name__ == '__main__':
    pairs = [(2,2), (3,1), (4,1), (5,2), (8,3), (12, 4)]
    for d,q in pairs:
        # write(f'gen-train-{d}-{q}.sh', gen_train(d,q))
        write(f'gen-test-{d}-{q}.sh', gen_test(d,q))
        # write(f'gen-sub-{d}-{q}.sh', gen_sub(d,q))

