cpu_boilerplate = '''
#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
'''

setup_boilerplate = '''
# setup tmp for job to avoid pytensor collision
JOB_TMPDIR="$HOME/tmp/pytensor_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export PYTENSOR_FLAGS="base_compiledir=$JOB_TMPDIR"

source $HOME/.bashrc
conda activate mb
cd $HOME/metabeta/metabeta/data
'''


def write(path: str, content: str):
    with open(path, 'w') as f:
        f.write(content)

def gen_synth(d: int, q: int) -> str:
    out = '#!/bin/bash'
    out += f'''
#SBATCH --job-name=gen-synth-{d}-{q}
#SBATCH --output=logs/gen-synth-{d}-{q}/%j.out
#SBATCH --error=logs/gen-synth-{d}-{q}/%j.err'''
    out += cpu_boilerplate
    out += setup_boilerplate
    out += f'''
python generate.py -d {d} -q {q} -b -1 --d_tag synth --slurm'''
    return out

def gen_specific(name: str, d: int, q: int) -> str:
    out = '#!/bin/bash'
    out += f'''
#SBATCH --job-name=gen-{name}-{d}-{q}
#SBATCH --output=logs/gen-{name}-{d}-{q}/%j.out
#SBATCH --error=logs/gen-{name}-{d}-{q}/%j.err'''
    out += cpu_boilerplate
    out += setup_boilerplate
    out += f'''
python generate.py -d {d} -q {q} -b -1 --d_tag {name} --sub --slurm'''
    return out


if __name__ == '__main__':
    synth_pairs = [(3,1), (5,2), (8,3), (12, 4)]
    for d,q in synth_pairs:
        write(f'gen-synth-{d}-{q}.sh', gen_synth(d,q))

    semi_pairs = [('sleep',2,2), ('gcse',3,1), ('collins',3,1), ('london',4,1), ('math',5,2),
                  ('titanic',8,3), ('schooling',8,3), ('churn',12,4), ('news',12,4),]
    for name, d,q in semi_pairs:
        write(f'gen-{name}-{d}-{q}.sh', gen_specific(name,d,q))

