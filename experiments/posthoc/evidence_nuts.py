"""NUTS reference draws for the reduced model of the evidence check's nested comparison.

evidence.py compares ln BF (random slope vs. random intercept only) against a reference whose
reduced-model half was bridge sampling on MB-IMH draws, i.e. built from the method under test.
This fits the reduced model (evidence.reducedItem: slopes dropped from the random effects, kept
as fixed effects) with NUTS at the reference budget instead: fit.py's level 2 with its default
seed, via its Fitter, so priors, settings and diagnostics match the full-model reference. Only
the global draws are kept; evidence.py bridges them.

One process per dataset, in the PyMC container (scripts/evidence-nuts.sh):
    python experiments/posthoc/evidence_nuts.py --tasks > tasks.txt  # nested rows, converged ref
    python experiments/posthoc/evidence_nuts.py --family poisson --size huge --idx 20
writes {out_dir}/nuts_red/{family}_{size}_{idx:03d}.npz
"""

import argparse
from pathlib import Path

import numpy as np

from metabeta.utils.experiments import experimentResultsPath
from metabeta.utils.padding import unpad

OUT_DIR = experimentResultsPath('evidence')
CSV_FAMILIES = {'normal': '', 'bernoulli': 'bernoulli_k8_', 'poisson': 'poisson_k8_'}
SIZES = ('small', 'medium', 'large', 'huge')


def outPath(family: str, size: str, idx: int) -> Path:
    return OUT_DIR / 'nuts_red' / f'{family}_{size}_{idx:03d}.npz'


def tasks() -> None:
    """One line per nested row whose full-model reference converged (the rows the tables keep)."""
    import pandas as pd

    for family, prefix in CSV_FAMILIES.items():
        for size in SIZES:
            df = pd.read_csv(OUT_DIR / f'{prefix}{size}_test_n512.csv')
            rows = df[df['logbf_ref'].notna() & df['nuts_converged'].astype(bool)]
            for idx in rows['idx']:
                print(family, size, idx)


def fitReduced(family: str, size: str, idx: int) -> None:
    path = outPath(family, size, idx)
    if path.exists():  # resubmitted array: keep finished fits
        print(f'exists {path}')
        return
    from metabeta.simulation.fit import Fitter  # PyMC stack; evidence.py imports only outPath

    cfg = argparse.Namespace(
        method='nuts',
        level=2,
        partition='test',
        epoch=None,
        idx=idx,
        seed=42,
        mp_ctx='forkserver',
        diagonal=False,
        data_id=f'{size}-{family[0]}-sampled',
    )
    fitter = Fitter(cfg)
    ds = {k: v[idx] for k, v in fitter.batch.items()}
    ds['q'] = np.array(1, dtype=ds['q'].dtype)  # buildPymc's Z = X[:, :q]: intercept only
    fitter.ds = unpad(ds, {k: int(ds[k]) for k in 'dqmn'})
    fit = fitter._fitNuts()['nuts2']
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in fit.items() if k != 'nuts2_rfx'})
    print(f'saved {path}')


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tasks', action='store_true', help='print the task list (family size idx) instead of fitting')
    p.add_argument('--family', choices=list(CSV_FAMILIES))
    p.add_argument('--size', choices=SIZES)
    p.add_argument('--idx', type=int)
    return p.parse_args()
# fmt: on


if __name__ == '__main__':
    args = setup()
    if args.tasks:
        tasks()
    else:
        fitReduced(args.family, args.size, args.idx)
