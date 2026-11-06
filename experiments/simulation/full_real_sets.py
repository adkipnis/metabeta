"""Whole real datasets beyond the checkpoints' limits, each as a one-dataset test set.

The real-world test sets draw their designs from the held-out collection with the
hierarchical bootstrap (``metabeta.simulation.emulator.Subsampler``), so InstEval, muscatine
and Chem97, which exceed the group budget, enter them only as subsamples (muscatine not at
all: every group has fewer than ``min_n`` rows). This script packages each of them whole, with
all groups, rows and predictors and a random intercept (q=1), applying the Subsampler's
transforms and Bambi's default priors, as ``{size}-{family}-full/test.npz``; ``size`` is the
fixed-effect band of the dataset's d. Rows are ordered by group, as the batch layout requires.

NUTS reference fits (one array task each):
    scripts/fit-ref.sh --method nuts --level 2 --data_id large-n-full --n_datasets 1

Usage from repo root:
    uv run python experiments/simulation/full_real_sets.py
"""

from pathlib import Path

import numpy as np

from metabeta.simulation.emulator import loadDataset
from metabeta.simulation.generate import Generator
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.experiments import DATA_DIR, PREPROCESSED_DATA_DIR
from metabeta.utils.padding import aggregate
from metabeta.utils.preprocessing import transformPredictors
from metabeta.utils.priors import bambiDefaultPriors

# source file stem -> (data id, likelihood family)
FULL_SETS = {
    'insteval__grp_group': ('large-n-full', 0),
    'muscatine__grp_group': ('medium-b-full', 1),
    'chem97__grp_group': ('small-p-full', 2),
}
Q = 1


def fullDataset(source: Path, likelihood_family: int) -> dict[str, np.ndarray]:
    """The whole source dataset in the layout of ``Generator._genDataset`` for ds_type 'real'."""
    src = loadDataset(source)
    order = np.argsort(src['groups'], kind='stable')
    groups = src['groups'][order]
    ns = np.bincount(groups)
    assert np.array_equal(ns, src['ns']), 'group sizes differ from the stored ns'

    x = transformPredictors(src['X'][order], axis=0, exclude_binary=True, transform_counts=True)
    x = np.concatenate([np.ones((len(x), 1), dtype=x.dtype), x], axis=-1)
    y = src['y'][order]
    sd_y = max(float(y.std()), 1e-6) if likelihood_family == 0 else 1.0
    if likelihood_family == 0:
        y = y / sd_y

    d, m = x.shape[1], len(ns)
    assert d == int(src['d']), f'{source.stem}: d={d}, stored d={src["d"]}'
    out = {
        'ffx': np.full(d, np.nan),
        'sigma_rfx': np.full(Q, np.nan),
        'corr_rfx': np.full((Q, Q), np.nan),
        'rfx': np.full((m, Q), np.nan),
        **bambiDefaultPriors(d, Q, likelihood_family),
        'y': y,
        'X': x,
        'groups': groups,
        'm': np.array(m),
        'n': np.array(len(y)),
        'ns': ns,
        'd': np.array(d),
        'q': np.array(Q),
        'sd_y': np.array(sd_y),
        'source': np.array(source.stem, dtype=object),
    }
    if hasSigmaEps(likelihood_family):
        out['sigma_eps'] = np.array(np.nan)
        out['r_squared'] = np.array(np.nan)
    return out


def main() -> None:
    for stem, (data_id, family) in FULL_SETS.items():
        path = DATA_DIR / data_id / 'test.npz'
        if path.exists():
            raise FileExistsError(f'{path} exists; this script never overwrites a test set')
        ds = fullDataset(PREPROCESSED_DATA_DIR / 'test' / f'{stem}.npz', family)
        batch = Generator._castCompactTypes(aggregate([ds]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **batch, allow_pickle=True)
        print(f'{data_id}: {stem}, d={ds["d"]}, m={ds["m"]}, n={ds["n"]}, -> {path}')


if __name__ == '__main__':
    main()
