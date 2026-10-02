"""Populate the PyTensor compile cache with the kernels of the compiled reference methods.

Fits ADVI, Pathfinder and Laplace briefly on two datasets of one test set (the first with
a random intercept only, the first with random slopes) so that a later fit of any dataset
of that likelihood family starts from the cache a user has after fitting one such GLMM.
The fits are throw-away; nothing is written. Run inside the same container and with the
same PYTENSOR_FLAGS as the campaign tasks (scripts/warm-cache.sh).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pymc as pm
from pymc_extras import fit_laplace, fit_pathfinder

from metabeta.simulation.fit import PATHFINDER_PATHS
from metabeta.utils.padding import unpad
from metabeta.utils.pymc import buildPymc

SRCDIR = Path(__file__).resolve().parents[1] / 'metabeta' / 'outputs' / 'data'


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Warm the PyTensor cache for one likelihood family.')
    parser.add_argument('--data_id', type=str, required=True, help='test set, e.g. small-n-sampled')
    return parser.parse_args()
# fmt: on


def pickDatasets(q: np.ndarray) -> list[int]:
    return [int(np.flatnonzero(q == 1)[0]), int(np.flatnonzero(q >= 2)[0])]


def warm(ds: dict) -> None:
    with buildPymc(ds, force_diagonal=False):
        approx = pm.fit(n=200, method='advi', progressbar=False, random_seed=0)
        approx.sample(10, random_seed=0)
        for paths in PATHFINDER_PATHS:
            fit_pathfinder(
                num_paths=paths, num_draws=10, random_seed=0, progressbar=False, parallel=False
            )
        fit_laplace(draws=10, random_seed=0, progressbar=False)


def main() -> None:
    cfg = setup()
    with np.load(SRCDIR / cfg.data_id / 'test.npz', allow_pickle=True) as batch:
        data = dict(batch)
    for idx in pickDatasets(data['q']):
        ds = {k: v[idx] for k, v in data.items()}
        ds = unpad(ds, {k: ds[k] for k in 'dqmn'})
        t0 = time.perf_counter()
        warm(ds)
        print(f'{cfg.data_id} dataset {idx} (q={int(ds["q"])}): {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
