from dataclasses import dataclass
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bambi as bmb

from metabeta.utils.io import datasetFilename


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit hierarchical datasets with Bambi.')
    # data dimensions
    parser.add_argument('-d', '--max_d', type=int, default=3, help='Maximum number of fixed effects (intercept + slopes) to draw per linear model (default = 16).')
    parser.add_argument('-q', '--max_q', type=int, default=1, help='Maximum number of random effects (intercept + slopes) to draw per linear model (default = 4).')
    parser.add_argument('--min_m', type=int, default=5, help='Minimum number of groups (default = 5).')
    parser.add_argument('--max_m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('--min_n', type=int, default=10, help='Minimum number of samples per group (default = 10).')
    parser.add_argument('--max_n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    # partitions and sources
    parser.add_argument('--type', type=str, default='toy', help='Type of predictors [toy, flat, scm, sampled], (default = toy)')
    parser.add_argument('--source', type=str, default='all', help='Source dataset if type==sampled (default = all)')
    # parser.add_argument('--loop', action='store_true', help='Loop dataset sampling instead of parallelizing it with joblib (default = False)')
    return parser.parse_args()


@dataclass
class Fitter:
    cfg: argparse.Namespace
    outdir: Path = Path('..', 'outputs', 'data')

    def __post_init__(self):
        self.cfg.partition = 'test'
        filename = datasetFilename(self.cfg)
        path = Path(self.outdir, filename)
        self.load(path)

    def load(self, path: Path) -> None:
        assert path.exists(), f'{path} does not exist'
        with np.load(path, allow_pickle=True) as batch:
            batch = dict(batch)
        self.batch = batch

    def __len__(self):
        return len(self.batch['X'])

    def _get(self, idx: int) -> dict[str, np.ndarray]:
        ''' extract a single dataset from batch and remove padding '''
        assert 0 <= idx < len(self), 'idx out of bounds'
        ds = {k: v[idx] for k,v in self.batch.items()}

        # --- unpad
        d, q, m, n = ds['d'], ds['q'], ds['m'], ds['n']

        # observations
        ds['y'] = ds['y'][:n]
        ds['X'] = ds['X'][:n, :d]
        ds['groups'] = ds['groups'][:n]
        ds['ns'] = ds['ns'][:m]

        # hyperparams
        ds['nu_ffx'] = ds['nu_ffx'][:d]
        ds['tau_ffx'] = ds['tau_ffx'][:d]
        ds['tau_rfx'] = ds['tau_rfx'][:q]

        # params
        ds['ffx'] = ds['ffx'][:d]
        ds['sigma_rfx'] = ds['sigma_rfx'][:q]
        ds['rfx'] = ds['rfx'][:m, :q]
        return ds

    def _pandify(self, ds: dict[str, np.ndarray]) -> pd.DataFrame:
        ''' get observations as dataframe '''
        n, d = ds['n'], ds['d']
        df = pd.DataFrame(index=range(n))
        df['i'] = ds['groups']
        df['y'] = ds['y'][:, None]
        for j in range(1, d):
            df[f'x{j}'] = ds['X'][..., j]
        return df

