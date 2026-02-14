import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from metabeta.simulation.sgld import SGLD
from metabeta.utils.preprocessing import checkConstant
from metabeta.utils.sampling import sampleCounts, counts2groups


# cached database
logger = logging.getLogger(__name__)
DATA_PATH = (Path(__file__).resolve().parent / '..' / 'datasets' / 'preprocessed').resolve()
# DATA_PATH = Path('..', 'datasets', 'preprocessed')
VAL_PATHS = list(Path(DATA_PATH, 'validation').glob('*.npz'))
TEST_PATHS = list(Path(DATA_PATH, 'test').glob('*.npz'))
PATHS = VAL_PATHS + TEST_PATHS
DATABASE: list[dict] | None = None


def getDatabase() -> list[dict]:
    # lazyload database on first use
    global DATABASE
    if DATABASE is None:
        DATABASE = [loadDataset(p) for p in PATHS]
    return DATABASE


def loadDataset(source: Path) -> dict:
    """wrapper for loading preprocessed npz dataset"""
    restore = lambda v: v.item() if v.shape == () else v

    # load preprocessed dataset
    assert source.exists(), 'source does not exist'
    data = np.load(source, allow_pickle=True)
    data = {k: restore(data[k]) for k in data.files if restore(data[k]) is not None}

    # check dims
    assert len(data['X']) == data['n'], 'dim mismatch (X, n)'
    groups = data.get('groups')
    if groups is not None:
        assert len(groups) == len(data['X']), 'dim mismatch (group, X)'
        assert len(np.unique(groups)) == data['m'], 'dim mismatch (group, m)'
        assert data['n'] == data['ns'].sum(), 'dim mismatch (n, ns)'

    return data


@dataclass
class Emulator:
    """class for sampling a design matrix and groups from source dataset"""

    rng: np.random.Generator
    source: str
    use_sgld: bool = True

    def __post_init__(self):
        if isinstance(self.rng, np.random.SeedSequence):
            self.rng = np.random.default_rng(self.rng)
        if self.use_sgld:
            self.sgld = SGLD()

    def _pull(self, d: int, m: int):
        # get dataset from database with matching dims
        database = getDatabase()
        if self.source == 'all':
            subset = [ds for ds in database if d <= ds['d'] and m <= ds.get('m', float('inf'))]
            n_ds = len(subset)
            idx = self.rng.integers(0, n_ds)
            self.ds = subset[idx]
        else:
            path0 = Path(DATA_PATH, 'test', f'{self.source}.npz')
            path1 = Path(DATA_PATH, 'validation', f'{self.source}.npz')
            if path0 in PATHS:
                idx = PATHS.index(path0)
            elif path1 in PATHS:
                idx = PATHS.index(path1)
            else:
                raise ValueError(f'{self.source} not in known paths.')
            self.ds = database[idx]
            assert d <= self.ds['d'] and m <= self.ds.get('m', float('inf')), 'dimension mismatch'

    def _subset(
        self,
        ds: dict,
        d: int,
        m: int,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ds['X'].copy()
        y = ds['y'].copy()

        # subset features
        idx_feat = self.rng.permutation(x.shape[1])[: d - 1]
        x = x[:, idx_feat]

        # subset observations
        ns = sampleCounts(self.rng, n, m)
        n = int(ns.sum())
        idx_obs = self.rng.permutation(len(x))[:n]
        x = x[idx_obs]
        y = y[idx_obs]
        return x, y, ns

    def _subsetGrouped(
        self,
        ds: dict,
        d: int,
        m: int,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ds['X'].copy()
        y = ds['y'].copy()

        # subset features
        idx_feat = self.rng.permutation(x.shape[1])[: d - 1]
        x_ = x[:, idx_feat]

        # hierarchically subset members / observations per member
        members = self.rng.permutation(ds['m'])[:m]
        ns = sampleCounts(self.rng, n, m)
        ns = np.minimum(ds['ns'][members], ns)   # avoid oversampling

        # subset observations per member
        member_mask = np.zeros(len(x)).astype(bool)
        for i, member in enumerate(members):
            idx_member = np.where(ds['groups'] == member)[0]
            n_i = len(idx_member)
            idx_obs = self.rng.permutation(n_i)[: ns[i]]
            member_mask[idx_member[idx_obs]] = True
        x_ = x_[member_mask]
        y_ = y[member_mask]
        return x_, y_, ns

    def sample(
        self,
        d: int,  # number of predictors
        ns: np.ndarray,  # n_obs per group (only used as a starting point)
    ) -> dict[str, np.ndarray]:
        # dims
        m = len(ns)   # number of groups
        n = int(ns.sum())   # total number of observations

        # pull source
        self._pull(d, m)
        source_is_grouped = 'm' in self.ds

        # check source dims
        if n > self.ds['n']:
            n = self.ds['n']
            logger.info(f'not enough observations in source, setting n={n}')
        if source_is_grouped:
            if m > self.ds['m']:
                m = self.ds['m']
                logger.info(f'not enough groups in source, setting m={m}')

        # subsample observations
        subset_fn = self._subsetGrouped if source_is_grouped else self._subset
        x, _, ns = subset_fn(self.ds, d, m, n)
        while checkConstant(x).any():
            x, _, ns = subset_fn(self.ds, d, m, n)

        # emulate predictors using SGLD
        if self.use_sgld:
            x = self.sgld(x, rng=self.rng)

        # get groups from counts
        groups = counts2groups(ns)

        # add intercept
        ones = np.ones_like(x[:, 0:1])
        x = np.concatenate([ones, x], axis=-1)
        return {'X': x, 'ns': ns, 'groups': groups}


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from joblib import Parallel, delayed

    b = 128
    n = 200
    m = 10
    d = 3
    seed = 0
    source = 'math'

    getDatabase()   # instantiate

    rng = np.random.default_rng(seed)
    main_seed = np.random.SeedSequence(seed)
    seeds = main_seed.spawn(b)

    ns = sampleCounts(rng, n, m)
    groups = counts2groups(ns)
    emulator = Emulator(rng, source)
    out = emulator.sample(d, ns)

    # --- sequential
    t0 = time.perf_counter()
    for rng in tqdm(seeds):
        Emulator(rng, source).sample(d, ns)   # type: ignore
    t1 = time.perf_counter()
    print(f'\n{t1-t0:.2f}s used for sequential sampling.')

    # --- parallel (only useful when using SGLD)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1)(
        delayed(Emulator(rng, source).sample)(d, ns) for rng in tqdm(seeds)  # type: ignore
    )
    t1 = time.perf_counter()
    print(f'\n{t1-t0:.2f}s used for parallel sampling.')
