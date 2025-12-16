from pathlib import Path
import numpy as np
from dataclasses import dataclass
from metabeta.simulation.sgld import SGLD
from metabeta.simulation.utils import checkConstant, sampleCounts, counts2groups


def loadDataset(source: Path) -> dict:
    ''' wrapper for loading preprocessed npz dataset '''
    restore = lambda v: v.item() if v.shape == () else v

    # load preprocessed dataset
    assert source.exists(), 'source does not exist'
    data = np.load(source, allow_pickle=True)
    data = {k: restore(data[k])
            for k in data.files
            if restore(data[k]) is not None}

    # check dims
    assert len(data['X']) == data['n'], 'dim mismatch (X, n)'
    groups = data.get('groups')
    if groups is not None:
        assert len(groups) == len(data['X']), 'dim mismatch (group, X)'
        assert len(np.unique(groups)) == data['m'] , 'dim mismatch (group, m)'
        assert data['n'] == data['ns'].sum(), 'dim mismatch (n, ns)'

    return data

data_path = Path('..', 'datasets', 'preprocessed')
train_paths = list(Path(data_path, 'train').glob('*.npz'))
test_paths = list(Path(data_path, 'test').glob('*.npz'))
PATHS = train_paths + test_paths
DATABASE = [loadDataset(source=path) for path in PATHS]

@dataclass
class Emulator:
    ''' class for sampling a design matrix and groups from source dataset '''
    source: str
    use_sgld: bool = True
    sgld = SGLD()


    def _pull(self, d: int, m: int):
        # get dataset from database with matching dims
        if self.source == 'all':
            subset = [
                ds for ds in DATABASE
                if d <= ds['d']+1 and m <= ds.get('m', float('inf'))
            ]
            idx = int(np.random.randint(0, len(subset)))
            self.ds = subset[idx]
        else:
            path0 = Path(data_path, 'test', f'{self.source}.npz')
            path1 = Path(data_path, 'train', f'{self.source}.npz')
            if path0 in PATHS:
                idx = PATHS.index(path0)
            elif path1 in PATHS:
                idx = PATHS.index(path1)
            else:
                raise ValueError(f'{self.source} not in known paths.')
            self.ds = DATABASE[idx]
            assert d <= self.ds['d']+1 and m <= self.ds.get('m', float('inf')), 'dimension mismatch'


    def _subset(self, ds: dict, d: int, m: int, n: int,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ds['X'].copy()
        y = ds['y'].copy()

        # subset features
        idx_feat = np.random.permutation(x.shape[1])[:d-1]
        x = x[:, idx_feat]

        # subset observations
        ns = sampleCounts(n, m)
        n = int(ns.sum())
        idx_obs = np.random.permutation(len(x))[:n]
        x = x[idx_obs]
        y = y[idx_obs]
        return x, y, ns


    def _subsetGrouped(self, ds: dict, d: int, m: int, n: int,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ds['X'].copy()
        y = ds['y'].copy()

        # subset features
        idx_feat = np.random.permutation(x.shape[1])[:d-1]
        x_ = x[:, idx_feat]

        # hierarchically subset members / observations per member
        members = np.random.permutation(ds['m'])[:m]
        ns = sampleCounts(n, m)
        ns = np.minimum(ds['ns'][members], ns) # avoid oversampling

        # subset observations per member
        member_mask = np.zeros(len(x)).astype(bool)
        for i, member in enumerate(members):
            idx_member = np.where(ds['groups'] == member)[0]
            n_i = len(idx_member)
            idx_obs = np.random.permutation(n_i)[:ns[i]]
            member_mask[idx_member[idx_obs]] = True
        x_ = x_[member_mask]
        y_ = y[member_mask]
        return x_, y_, ns


    def sample(self, d: int, ns: np.ndarray) -> dict[str, np.ndarray]:
        # dims
        m = len(ns)
        n = int(ns.sum())

        # pull source
        self._pull(d, m)
        source_is_grouped = 'm' in self.ds

        # check source dims
        if source_is_grouped:
            if m > self.ds['m']:
                m = self.ds['m']
                print(f'Warning: not enough groups in source, setting m={m}')
            if n > self.ds['n']:
                n = self.ds['n']
                print(f'Warning: not enough observations in source, setting n={n}')

        # subsample observations
        subset_fn = self._subsetGrouped if source_is_grouped else self._subset
        x, y, ns = subset_fn(self.ds, d, m, n)
        while checkConstant(x).any():
            x, y, ns = subset_fn(self.ds, d, m, n)

        # emulate dataset using SGLD
        if self.use_sgld:
            x = self.sgld(x)
 
        # get groups from counts
        groups = counts2groups(ns)
 
        # add intercept
        ones = np.ones_like(x[:, 0:1])
        x = np.concat([ones, x], axis=-1)
        return {'X': x, 'y': y, 'ns': ns, 'groups': groups}


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from joblib import Parallel, delayed

    seed = 1
    _ = np.random.seed(seed)
    b = 128
    n = 200
    m = 10
    d = 3

    ns = sampleCounts(n, m)
    groups = counts2groups(ns)
    emulator = Emulator(source='math')
    out = emulator.sample(d, ns)

    # --- sequential
    t0 = time.perf_counter()
    for _ in tqdm(range(b)):
        Emulator(source='math').sample(d, ns)
    t1 = time.perf_counter()
    print(f'\n{t1-t0:.2f}s used for sequential sampling.')

    # --- parallel
    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1)(
        delayed(Emulator(source='math').sample)(d, ns) for _ in tqdm(range(b))
        )
    t1 = time.perf_counter()
    print(f'\n{t1-t0:.2f}s used for parallel sampling.')

