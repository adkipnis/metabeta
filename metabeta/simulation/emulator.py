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
    data['source'] = source

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
    min_m: int = 1
    min_n: int = 1
    max_n: int | None = None
    max_attempts: int = 20
    strict_request_filter: bool = False

    def __post_init__(self):
        if isinstance(self.rng, np.random.SeedSequence):
            self.rng = np.random.default_rng(self.rng)
        if self.min_m < 1:
            raise ValueError(f'min_m must be >= 1, but got {self.min_m}')
        if self.min_n < 1:
            raise ValueError(f'min_n must be >= 1, but got {self.min_n}')
        if self.max_n is not None and self.max_n < self.min_n:
            raise ValueError(f'max_n must be >= min_n ({self.min_n}), but got max_n={self.max_n}')
        if self.max_attempts < 1:
            raise ValueError(f'max_attempts must be >= 1, but got {self.max_attempts}')
        if self.use_sgld:
            self.sgld = SGLD()

    def _sampleCountsBounded(self, n: int, m: int, max_n: int | None = None) -> np.ndarray:
        if n < m * self.min_n:
            raise ValueError(
                f'cannot sample counts: n={n} is too small for m={m} with min_n={self.min_n}'
            )

        if max_n is None:
            max_n = self.max_n
        if max_n is not None and n > m * max_n:
            raise ValueError(
                f'cannot sample counts: n={n} is too large for m={m} with max_n={max_n}'
            )

        ns = np.full(m, self.min_n, dtype=int)
        rem = n - int(ns.sum())
        if rem == 0:
            return ns

        caps = None
        if max_n is not None:
            caps = np.full(m, max_n - self.min_n, dtype=int)

        while rem > 0:
            alpha = self.rng.uniform(2.0, 20.0)
            p = self.rng.dirichlet(np.ones(m) * alpha)
            extra = self.rng.multinomial(rem, p)
            if caps is not None:
                extra = np.minimum(extra, caps)
            added = int(extra.sum())
            if added == 0:
                idx = np.where((caps is None) | (caps > 0))[0]
                if len(idx) == 0:
                    break
                j = int(self.rng.choice(idx))
                extra[j] = 1
                added = 1
            ns += extra
            rem -= added
            if caps is not None:
                caps -= extra

        assert rem == 0, 'failed to distribute all observations under bounds'
        return ns

    def _maxGroups(self, ds: dict) -> int:
        n = int(ds['n'])
        if 'm' in ds and 'ns' in ds and 'groups' in ds:
            eligible = int(np.sum(ds['ns'] >= self.min_n))
            return min(int(ds['m']), eligible, n // self.min_n)
        return n // self.min_n

    def _compatible(
        self, ds: dict, d: int, req_m: int | None = None, req_n: int | None = None
    ) -> bool:
        if d > int(ds['d']):
            return False
        max_groups = self._maxGroups(ds)
        if max_groups < self.min_m:
            return False
        if self.strict_request_filter:
            if req_m is not None and max_groups < req_m:
                return False
            if req_n is not None and int(ds['n']) < req_n:
                return False
        return True

    def _pull(self, d: int, req_m: int | None = None, req_n: int | None = None):
        # get dataset from database with matching dims
        database = getDatabase()
        if self.source == 'all':
            subset = [ds for ds in database if self._compatible(ds, d, req_m=req_m, req_n=req_n)]
            n_ds = len(subset)
            if n_ds == 0:
                raise ValueError(
                    f'no source dataset can support d={d}, min_m={self.min_m}, min_n={self.min_n}, '
                    f'req_m={req_m}, req_n={req_n}'
                )
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
            if not self._compatible(self.ds, d, req_m=req_m, req_n=req_n):
                raise ValueError(
                    f'dimension mismatch for source={self.source}: '
                    f'requested d={d}, min_m={self.min_m}, min_n={self.min_n}, '
                    f'req_m={req_m}, req_n={req_n}, '
                    f"but source has d={self.ds['d']}, n={self.ds['n']}, max_groups={self._maxGroups(self.ds)}"
                )

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
        ns = self._sampleCountsBounded(n, m)
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
        eligible = np.where(ds['ns'] >= self.min_n)[0]
        members = self.rng.permutation(eligible)[:m]
        member_caps = ds['ns'][members].astype(int, copy=False)
        if self.max_n is not None:
            member_caps = np.minimum(member_caps, self.max_n)
        n_cap = int(member_caps.sum())
        n = min(n, n_cap)
        n = max(n, m * self.min_n)
        ns = self._sampleCountsBounded(n, m, max_n=None)
        ns = np.minimum(member_caps, ns)  # avoid oversampling and enforce cap

        # redistribute remainder if clipping reduced total count
        rem = int(n - ns.sum())
        if rem > 0:
            spare = member_caps - ns
            while rem > 0:
                idx = np.where(spare > 0)[0]
                if len(idx) == 0:
                    break
                alpha = self.rng.uniform(2.0, 20.0)
                p = self.rng.dirichlet(np.ones(len(idx)) * alpha)
                extra = self.rng.multinomial(rem, p)
                room = spare[idx]
                extra = np.minimum(extra, room)
                added = int(extra.sum())
                if added == 0:
                    j = int(self.rng.choice(idx))
                    extra[idx == j] = 1
                    added = 1
                ns[idx] += extra
                spare[idx] -= extra
                rem -= added
            assert rem == 0, 'failed to allocate grouped observations under bounds'

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
        req_m = len(ns)  # number of groups
        req_n = int(ns.sum())  # total number of observations

        for _ in range(self.max_attempts):
            # pull source and derive feasible dims
            self._pull(d, req_m=req_m, req_n=req_n)
            source_is_grouped = 'm' in self.ds and 'ns' in self.ds and 'groups' in self.ds
            max_groups = self._maxGroups(self.ds)
            if max_groups < self.min_m:
                continue

            m = min(req_m, max_groups)
            if m < self.min_m:
                continue

            n = min(req_n, int(self.ds['n']))
            if self.max_n is not None:
                n = min(n, m * self.max_n)
            n = max(n, m * self.min_n)
            if n > int(self.ds['n']):
                continue

            if source_is_grouped:
                eligible = np.where(self.ds['ns'] >= self.min_n)[0]
                if len(eligible) < m:
                    continue
                member_cap_all = self.ds['ns'][eligible].astype(int, copy=False)
                if self.max_n is not None:
                    member_cap_all = np.minimum(member_cap_all, self.max_n)
                if np.sum(member_cap_all) < n:
                    n = int(np.sum(member_cap_all))
                    n = max(n, m * self.min_n)
                    if np.sum(member_cap_all) < n:
                        continue

            logger.debug(
                f'sampling from source with adjusted dims d={d}, m={m}, n={n}, max_groups={max_groups}'
            )

            # subsample observations
            subset_fn = self._subsetGrouped if source_is_grouped else self._subset
            x, _, ns = subset_fn(self.ds, d, m, n)
            if self.max_n is not None and int(ns.max()) > self.max_n:
                continue
            if checkConstant(x).any():
                continue

            # emulate predictors using SGLD
            if self.use_sgld:
                x = self.sgld(x, rng=self.rng)

            # get groups from counts
            groups = counts2groups(ns)

            # add intercept
            ones = np.ones_like(x[:, 0:1])
            x = np.concatenate([ones, x], axis=-1)
            return {'X': x, 'ns': ns, 'groups': groups}

        raise RuntimeError(
            f'failed to sample emulator design after {self.max_attempts} attempts '
            f'for d={d}, req_m={req_m}, req_n={req_n}, source={self.source}'
        )


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

    getDatabase()  # instantiate

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
        Emulator(rng, source).sample(d, ns)  # type: ignore
    t1 = time.perf_counter()
    print(f'\n{t1 - t0:.2f}s used for sequential sampling.')

    # --- parallel (only useful when using SGLD)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1)(
        delayed(Emulator(rng, source).sample)(d, ns) for rng in tqdm(seeds)  # type: ignore
    )
    t1 = time.perf_counter()
    print(f'\n{t1 - t0:.2f}s used for parallel sampling.')
