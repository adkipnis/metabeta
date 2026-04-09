import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from metabeta.simulation.sgld import SGLD
from metabeta.utils.preprocessing import checkConstant, transformPredictors
from metabeta.utils.sampling import sampleCounts, counts2groups


# cached database
logger = logging.getLogger(__name__)
DATA_PATH = (Path(__file__).resolve().parent / '..' / 'datasets' / 'preprocessed').resolve()
VAL_PATHS = list(Path(DATA_PATH, 'validation').glob('*.npz'))
TEST_PATHS = list(Path(DATA_PATH, 'test').glob('*.npz'))
PATHS = VAL_PATHS + TEST_PATHS

# Metadata-only index: scalars + ns arrays (tiny).  Populated lazily.
_META_DB: list[dict] | None = None
_TEST_META_DB: list[dict] | None = None
# Full datasets loaded on demand and cached by path.
_DATA_CACHE: dict[Path, dict] = {}


# y_type strings saved by preprocess.py → compatible likelihood families
# family 0=normal (always), 1=bernoulli, 2=poisson
_Y_TYPE_FAMILIES: dict[str, set[int]] = {
    'continuous': {0},
    'binary': {0, 1},
    'count': {0, 2},
    'multiclass': {0},
}


def _loadMeta(path: Path) -> dict:
    """Load only lightweight fields (no X/y/groups arrays).

    Derives y-compatibility from the stored ``y_type`` string so the large
    ``y`` array is never read during metadata loading.
    """
    restore = lambda v: v.item() if v.shape == () else v
    data = np.load(path, allow_pickle=True)
    meta: dict = {}
    for k in data.files:
        if k in ('X', 'y', 'groups'):
            continue
        v = restore(data[k])
        if v is not None:
            meta[k] = v
    y_type = str(meta.get('y_type', ''))
    meta['y_families'] = _Y_TYPE_FAMILIES.get(y_type, set()) if 'y' in data.files else set()
    meta['source'] = path
    return meta


def getDatabase() -> list[dict]:
    """Lazyload metadata-only index for all datasets."""
    global _META_DB
    if _META_DB is None:
        _META_DB = [_loadMeta(p) for p in PATHS]
    return _META_DB


def getTestDatabase() -> list[dict]:
    """Lazyload metadata-only index for test-pool datasets."""
    global _TEST_META_DB
    if _TEST_META_DB is None:
        _TEST_META_DB = [_loadMeta(p) for p in TEST_PATHS]
    return _TEST_META_DB


def loadDataset(source: Path) -> dict:
    """Load full dataset (X, y, ...) on demand; cache by path."""
    if source in _DATA_CACHE:
        return _DATA_CACHE[source]
    restore = lambda v: v.item() if v.shape == () else v
    assert source.exists(), 'source does not exist'
    data = np.load(source, allow_pickle=True)
    result = {k: restore(data[k]) for k in data.files if restore(data[k]) is not None}
    result['source'] = source
    assert len(result['X']) == result['n'], 'dim mismatch (X, n)'
    groups = result.get('groups')
    if groups is not None:
        assert len(groups) == len(result['X']), 'dim mismatch (group, X)'
        assert len(np.unique(groups)) == result['m'], 'dim mismatch (group, m)'
        assert result['n'] == result['ns'].sum(), 'dim mismatch (n, ns)'
    _DATA_CACHE[source] = result
    return result


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
        if 'm' in ds and 'ns' in ds:
            eligible = int(np.sum(ds['ns'] >= self.min_n))
            return min(int(ds['m']), eligible, n // self.min_n)
        return n // self.min_n

    def _compatible(self, ds: dict, d: int) -> bool:
        if d > int(ds['d']):
            return False
        return self._maxGroups(ds) >= self.min_m

    def _pull(self, d: int):
        # select from lightweight metadata index, then load full data for chosen path only
        meta_db = getDatabase()
        if self.source == 'all':
            subset = [m for m in meta_db if self._compatible(m, d)]
            n_ds = len(subset)
            if n_ds == 0:
                raise ValueError(
                    f'no source dataset can support d={d}, min_m={self.min_m}, min_n={self.min_n}'
                )
            idx = self.rng.integers(0, n_ds)
            self.ds = loadDataset(subset[idx]['source'])
        else:
            path0 = Path(DATA_PATH, 'test', f'{self.source}.npz')
            path1 = Path(DATA_PATH, 'validation', f'{self.source}.npz')
            if path0 in PATHS:
                path = path0
            elif path1 in PATHS:
                path = path1
            else:
                raise ValueError(f'{self.source} not in known paths.')
            meta = next(m for m in meta_db if m['source'] == path)
            if not self._compatible(meta, d):
                raise ValueError(
                    f'dimension mismatch for source={self.source}: '
                    f'requested d={d}, min_m={self.min_m}, min_n={self.min_n}, '
                    f"but source has d={meta['d']}, n={meta['n']}, max_groups={self._maxGroups(meta)}"
                )
            self.ds = loadDataset(path)

    def _subset(
        self,
        ds: dict,
        d: int,
        m: int,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ds['X'].copy()
        y = ds['y'].copy() if 'y' in ds else np.zeros(len(ds['X']))

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
        y = ds['y'].copy() if 'y' in ds else np.zeros(len(ds['X']))

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
            self._pull(d)
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


def _yCompatible(y: np.ndarray, likelihood_family: int) -> bool:
    """Check whether y values are appropriate for the given likelihood family."""
    if likelihood_family == 0:  # normal: any continuous values
        return True
    elif likelihood_family == 1:  # bernoulli: y ∈ {0, 1}
        unique = np.unique(y)
        return set(unique.tolist()) <= {0.0, 1.0}
    elif likelihood_family == 2:  # poisson: non-negative integers
        if np.any(y < -1e-12):
            return False
        return bool(np.allclose(y, np.round(y), atol=1e-12))
    return False


@dataclass
class Subsampler:
    """Subsample (X, y, groups) from preprocessed datasets, keeping y as-is.

    Unlike Emulator, this class preserves the original outcome variable and does
    not apply SGLD to predictors.  Used for out-of-distribution evaluation where
    no parameter simulation is needed.
    """

    rng: np.random.Generator
    source: str
    likelihood_family: int
    min_m: int = 1
    min_n: int = 1
    max_n: int | None = None
    max_attempts: int = 20

    def __post_init__(self):
        if isinstance(self.rng, np.random.SeedSequence):
            self.rng = np.random.default_rng(self.rng)
        if self.min_m < 1:
            raise ValueError(f'min_m must be >= 1, but got {self.min_m}')
        if self.min_n < 1:
            raise ValueError(f'min_n must be >= 1, but got {self.min_n}')
        if self.max_n is not None and self.max_n < self.min_n:
            raise ValueError(f'max_n must be >= min_n ({self.min_n}), but got max_n={self.max_n}')

    # -- reuse Emulator helpers via composition ----------------------------------

    def _sampleCountsBounded(self, n: int, m: int, max_n: int | None = None) -> np.ndarray:
        return Emulator._sampleCountsBounded(self, n, m, max_n)

    def _maxGroups(self, ds: dict) -> int:
        return Emulator._maxGroups(self, ds)

    def _compatible(self, ds: dict, d: int) -> bool:
        if d > int(ds['d']):
            return False
        if self._maxGroups(ds) < self.min_m:
            return False
        # use precomputed y_families flag from metadata (no y array needed)
        return self.likelihood_family in ds.get('y_families', set())

    def _pull(self, d: int):
        # select from lightweight metadata index, then load full data for chosen path only
        meta_db = getTestDatabase()
        if self.source == 'all':
            subset = [m for m in meta_db if self._compatible(m, d)]
            n_ds = len(subset)
            if n_ds == 0:
                raise ValueError(
                    f'no source dataset supports d={d}, min_m={self.min_m}, '
                    f'min_n={self.min_n}, likelihood_family={self.likelihood_family}'
                )
            idx = self.rng.integers(0, n_ds)
            self.ds = loadDataset(subset[idx]['source'])
        else:
            path = Path(DATA_PATH, 'test', f'{self.source}.npz')
            if path not in TEST_PATHS:
                raise ValueError(f'{self.source} not in test pool.')
            meta = next(m for m in meta_db if m['source'] == path)
            if not self._compatible(meta, d):
                raise ValueError(
                    f'source={self.source} incompatible: d={d}, '
                    f'likelihood_family={self.likelihood_family}'
                )
            self.ds = loadDataset(path)

    def _subset(
        self, ds: dict, d: int, m: int, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ds['X'].copy()
        y = ds['y'].copy()

        idx_feat = self.rng.permutation(x.shape[1])[: d - 1]
        x = x[:, idx_feat]

        ns = self._sampleCountsBounded(n, m)
        n = int(ns.sum())
        idx_obs = self.rng.permutation(len(x))[:n]
        x = x[idx_obs]
        y = y[idx_obs]
        return x, y, ns

    def _subsetGrouped(
        self, ds: dict, d: int, m: int, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ds['X'].copy()
        y = ds['y'].copy()

        idx_feat = self.rng.permutation(x.shape[1])[: d - 1]
        x_ = x[:, idx_feat]

        eligible = np.where(ds['ns'] >= self.min_n)[0]
        members = self.rng.permutation(eligible)[:m]
        member_caps = ds['ns'][members].astype(int, copy=False)
        if self.max_n is not None:
            member_caps = np.minimum(member_caps, self.max_n)
        n_cap = int(member_caps.sum())
        n = min(n, n_cap)
        n = max(n, m * self.min_n)
        ns = self._sampleCountsBounded(n, m, max_n=None)
        ns = np.minimum(member_caps, ns)

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
        d: int,
        ns: np.ndarray,
    ) -> dict[str, np.ndarray]:
        req_m = len(ns)
        req_n = int(ns.sum())

        for _ in range(self.max_attempts):
            self._pull(d)
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

            subset_fn = self._subsetGrouped if source_is_grouped else self._subset
            x, y, ns = subset_fn(self.ds, d, m, n)
            if self.max_n is not None and int(ns.max()) > self.max_n:
                continue
            if checkConstant(x).any():
                continue

            groups = counts2groups(ns)

            # standardize predictors (no SGLD)
            x = transformPredictors(x, axis=0, exclude_binary=True, transform_counts=True)

            # add intercept
            ones = np.ones_like(x[:, 0:1])
            x = np.concatenate([ones, x], axis=-1)

            # normalize y to unit sd for normal likelihood
            if self.likelihood_family == 0:
                sd_y = max(float(y.std()), 1e-6)
                y = y / sd_y
            else:
                sd_y = 1.0

            return {'X': x, 'y': y, 'ns': ns, 'groups': groups, 'sd_y': np.array(sd_y)}

        raise RuntimeError(
            f'failed to subsample after {self.max_attempts} attempts '
            f'for d={d}, req_m={req_m}, req_n={req_n}, source={self.source}, '
            f'likelihood_family={self.likelihood_family}'
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
