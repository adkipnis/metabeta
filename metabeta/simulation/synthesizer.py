import numpy as np
from dataclasses import dataclass
from metabeta.utils.preprocessing import checkContinuous
from metabeta.utils.sampling import counts2groups, wishartCorrelation
from metabeta.simulation.distributions import (
    Normal, Student, LogNormal, Uniform,
    ScaledBeta, Bernoulli, NegativeBinomial,
)

DISTDICT = {
    Normal: 0.05,
    Student: 0.25,
    LogNormal: 0.10,
    Uniform: 0.10,
    ScaledBeta: 0.15,
    Bernoulli: 0.20,
    NegativeBinomial: 0.15,
}
PROBS = np.array(list(DISTDICT.values()))
DISTS = np.array(list(DISTDICT.keys()))

@dataclass
class Synthesizer:
    ''' class for sampling a design matrix and groups using synthetic distributions '''
    rng: np.random.Generator
    toy: bool = False
    correlate: bool = True

    def __post_init__(self):
        if isinstance(self.rng, np.random.SeedSequence):
            self.rng = np.random.default_rng(self.rng)

    def _induceCorrelation(self, x: np.ndarray) -> np.ndarray:
        # x (n, d), L (d, d)

        # get continuous columns
        cont_cols = checkContinuous(x)
        d_ = cont_cols.sum()
        if d_ < 2:
            return x

        out = x.copy()
        x_ = out[:, cont_cols]

        # get lower triangular of corr matrix
        C = wishartCorrelation(self.rng, d_, nu=d_+9)
        L = np.linalg.cholesky(C)

        # standardize
        mean = np.mean(x_, axis=0, keepdims=True)
        std = np.std(x_, axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        x_ = (x_ - mean) / std

        # correlate continuous and enforce unit std
        x_cor = (x_ @ L.T)
        std_cor = np.std(x_cor, axis=0, keepdims=True)
        std_cor = np.where(std_cor < 1e-12, 1.0, std_cor)
        x_cor = x_cor / std_cor

        # unstandardize and insert
        x_cor = x_cor * std + mean
        out[:, cont_cols] = x_cor

        return out


    def _sample(self, n: int, d: int) -> np.ndarray:
        # init design matrix
        x = np.zeros((n, d))
        x[..., 0] = 1.

        # toy samples
        if self.toy:
            x[..., 1:] = self.rng.normal(size=(n, d-1))
            return x

        # choose distributions
        sample_dists = self.rng.choice(DISTS, size=d-1, replace=True, p=PROBS)

        # sample covariates from chosen distributions
        samples = [D(self.rng).sample(n) for D in sample_dists]
        samples = np.column_stack(samples)
        if self.correlate:
            samples = self._induceCorrelation(samples)
        x[..., 1:] = samples
        return x


    def sample(self, d: int, ns: np.ndarray) -> dict[str, np.ndarray]:
        n = int(ns.sum())
        x = self._sample(n, d)
        groups = counts2groups(ns)
        return {'X': x, 'groups': groups}


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from joblib import Parallel, delayed

    b = 4096
    n = 1000
    d = 12
    seed = 0

    _ = np.random.seed(seed)
    main_seed = np.random.SeedSequence(seed)
    seeds = main_seed.spawn(b)

    # --- sequential
    t0 = time.perf_counter()
    for rng in tqdm(seeds):
        Synthesizer(rng)._sample(n, d)
    t1 = time.perf_counter()
    print(f'\n{t1-t0:.2f}s used for sequential sampling.')

    # --- parallel
    def sample_batch(rng):
        return Synthesizer(rng)._sample(n, d)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1)(
        delayed(sample_batch)(rng) for rng in tqdm(seeds)
        )
    t1 = time.perf_counter()
    print(f'\n{t1-t0:.2f}s used for parallel sampling.')

    # --- outer wrapper
    from metabeta.utils.sampling import sampleCounts
    ns = sampleCounts(n, 10)
    rng = np.random.default_rng(seed)
    Synthesizer(rng).sample(d, ns)

