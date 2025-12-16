from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, t
from metabeta.simulation.utils import standardize

def hypersample(d: int, q: int) -> dict[str, np.ndarray]:
    ''' sample hyperparameters to instantiate prior '''
    out = {}
    out['nu_ffx'] = np.random.uniform(-3, 3, (d,))
    out['tau_ffx'] = np.random.uniform(0, 3, (d,))
    out['tau_rfx'] = np.random.uniform(0, 3, (q,))
    out['tau_eps'] = np.random.uniform(0, 3)
    return out


@dataclass
class Prior:
    ''' class for drawing parameters from prior '''
    params: dict[str, np.ndarray]
    rng: np.random.Generator

    def __post_init__(self):
        self.d = len(self.params['tau_ffx']) # number of fixed effects
        self.q = len(self.params['tau_rfx']) # number of random effects

    def _sampleFfx(self) -> np.ndarray:
        dist = norm(self.nu_ffx, self.tau_ffx)
        ffx = dist.rvs(size=self.nu_ffx.shape,
                       random_state=self.rng)
        return ffx

    def _sampleSigmaRfx(self) -> np.ndarray:
        dist = norm(0, self.tau_rfx)
        sigma_rfx = dist.rvs(size=self.tau_rfx.shape,
                             random_state=self.rng)
        return np.abs(sigma_rfx)

    def _sampleSigmaEps(self) -> np.ndarray:
        dist = t(4, 0, self.tau_eps)
        sigma_eps = dist.rvs(size=self.tau_eps.shape,
                             random_state=self.rng)
        return np.abs(sigma_eps)

    def _sampleRfx(self, m: int, sigma_rfx: np.ndarray) -> np.ndarray:
        # m: number of groups
        q = len(sigma_rfx)
        size = (m, q)
        rfx = norm(0, 1).rvs(size, random_state=self.rng)
        rfx = standardize(rfx, axis=0) * sigma_rfx[None, :]
        return rfx
 
    def sample(self, m: int) -> dict[str, np.ndarray]:
        ffx = self._sampleFfx()
        sigma_rfx = self._sampleSigmaRfx()
        sigma_eps = self._sampleSigmaEps()
        rfx = self._sampleRfx(m, sigma_rfx)
        return dict(ffx=ffx, sigma_rfx=sigma_rfx, sigma_eps=sigma_eps, rfx=rfx)


if __name__ == '__main__':
    seed = 0
    _ = np.random.seed(seed)
    rng = np.random.default_rng(seed)
 
    d, q = 3, 1
    m = 10

    hyperparams = hypersample(d, q)
    prior = Prior(*hyperparams, rng=rng)
    params = prior.sample(m)

