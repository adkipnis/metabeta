from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, t

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
        nu = self.params['nu_ffx']
        tau = self.params['tau_ffx']
        dist = norm(nu, tau)
        ffx = dist.rvs(size=nu.shape,
                       random_state=self.rng)
        return ffx

    def _sampleSigmaRfx(self) -> np.ndarray:
        tau = self.params['tau_rfx']
        dist = norm(0, tau)
        sigma_rfx = dist.rvs(size=tau.shape,
                             random_state=self.rng)
        return np.abs(sigma_rfx)

    def _sampleSigmaEps(self) -> np.ndarray:
        tau = self.params['tau_eps']
        dist = t(4, 0, tau)
        sigma_eps = dist.rvs(random_state=self.rng)
        return np.abs(sigma_eps)

    def _sampleRfx(self, m: int, sigma_rfx: np.ndarray) -> np.ndarray:
        # m: number of groups
        size = (m, self.q)
        dist = norm(loc=0, scale=sigma_rfx)
        rfx = dist.rvs(size, random_state=self.rng)
        return rfx
 
    def sample(self, m: int) -> dict[str, np.ndarray]:
        out = {}
        out['ffx'] = self._sampleFfx()
        out['sigma_rfx'] = self._sampleSigmaRfx()
        out['sigma_eps'] = self._sampleSigmaEps()
        out['rfx'] = self._sampleRfx(m, out['sigma_rfx'])
        return out


if __name__ == '__main__':
    seed = 0
    _ = np.random.seed(seed)
    rng = np.random.default_rng(seed)
 
    d, q = 3, 1
    m = 10

    hyperparams = hypersample(d, q)
    prior = Prior(hyperparams, rng=rng)
    params = prior.sample(m)

