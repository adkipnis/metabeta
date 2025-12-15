from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, t
from metabeta.simulation.utils import standardize

def hypersample(d: int, q: int, b: int = 1):
    # sample hyperparameters needed for prior
    nu_ffx = np.random.uniform(-3, 3, (d,))
    tau_ffx = np.random.uniform(0, 3, (d,))
    tau_rfx = np.random.uniform(0, 3, (q,))
    tau_eps = np.random.uniform(0, 3, (1,))
    return nu_ffx, tau_ffx, tau_rfx, tau_eps

@dataclass
class Prior:
    nu_ffx: np.ndarray
    tau_ffx: np.ndarray
    tau_rfx: np.ndarray
    tau_eps: np.ndarray
    rng: np.random.Generator

    def __post_init__(self):
        self.d = len(self.tau_ffx) # number of fixed effects
        self.q = len(self.tau_rfx) # number of random effects

    def _sampleFfx(self) -> np.ndarray:
        ffx = norm(self.nu_ffx, self.tau_ffx).rvs(random_state=self.rng)
        return ffx

    def _sampleSigmaRfx(self) -> np.ndarray:
        sigma_rfx = norm(0, self.tau_rfx).rvs(random_state=self.rng)
        return np.abs(sigma_rfx)

    def _sampleSigmaEps(self) -> np.ndarray:
        sigma_eps = t(4, 0, self.tau_eps).rvs(random_state=self.rng)
        return np.abs(sigma_eps)

    def _sampleRfx(self, m: int, sigma_rfx: np.ndarray) -> np.ndarray:
        # m: number of groups
        b, q = sigma_rfx.shape
        size = (b, m, q)
        rfx = norm(0, 1).rvs(size, random_state=self.rng)
        rfx = standardize(rfx, axis=1) * sigma_rfx[:, None]
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
 
    b = 32
    d, q = 3, 1
    m = 10

    nu_ffx = np.random.uniform(-3, 3, (b,d))
    tau_ffx = np.random.uniform(0, 3, (b,d))
    tau_rfx = np.random.uniform(0, 3, (b,q))
    tau_eps = np.random.uniform(0, 3, (b,))

    prior = Prior(nu_ffx, tau_ffx, tau_rfx, tau_eps, rng=rng)
    params = prior.sample(m)

