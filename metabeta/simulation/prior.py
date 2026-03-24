from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, t, multivariate_normal

from metabeta.utils.sampling import truncLogUni, lkjCorrelation


def hypersample(
    rng: np.random.Generator,
    d: int,
    q: int,
    correlated_rfx: bool = False,
) -> dict[str, np.ndarray]:
    """sample hyperparameters to instantiate prior"""
    out = {}
    out['nu_ffx'] = rng.uniform(-2.0, 2.0, size=d)
    out['tau_ffx'] = truncLogUni(rng, 1.0, 12.0, size=d)
    out['tau_rfx'] = truncLogUni(rng, 0.01, 4.0, size=q)
    out['tau_eps'] = float(truncLogUni(rng, 0.01, 4.0, size=1)[0])
    out['correlated_rfx'] = correlated_rfx
    if correlated_rfx and q > 1:
        out['eta_rfx'] = float(rng.uniform(1.0, 2.0))
    return out


@dataclass
class Prior:
    """class for drawing parameters from prior"""

    rng: np.random.Generator
    hyperparams: dict[str, np.ndarray]

    def __post_init__(self):
        self.d = len(self.hyperparams['tau_ffx'])   # number of fixed effects
        self.q = len(self.hyperparams['tau_rfx'])   # number of random effects
        self.correlated_rfx = self.hyperparams.get('correlated_rfx', False)

    def _sampleFfx(self) -> np.ndarray:
        tau = self.hyperparams['tau_ffx']
        dist = norm(nu, tau)
        ffx = dist.rvs(size=nu.shape, random_state=self.rng)
        return ffx

    def _sampleSigmaRfx(self) -> np.ndarray:
        tau = self.hyperparams['tau_rfx']
        dist = norm(loc=0, scale=tau)
        sigma_rfx = dist.rvs(size=tau.shape, random_state=self.rng)
        return np.abs(sigma_rfx)

    def _sampleSigmaEps(self) -> np.ndarray:
        tau = self.hyperparams['tau_eps']
        dist = t(df=4, loc=0, scale=tau)
        sigma_eps = dist.rvs(random_state=self.rng)
        return np.abs(sigma_eps)

    def _sampleCorrMat(self) -> np.ndarray:
        eta = self.hyperparams['eta_rfx']
        return lkjCorrelation(self.rng, self.q, eta=eta)

    def _sampleRfx(
            self, m: int, sigma_rfx: np.ndarray, corr_mat: np.ndarray | None = None
    ) -> np.ndarray:
        if corr_mat is None:
            # uncorrelated: independent normal per dimension
            dist = norm(loc=0, scale=sigma_rfx)
            return dist.rvs(size=(m, self.q), random_state=self.rng)
        # correlated: multivariate normal with LKJ correlation
        cov = np.diag(sigma_rfx) @ corr_mat @ np.diag(sigma_rfx)
        dist = multivariate_normal(mean=np.zeros(self.q), cov=cov)
        return dist.rvs(size=(m, self.q), random_state=self.rng)

    def sample(self, m: int) -> dict[str, np.ndarray]:
        out = {}
        out['ffx'] = self._sampleFfx()
        out['sigma_rfx'] = self._sampleSigmaRfx()
        out['sigma_eps'] = self._sampleSigmaEps()
        if self.correlated_rfx and self.q > 1:
            out['corr_rfx'] = self._sampleCorrMat()
        else:
            out['corr_rfx'] = None
        out['rfx'] = self._sampleRfx(m, out['sigma_rfx'], corr_mat=out['corr_rfx'])
        return out


if __name__ == '__main__':
    seed = 0
    rng = np.random.default_rng(seed)

    d, q = 3, 1
    m = 10

    hyperparams = hypersample(rng, d, q)
    prior = Prior(rng, hyperparams)
    params = prior.sample(m)
