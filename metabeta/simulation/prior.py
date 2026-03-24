from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, t, multivariate_normal

from metabeta.utils.sampling import truncLogUni, lkjCorrelation

MIN_STD = 1e-5

def hypersample(
    rng: np.random.Generator,
    d: int,
    q: int,
    correlated_rfx: bool = True,
) -> dict[str, np.ndarray]:
    """sample hyperparameters to instantiate prior"""
    out = {}
    out['tau_ffx'] = truncLogUni(rng, 1.0, 12.0, size=d)
    out['tau_rfx'] = truncLogUni(rng, 0.01, 4.0, size=q)
    out['tau_eps'] = truncLogUni(rng, 0.01, 4.0, size=1)[0]
    out['correlated_rfx'] = correlated_rfx
    if correlated_rfx and q > 1:
        out['eta_rfx'] = rng.uniform(1.0, 2.0)
    else:
        out['eta_rfx'] = np.array(0.0)
    # TODO: sample prior families
    # TODO: sample link function (normal, t, bernoulli, poisson)
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
        # TODO: choice between normal and t
        tau = self.hyperparams['tau_ffx']
        dist = norm(loc=0, scale=tau)
        ffx = dist.rvs(size=tau.shape, random_state=self.rng)
        return ffx

    def _sampleSigmaRfx(self) -> np.ndarray:
        # TODO: choice between halfnormal, half-t(4), exponential(lambda)
        tau = self.hyperparams['tau_rfx']
        dist = norm(loc=0, scale=tau)
        sigma_rfx = dist.rvs(size=tau.shape, random_state=self.rng)
        return np.maximum(np.abs(sigma_rfx), MIN_STD)

    def _sampleSigmaEps(self) -> np.ndarray:
        # TODO: choice between halfnormal, half-t(4), exponential(1)
        tau = self.hyperparams['tau_eps']
        dist = t(df=4, loc=0, scale=tau)
        sigma_eps = dist.rvs(random_state=self.rng)
        return np.maximum(np.abs(sigma_eps), MIN_STD)

    def _sampleCorrMat(self) -> np.ndarray:
        eta = float(self.hyperparams['eta_rfx'])
        if eta > 0:
            return lkjCorrelation(self.rng, self.q, eta=eta)
        return np.eye(self.q)

    def _sampleRfx(
            self, m: int, sigma_rfx: np.ndarray, corr_mat: np.ndarray
    ) -> np.ndarray:
        cov = np.diag(sigma_rfx) @ corr_mat @ np.diag(sigma_rfx)
        dist = multivariate_normal(mean=np.zeros(self.q), cov=cov) # type: ignore
        rfx = dist.rvs(size=m, random_state=self.rng) # type: ignore
        return rfx.reshape(m, self.q)

    def sample(self, m: int) -> dict[str, np.ndarray]:
        out = {}
        out['ffx'] = self._sampleFfx()
        out['sigma_rfx'] = self._sampleSigmaRfx()
        out['sigma_eps'] = self._sampleSigmaEps()
        out['corr_rfx'] = self._sampleCorrMat()
        out['rfx'] = self._sampleRfx(m, sigma_rfx=out['sigma_rfx'], corr_mat=out['corr_rfx'])
        return out

 
if __name__ == '__main__':
    seed = 0
    rng = np.random.default_rng(seed)

    d, q = 3, 2
    m = 10

    # uncorrelated
    hyperparams = hypersample(rng, d, q, correlated_rfx=False)
    prior = Prior(rng, hyperparams)
    params = prior.sample(m)
    print('Uncorrelated rfx shape:', params['rfx'].shape)

    # correlated
    hyperparams = hypersample(rng, d, q, correlated_rfx=True)
    prior = Prior(rng, hyperparams)
    params = prior.sample(m)
    print('Correlated rfx shape:', params['rfx'].shape)
    print('Correlation matrix:\n', params['corr_rfx'])
