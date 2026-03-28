from dataclasses import dataclass
import numpy as np
from scipy.stats import multivariate_normal

from metabeta.utils.sampling import lkjCorrelation, spikeAndSlab, skewedBeta
from metabeta.utils.families import (
    FFX_FAMILIES,
    SIGMA_FAMILIES,
    FFX_FAMILY_PROBS,
    SIGMA_RFX_FAMILY_PROBS,
    SIGMA_EPS_FAMILY_PROBS,
    hasSigmaEps,
    sampleFfxNp,
    sampleSigmaNp,
)

MIN_STD = 1e-3


def hypersample(
    rng: np.random.Generator,
    d: int,
    q: int,
    likelihood_family: int = 0,
) -> dict[str, np.ndarray]:
    """sample hyperparameters to instantiate prior.
    eta_rfx > 0 indicates correlated rfx (LKJ prior), eta_rfx = 0 means uncorrelated.
    family_* are integer indices into FFX_FAMILIES / SIGMA_FAMILIES.
    """
    out = {}
    out['likelihood_family'] = np.array(likelihood_family)

    # hyperparameter ranges (tighter for log/logit-link likelihoods)
    if likelihood_family == 2:  # poisson (log link)
        out['nu_ffx'] = spikeAndSlab(rng, size=d, scale=0.5)
        out['tau_ffx'] = skewedBeta(rng, 0.01, 2.0, mode=0.5, concentration=6.0, size=d)
        out['tau_rfx'] = skewedBeta(rng, 0.01, 1.5, mode=0.3, concentration=5.0, size=q)
    else:  # normal, bernoulli
        out['nu_ffx'] = spikeAndSlab(rng, size=d)
        out['tau_ffx'] = skewedBeta(rng, 0.01, 10.0, mode=1.0, concentration=6.0, size=d)
        out['tau_rfx'] = skewedBeta(rng, 0.01, 5.0, mode=1.0, concentration=5.0, size=q)

    if hasSigmaEps(likelihood_family):
        out['tau_eps'] = skewedBeta(rng, 0.01, 5.0, mode=1.0, concentration=4.0, size=1)[0]

    # stochastic choice of rfx correlation (probability depends on q)
    if q == 1:
        correlated = False
    elif q == 2:
        correlated = rng.random() < 0.8
    else:
        correlated = rng.random() < 0.5
    out['eta_rfx'] = rng.uniform(1.0, 2.0) if correlated else np.array(0.0)

    # prior families (weighted over available families)
    out['family_ffx'] = np.array(rng.choice(len(FFX_FAMILIES), p=FFX_FAMILY_PROBS))
    out['family_sigma_rfx'] = np.array(rng.choice(len(SIGMA_FAMILIES), p=SIGMA_RFX_FAMILY_PROBS))
    if hasSigmaEps(likelihood_family):
        out['family_sigma_eps'] = np.array(
            rng.choice(len(SIGMA_FAMILIES), p=SIGMA_EPS_FAMILY_PROBS)
        )

    return out


@dataclass
class Prior:
    """class for drawing parameters from prior"""

    rng: np.random.Generator
    hyperparams: dict[str, np.ndarray]

    def __post_init__(self):
        self.d = len(self.hyperparams['tau_ffx'])  # number of fixed effects
        self.q = len(self.hyperparams['tau_rfx'])  # number of random effects
        self.correlated_rfx = float(self.hyperparams.get('eta_rfx', 0)) > 0
        self.likelihood_family = int(self.hyperparams.get('likelihood_family', 0))

    def _sampleFfx(self) -> np.ndarray:
        nu = self.hyperparams['nu_ffx']
        tau = self.hyperparams['tau_ffx']
        family = int(self.hyperparams['family_ffx'])
        return sampleFfxNp(family, nu, tau, self.rng, size=nu.shape)

    def _sampleSigmaRfx(self) -> np.ndarray:
        tau = self.hyperparams['tau_rfx']
        family = int(self.hyperparams['family_sigma_rfx'])
        sigma_rfx = sampleSigmaNp(family, tau, self.rng, size=tau.shape)
        return np.maximum(sigma_rfx, MIN_STD)

    def _sampleSigmaEps(self) -> np.ndarray:
        tau = self.hyperparams['tau_eps']
        family = int(self.hyperparams['family_sigma_eps'])
        sigma_eps = sampleSigmaNp(family, np.atleast_1d(tau), self.rng, size=(1,))
        return np.maximum(sigma_eps[0], MIN_STD)

    def _sampleCorrMat(self) -> np.ndarray:
        eta = float(self.hyperparams['eta_rfx'])
        if eta > 0:
            return lkjCorrelation(self.rng, self.q, eta=eta)
        return np.eye(self.q)

    def _sampleRfx(self, m: int, sigma_rfx: np.ndarray, corr_mat: np.ndarray) -> np.ndarray:
        # if corr_mat == np.eye(self.q):
        #     dist = norm(loc=0, scale=sigma_rfx)
        #     return dist.rvs(size=(m, self.q), random_state=self.rng)
        cov = np.diag(sigma_rfx) @ corr_mat @ np.diag(sigma_rfx)
        dist = multivariate_normal(mean=np.zeros(self.q), cov=cov, allow_singular=True)  # type: ignore
        rfx = dist.rvs(size=m, random_state=self.rng)  # type: ignore
        return rfx.reshape(m, self.q)

    def sample(self, m: int) -> dict[str, np.ndarray]:
        out = {}
        out['ffx'] = self._sampleFfx()
        out['sigma_rfx'] = self._sampleSigmaRfx()
        if hasSigmaEps(self.likelihood_family):
            out['sigma_eps'] = self._sampleSigmaEps()
        out['corr_rfx'] = self._sampleCorrMat()
        out['rfx'] = self._sampleRfx(m, sigma_rfx=out['sigma_rfx'], corr_mat=out['corr_rfx'])
        return out


if __name__ == '__main__':
    seed = 0
    rng = np.random.default_rng(seed)

    d, q = 3, 2
    m = 10

    hyperparams = hypersample(rng, d, q)
    prior = Prior(rng, hyperparams)
    params = prior.sample(m)
    print('eta_rfx:', hyperparams['eta_rfx'])
    print('correlated:', prior.correlated_rfx)
    print('rfx shape:', params['rfx'].shape)
    print('Correlation matrix:\n', params['corr_rfx'])
