import logging
from dataclasses import dataclass
import numpy as np
from metabeta.utils.preprocessing import transformPredictors
from metabeta.utils.families import (
    hasSigmaEps,
    simulateYNp,
    BERNOULLI_ETA_ABS_MAX,
    BERNOULLI_REROLL_EXTREME_FRACTION_MAX,
    BERNOULLI_REROLL_MAX_ATTEMPTS,
    BERNOULLI_LP_SD_CAP_LOW,
    BERNOULLI_LP_SD_CAP_HIGH,
    POISSON_ETA_CLIP_MAX,
    POISSON_X_CLIP_ABS,
    POISSON_REROLL_CLIP_FRACTION_MAX,
    POISSON_REROLL_MAX_ATTEMPTS,
    POISSON_LP_SD_CAP_LOW,
    POISSON_LP_SD_CAP_HIGH,
)
from metabeta.simulation import Prior, Synthesizer, Scammer, Emulator
from metabeta.plotting import plotDataset

SCALE_PARAMS = {'ffx', 'sigma_rfx', 'sigma_eps', 'rfx'}
SCALE_HYPERPARAMS = {'nu_ffx', 'tau_ffx', 'tau_rfx', 'tau_eps'}
NORMAL_R2_CAP_BASE = 0.68
NORMAL_R2_CAP_FFX_SLOPE = 0.015
NORMAL_R2_CAP_RFX_SLOPE = 0.010
NORMAL_R2_CAP_SD = 0.10
NORMAL_R2_CAP_MIN = 0.45
NORMAL_R2_CAP_MAX = 0.95
logger = logging.getLogger(__name__)


def linearPredictor(
    parameters: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute eta = X beta + Z b for a single grouped dataset."""
    ffx = parameters['ffx']
    rfx = parameters['rfx']
    q = rfx.shape[1]
    X = observations['X']
    Z = X[:, :q]
    groups = observations['groups']
    return X @ ffx + (Z * rfx[groups]).sum(-1)


def simulate(
    rng: np.random.Generator,
    parameters: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    likelihood_family: int = 0,
) -> np.ndarray:
    """draw y given X and theta for a single dataset"""
    sigma_eps = float(parameters.get('sigma_eps', 0.0))
    eta = linearPredictor(parameters, observations)
    return simulateYNp(rng, eta, sigma_eps, likelihood_family)


@dataclass
class Simulator:
    rng: np.random.Generator
    prior: Prior
    design: Synthesizer | Scammer | Emulator
    ns: np.ndarray  # number of observations per group
    plot: bool = False

    def __post_init__(self):
        self.d = self.prior.d  # number of ffx
        self.q = self.prior.q  # number of rfx
        if isinstance(self.ns, list):
            self.ns = np.array(self.ns)

    @property
    def m(self) -> int:
        return len(self.ns)

    @staticmethod
    def _poissonClipFraction(
        params: dict[str, np.ndarray],
        observations: dict[str, np.ndarray],
    ) -> float:
        ffx = params['ffx']
        rfx = params['rfx']
        q = rfx.shape[1]
        X = observations['X']
        Z = X[:, :q]
        groups = observations['groups']
        eta = X @ ffx + (Z * rfx[groups]).sum(-1)
        return float(np.mean(eta > POISSON_ETA_CLIP_MAX))

    @staticmethod
    def _bernoulliExtremeEtaFraction(
        params: dict[str, np.ndarray],
        observations: dict[str, np.ndarray],
    ) -> float:
        eta = linearPredictor(params, observations)
        return float(np.mean(np.abs(eta) > BERNOULLI_ETA_ABS_MAX))

    def _sampleBernoulliLpCap(self) -> float:
        return float(self.rng.uniform(BERNOULLI_LP_SD_CAP_LOW, BERNOULLI_LP_SD_CAP_HIGH))

    def _calibrateBernoulliEtaScale(
        self,
        params: dict[str, np.ndarray],
        hyperparams: dict[str, np.ndarray],
        observations: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Scale Bernoulli params down if sd(eta) exceeds a sampled cap.

        Keeps stored hyperparams coherent with the rescaled parameters so that
        the NPE sees a consistent prior context after calibration.
        """
        eta = linearPredictor(params, observations)
        eta_sd = float(np.std(eta))
        if eta_sd <= 1e-12:
            return params, hyperparams
        cap = self._sampleBernoulliLpCap()
        if eta_sd <= cap:
            return params, hyperparams
        scale = cap / eta_sd
        params = {k: v * scale if k in SCALE_PARAMS else v for k, v in params.items()}
        hyperparams = {
            k: v * scale if k in SCALE_HYPERPARAMS else v for k, v in hyperparams.items()
        }
        return params, hyperparams

    def _samplePoissonLpCap(self) -> float:
        return float(self.rng.uniform(POISSON_LP_SD_CAP_LOW, POISSON_LP_SD_CAP_HIGH))

    def _calibratePoissonEtaScale(
        self,
        params: dict[str, np.ndarray],
        hyperparams: dict[str, np.ndarray],
        observations: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Scale Poisson params down if sd(eta) exceeds a sampled cap.

        Keeps stored hyperparams coherent with the rescaled parameters so that
        the NPE sees a consistent prior context after calibration.
        """
        eta = linearPredictor(params, observations)
        eta_sd = float(np.std(eta))
        if eta_sd <= 1e-12:
            return params, hyperparams
        cap = self._samplePoissonLpCap()
        if eta_sd <= cap:
            return params, hyperparams
        scale = cap / eta_sd
        params = {k: v * scale if k in SCALE_PARAMS else v for k, v in params.items()}
        hyperparams = {
            k: v * scale if k in SCALE_HYPERPARAMS else v for k, v in hyperparams.items()
        }
        return params, hyperparams

    def _sampleNormalR2Cap(self) -> float:
        ffx_covariates = max(self.d - 1, 0)
        rfx_slopes = max(self.q - 1, 0)
        mean = (
            NORMAL_R2_CAP_BASE
            + NORMAL_R2_CAP_FFX_SLOPE * ffx_covariates
            + NORMAL_R2_CAP_RFX_SLOPE * rfx_slopes
        )
        cap = self.rng.normal(mean, NORMAL_R2_CAP_SD)
        return float(np.clip(cap, NORMAL_R2_CAP_MIN, NORMAL_R2_CAP_MAX))

    def _calibrateNormalResidualShare(
        self,
        params: dict[str, np.ndarray],
        hyperparams: dict[str, np.ndarray],
        observations: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        eta = linearPredictor(params, observations)
        signal_var = float(np.var(eta))
        if signal_var <= 1e-12:
            return params, hyperparams

        sigma_eps = max(float(params['sigma_eps']), 1e-8)
        expected_r2 = signal_var / (signal_var + sigma_eps**2)
        r2_cap = self._sampleNormalR2Cap()
        if expected_r2 <= r2_cap:
            return params, hyperparams

        target_sigma_eps = np.sqrt(signal_var * (1.0 - r2_cap) / max(r2_cap, 1e-8))
        scale = target_sigma_eps / sigma_eps

        params = dict(params)
        hyperparams = dict(hyperparams)
        params['sigma_eps'] = np.array(target_sigma_eps)
        hyperparams['tau_eps'] = np.asarray(hyperparams['tau_eps']) * scale
        return params, hyperparams

    def sample(self) -> dict[str, np.ndarray]:
        likelihood_family = int(self.prior.hyperparams.get('likelihood_family', 0))

        # sample and standardize observations
        obs = self.design.sample(self.d, self.ns)
        obs['X'] = transformPredictors(obs['X'], axis=0, exclude_binary=True, transform_counts=True)
        if likelihood_family == 2 and obs['X'].shape[-1] > 1:
            obs['X'][:, 1:] = np.clip(obs['X'][:, 1:], -POISSON_X_CLIP_ABS, POISSON_X_CLIP_ABS)
        if 'ns' in obs:  # in case of update through Emulator
            self.ns = obs['ns']

        # sanity check for group indices
        groups = obs['groups']
        assert groups.min() >= 0 and groups.max() < self.m

        # sample parameters and optionally reroll for extreme Poisson eta clipping
        params = self.prior.sample(self.m)
        hyperparams = dict(self.prior.hyperparams)
        if likelihood_family == 2:
            clip_fraction = self._poissonClipFraction(params, obs)
            attempts = 1
            while (
                clip_fraction > POISSON_REROLL_CLIP_FRACTION_MAX
                and attempts < POISSON_REROLL_MAX_ATTEMPTS
            ):
                params = self.prior.sample(self.m)
                clip_fraction = self._poissonClipFraction(params, obs)
                attempts += 1
            if clip_fraction > POISSON_REROLL_CLIP_FRACTION_MAX:
                logger.warning(
                    (
                        'Poisson eta clipping remained high after rerolls: %.2f%% > %.2f%% '
                        '(attempts=%d/%d). Applying LP scale calibration.'
                    ),
                    100.0 * clip_fraction,
                    100.0 * POISSON_REROLL_CLIP_FRACTION_MAX,
                    attempts,
                    POISSON_REROLL_MAX_ATTEMPTS,
                )
            params, hyperparams = self._calibratePoissonEtaScale(params, hyperparams, obs)
        elif likelihood_family == 1:
            extreme_fraction = self._bernoulliExtremeEtaFraction(params, obs)
            attempts = 1
            while (
                extreme_fraction > BERNOULLI_REROLL_EXTREME_FRACTION_MAX
                and attempts < BERNOULLI_REROLL_MAX_ATTEMPTS
            ):
                params = self.prior.sample(self.m)
                extreme_fraction = self._bernoulliExtremeEtaFraction(params, obs)
                attempts += 1
            if extreme_fraction > BERNOULLI_REROLL_EXTREME_FRACTION_MAX:
                logger.warning(
                    (
                        'Bernoulli extreme logits remained high after rerolls: %.2f%% > %.2f%% '
                        '(attempts=%d/%d). Applying LP scale calibration.'
                    ),
                    100.0 * extreme_fraction,
                    100.0 * BERNOULLI_REROLL_EXTREME_FRACTION_MAX,
                    attempts,
                    BERNOULLI_REROLL_MAX_ATTEMPTS,
                )
            params, hyperparams = self._calibrateBernoulliEtaScale(params, hyperparams, obs)
        elif likelihood_family == 0:
            params, hyperparams = self._calibrateNormalResidualShare(params, hyperparams, obs)

        # sample outcomes
        y = simulate(self.rng, params, obs, likelihood_family)

        # normalize to unit SD (continuous likelihoods only)
        if hasSigmaEps(likelihood_family):
            sd = max(y.std(0), 1e-6)
            y /= sd
            params = {k: v / sd if k in SCALE_PARAMS else v for k, v in params.items()}
            hyperparams = {
                k: v / sd if k in SCALE_HYPERPARAMS else v for k, v in hyperparams.items()
            }
        else:
            sd = 1.0

        # optional plot
        if self.plot:
            data = np.concatenate([y[:, None], obs['X'][:, 1:]], axis=-1)
            names = ['y'] + [f'x{j}' for j in range(1, self.d)]
            plotDataset(data, names)

        # bundle
        out = {
            # parameters
            **params,
            **hyperparams,
            # observations
            'y': y,
            'X': obs['X'],
            'groups': obs['groups'],
            # dimensions
            'm': self.m,
            'n': len(y),
            'ns': self.ns,
            'd': self.d,
            'q': self.q,
            # miscellanious
            'sd_y': sd,
        }
        if hasSigmaEps(likelihood_family):
            out['r_squared'] = 1 - params['sigma_eps'] ** 2

        # package scalars as numpy arrays
        for k, v in out.items():
            if not isinstance(v, np.ndarray):
                out[k] = np.array(v)
        return out


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    from metabeta.utils.sampling import sampleCounts
    from metabeta.simulation import hypersample

    # set seed
    seed = 1
    rng = np.random.default_rng(seed)

    # dims
    n = 100
    m = 5
    d = 3
    q = 2
    ns = sampleCounts(rng, n, m)

    # sample parameters
    hyperparams = hypersample(rng, d, q)
    prior = Prior(rng, hyperparams)

    # --- test: simulator
    # simulator 1
    design = Synthesizer(rng)
    simulator1 = Simulator(rng, prior, design, ns, plot=True)
    dataset1 = simulator1.sample()

    # simulator 2
    design = Emulator(rng, 'math')
    simulator2 = Simulator(rng, prior, design, ns, plot=True)
    dataset2 = simulator2.sample()
