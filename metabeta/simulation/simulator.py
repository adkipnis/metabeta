import logging
from dataclasses import dataclass
import numpy as np
from metabeta.utils.preprocessing import transformPredictors
from metabeta.utils.families import (
    hasSigmaEps,
    simulateYNp,
    POISSON_ETA_CLIP_MAX,
    POISSON_X_CLIP_ABS,
    POISSON_REROLL_CLIP_FRACTION_MAX,
    POISSON_REROLL_MAX_ATTEMPTS,
)
from metabeta.simulation import Prior, Synthesizer, Scammer, Emulator
from metabeta.plot import plotDataset

SCALE_PARAMS = {'ffx', 'sigma_rfx', 'sigma_eps', 'rfx'}
SCALE_HYPERPARAMS = {'nu_ffx', 'tau_ffx', 'tau_rfx', 'tau_eps'}
logger = logging.getLogger(__name__)


def simulate(
    rng: np.random.Generator,
    parameters: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    likelihood_family: int = 0,
) -> np.ndarray:
    """draw y given X and theta for a single dataset"""
    # unpack parameters
    ffx = parameters['ffx']  # (d,)
    rfx = parameters['rfx']  # (m, q)
    sigma_eps = float(parameters.get('sigma_eps', 0.0))
    q = rfx.shape[1]

    # unpack (standardized) observations
    X = observations['X']  # (n, d)
    Z = X[:, :q]  # (n, q)
    groups = observations['groups']  # (n, )

    # linear predictor
    rfx_ext = rfx[groups]  # (n, q)
    eta = X @ ffx + (Z * rfx_ext).sum(-1)  # (n, )

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
                        '(attempts=%d/%d). Accepting dataset.'
                    ),
                    100.0 * clip_fraction,
                    100.0 * POISSON_REROLL_CLIP_FRACTION_MAX,
                    attempts,
                    POISSON_REROLL_MAX_ATTEMPTS,
                )

        # sample outcomes
        y = simulate(self.rng, params, obs, likelihood_family)

        # normalize to unit SD (continuous likelihoods only)
        if hasSigmaEps(likelihood_family):
            sd = max(y.std(0), 1e-6)
            y /= sd
            params = {
                k: v / sd if k in SCALE_PARAMS else v
                for k, v in params.items()
            }
            hyperparams = {
                k: v / sd if k in SCALE_HYPERPARAMS else v
                for k, v in self.prior.hyperparams.items()
            }
        else:
            sd = 1.0
            hyperparams = dict(self.prior.hyperparams)

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
