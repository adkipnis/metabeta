from dataclasses import dataclass
import numpy as np
from metabeta.utils.preprocessing import standardize
from metabeta.simulation import Prior, Synthesizer, Emulator
from metabeta.plot import Plot


def simulate(
    rng: np.random.Generator,
    parameters: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    ) -> np.ndarray:
    ''' draw y given X and theta for a single dataset '''
    # unpack parameters
    ffx = parameters['ffx'] # (d,)
    rfx = parameters['rfx'] # (m, q)
    sigma_eps = parameters['sigma_eps']
    q = rfx.shape[1]

    # unpack observations
    X = observations['X'] # assumed to be standardized
    Z = X[:, :q]
    groups = observations['groups']
    n = len(X)

    # generate noise
    eps = rng.normal(size=(n,))
    eps = standardize(eps, axis=0) * sigma_eps

    # outcome
    y = X @ ffx + (Z * rfx[groups]).sum(-1) + eps
    return y


@dataclass
class Simulator:
    rng: np.random.Generator
    prior: Prior
    design: Synthesizer | Emulator
    ns: np.ndarray # number of observations per group
    plot: bool = False

    def __post_init__(self):
        self.d = self.prior.d # number of ffx
        self.q = self.prior.q # number of rfx
        if isinstance(self.ns, list):
            self.ns = np.array(self.ns)

    @property
    def m(self) -> int:
        return len(self.ns)

    def _covsum(self,
                parameters: dict[str, np.ndarray],
                observations: dict[str, np.ndarray],
                ) -> float:
        ''' get sum of covariance matrix (minus the first element) '''
        if self.m < 2 or self.q < 2:
            return 0.0
        rfx = parameters['rfx']
        q = rfx.shape[1]
        Z = observations['X'][:, :q]
        weighted_rfx = Z.mean(0, keepdims=True) * rfx
        cov = np.cov(weighted_rfx, rowvar=False)
        cov_sum = cov.sum() - cov[0,0]
        return cov_sum

    def sample(self) -> dict[str, int | float | np.ndarray]:
        # sample and standardize observations
        obs = self.design.sample(self.d, self.ns)
        obs['X'] = standardize(obs['X'], axis=0, exclude_binary=True)
        if 'ns' in obs: # in case of update through Emulator
            self.ns = obs['ns']

        # sanity check for group indices
        groups = obs['groups']
        assert groups.min() >= 0 and groups.max() < self.m

        # sample parameters
        params = self.prior.sample(self.m)

        # sample outcomes and normalize to unit SD
        y = simulate(self.rng, params, obs)
        sd = y.std(0)
        y /= sd

        # normalize parameters
        params = {k: v/sd for k,v in params.items()}

        # normalize hyperparameters (our priors are scale families)
        hyperparams = {k: v/sd for k,v in self.prior.params.items()}

        # optional plot
        if self.plot:
            data = np.concat([y[:, None], obs['X'][:, 1:]], axis=-1)
            names = ['y'] + [f'x{j}' for j in range(1, self.d)]
            Plot.dataset(data, names)

        # bundle
        out = {
            # parameters
            **params,
            **hyperparams,

            # observations
            **obs,
            'y': y,

            # dimensions
            'm': self.m,
            'n': len(y),
            'ns': self.ns,
            'd': self.d,
            'q': self.q,

            # miscellanious
            'r_squared': 1 - params['sigma_eps']**2,
            'cov_sum': self._covsum(params, obs),
        }

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

    # --- test: simulate
    # sample parameters
    hyperparams = hypersample(rng, d, q)
    prior = Prior(rng, hyperparams)
    params = prior.sample(m)
 
    # sample observations
    design = Synthesizer(rng)
    obs = design.sample(d, ns)

    # forward pass
    y = simulate(rng, params, obs)

    # --- test: simulator
    # simulator 1
    simulator1 = Simulator(rng, prior, design, ns, plot=True)
    dataset1 = simulator1.sample()

    # simulator 2
    design = Emulator(rng, 'math')
    simulator2 = Simulator(rng, prior, design, ns, plot=True)
    dataset2 = simulator2.sample()

