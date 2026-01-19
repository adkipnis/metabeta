from dataclasses import dataclass
import numpy as np
from metabeta.utils.preprocessing import standardize
from metabeta.simulation import Prior, Synthesizer, Emulator
from metabeta import plot


def simulate(
    parameters: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    ) -> np.ndarray:
    ''' draw y given X and theta '''
    # unpack parameters
    ffx = parameters['ffx']
    rfx = parameters['rfx']
    sigma_rfx = parameters['sigma_rfx']
    sigma_eps = parameters['sigma_eps']
    q = len(sigma_rfx)

    # unpack observations
    X = observations['X'] # assumed to be standardized
    Z = X[:, :q]
    groups = observations['groups']
    n = len(X)

    # generate noise
    eps = np.random.normal(size=(n,))
    eps = standardize(eps, axis=0) * sigma_eps

    # outcome
    y = X @ ffx + (Z * rfx[groups]).sum(-1) + eps
    return y


@dataclass
class Simulator:
    prior: Prior
    design: Synthesizer | Emulator
    ns: np.ndarray # number of observations per group
    plot: bool = False

    def __post_init__(self):
        self.d = self.prior.d # number of ffx
        self.q = self.prior.q # number of rfx
        self.m = len(self.ns)
        if isinstance(self.ns, list):
            self.ns = np.array(self.ns)

    def _covsum(self,
                parameters: dict[str, np.ndarray],
                observations: dict[str, np.ndarray],
                ) -> float:
        ''' get sum of covariance matrix (minus the first element) '''
        rfx = parameters['rfx']
        q = rfx.shape[1]
        Z = observations['X'][:, :q]
        weighted_rfx = Z.mean(0, keepdims=True) * rfx
        cov = np.cov(weighted_rfx, rowvar=False)
        cov_sum = cov.sum() - cov[0,0]
        return cov_sum

    def sample(self) -> dict[str, np.ndarray]:
        # sample parameters
        params = self.prior.sample(self.m)

        # sample and standardize observations
        obs = self.design.sample(self.d, self.ns)
        obs['X'] = standardize(obs['X'], axis=0, exclude_binary=True)

        # sample outcomes and normalize to unit SD
        y = simulate(params, obs)
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
            plot.dataset(data, names)

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
    prior = Prior(hyperparams, rng=rng)
    params = prior.sample(m)
 
    # sample observations
    design = Synthesizer(rng=rng)
    obs = design.sample(d, ns)

    # forward pass
    y = simulate(params, obs)

    # --- test: simulator
    # simulator 1
    simulator1 = Simulator(prior, design, ns, plot=True)
    dataset1 = simulator1.sample()

    # simulator 2
    design = Emulator('math', rng=rng)
    simulator2 = Simulator(prior, design, ns, plot=True)
    dataset2 = simulator2.sample()

