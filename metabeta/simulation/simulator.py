import numpy as np
from metabeta.simulation.utils import standardize


def simulate(parameters: dict[str, np.ndarray],
             observations: dict[str, np.ndarray],
             ) -> np.ndarray:
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


    # # standardize observations
    # X_ = standardize(X, axis=0, exclude_binary=True)
    # Z_ = X_[:, :q]
    #
    # # outcome
    # y = X_ @ ffx + (Z_ * rfx[groups]).sum(-1) + eps

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    from metabeta.simulation.utils import sampleCounts
    from metabeta.simulation.synthesizer import Synthesizer
    from metabeta.simulation.prior import Prior, hypersample

    # set seed
    seed = 0
    _ = np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # dims
    n = 100
    m = 5
    d = 3
    q = 2
    ns = sampleCounts(n, m)

    # sample observations
    synthesizer = Synthesizer(rng)
    obs = synthesizer.sample(d, ns)

    # sample parameters
    prior = Prior(*hypersample(d, q), rng=rng)
    params = prior.sample(m)

    # forward pass
    y = simulate(params, obs)

