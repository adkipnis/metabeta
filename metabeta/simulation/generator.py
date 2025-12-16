from dataclasses import dataclass
import numpy as np
from metabeta.simulation.utils import standardize
from metabeta.simulation import Prior, Synthesizer, Emulator


def sampleOutcomes(
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

