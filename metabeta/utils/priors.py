import numpy as np

from metabeta.utils.constants import hasSigmaEps

# Smallest and largest value the network saw of each hyperparameter it conditions on, by
# likelihood family, rounded outward: the stored values of the sampled test and validation sets
# of every size, which share the training generator (simulation/prior.py, hypersample;
# experiments/simulation/prior_misspec.py, trainedRange). Stored values are standardized, so
# the range is not the drawn one: Gaussian scales are divided by sd(y), which widens it, and
# Bernoulli and Poisson scales are shrunk by the linear-predictor cap, which ends it below the
# draw's bound. nu_ffx is symmetric about 0; eta_rfx is its draw's (0 means uncorrelated).
TRAINED_RANGE: dict[int, dict[str, tuple[float, float]]] = {
    0: {
        'nu_ffx': (-3.42, 3.42),
        'tau_ffx': (0.0015, 44.35),
        'tau_rfx': (0.0017, 17.74),
        'tau_eps': (0.107, 1263.3),
        'eta_rfx': (1.0, 2.0),
    },
    1: {
        'nu_ffx': (-5.35, 5.35),
        'tau_ffx': (0.013, 2.77),
        'tau_rfx': (0.021, 1.91),
        'eta_rfx': (1.0, 2.0),
    },
    2: {
        'nu_ffx': (-3.71, 3.71),
        'tau_ffx': (0.0074, 1.42),
        'tau_rfx': (0.010, 0.93),
        'eta_rfx': (1.0, 2.0),
    },
}

# The Api's default scales where Bambi's 2.5 lies outside TRAINED_RANGE: the largest round
# value with at least 1% of the trained scales above it (issue #53).
API_DEFAULT_SCALES: dict[int, dict[str, float]] = {
    1: {'tau_rfx': 1.5},
    2: {'tau_ffx': 1.0, 'tau_rfx': 0.75},
}


def apiDefaultPriors(d: int, q: int, likelihood_family: int = 0) -> dict[str, np.ndarray]:
    """Bambi's defaults (bambiDefaultPriors) with the scales the network was not trained on
    lowered into its range (API_DEFAULT_SCALES)."""
    out = bambiDefaultPriors(d, q, likelihood_family)
    for key, value in API_DEFAULT_SCALES.get(likelihood_family, {}).items():
        out[key] = np.full_like(out[key], value)
    return out


def outsideTrainedRange(hyper: dict[str, np.ndarray], likelihood_family: int) -> np.ndarray:
    """(B,) per dataset, its hyperparameters outside TRAINED_RANGE, e.g. 'tau_rfx 2.5 outside
    [0.021, 1.91]', and '' where all lie inside.  ``hyper`` holds batch-stage arrays: nu_ffx and
    tau_ffx (B, d) with mask_d, tau_rfx (B, q) with mask_q, tau_eps and eta_rfx (B,)."""
    B = len(hyper['mask_d'])
    masks = {'nu_ffx': hyper['mask_d'], 'tau_ffx': hyper['mask_d'], 'tau_rfx': hyper['mask_q']}
    found: list[list[str]] = [[] for _ in range(B)]
    for key, (lo, hi) in TRAINED_RANGE[likelihood_family].items():
        if key not in hyper:
            continue
        v = np.asarray(hyper[key], dtype=float).reshape(B, -1)  # (B, k)
        if key in masks:
            active = np.asarray(masks[key], dtype=bool)
        else:  # eta_rfx 0 marks uncorrelated random effects
            active = v > 0 if key == 'eta_rfx' else np.ones_like(v, dtype=bool)
        outside = active & ((v < lo) | (v > hi))
        for i in np.flatnonzero(outside.any(1)):
            values = ', '.join(f'{x:g}' for x in np.unique(v[i, outside[i]]))
            found[i].append(f'{key} {values} outside [{lo:g}, {hi:g}]')
    return np.array(['; '.join(f) for f in found])


def bambiDefaultPriors(
    d: int,
    q: int,
    likelihood_family: int = 0,
) -> dict[str, np.ndarray]:
    """Replicate bambi's default priors for a hierarchical regression model.

    Assumes predictors are standardized (mean 0, sd 1) and, for gaussian
    likelihood, y is normalized to unit sd. Under these conditions bambi assigns:

        Gaussian:
            Intercept/slopes  ~ Normal(0, 2.5)
            sigma_rfx         ~ HalfNormal(2.5)
            sigma_eps          ~ HalfStudentT(nu=4, sigma=1)
        Bernoulli:
            Intercept          ~ Normal(0, 1.5)
            Slopes             ~ Normal(0, 1.0)
            sigma_rfx          ~ HalfNormal(2.5)
        Poisson:
            Intercept/slopes   ~ Normal(0, 2.5)
            sigma_rfx          ~ HalfNormal(2.5)
        corr_rfx               ~ LKJ(1)
    """
    out: dict[str, np.ndarray] = {}
    out['likelihood_family'] = np.array(likelihood_family)

    out['nu_ffx'] = np.zeros(d)
    if likelihood_family == 1:
        tau = np.full(d, 1.0)
        tau[0] = 1.5
    else:
        tau = np.full(d, 2.5)
    out['tau_ffx'] = tau

    out['tau_rfx'] = np.full(q, 2.5)

    if hasSigmaEps(likelihood_family):
        out['tau_eps'] = np.array(1.0)

    out['eta_rfx'] = np.array(1.0) if q > 1 else np.array(0.0)

    out['family_ffx'] = np.array(0)
    out['family_sigma_rfx'] = np.array(0)
    if hasSigmaEps(likelihood_family):
        out['family_sigma_eps'] = np.array(1)

    return out
