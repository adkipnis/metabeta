import numpy as np

from metabeta.utils.constants import hasSigmaEps


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
