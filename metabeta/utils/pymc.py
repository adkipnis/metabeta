from __future__ import annotations

from typing import Any

import numpy as np

from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF, hasSigmaEps


def buildPymc(ds: dict[str, np.ndarray], force_diagonal: bool = False) -> Any:
    """Build a PyMC GLMM for a single unpadded dataset.

    The PyMC dependency is imported lazily so this shipped helper can live in the
    base package without making ordinary imports require the scientific stack.
    """
    import pymc as pm
    import pytensor.tensor as pt

    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2 and not force_diagonal

    y_obs = ds['y'].astype(np.float64)
    X = ds['X'].astype(np.float64).copy()
    Z = X[:, :q].copy()
    groups = ds['groups'].astype(int)

    nu_ffx = ds['nu_ffx'].astype(float)
    tau_ffx = ds['tau_ffx'].astype(float)
    tau_rfx = ds['tau_rfx'].astype(float)

    ffx_family = FFX_FAMILIES[int(ds.get('family_ffx', 0))]
    sigma_family = SIGMA_FAMILIES[int(ds.get('family_sigma_rfx', 0))]
    eps_family = SIGMA_FAMILIES[int(ds.get('family_sigma_eps', 0))]
    likelihood = int(ds.get('likelihood_family', 0))

    def rlabel(j: int, suffix: str = '') -> str:
        return ('1|i' if j == 0 else f'x{j}|i') + suffix

    def half_rv(name: str, family: str, scale, as_dist: bool = False):
        if family == 'halfnormal':
            cls, kw = pm.HalfNormal, {'sigma': scale}
        elif family == 'halfstudent':
            cls, kw = pm.HalfStudentT, {'nu': STUDENT_DF, 'sigma': scale}
        elif family == 'exponential':
            cls, kw = pm.Exponential, {'lam': 1.0 / (np.asarray(scale) + 1e-12)}
        else:
            raise ValueError(f'unsupported sigma family: {family}')
        return cls.dist(**kw) if as_dist else cls(name, **kw)

    with pm.Model() as model:
        betas = []
        for j in range(d):
            name = 'Intercept' if j == 0 else f'x{j}'
            if ffx_family == 'normal':
                betas.append(pm.Normal(name, mu=nu_ffx[j], sigma=tau_ffx[j]))
            elif ffx_family == 'student':
                betas.append(pm.StudentT(name, nu=STUDENT_DF, mu=nu_ffx[j], sigma=tau_ffx[j]))
            else:
                raise ValueError(f'unsupported ffx family: {ffx_family}')

        if correlated:
            chol, _, sigma_vec = pm.LKJCholeskyCov(
                '_lkj_rfx',
                n=q,
                eta=float(ds['eta_rfx']),
                sd_dist=half_rv('', sigma_family, tau_rfx, as_dist=True),
                compute_corr=True,
            )
            for j in range(q):
                pm.Deterministic(rlabel(j, '_sigma'), sigma_vec[j])
            z = pm.Normal('_rfx_offset', 0.0, 1.0, shape=(m, q))
            b = pm.Deterministic('_rfx', pt.dot(z, chol.T))
            for j in range(q):
                pm.Deterministic(rlabel(j), b[:, j])
        else:
            cols = []
            for j in range(q):
                s = half_rv(rlabel(j, '_sigma'), sigma_family, float(tau_rfx[j]))
                z = pm.Normal(rlabel(j, '_offset'), 0.0, 1.0, shape=(m,))
                cols.append(pm.Deterministic(rlabel(j), z * s))
            b = pt.stack(cols, axis=1)

        mu = pt.dot(pt.as_tensor_variable(X), pt.stack(betas))
        mu = mu + (pt.as_tensor_variable(Z) * b[groups]).sum(axis=1)

        if likelihood == 0:
            sigma_eps = half_rv('sigma', eps_family, float(ds.get('tau_eps', 1.0)))
            pm.Normal('y_obs', mu=mu, sigma=sigma_eps, observed=y_obs)
        elif likelihood == 1:
            pm.Bernoulli('y_obs', logit_p=mu, observed=y_obs.astype(int))
        elif likelihood == 2:
            pm.Poisson('y_obs', mu=pt.exp(mu), observed=y_obs.astype(int))
        else:
            raise ValueError(f'unsupported likelihood_family: {likelihood}')

    return model


def extractSingle(trace: Any, name: str) -> np.ndarray:
    """Flatten (chains, draws, ...) into (1, n_s, ...)."""
    x = trace.posterior[name].to_numpy()
    x = x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
    return x[None, ...]


def extractAll(
    trace: Any,
    ds: dict[str, np.ndarray],
    d: int,
    q: int,
    prefix: str,
    force_diagonal: bool = False,
) -> dict[str, np.ndarray]:
    """Extract posterior arrays from a trace into the (1, n_s, ...) convention."""
    likelihood_family = int(ds.get('likelihood_family', 0))
    ffx = np.concatenate(
        [extractSingle(trace, 'Intercept' if j == 0 else f'x{j}') for j in range(d)], axis=0
    )
    sigma_rfx = np.concatenate(
        [extractSingle(trace, '1|i_sigma' if j == 0 else f'x{j}|i_sigma') for j in range(q)],
        axis=0,
    )
    rfx = np.concatenate(
        [extractSingle(trace, '1|i' if j == 0 else f'x{j}|i') for j in range(q)], axis=0
    ).swapaxes(2, 1)

    out = {f'{prefix}_ffx': ffx, f'{prefix}_sigma_rfx': sigma_rfx, f'{prefix}_rfx': rfx}
    if hasSigmaEps(likelihood_family):
        out[f'{prefix}_sigma_eps'] = extractSingle(trace, 'sigma')
    correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2 and not force_diagonal
    if correlated:
        out[f'{prefix}_corr_rfx'] = extractSingle(trace, '_lkj_rfx_corr')
    else:
        n_s = ffx.shape[-1]
        out[f'{prefix}_corr_rfx'] = np.tile(np.eye(q, dtype=ffx.dtype)[None, None], (1, n_s, 1, 1))
    return out
