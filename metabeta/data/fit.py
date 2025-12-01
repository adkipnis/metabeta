import torch
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import bambi as bmb
import arviz as az
import time


# -----------------------------------------------------------------------------
# PyMC wrappers

def extractPymc(trace, name: str) -> torch.Tensor:
    x = trace.posterior[name].to_numpy()
    return torch.tensor(x).flatten(0, 1).movedim(0, -1)


def autoScalePriors(X: np.ndarray, y: np.ndarray, multiplier: float = 2.5):
    # automatic priors based on Bambi's procedure

    # init
    nu_ffx = np.zeros_like(X[0])
    tau_ffx = np.zeros_like(X[0])
    tau_rfx = np.zeros_like(X[0]) # mask out d-q later
    tau_eps = 0.

    # moments
    y_mean = np.mean(y)
    y_std = np.std(y)
    X_mean = np.mean(X[:, 1:], axis=0)
    X_std = np.std(X[:, 1:], axis=0)

    # slopes
    tau_ffx[1:] = multiplier * y_std / X_std

    # intercept
    nu_ffx[0] = y_mean
    tau_ffx[0] = ( (multiplier * y_std)**2 + np.dot(tau_ffx[1:]**2, X_mean**2) )**0.5

    # variance components
    tau_eps = y_std
    tau_rfx = tau_ffx

    return nu_ffx, tau_ffx, tau_rfx, tau_eps



def prepare(ds: dict[str, torch.Tensor],
            parameterization: str = 'matched', respecify_ffx: bool = True):
    # unpack for pymc
    n = ds['n'].item()
    m = ds['m'].item()
    d = ds['d'].item()
    q = ds['q'].item()
    tau_eps = ds['tau_eps'].numpy()
    nu_ffx = ds['nu_ffx'][:d].numpy()
    tau_ffx = ds['tau_ffx'][:d].numpy()
    tau_rfx = ds['tau_rfx'][:q].numpy()
    y = ds['y'].numpy()
    X = ds['X'].numpy()
    groups = ds['groups'].numpy()

    # bambi-like ffx priors
    if respecify_ffx:
        nu_ffx, tau_ffx, _, _ = autoScalePriors(X, y)
 
    # same parameterization as during generation
    if parameterization == 'matched':
        with pm.Model() as model:
            # data
            y = pm.Data('y', y) # (n, )
            Z = pm.Data('Z', X[:, :q]) # (n, q)
            X = pm.Data('X', X) # (n, d)

            groups = pm.Data('groups', groups) # (n, )

            # fixed effects
            ffx = pm.Normal('ffx', mu=nu_ffx, sigma=tau_ffx, shape=d)

            # additional random effects
            sigmas_rfx = pm.HalfNormal('sigmas_rfx', sigma=tau_rfx, shape=q)
            rfx_norm = pm.Normal('rfx_norm', mu=0.0, sigma=1.0, shape=(m,q))
            rfx = pm.Deterministic('rfx', rfx_norm * sigmas_rfx[None, :])

            # outcome
            mu = pt.dot(X, ffx) + pt.sum(Z * rfx[groups], axis=1)
            # sigma_eps = pm.HalfNormal('sigma_eps', sigma=tau_eps)
            sigma_eps = pm.HalfStudentT('sigma_eps', nu=4, sigma=tau_eps)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_eps, observed=y, shape=n)
        return model

    # hierarchical parameterization like in pymc example for simpsons paradoxon
    elif parameterization == 'hierarchical':
        with pm.Model() as model:
            # data
            y = pm.Data("y", y)
            Z = pm.Data("Z", X[:, :q])
            X = pm.Data("X", X[:, q:])
            groups = pm.Data("groups", groups)

            # truly fixed effects
            ffx = pm.Normal("ffx", mu=nu_ffx[q:], sigma=tau_ffx[q:], shape=d-q)

            # separate mixed effects
            beta = pm.Normal("beta", mu=nu_ffx[:q], sigma=tau_ffx[:q], shape=q)
            sigmas_rfx = pm.HalfNormal("sigmas_rfx", sigma=tau_rfx, shape=q)
            rfx_norm = pm.Normal("rfx_norm", mu=0, sigma=1, shape=(m,q))
            rfx = pm.Deterministic("rfx", rfx_norm * sigmas_rfx[None, :])
            mfx = beta[None, :] + rfx

            # outcome
            mu = pt.dot(X, ffx) + pt.sum(Z * mfx[groups], axis=1)
            # sigma_eps = pm.HalfNormal('sigma_eps', sigma=tau_eps)
            sigma_eps = pm.HalfStudentT('sigma_eps', nu=4, sigma=tau_eps)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_eps, observed=y, shape=n)
        return model
    else:
        raise ValueError


def fitPyMC(ds: dict[str, torch.Tensor],
        tune=2500,
        draws=1000,
        chains=4,
        seed=0,
        method='nuts',
        respecify_ffx: bool = True,
        parameterization: str = 'matched',
        use_multiprocessing: bool = True,
        ) -> dict[str, torch.Tensor]:
    assert method in ['nuts', 'advi'], 'unknown method selected'
    assert parameterization in ['matched', 'hierarchical'], 'unknown parameterization selected'
 
    # run pymc
    t0 = time.perf_counter()
    with prepare(ds, parameterization, respecify_ffx) as model:
        if method == 'nuts':
            trace = pm.sample(tune=tune, draws=draws, chains=chains,
                              cores=(chains if use_multiprocessing else 1),
                              random_seed=seed,
                              return_inferencedata=True)
        elif method == 'advi':
            mean_field = pm.fit(method='advi', 
                                n=100_000, 
                                obj_optimizer=pm.adam(learning_rate=5e-3)
                                )
            trace = mean_field.sample(draws=draws*chains, 
                                      random_seed=seed,
                                      return_inferencedata=True)  
    t1 = time.perf_counter()

    # extract samples
    if parameterization == 'matched':
        ffx = extractPymc(trace, 'ffx')
    elif parameterization == 'hierarchical':
        ffx = torch.cat([extractPymc(trace, 'beta'), extractPymc(trace, 'ffx')])
    sigma_eps = extractPymc(trace, 'sigma_eps').unsqueeze(0)
    sigmas_rfx = extractPymc(trace, 'sigmas_rfx')
    rfx = extractPymc(trace, 'rfx').movedim(0,1)

    out = {
        f'{method}_ffx': ffx,
        f'{method}_sigma_eps': sigma_eps,
        f'{method}_sigmas_rfx': sigmas_rfx,
        f'{method}_rfx': rfx,
        f'{method}_duration': torch.tensor(t1 - t0),
        }

    # extract fit info
    summary = az.summary(trace, kind='diagnostics')
    out[f'{method}_ess'] = torch.tensor(summary['ess_bulk'].to_numpy())
    if method == 'nuts':
        divergent_count = torch.tensor(trace.sample_stats['diverging'].values.sum(-1))
        rhat = torch.tensor(summary['r_hat'].to_numpy())
        out.update({
            f'{method}_divergences': divergent_count,
            f'{method}_rhat': rhat,
            })

    # # unnormalized log posterior
    # if 'ffx' in ds:
    #     logp_fn = model.compile_logp(sum=False)
    #     logp_ffx, logp_sigmas_rfx, logp_rfx, logp_sigma_eps, logp_y = logp_fn({
    #         'ffx': ds['ffx'],
    #         'rfx_norm': ds['rfx']/ds['sigmas_rfx'],
    #         'sigmas_rfx_log__': ds['sigmas_rfx'].log(),
    #         'sigma_eps_log__': ds['sigma_eps'].log(),
    #         })
    #     out.update({
    #         f'{method}_logp_ffx': logp_ffx,
    #         f'{method}_logp_sigmas_rfx': logp_sigmas_rfx,
    #         f'{method}_logp_rfx': logp_rfx,
    #         f'{method}_logp_sigma_eps': logp_sigma_eps,
    #         })
    return out


# -----------------------------------------------------------------------------
# Bambi wrappers
def pandify(ds: dict[str, torch.Tensor]) -> pd.DataFrame:
    n, d = int(ds['n']), int(ds['d'])
    df = pd.DataFrame(index=range(n))
    df['i'] = ds['groups'].numpy()
    df['y'] = ds['y'].unsqueeze(-1).numpy()
    for j in range(1, d):
        df[f'x{j}'] = ds['X'][..., j].numpy()
    return df


def formulate(ds: dict[str, torch.Tensor]) -> str:
    d, q = int(ds['d']), int(ds['q'])
    fixed = ' + '.join(f'x{j}' for j in range(1, d))
    random = ' + '.join(f'x{j}' for j in range(1, q))
    if not random:
        random = '1'
    out = f'y ~ 1 + {fixed} + ({random} | i)'
    return out


def priorize(ds: dict[str, torch.Tensor], respecify_ffx: bool = False) -> dict[str, bmb.Prior]:
    # unpack
    d, q = int(ds['d']), int(ds['q'])
    nu_ffx = ds['nu_ffx'][:d].numpy()
    tau_ffx = ds['tau_ffx'][:d].numpy()
    tau_rfx = ds['tau_rfx'][:q].numpy()
    tau_eps = ds['tau_eps'].numpy()
    include_ffx = not respecify_ffx
    priors = {}

    # fixed effects
    if include_ffx:
        priors.update({
            'Intercept': bmb.Prior('Normal', mu=nu_ffx[0], sigma=tau_ffx[0])
            })
        priors.update({
            f'x{j}': bmb.Prior('Normal', mu=nu_ffx[j], sigma=tau_ffx[j])
            for j in range(1, d)
            })

    #  noise variance and random intercept variance
    priors.update({
        # 'sigma': bmb.Prior('HalfNormal', sigma=tau_eps),
        'sigma': bmb.Prior('HalfStudentT', nu=4, sigma=tau_eps),
        '1|i': bmb.Prior('Normal', mu=0, sigma=bmb.Prior('HalfNormal', sigma=tau_rfx[0])),
        })

    # random slope variances
    priors.update({
        f'x{j}|i': bmb.Prior('Normal', mu=0, sigma=bmb.Prior('HalfNormal', sigma=tau_rfx[j]))
        for j in range(1, q)
        })
 
    return priors


def bambify(ds: dict[str, torch.Tensor], specify_priors=True, respecify_ffx=False) -> bmb.Model:
    data = pandify(ds)
    form = formulate(ds)
    priors = priorize(ds, respecify_ffx=respecify_ffx) if specify_priors else None
    model = bmb.Model(form, data, categorical='i', priors=priors)
    model.build()
    return model


def extractBambi(trace, name: str) -> torch.Tensor:
    x = trace.posterior[name].to_numpy()
    shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
    return torch.tensor(x.reshape(shape)).unsqueeze(0)


def fitBambi(ds: dict[str, torch.Tensor],
             tune: int = 2500,
             draws: int = 1000,
             chains: int = 4,
             seed: int = 0,
             method='nuts',
             respecify_ffx: bool = True,
             specify_priors: bool = True,
             use_multiprocessing: bool = True,
             ) -> dict[str, torch.Tensor]:
    assert method in ['nuts', 'advi'], 'unknown method selected'
    d, q = int(ds['d']), int(ds['q'])
    model = bambify(ds, specify_priors=specify_priors, respecify_ffx=respecify_ffx)
 
    t0 = time.perf_counter()
    if method == 'nuts':
        trace = model.fit(draws=draws, tune=tune, chains=chains,
                          cores=(chains if use_multiprocessing else 1),
                          inference_method='pymc',
                          random_seed=seed,
                          return_inferencedata=True)
    elif method == 'advi':
        mean_field = model.fit(inference_method='vi',
                               n=100_000, 
                               obj_optimizer=pm.adam(learning_rate=5e-3),
                               )

        trace = mean_field.sample(draws=draws*chains, 
                                  random_seed=seed,
                                  return_inferencedata=True)
    t1 = time.perf_counter()

    # extract samples
    ffx = torch.cat(
        [extractBambi(trace, 'Intercept')] +
        [extractBambi(trace, f'x{j}') for j in range(1, d)],
    )
    sigmas_rfx = torch.cat(
        [extractBambi(trace, '1|i_sigma')] +
        [extractBambi(trace, f'x{j}|i_sigma') for j in range(1, q)],
    )
    sigma_eps = extractBambi(trace, 'sigma')
    rfx = torch.cat(
        [extractBambi(trace, '1|i')] +
        [extractBambi(trace, f'x{j}|i') for j in range(1, q)],
    ).movedim(2, 1)
 
    # finalize
    out = {
        f'{method}_ffx': ffx,
        f'{method}_sigma_eps': sigma_eps,
        f'{method}_sigmas_rfx': sigmas_rfx,
        f'{method}_rfx': rfx,
        f'{method}_duration': torch.tensor(t1 - t0),
    }

    # extract fit info
    summary = az.summary(trace, kind='diagnostics')
    out[f'{method}_ess'] = torch.tensor(summary['ess_bulk'].to_numpy())
    if method == 'nuts':
        divergent_count = torch.tensor(trace.sample_stats['diverging'].values.sum(-1))
        rhat = torch.tensor(summary['r_hat'].to_numpy())
        out.update({
            f'{method}_divergences': divergent_count,
            f'{method}_rhat': rhat,
            })

    return out

