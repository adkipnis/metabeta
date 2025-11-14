import torch
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import time


def extract(trace, name: str) -> torch.Tensor:
    x = trace.posterior[name].to_numpy()
    return torch.tensor(x).flatten(0, 1).movedim(0, -1)


def prepare(ds: dict[str, torch.Tensor], parameterization: str = 'default'):
    # unpack
    n = ds['n'].item()
    m = ds['m'].item()
    d = ds['d'].item()
    q = ds['q'].item()
    tau_eps = ds['tau_eps'].numpy()
    nu_ffx = ds['nu_ffx'][:d].numpy()
    tau_ffx = ds['tau_ffx'][:d].numpy()
    tau_rfx = ds['tau_rfx'][:q].numpy()
 
    # same parameterization as during generation
    if parameterization == 'default':
        with pm.Model() as model:
            # data
            y = pm.Data('y', ds['y'].numpy())
            X = pm.Data('X', ds['X'].numpy())
            Z = pm.Data('Z', ds['X'][:, :q].numpy())
            groups = pm.Data('groups', ds['groups'].numpy())

            # fixed effects
            ffx = pm.Normal('ffx', mu=nu_ffx, sigma=tau_ffx, shape=d)

            # additional random effects
            sigmas_rfx = pm.HalfNormal('sigmas_rfx', sigma=tau_rfx, shape=q)
            rfx_norm = pm.Normal('rfx_norm', mu=0.0, sigma=1.0, shape=(m,q))
            rfx = pm.Deterministic('rfx', rfx_norm * sigmas_rfx)

            # outcome
            mu = pt.dot(X, ffx) + pt.sum(Z * rfx[groups], axis=1)
            sigma_eps = pm.HalfNormal('sigma_eps', sigma=tau_eps)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_eps, observed=y, shape=n)
        return model

    elif parameterization == 'hierarchical':
        # hierarchical parameterization
        with pm.Model() as model:
            # data
            y = pm.Data("y", ds['y'].numpy())
            X = pm.Data("X", ds['X'][:, q:].numpy())
            Z = pm.Data("Z", ds['X'][:, :q].numpy())
            groups = pm.Data("groups", ds['groups'].numpy())

            # truly fixed effects
            ffx = pm.Normal("ffx", mu=nu_ffx[q:], sigma=tau_ffx[q:])

            # separate mixed effects
            beta = pm.Normal("beta", mu=nu_ffx[:q], sigma=tau_ffx[:q])
            sigmas_rfx = pm.HalfNormal("sigmas_rfx", sigma=tau_rfx)
            rfx_norm = pm.Normal("rfx_norm", mu=0, sigma=1, shape=(m,q))
            rfx = pm.Deterministic("rfx", rfx_norm * sigmas_rfx)
            mfx = beta + rfx

            # outcome
            mu = pt.dot(X, ffx) + pt.sum(Z * mfx[groups], axis=1)
            sigma_eps = pm.HalfNormal('sigma_eps', sigma=tau_eps)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_eps, observed=y, shape=n)
        return model
    else:
        raise ValueError


def fit(ds: dict[str, torch.Tensor],
        tune=2000,
        draws=1000,
        cores=4,
        seed=0,
        parameterization='default',
        method='advi') -> dict[str, torch.Tensor]:
    assert method in ['nuts', 'advi'], 'unknown method selected'
    assert parameterization in ['default', 'hierarchical'], 'unknown parameterization selected'
 
    # run pymc
    t0 = time.perf_counter()
    with prepare(ds, parameterization):
        if method == 'advi':
            mean_field = pm.fit(n=100_000, method='advi',
                                obj_optimizer=pm.adam(learning_rate=5e-3))
            trace = mean_field.sample(draws=draws*cores, random_seed=seed,
                                      return_inferencedata=True)
        elif method == 'nuts':
            trace = pm.sample(tune=tune, draws=draws, cores=cores,
                              random_seed=seed, return_inferencedata=True)
        else:
            raise ValueError
    t1 = time.perf_counter()

    # extract samples
    if parameterization == 'default':
        ffx = extract(trace, 'ffx')
    elif parameterization == 'hierarchical':
        ffx = torch.cat([extract(trace, 'beta'), extract(trace, 'ffx')])
    sigma_eps = extract(trace, 'sigma_eps').unsqueeze(0)
    sigmas_rfx = extract(trace, 'sigmas_rfx')
    rfx = extract(trace, 'rfx').movedim(0,1)

    # extract fit info
    divergent_count = torch.tensor(trace.sample_stats['diverging'].values.sum(-1))
    summary = az.summary(trace)
    ess = summary['ess_bulk'].to_numpy()
    rhat = summary['r_hat'].to_numpy()

    # finalize
    out = {
        'mcmc_ffx': ffx,
        'mcmc_sigma_eps': sigma_eps,
        'mcmc_sigmas_rfx': sigmas_rfx,
        'mcmc_rfx': rfx,
        'mcmc_divergences': divergent_count,
        'mcmc_ress': torch.tensor(ess),
        'mcmc_rhat': torch.tensor(rhat),
        'mcmc_duration': torch.tensor(t1 - t0)
    }
    return out

