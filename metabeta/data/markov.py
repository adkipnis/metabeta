import torch
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import time


def extract(trace, name: str) -> torch.Tensor:
    x = trace.posterior[name].to_numpy()
    return torch.tensor(x).flatten(0, 1).movedim(0, -1)


def prepare(ds: dict[str, torch.Tensor], mono=False):
    # inpack
    m = ds['m'].item()
    d = ds['d'].item()
    q = ds['q'].item()

    y = ds['y']  # (n, )
    X = ds['X']  # (n, d)
    Z = X[..., :q]  # (n, q)
    groups = ds['groups']  # (n, )

    tau_eps = ds['tau_eps'].numpy()
    nu_ffx = ds['nu_ffx'][:d].numpy()
    tau_ffx = ds['tau_ffx'][:d].numpy()
    tau_rfx = ds['tau_rfx'][:q].numpy()

    # optionally use one-dimensional priors
    if mono:  
        nu_ffx = nu_ffx[0]
        tau_ffx = tau_ffx[0]
        tau_rfx = tau_rfx[0]

    # specify model
    with pm.Model() as model:
        y_shared = pm.Data('y', y.numpy())
        X_shared = pm.Data('X', X.numpy())
        Z_shared = pm.Data('Z', Z.numpy())
        groups_shared = pm.Data('groups', groups.numpy())

        beta = pm.Normal('beta', mu=nu_ffx, sigma=tau_ffx, shape=d)
        sigma_a = pm.HalfNormal('sigma_alpha', sigma=tau_rfx, shape=q)
        z = pm.Normal('z', mu=0.0, sigma=1.0, shape=(m, q))
        alpha = pm.Deterministic('alpha', z * sigma_a)
        A = alpha[groups_shared]

        mu = pt.dot(X_shared, beta) + pt.sum(Z_shared * A, axis=1)  # linear predictor
        sigma_e = pm.HalfNormal('sigma_eps', sigma=tau_eps)  # noise SD
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_e, observed=y_shared)
    return model


def fitMCMC(ds: dict[str, torch.Tensor], 
            tune=1000,
            draws=1000,
            cores=4,
            mono=False,
            sampler: str = 'nuts',
            seed=0) -> dict[str, torch.Tensor]:
    assert sampler in ['hmc', 'nuts'], f'unknown sampler: {sampler}'

    # run mcmc
    t0 = time.perf_counter()
    with prepare(ds, mono=mono):
        step = pm.HamiltonianMC() if sampler == 'hmc' else None
        trace = pm.sample(tune=tune, draws=draws, cores=cores,
                          random_seed=seed, step=step)
    t1 = time.perf_counter()

    # extract samples
    ffx = extract(trace, 'beta')
    sigma_eps = extract(trace, 'sigma_eps').unsqueeze(0)
    sigmas_rfx = extract(trace, 'sigma_alpha')
    rfx = extract(trace, 'alpha').movedim(0, 1)

    # extract fit info
    divergent_count = torch.tensor(trace.sample_stats['diverging'].values.sum(-1)) # type: ignore
    summary = torch.tensor(az.summary(trace).to_numpy())

    # finalize
    out = {
        'mcmc_ffx': ffx,
        'mcmc_sigma_eps': sigma_eps,
        'mcmc_sigmas_rfx': sigmas_rfx,
        'mcmc_rfx': rfx,
        'mcmc_divergences': divergent_count,
        'mcmc_summary': summary,
        'mcmc_duration': torch.tensor(t1 - t0)
    }
    return out

