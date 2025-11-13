import pandas as pd
import torch
import bambi as bmb
from bambi import Prior
import arviz as az
import time

from torch import distributions as D
import matplotlib.pyplot as plt

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


def priorize(ds: dict[str, torch.Tensor]) -> dict[str, Prior]:
    # unpack
    d, q = int(ds['d']), int(ds['q'])
    nu_ffx = ds['nu_ffx'][:d].numpy()
    tau_ffx = ds['tau_ffx'][:d].numpy()
    tau_rfx = ds['tau_rfx'][:q].numpy()
    tau_eps = ds['tau_eps'].numpy()
 
    # setup priors
    priors = {
        'Intercept': Prior('Normal', mu=nu_ffx[0], sigma=tau_ffx[0]),
        '1|i': Prior('Normal', mu=0, sigma=Prior('HalfNormal', sigma=tau_rfx[0])),
        'sigma': Prior('HalfNormal', sigma=tau_eps)
    }

    # fixed slopes
    priors.update(
        {f'x{j}': Prior('Normal', mu=nu_ffx[j], sigma=tau_ffx[j])
         for j in range(1, d)}
    )

    # random slopes
    priors.update(
        {f'x{j}|i': Prior('Normal', mu=0, sigma=Prior('HalfNormal', sigma=tau_rfx[j]))
         for j in range(1, q)}
    )
 
    return priors


def bambify(ds: dict[str, torch.Tensor]) -> bmb.Model:
    data = pandify(ds)
    form = formulate(ds)
    priors = priorize(ds)
    model = bmb.Model(form, data, categorical='i', priors=priors)
    model.build()
    return model


def extract(trace, name: str) -> torch.Tensor:
    x = trace.posterior[name].to_numpy()
    shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
    return torch.tensor(x.reshape(shape)).unsqueeze(0)


def fitMCMC(ds: dict[str, torch.Tensor],
            tune: int = 2000,
            draws: int = 1000,
            chains: int = 4,
            seed: int = 0,
            **kwargs) -> dict[str, torch.Tensor]:
    d, q = int(ds['d']), int(ds['q'])
    t0 = time.perf_counter()
    model = bambify(ds)
    trace = model.fit(draws=draws, tune=tune, chains=chains,
                      random_seed=seed, **kwargs)
    t1 = time.perf_counter()

    # extract samples
    ffx = torch.cat(
        [extract(trace, 'Intercept')] +
        [extract(trace, f'x{j}') for j in range(1, d)],
    )
    sigmas_rfx = torch.cat(
        [extract(trace, '1|i_sigma')] +
        [extract(trace, f'x{j}|i_sigma') for j in range(1, q)],
    )
    sigma_eps = extract(trace, 'y_sigma')
    rfx = torch.cat(
        [extract(trace, '1|i')] +
        [extract(trace, f'x{j}|i') for j in range(1, q)],
    ).movedim(2, 1)
 
    # extract fit info
    divergent_count = torch.tensor(trace.sample_stats['diverging'].values.sum(-1)) # type: ignore
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
        'mcmc_ess': torch.tensor(ess),
        'mcmc_rhat': torch.tensor(rhat),
        'mcmc_duration': torch.tensor(t1 - t0)
    }
    return out


