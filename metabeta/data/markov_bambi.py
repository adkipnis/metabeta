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
