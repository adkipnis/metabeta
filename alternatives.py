from typing import Dict
import numpy as np
import torch
import pymc as pm
from pymc import variational
import pytensor.tensor as pt

def fitFfxVI(y: torch.Tensor, X: torch.Tensor) -> variational.approximations.MeanField:
    ''' perform variational inference with automatic diffenentiation '''
    d = X.shape[1]
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0., sigma=3., shape=d) # priors
        mu = pt.dot(X.numpy(), beta)
        sigma = pm.HalfNormal("sigma", sigma=1.)  # Noise standard deviation
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y.numpy())
        approx = pm.fit(method="advi", n=10000)  # Use ADVI with 10,000 iterations
    return approx

def fitMfxVI(y: torch.Tensor, X: torch.Tensor, groups: torch.Tensor, q: int, m: int) -> variational.approximations.MeanField:
    ''' perform variational inference with automatic diffenentiation '''
    d = X.shape[1]
    Z = X[:,:q]
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0., sigma=3., shape=d) # priors
        sigma_b = pm.HalfNormal("sigma_b", sigma=1., shape=q) # rfx SD
        b = pm.Normal("b", mu=0., sigma=sigma_b, shape=(m, q)) # rfx
        B = b[groups.numpy()]
        mu = pt.dot(X.numpy(), beta) + pt.sum(Z.numpy() * B, axis=1) # linear predictor
        sigma_e = pm.HalfNormal("sigma_e", sigma=1.)  # noise SD
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_e, observed=y.numpy())
        approx = pm.fit(method="advi", n=10000) 
    return approx

def evalVI(approx: variational.approximations.MeanField,
           param_name: str,
           n_samples: int = 1000,
           ) -> Dict[str, torch.Tensor]:
    posterior_samples = approx.sample(n_samples).posterior[param_name][0] # type: ignore
    means = posterior_samples.mean(dim='draw').to_numpy()
    stds = posterior_samples.std(dim='draw').to_numpy()
    ci95 = np.percentile(posterior_samples.to_numpy(), [2.5, 50., 97.5], axis=0)
    return {'loc': torch.tensor(means),
            'scale': torch.tensor(stds),
            'ci95': torch.tensor(ci95),}

def allVariationalPosteriors(y: torch.Tensor,
                             X: torch.Tensor,
                             groups: torch.Tensor,
                             q: int,
                             m: int) -> Dict[str, torch.Tensor]:
    approx = fitMfxVI(y, X, groups, q, m)
    ffx_quantiles   = evalVI(approx, "beta")["ci95"].T
    rfx_quantiles   = evalVI(approx, "sigma_b")["ci95"].T
    noise_quantiles = evalVI(approx, "sigma_e")["ci95"]
    return {"ffx_vp":   ffx_quantiles,
            "rfx_vp":   rfx_quantiles,
            "noise_vp": noise_quantiles}


