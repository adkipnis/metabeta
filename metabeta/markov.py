import torch
import pymc as pm
import pytensor.tensor as pt

def extract(posterior, name: str) -> torch.Tensor:
    return torch.tensor(posterior[name].to_numpy()).flatten(0, 1).movedim(0, -1)

# MCMC for FFX models (for comparison)
def prepareFFX(y: torch.Tensor, X: torch.Tensor, *args):
    d = X.shape[-1]
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0., sigma=3., shape=d) # priors
        mu = pt.dot(X.numpy(), beta) # linear predictor
        sigma_e = pm.HalfNormal("sigma_e", sigma=1.)  # noise SD
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_e, observed=y.numpy())
    return model

def fitFFX(y: torch.Tensor, X: torch.Tensor, **kwargs):
    with prepareFFX(y, X):
        step = pm.HamiltonianMC()
        trace = pm.sample(tune=1000, draws=1000, cores=4, return_inferencedata=True, step=step)
        posterior = trace.posterior
        ffx = extract(posterior, 'beta')
        sigma_error = extract(posterior, 'sigma_e').unsqueeze(0)
    return dict(ffx=ffx, sigma_error=sigma_error, sigmas_rfx=None, rfx=None)


# MCMC for MFX models
def prepareMFX(y: torch.Tensor, X: torch.Tensor, groups: torch.Tensor, q: int, m: int, *args):
    d = X.shape[-1]
    Z = X[..., :q]
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0., sigma=3., shape=d) # priors
        sigma_b = pm.HalfNormal("sigma_b", sigma=1., shape=q) # rfx SD
        b = pm.Normal("b", mu=0., sigma=sigma_b, shape=(m, q)) # rfx
        B = b[groups.numpy()]
        mu = pt.dot(X.numpy(), beta) + pt.sum(Z.numpy() * B, axis=1) # linear predictor
        sigma_e = pm.HalfNormal("sigma_e", sigma=1.)  # noise SD
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_e, observed=y.numpy())
    return model

def fitMFX(y: torch.Tensor, X: torch.Tensor, groups: torch.Tensor, q: int, m: int, **kwargs):
    with prepareMFX(y, X, groups, int(q), int(m)):
        step = pm.HamiltonianMC()
        trace = pm.sample(tune=1000, draws=1000, cores=4, return_inferencedata=True, step=step)
        posterior = trace.posterior
        ffx = extract(posterior, 'beta')
        sigma_error = extract(posterior, 'sigma_e').unsqueeze(0)
        sigmas_rfx = extract(posterior, 'sigma_b')
        rfx = extract(posterior, 'b').movedim(0, 1)
    return dict(ffx=ffx, sigma_error=sigma_error, sigmas_rfx=sigmas_rfx, rfx=rfx)

