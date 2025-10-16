import torch
import pymc as pm
import pytensor.tensor as pt


def extract(posterior, name: str) -> torch.Tensor:
    return torch.tensor(posterior[name].to_numpy()).flatten(0, 1).movedim(0, -1)


# MCMC for FFX models (for comparison)
def prepareFFX(ds: dict[str, torch.Tensor]):
    nu_ffx, tau_ffx, tau_eps = ds["nu_ffx"], ds["tau_ffx"], ds["tau_eps"]
    X, y = ds["X"], ds["y"]
    d = X.shape[-1]
    with pm.Model() as model:
        beta = pm.Normal(
            "beta", mu=nu_ffx.numpy(), sigma=tau_ffx.numpy(), shape=d
        )  # priors
        mu = pt.dot(X.numpy(), beta)  # linear predictor
        sigma_e = pm.HalfNormal("sigma_e", sigma=tau_eps.numpy())  # noise SD
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_e, observed=y.numpy())
    return model


def fitFFX(ds: dict[str, torch.Tensor], tune=1000, draws=1000):
    with prepareFFX(ds):
        step = pm.HamiltonianMC()
        trace = pm.sample(
            tune=tune, draws=draws, cores=4, return_inferencedata=True, step=step
        )
        posterior = trace.posterior
        ffx = extract(posterior, "beta")
        sigma_eps = extract(posterior, "sigma_e").unsqueeze(0)
    out = {
        "mcmc_ffx": ffx,
        "mcmc_sigma_eps": sigma_eps,
    }
    return out


# MCMC for MFX models
def prepareMFX(ds: dict[str, torch.Tensor], mono=False):
    m = ds["m"].item()
    d = ds["d"].item()
    q = ds["q"].item()
    tau_eps = ds["tau_eps"].numpy()
    nu_ffx = ds["nu_ffx"][:d].numpy()
    tau_ffx = ds["tau_ffx"][:d].numpy()
    tau_rfx = ds["tau_rfx"][:q].numpy()
    if mono:  # use one-dimensional priors, assumes that all elements are the same
        nu_ffx = nu_ffx[0]
        tau_ffx = tau_ffx[0]
        tau_rfx = tau_rfx[0]
    y = ds["y"]  # (n, )
    X = ds["X"]  # (n, d)
    Z = X[..., :q]  # (n, q)
    groups = ds["groups"]  # (n, )
    # note that no padding has been applied to the observations, yet

    with pm.Model() as model:
        y_shared = pm.Data("y", y.numpy())
        X_shared = pm.Data("X", X.numpy())
        Z_shared = pm.Data("Z", Z.numpy())
        groups_shared = pm.Data("groups", groups.numpy())

        beta = pm.Normal("beta", mu=nu_ffx, sigma=tau_ffx, shape=d)  # priors
        sigma_a = pm.HalfNormal("sigma_alpha", sigma=tau_rfx, shape=q)  # rfx SD
        # alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_a, shape=(m, q))  # rfx
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(m, q)) # non-central parameterization
        alpha = pm.Deterministic("alpha", z * sigma_a) # rfx
        A = alpha[groups_shared]

        mu = pt.dot(X_shared, beta) + pt.sum(Z_shared * A, axis=1)  # linear predictor
        sigma_e = pm.HalfNormal("sigma_eps", sigma=tau_eps)  # noise SD
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_e, observed=y_shared)
    return model


def fitMFX(ds: dict[str, torch.Tensor], tune=1000, draws=1000, mono=False):
    with prepareMFX(ds, mono=mono):
        step = pm.HamiltonianMC()
        trace = pm.sample(
            tune=tune, draws=draws, cores=4, return_inferencedata=True, step=step
        )
        divergent_count = torch.tensor(trace.sample_stats["diverging"].values.sum(-1))
        posterior = trace.posterior
        ffx = extract(posterior, "beta")
        sigma_eps = extract(posterior, "sigma_e").unsqueeze(0)
        if int(ds["q"]) > 0:
            sigmas_rfx = extract(posterior, "sigma_b")
            rfx = extract(posterior, "b").movedim(0, 1)
        else:
            sigmas_rfx = torch.zeros((1, 4000))
            rfx = torch.zeros((1, 1, 4000))
    out = {
        "mcmc_ffx": ffx,
        "mcmc_sigma_eps": sigma_eps,
        "mcmc_sigmas_rfx": sigmas_rfx,
        "mcmc_rfx": rfx,
        "mcmc_divergences": divergent_count,
    }
    return out

