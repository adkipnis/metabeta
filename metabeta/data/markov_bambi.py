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


# ----- simulation 
def parameters(ds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])

    # fixed effects
    ffx = torch.randn(d)
    ffx = (ffx - ffx.mean())/ffx.std()
    ffx = ffx * ds['tau_ffx'] + ds['nu_ffx']

    # variances
    sigmas_rfx = D.HalfNormal(ds['tau_rfx']).sample((1,))
    sigma_eps = D.HalfNormal(ds['tau_eps']).sample()

    # random effects
    rfx = torch.randn(m, q)
    rfx = (rfx - rfx.mean(0, keepdim=True)) / rfx.std(0, keepdim=True) * sigmas_rfx # type: ignore

    out = dict(ffx=ffx, sigmas_rfx=sigmas_rfx, sigma_eps=sigma_eps, rfx=rfx)
    ds.update(out)
    return ds


def outcome(ds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    n, q = int(ds['n']), int(ds['q'])
    X = ds['X']
    Z = X[:, :q]
    groups = ds['groups']
    ffx = ds['ffx']
    rfx = ds['rfx']
    sigma_eps = ds['sigma_eps']
    eps = torch.randn(n)
    eps = (eps - eps.mean()) / eps.std() * sigma_eps
    mu = X @ ffx + (Z * rfx[groups]).sum(-1)
    ds['y'] = mu + eps
    return ds


def plot_data(data: pd.DataFrame):
    m = len(df['i'].unique())
    fig, axes = plt.subplots(
        2, m//2, figsize=(m * 2.5, 10), dpi=300,
        sharey=True, sharex=True, constrained_layout=False)
    fig.subplots_adjust(
        left=0.075, right=0.975, bottom=0.075, top=0.925, wspace=0.03)
    axes_flat = axes.ravel()

    for i, subject in enumerate(data["i"].unique()):
        ax = axes_flat[i]
        idx = data.index[data["i"] == subject].tolist()
        x1 = data.loc[idx, "x1"].values
        y = data.loc[idx, "y"].values

        # Plot observed data points
        ax.scatter(x1, y, color="C0", ec="black", alpha=0.7)

        # Add a title
        ax.set_title(f"Subject: {i}", fontsize=14)

        # Add a grid
        ax.grid(True, linestyle='--', linewidth=0.5)

        # Add lines at x=0 and y=0
        ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.axvline(0, color='black', linestyle=':', linewidth=1)

    ax.xaxis.set_ticks([0, 2, 4, 6, 8]) # type: ignore
    fig.text(0.5, 0.02, "x1", fontsize=14)
    fig.text(0.03, 0.5, "y", rotation=90, fontsize=14, va="center")

    return axes

if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    n = 100
    m = 10
    d = 2
    q = 2
    ds = {
        'n': torch.tensor(n), 'm': torch.tensor(m),
        'd': torch.tensor(d), 'q': torch.tensor(q),
        'X': torch.cat([torch.ones(n, 1), torch.randn(n, d-1)], dim=-1),
        'groups': torch.randint(low=0, high=m, size=(n,)),
        'nu_ffx': torch.zeros(d),
        'tau_ffx': D.HalfNormal(1).sample((d,)),
        'tau_rfx': D.HalfNormal(1).sample((q,)),
        'tau_eps': D.HalfNormal(1).sample((1,)),
        }
    ds = parameters(ds)
    ds = outcome(ds)
    df = pandify(ds)
    form = formulate(ds)
    priors = priorize(ds)
    model = bambify(ds)
    plot_data(df)
    results = fitMCMC(ds, tune=100, draws=100, seed=seed)

    # ffx
    print(ds['ffx'].numpy(), results['mcmc_ffx'].mean(-1).numpy())

    # sigmas
    print(ds['sigmas_rfx'].numpy(), results['mcmc_sigmas_rfx'].mean(-1).numpy())
    print(ds['sigma_eps'].numpy(), results['mcmc_sigma_eps'].mean(-1).numpy())

    # rfx
    print(ds['rfx'].numpy())
    print(results['mcmc_rfx'].mean(-1).movedim(1,0).numpy())

