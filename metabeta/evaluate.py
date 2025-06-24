import argparse
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde, binom
import torch
from torch import distributions as D
from metabeta.utils import dsFilename, getConsoleWidth, modelID, getDataLoader
from metabeta.models.approximators import Approximator, ApproximatorFFX#, ApproximatorMFX
from metabeta.importance import importanceSample
plt.rcParams['figure.dpi'] = 300
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(cmap.N)]

# --- for given snapshot:
# 1. evaluate performance (over batch):
# - plot recovery  [✓]
# - plot error bars [✓]
# - add rmse + r  [✓]

# 2. calibration
# - coverage over different CI's [✓]
# - plot calibration curve (nominal vs. empirical) [✓]
# - SBC histogram [✓] 
# - SBC shade 
# - ECDF difference plot [✓] 

# 3. visualize posterior (conditional on single dataset):
# - plot posterior distribution with vline for true values [✓]
# - (1) mixture [✓]
# - (2) discrete [✓]
# - (3) flow [✓]

# 4. visualize posterior devolopment over n (conditional on single dataset):
# - evaluate quantiles [✓]
# - subset dataset and run model on each [✓]
# - plot quantiles [✓]

# 5. posterior predictive checks
# - read up on theory

# --- over snapshots:
# 1. visualize training and testing NLLL [✓]
# 2. evaluate [✓] and visualize KLD [✓]

# -----------------------------------------------------------------------------
# coverage

def empiricalCoverage(quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #  how often are the targets actually inside a given credibility interval?
    mask = (targets != 0.)
    above = targets > quantiles[..., 0]
    below = targets < quantiles[..., -1]
    inside = above * below * mask
    coverage = inside.float().sum(0)/(mask.sum(0) + 1e-12)
    return coverage # (d,)

def getCoverage(model: Approximator,
                proposed: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    quantiles50 = model.quantiles(proposed, [.25, .75])
    quantiles68 = model.quantiles(proposed, [.16, .84])
    quantiles80 = model.quantiles(proposed, [.10, .90])
    quantiles90 = model.quantiles(proposed, [.05, .95])
    quantiles95 = model.quantiles(proposed, [.025, .975])
    c50 = empiricalCoverage(quantiles50, targets)
    c68 = empiricalCoverage(quantiles68, targets)
    c80 = empiricalCoverage(quantiles80, targets)
    c90 = empiricalCoverage(quantiles90, targets)
    c95 = empiricalCoverage(quantiles95, targets)
    return {'50': c50, '68': c68, '80': c80, '90': c90, '95': c95}

def coverageError(coverage: dict):
    concatenated = torch.cat([v.unsqueeze(0) for _, v in coverage.items()])
    mask = (concatenated != 0.)
    nominal = torch.tensor([int(k) for k in coverage.keys()]).unsqueeze(1) / 100
    errors = (concatenated - nominal) * mask
    mean_error = errors.sum(0) / (mask.sum(0) + 1e-12)
    return mean_error

def plotCalibration(coverage: Dict[str, torch.Tensor], names, source: str = '') -> None:
    nominal = [int(k) for k in coverage.keys()]
    matrix = torch.cat([t.unsqueeze(-1) for _,t in coverage.items()], dim=-1)
    _, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        color = colors[i]
        coverage_i = matrix[i] * 100.
        ax.plot(nominal, coverage_i, color=color, label=name)
    ax.plot([50, 95], [50, 95], '--', lw=2, zorder=1,
            color='grey', label='identity')
    ax.set_xticks(nominal)
    ax.set_yticks(nominal)
    ax.set_xlabel('nominal CI')
    ax.set_ylabel('empirical CI')
    ax.legend()
    ax.grid(True)
    plt.title(f'coverage {source}')

# -----------------------------------------------------------------------------
# simulation based calibration

def getRanks(targets: torch.Tensor, proposed: dict, absolute=False) -> torch.Tensor:
    # targets (b, d), samples (b, s, d)
    samples = proposed['samples']
    weights = proposed.get('weights', None)
    targets_ = targets.unsqueeze(-1)
    if absolute:
        samples = samples.abs()
        targets_ = targets_.abs()        
    smaller = (samples < targets_).float()
    if weights is not None:
        s = samples.shape[-1]
        ranks = (smaller * weights.unsqueeze(1)).sum(-1) / s
    else:
        ranks = smaller.mean(-1)
    return ranks


def plotSBC(ranks: torch.Tensor, names: list, color='darkgreen') -> None:
    eps = 0.02
    n = len(names)
    w = int(torch.tensor(n).sqrt().ceil())
    _, axs = plt.subplots(figsize=(8 * w, 6 * w), ncols=w, nrows=w)
    axs = axs.flatten()
    for i, name in enumerate(names):              
        ax = axs[i]
        ax.set_axisbelow(True)
        ax.grid(True)
        xx = ranks[:, i]
        sns.histplot(xx, kde=True, ax=ax, binwidth=0.05, binrange=(0,1),
                     color=color, alpha=0.5, stat="density", label=names[i])
        ax.set_xlim(0-eps,1+eps)
        ax.set_xlabel('U', fontsize=20)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelcolor='w')
        ax.legend()
    for i in range(n, w*w):
        axs[i].set_visible(False)
    
def getWasserstein(sbc: torch.Tensor, n_points=1000):
    b, d = sbc.shape
    
    # support
    x = np.linspace(0, 1, n_points)
    dx = x[1] - x[0]
    
    # distributions
    p = [gaussian_kde(sbc[:,i], bw_method='scott')(x) for i in range(d)]
    q = np.ones_like(x)

    # cdfs
    ecdf = [np.cumsum(p_i) * dx for p_i in p] 
    ucdf = np.cumsum(q) * dx
    
    wd = [wasserstein_distance(ecdf_i, ucdf) for ecdf_i in ecdf]
    return float(sum(wd)/d)


def getDistanceRanks(targets: torch.Tensor, proposed: dict):
    samples = proposed['samples']
    dist_prior = targets.unsqueeze(-1).abs()
    dist_post = samples.abs()
    ranks = (dist_post < dist_prior).float().mean(-1)
    return ranks

def boundECDF(n_ranks: int, n_sim: int = 1000, alpha: float = 0.05):
    p = np.linspace(0, 1, n_ranks)
    lower = binom.ppf(alpha / 2, n_sim, p) / n_sim - p
    upper = binom.ppf(1 - alpha / 2, n_sim, p) / n_sim - p
    return p, lower, upper

def plotECDF(ranks: torch.Tensor, names: list, color='darkgreen') -> None:
    eps = 0.02
    n = len(names)
    w = int(torch.tensor(n).sqrt().ceil())
    xx_theo, lower, upper = boundECDF(len(ranks))
    _, axs = plt.subplots(figsize=(8 * w, 6 * w), ncols=w, nrows=w)
    axs = axs.flatten()
    for i, name in enumerate(names):              
        ax = axs[i]
        ax.set_axisbelow(True)
        ax.grid(True)
        xx = ranks[:, i].sort()[0].numpy()
        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
        yy = np.linspace(0, 1, num=xx.shape[-1]) - xx
        ax.plot(xx, yy, color=color, label='sample')
        ax.fill_between(xx_theo, lower, upper, color=color, alpha=0.1, label='theoretical')
        ax.set_xlim(0-eps,1+eps)
        ax.set_xlabel('U', fontsize=20)
        ax.set_ylabel(r'$\Delta$ ECDF')
        # ax.tick_params(axis='y', labelcolor='w')
        ax.set_title(names[i])
        ax.legend()
    for i in range(n, w*w):
        axs[i].set_visible(False)

# -----------------------------------------------------------------------------
# compare posterior intervals with mcmc
def plotIntervals(ax,
                  quantiles1: torch.Tensor,
                  quantiles2: torch.Tensor,
                  target: torch.Tensor,
                  name: str, n: int = 16):
    # calculate overlap
    width1 = quantiles1[:, 3] - quantiles1[:, 1]
    width2 = quantiles2[:, 3] - quantiles2[:, 1]
    d_50 = (width1 - width2).mean()
    width1 = quantiles1[:, 4] - quantiles1[:, 0]
    width2 = quantiles2[:, 4] - quantiles2[:, 0]
    d_95 = (width1 - width2).mean()
    
    # sort targets
    target, idx = torch.sort(target)
    
    # get evenly spaced subset of targets
    u = torch.linspace(0.05, 0.95, n)
    idx_ = torch.round(u * (len(target) - 1)).long()
    
    # subset the posterior quantiles
    q1 = quantiles1[idx][idx_]
    q2 = quantiles2[idx][idx_]
    
    # prepare axes
    x = np.arange(n)
    bar_gap = 0.05
    x1 = x - (0.20 + bar_gap/2)
    x2 = x + (0.20 + bar_gap/2)
    
    # plot 
    ax.bar(x1, bottom=q1[:, 0], height=q1[:, 4]-q1[:, 0],
           width=0.35, color='darkgreen', alpha=0.4)
    ax.bar(x1, bottom=q1[:, 1], height=q1[:, 3]-q1[:, 1],
           width=0.40, color='darkgreen', alpha=0.9, label='MB')
    ax.bar(x2, bottom=q2[:, 0], height=q2[:, 4]-q2[:, 0],
           width=0.35, color='darkorange', alpha=0.4)
    ax.bar(x2, bottom=q2[:, 1], height=q2[:, 3]-q2[:, 1],
           width=0.40, color='darkorange', alpha=0.9, label='HMC')
    for i in range(n):
        plt.hlines(y=q1[i, 2], xmin=x1[i]-0.2, xmax=x1[i]+0.2,
                   color='white', linewidth=1.5)
        plt.hlines(y=q2[i, 2], xmin=x2[i]-0.2, xmax=x2[i]+0.2,
                   color='white', linewidth=1.5)
    
    ax.text(
        0.75, 0.1,
        fr'$d_{{50}} = {d_50.item():.3f}$' + '\n' + fr'$d_{{95}} = {d_95.item():.3f}$',
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=16,
        bbox=dict(boxstyle='round',
                  facecolor=(1, 1, 1, 0.7),
                  edgecolor=(0, 0, 0, 0.2),
                  ),
    )
    
    ax.set_title(name)
    ax.set_ylabel('credibility intervals')
    ax.set_xticks([])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.legend()
    
    
def plotAllIntervals(model: Approximator,
                     proposed: torch.Tensor,
                     mcmc: torch.Tensor,
                     targets: torch.Tensor,
                     names: list):
    q1 = model.quantiles(proposed, [.025, .25, .50, .75, .975])
    q2 = model.quantiles(mcmc, [.025, .25, .50, .75, .975])
    n = len(names)
    w = int(torch.tensor(n).sqrt().ceil())
    _, axs = plt.subplots(figsize=(6 * w, 4 * w), ncols=w, nrows=w)
    axs = axs.flatten()
    for i, name in enumerate(names):
        plotIntervals(axs[i], q1[:, i], q2[:, i], targets[:, i], name)
    for i in range(n, w*w):
        axs[i].set_visible(False)
    
# -----------------------------------------------------------------------------
# comparison with analytical posterior
def kldFull(mean_lin: torch.Tensor, Sigma_lin: torch.Tensor,
            mean_prop: torch.Tensor, Sigma_prop: torch.Tensor) -> torch.Tensor:
    ''' looped version for KLD between two MVNs with non-diagonal variance '''
    mask = (mean_lin != 0.)
    b = mean_prop.shape[0]
    losses = torch.zeros(b)
    for i in range(b):
        mask_i = mask[i] 
        mean_lin_i = mean_lin[i, mask_i]
        mean_i = mean_prop[i, mask_i]
        Sigma_lin_i = Sigma_lin[i, mask_i][..., mask_i]
        Sigma_i = Sigma_prop[i, mask_i][..., mask_i]
        post_lin = D.MultivariateNormal(mean_lin_i, Sigma_lin_i)
        post_prop = D.MultivariateNormal(mean_i, Sigma_i)
        losses[i] = D.kl.kl_divergence(post_lin, post_prop)
    return losses


def kldMarginal(mean_lin: torch.Tensor, var_lin: torch.Tensor,
                mean_prop: torch.Tensor, var_prop: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    ''' vectorized version for KLD between two MVNs with diagonal variance '''
    mask = (mean_lin != 0).float()
    var_prop = var_prop + eps
    var_lin = var_lin + eps
    term1 = (mean_lin - mean_prop).square() / var_prop
    term2 = var_lin / var_prop
    term3 = var_prop.log() - var_lin.log()
    kl = 0.5 * (term1 + term2 + term3 - 1.) * mask
    return kl.sum(dim=-1)


def compareWithAnalytical(batch: dict,
                          loc: torch.Tensor, scale: torch.Tensor,
                          marginal: bool = True) -> torch.Tensor:
    # prepare proposed posterior
    mean_prop, var_prop = loc[..., :-1], scale[..., :-1].square()
    # prepare analytical solution
    params = batch['analytical']['ffx']
    mean_lin, Sigma_lin = params['mu'], params['Sigma']
    sigma_error = batch['sigma_error'].square()
    # calculate KL Divergence
    if marginal:
        var_lin = torch.diagonal(Sigma_lin, dim1=-2, dim2=-1) * sigma_error.view(-1,1)
        kld = kldMarginal(mean_lin, var_lin, mean_prop, var_prop)
    else:
        Sigma_prop = torch.diag_embed(var_prop)
        Sigma_lin = Sigma_lin * sigma_error.view(-1,1,1)
        kld = kldFull(mean_lin, Sigma_lin, mean_prop, Sigma_prop)
    return kld


def plotOverT(time: torch.Tensor, losses: torch.Tensor,
              q: list = [.025, .500, .975], kl: bool = False):
    # time: (n_iter) losses: (n_iter, batch)
    # center = losses.mean(-1)
    # std = losses.std(-1)
    # lower, upper = center - std, center + std
    lower = torch.quantile(losses, q[0], dim=-1)
    center = torch.quantile(losses, q[1], dim=-1)
    upper = torch.quantile(losses, q[2], dim=-1)
    _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time, center, color='darkgreen')
    ax.fill_between(time, lower, upper, color='darkgreen', alpha=0.3)
    # ax.set_xticks(time)
    ax.set_xlabel('datasets [10k]')
    ylabel = 'D(Optimal | Model)' if kl else '-log p(theta)'
    ax.set_ylabel(ylabel)
    ax.grid(True) 


# -----------------------------------------------------------------------------
# plot over n
def subsetFFX(batch: dict, batch_idx: int = 0) -> dict:
    ''' for dataset {batch_idx} in batch,
        generate a new batch out of progressive subsamples '''
    # extract batch_idx
    ds = {k: v[batch_idx:batch_idx+1].clone() for k, v in batch.items()
          if isinstance(v, torch.Tensor)}

    # repeat all tensors n times
    n = int(ds['n'])
    ds = {k: v.repeat(n, *[1]*(v.ndim-1)) for k, v in ds.items()}

    # dynamically subset
    ns, mask_n =  ds['n'], ds['mask_n']
    X, y = ds['X'], ds['y']
    for i in range(n):
        ns[i] = i + 1
        mask_n[i, i+1:n] = False
        X[i, i+1:n] = torch.zeros_like(X[i, i+1:n])
        y[i, i+1:n] = torch.zeros_like(y[i, i+1:n])
    ds.update(dict(n=ns, mask_n=mask_n, X=X, y=y))
    return ds
        
        
