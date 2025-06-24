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
        
        
def subsetMFX(batch: dict, batch_idx: int = 0) -> dict:
    # extract batch_idx
    ds = {k: v[batch_idx:batch_idx+1].clone() for k, v in batch.items()
          if isinstance(v, torch.Tensor)}

    # repeat all tensors n times
    n = ds['X'].shape[-2]
    ds = {k: v.repeat(n, *[1]*(v.ndim-1)) for k, v in ds.items()}

    # dynamically subset
    ns, mask_n =  ds['n_i'], ds['mask_n']
    X, y = ds['X'], ds['y']
    for i in range(n):
        i_ = i + 1 + torch.zeros_like(ns[i])
        ns[i] = torch.min(i_, ns[i])
        mask_n[i,:, i+1:n] = False
        X[i, :, i+1:n] = torch.zeros_like(X[i, :, i+1:n])
        y[i, :, i+1:n] = torch.zeros_like(y[i, :, i+1:n])        
    ds.update(dict(n=ns, mask_n=mask_n, X=X, y=y))
    return ds


def plotOverN(quantiles: torch.Tensor, targets: torch.Tensor, names) -> None:
    # prepare targets and quantiles
    if quantiles.shape[1] == targets.shape[1] + 1:
        quantiles = quantiles[:, :-1]
    target = targets[0]
    mask = (target != 0.)
    target_ = target[mask]
    quantiles_ = quantiles[:, mask]
    names_ = names[mask.numpy()]

    # prepare colors and x axiscolors = [cmap(i) for i in range(cmap.N)]
    d = int(mask.sum())
    ns = torch.arange(1, quantiles.shape[0]+1)
    min_val = float(torch.tensor([quantiles.min(), quantiles.min()]).min())
    max_val = float(torch.tensor([quantiles.max(), quantiles.max()]).max())

    _, ax = plt.subplots(figsize=(8, 6))
    for i in range(d):
        color = colors[i]
        quantiles_i = quantiles_[:, i]
        ax.plot(ns, quantiles_i[..., 1], label=names_[i], color=color)
        ax.fill_between(ns, quantiles_i[..., 0], quantiles_i[..., 2],
                        color=color, alpha=0.15)
        ax.axhline(y=target_[i], color=color, linestyle=':', linewidth=1.5) # type: ignore

    # Adding labels and title
    ax.set_xlabel('n')  # X-axis label
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(max(-7.5, min_val), min(7.5, max_val))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)


def unfold(model: Approximator, full: dict, batch_idx: int) -> None:
        subset = subsetMFX if cfg.fx_type == 'mfx' else subsetFFX
        batch = subset(full, batch_idx)
        _, proposed, _ = model(batch, sample=True)
        quantiles = model.quantiles(proposed['global'], [.025, .500, .975])
        plotOverN(quantiles, *model.targets(batch))


# -----------------------------------------------------------------------------
# main functions

def load(model: Approximator, iteration: int) -> None:
    model_path = Path('outputs', 'checkpoints', modelID(cfg))
    fname = Path(model_path, f"checkpoint_i={iteration}.pt")
    state = torch.load(fname, weights_only=False)
    model.load_state_dict(state["model_state_dict"])


def validate(model: Approximator, batch: dict, 
             kld=True, plot=True, importance=False):
    with torch.no_grad():
        targets, names = model.targets(batch)
        losses_nll, proposed, _ = model(batch, sample=True, n=1000, log_prob=True)
        mcmc = {'samples': batch['mcmc']['global']} if 'mcmc' in batch else None

        # get kl divergence
        if kld and cfg.fx_type == 'ffx':
            loc, scale = model.moments(proposed['global'])
            losses_kl = compareWithAnalytical(batch, loc, scale, marginal=True)
            print(f"KL Divergence (to analytical): {losses_kl.mean().item():.3f}")
            
        # importance sampling
        if 'flow' in cfg.post_type and importance:
            is_results = importanceSample(batch, proposed['global'])
            proposed['global'].update(**is_results)
            
        # plots
        if plot:
            # collapse samples for flow models
            if 'flow' in cfg.post_type and cfg.fx_type == 'mfx':
                proposed['global'] = torch.cat([
                    proposed['global']['ffx'][:, :-1],
                    proposed['global']['sigmas'][:, :-1]
                    ], dim=1)
 
            # parameter recovery plot
            model.plotRecovery(
                batch, proposed['global'], show_error=False, color='darkgreen', alpha=0.3)
            if mcmc is not None:
                model.plotRecovery(
                    batch, mcmc, show_error=False, color='darkorange', alpha=0.3)
            if cfg.fx_type == 'mfx':
                model.plotRecoveryRFX(batch, proposed['local'], show_error=False)
                
            # SBC plot
            ranks = getRanks(targets, proposed['global'], absolute=False)
            plotSBC(ranks, names, color='darkgreen')
            # wd = getWasserstein(ranks)
            if mcmc is not None:
                ranks_ = getRanks(targets, mcmc)
                plotSBC(ranks_, names, color='darkorange')
                wd_ = getWasserstein(ranks_)
                print(f"Wasserstein Distance (SBC): {wd_:.3f}")
            
            # ECDF diff plot
            ranks = getRanks(targets, proposed['global'], absolute=True)
            plotECDF(ranks, names, color='darkgreen')
            if mcmc is not None:
                ranks_ = getRanks(targets, mcmc, absolute=True)
                plotECDF(ranks_, names, color='darkorange')                
            
            # CI calibration plot
            covered = getCoverage(model, proposed['global'], targets)
            plotCalibration(covered, names, source='(MB)')
            if mcmc is not None:
                covered_ = getCoverage(model, mcmc, targets)
                plotCalibration(covered_, names, source='(HMC)')
            coverage_error = coverageError(covered)
            print(f"Empirical coverage errors: {coverage_error.numpy()}")
            

            # plot posterior interval palette
            plotAllIntervals(model, proposed['global'], mcmc, targets, names)


        return {'proposed': proposed}


def inspect(model: Approximator, batch: dict,
            proposed, batch_indices: list,
            over_n: bool = False):
    targets, names = model.targets(batch)
    mcmc = {'samples': batch['mcmc']['global']} if 'mcmc' in batch else None
    
    # visualize some posteriors
    for b in batch_indices:
        model.posterior.plot(proposed['global'], targets, names, b, mcmc=mcmc)

    # visualize model performance over n
    if over_n:
        for b in batch_indices:
            unfold(model, batch, b)
            

###############################################################################

def setup() -> argparse.Namespace:
    ''' Parse command line arguments. '''
    parser = argparse.ArgumentParser()

    # misc
    parser.add_argument("-s", "--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--cores", type=int, default=4, help="Nubmer of processor cores to use (default = 4)")
    parser.add_argument("-i", "--iteration", type=int, default=10, help="Preload model from iteration #i")
    parser.add_argument("--importance", action='store_false', help="Do importance sampling (default = True)")

    # data & training
    parser.add_argument("-t", "--fx_type", type=str, default="ffx", help="Type of dataset [ffx, mfx] (default = ffx)")
    parser.add_argument("-d", type=int, default=1, help="Number of fixed effects (without bias, default = 3)")
    parser.add_argument("-m", type=int, default=0, help="Maximum number of groups (default = 30).")
    parser.add_argument("-n", type=int, default=50, help="Maximum number of samples per group (default = 20).")

    # model
    parser.add_argument("--emb_type", type=str, default="joint", help="Embedding architecture [joint, separate, sequence] (default = joint)")
    parser.add_argument("--sum_type", type=str, default="poolformer", help="Summarizer architecture [deepset, poolformer] (default = deepset)")
    parser.add_argument("--post_type", type=str, default="flow-affine", help="Posterior architecture [point, discrete, mixture, flow-affine, flow-spline, flow-matching] (default = flow)")
    parser.add_argument("--bins", type=int, default=500, help="Number of bins in discrete posterior (default = 500)")
    parser.add_argument("--components", type=int, default=3, help="Number of mixture components (default = 3)")
    parser.add_argument("--flows", type=int, default=3, help="Number of normalizing flow blocks (default = 3)")
    parser.add_argument("--dropout", type=float, default=0.01, help="Dropout rate (default = 0.01)")
    parser.add_argument("--act", type=str, default="Mish", help="Activation funtction [anything implemented in torch.nn] (default = GELU)")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension (default = 64)")
    parser.add_argument("--ff", type=int, default=128, help="Feedforward dimension (default = 128)")
    parser.add_argument("--out", type=int, default=32, help="Summary dimension (default = 32)")
    parser.add_argument("--blocks", type=int, default=3, help="Number of blocks in summarizer (default = 3)")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads (poolformer, default = 8)")
    
    return parser.parse_args()


# =============================================================================
if __name__ == "__main__":
    # --- setup 
    cfg = setup()
    console_width = getConsoleWidth()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cores)
    Approx = ApproximatorMFX if cfg.fx_type == "mfx" else ApproximatorFFX
    
    # --- load model
    model = Approx.build(cfg.d, cfg.hidden, cfg.ff, cfg.out,
                               cfg.dropout, cfg.act,
                               cfg.heads, cfg.blocks,
                               cfg.emb_type, cfg.sum_type, cfg.post_type,
                               bins=cfg.bins, components=cfg.components, flows=cfg.flows,
                               max_m=cfg.m,
                               )
    model.eval()
    print(f'{"-"*console_width}\nmodel: {modelID(cfg)}')

    # --- load model and data
    load(model, cfg.iteration)
    b = 250
    fn = dsFilename(cfg.fx_type, 'test', cfg.d, cfg.m, cfg.n, b, -1)
    batch = next(iter(getDataLoader(fn, b)))
    print(f'preloaded model from iteration {cfg.iteration} and test set of size {b}...\n{"-"*console_width}')
    
    # --- run full validation for desired snapshot
    results = validate(model, batch, importance=cfg.importance)
    inspect(model, batch, results['proposed'], batch_indices=range(3))

    # # --- run base validation over snapshot history
    # iters = range(5, cfg.iteration + 1, 5)
    # nll = torch.zeros((len(iters), cfg.b))
    # kld = torch.zeros_like(nll)
    # for i, iteration in enumerate(tqdm(iters)):
    #     load(model, iteration)
    #     results = validate(model, batch, kld=True)
    #     nll[i] = results['nll']
    #     kld[i] = results['kld']
    # plotOverT(iters, nll)
    # plotOverT(iters, kld, kl=True, q=[.16, .50, .84])
    # outname = Path('outputs', 'losses', modelID(cfg), 'losses.pt')
    # torch.save({'nll': nll, 'kld': kld}, outname)
