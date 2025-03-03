import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from dataset import LMDataset
from utils import dsFilenameVal
from train import mixMean, mixVariance, mixLogProb
from torch import Value, distributions as D
cmap = colors.LinearSegmentedColormap.from_list("custom_blues", ["#add8e6", "#000080"])


# -----------------------------------------------------------------------------------------
# data wrangling

def locScaleWeight(prefix: str, predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = [f'{prefix}{suffix}' for suffix in ['_loc', '_scale', '_weight']]
    out = [torch.cat([x[key] for x in predictions]) for key in keys]
    return dict(zip(keys, out))


def preloadPredictions(date: str, model_id: str, iteration: int = 100, n_batches: int = 45, fixed: float = 0, ds_type: str = "ffx", num_components: int = 1) -> Dict[str, torch.Tensor]:
    # gather predicted posteriors
    paths = [Path('predictions', model_id, date,
                  f'predictions_i={iteration}_b={batch}.pt')
             for batch in range(n_batches)]
    predictions = [torch.load(paths[batch], weights_only=False)
                   for batch in range(n_batches)]
    out = []
    out.append(locScaleWeight('ffx', predictions))
    if "rfx_loc" in predictions[0]:
        out.append(locScaleWeight('rfx', predictions))
    out.append(locScaleWeight('noise', predictions))

    # gather validation data
    filename = dsFilenameVal(ds_type, 8, 50, fixed)
    ds_val_raw = torch.load(filename, weights_only=False)
    ds_val = LMDataset(**ds_val_raw, permute=False)
    ffx_target = torch.stack([x["beta"] for x in ds_val], dim=0)
    noise_target = torch.stack([x["sigma_error"] for x in ds_val], dim=0).unsqueeze(-1)
    out.append({"ffx_target": ffx_target, "noise_target": noise_target})

    # optionally get analytical solutions for ffx
    if ds_type == "ffx" and num_components == 1:
        ffx_loc_a = torch.stack([x["mu_n"] for x in ds_val], dim=0)
        ffx_scale_a = [torch.diagonal(x["Sigma_n"], dim1=-2, dim2=-1).sqrt() for x in ds_val]
        ffx_scale_a = torch.stack(ffx_scale_a, dim=0)
        # as_a = torch.stack([x["a_n"] for x in ds_val], dim=0).unsqueeze(-1)
        # bs_a = torch.stack([x["b_n"] for x in ds_val], dim=0).unsqueeze(-1)
        # abs_a = torch.cat([as_a, bs_a], dim=-1)
        out.append({"ffx_loc_a": ffx_loc_a, "ffx_scale_a": ffx_scale_a})

    # optionally get rfx structure
    if ds_type == "mfx":
        S = torch.stack([x["S"].sqrt() for x in ds_val], dim=0)
        S_emp = torch.stack([x["S_emp"].sqrt() for x in ds_val], dim=0)
        out.append({"S": S, "S_emp": S_emp})
    return {k: v for d in out for k, v in d.items()}


def multivariateDataFrame(loc: torch.Tensor, quants: torch.Tensor) -> pd.DataFrame:
    n, d = loc.shape
    values_l = loc.numpy().transpose(1,0).flatten()
    values_q1 = quants[...,0].numpy().transpose(1,0).flatten()
    values_q2 = quants[...,1].numpy().transpose(1,0).flatten()
    values_q3 = quants[...,2].numpy().transpose(1,0).flatten()
    values_d = np.repeat(np.arange(d), n)
    values_n = np.tile(np.arange(n), d) + 1
    out = {'n': values_n, 'd': values_d, 'loc': values_l,
           'q1': values_q1, 'q2': values_q2, 'q3': values_q3}
    return pd.DataFrame(out)


def lossFromPredictions(data: Dict[str, torch.Tensor],
                        target_type: str,
                        source = "proposed") -> torch.Tensor:
    target = data[f'{target_type}_target']
    if source == 'proposed':
        loc = data[f'{target_type}_loc']
        scale = data[f'{target_type}_scale']
        weight = data[f'{target_type}_weight']
        return mixLogProb(loc, scale, weight, target, target_type)
    elif source == 'analytical' and target_type == 'ffx':
        loc = data[f'{target_type}_loc_a'].unsqueeze(-1)
        scale = data[f'{target_type}_scale_a'].unsqueeze(-1)
        scale = scale + (scale == 0.).float()
        weight = torch.ones_like(loc)
        return mixLogProb(loc, scale, weight, target, target_type)
    else:
        raise ValueError(f'source {source} unknown.')


def batchLoss(losses, targets, batch):
    losses_batch = losses[batch]
    n = losses_batch.shape[0]
    betas = targets.unsqueeze(1).expand_as(losses)[batch]
    mask = (betas != 0.).float()
    d = int(mask[0].sum())
    masked_losses = losses_batch * mask
    average_losses = masked_losses.sum(dim=1) / d
    out = {"loss": average_losses.numpy(),
           "n": np.arange(n) + 1,
           "d": d - 1,
           "batch": batch + 1}
    return pd.DataFrame(out)


def loss2df(data, target_type: str, source = 'proposed'):
    targets = data[f'{target_type}_target']
    losses = lossFromPredictions(data, target_type, source)
    b = losses.shape[0]
    batch_losses = [batchLoss(losses, targets, i) for i in range(b)]
    return pd.concat(batch_losses) 
   
    
# -----------------------------------------------------------------------------------------
# subplots

# training loss over iterations
def plotTrain(date: str, model_id: str):
    path = Path('losses', model_id, date, 'loss_train.csv')
    df = pd.read_csv(path)
    df = df.groupby(['iteration'])['loss'].agg(['mean', 'std', 'max', 'min']).reset_index()
    df = df[df['iteration'] != 1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['iteration'], df['mean'], label='Mean Loss', color='blue')
    
    # Shade 95% CI area
    ax.fill_between(df['iteration'],
                    df['mean'] - 1.96 * df['std'], 
                    df['mean'] + 1.96 * df['std'], 
                    color='blue', alpha=0.3)
    
    plt.xlabel('Iteration')
    plt.ylabel('-log p(target)')
    # plt.ylim(-1, 3)
    plt.grid(True) 
    plt.show()


# validation loss over iterations
def plotVal(date: str, model_id: str, suffix: str = "val", focus: int = -1):
    path = Path('losses', model_id, date, f'loss_{suffix}.csv')
    if suffix == "val":
        ylabel = '-log p(target)' 
    elif suffix == "kl":
        ylabel = 'KL Divergence'
    else:
        raise ValueError
    df = pd.read_csv(path)
    unique_partitions = sum(df['iteration'] == 1)
    i = df.shape[0] // unique_partitions
    unique_d = unique_partitions // 5
    d_values = np.repeat(np.arange(unique_d), 5)
    df['d'] = np.tile(d_values, i)
    df_agg = df.groupby(['d', 'iteration'])['loss'].agg(['mean', 'std', 'min', 'max']).reset_index()
    if focus >= 0:
        df_agg = df_agg[df_agg['d'] == focus]
    norm = colors.Normalize(vmin=0, vmax=unique_d)
    fig, ax = plt.subplots(figsize=(10, 6))
    for d_value, group in df_agg.groupby('d'):
        color = cmap(norm(d_value))
        ax.plot(group['iteration'], group['mean'], label=f'd={d_value}', color=color)
        ax.fill_between(group['iteration'], 
                        # group['mean'] - group['std'], 
                        # group['mean'] + group['std'], 
                        group['min'],
                        group['max'],
                        color=color, alpha=0.3)  # Shade ± SD
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True) 
    plt.show()


# proposed posterior
def plotMultivariateParams(df, targets, quantiles, ylims, est_type: str, ax):
    ''' plot first quartile, median, and third quartile '''
    unique_d = df['d'].unique()
    norm = colors.Normalize(vmin=unique_d.min(), vmax=unique_d.max())
 
    # Create the plot
    for d_value, group in df.groupby('d'):
        target = targets[d_value].item()
        color = cmap(norm(d_value))
        ax.plot(group['n'], group['q2'], label=f'd={d_value}', color=color)
        ax.fill_between(group['n'], 
                        group['q1'],
                        group['q3'],
                        color=color, alpha=0.1)  # Shade ± SD
        ax.axhline(y=target, color=color, linestyle=':', linewidth=1.5)
    
    # Adding labels and title
    ax.set_xlabel('n')  # X-axis label
    ax.set_ylabel(f'{est_type}')
    if est_type == "analytical": 
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(ylims)
    ax.grid(True)
     

# validation loss over n
def plotValN(df, quantiles, iteration, source, focus: int = -1):
    unique_d = df['d'].unique().shape[0]
    q1 = lambda x: x.quantile(quantiles[0])
    q2 = lambda x: x.quantile(quantiles[1])
    q3 = lambda x: x.quantile(quantiles[2])
    df_agg = df.groupby(['d', 'n'])['loss'].agg([('q1', q1), 
                                                 ('q2', q2),
                                                 ('q3', q3)]).reset_index()
    if focus >= 0:
        df_agg = df_agg[df_agg['d'] == focus]
    norm = colors.Normalize(vmin=0, vmax=unique_d)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for d_value, group in df_agg.groupby('d'):
        color = cmap(norm(d_value))
        ax.plot(group['n'], group['q2'], label=f'd={d_value}', color=color)
        ax.fill_between(group['n'], 
                        group['q1'], 
                        group['q3'], 
                        color=color, alpha=0.2)  # Shade ± SD
    plt.ylim(-2, 6)
    plt.xlabel('n')
    plt.ylabel('-log p(target)')
    plt.legend()
    plt.grid(True) 
    title = source
    if source == "proposed":
        title += f' (iteration {iteration})'
    plt.title(title)
    plt.show()
    
    
# -----------------------------------------------------------------------------
# plot wrappers

def plotMixtureDensity(target, loc, scale, weight, i):
    comp = D.LogNormal(loc, scale)
    mix = D.Categorical(weight)
    proposal = D.MixtureSameFamily(mix, comp)
    n_values = 1000  # Number of values to evaluate
    x = torch.linspace(0.01, 3, n_values)  # Adjust the range as needed
    x_expanded = x.view(1, -1).expand(50, -1)
    log_probs = proposal.log_prob(x_expanded)
    densities = torch.exp(log_probs).detach().numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, densities[i])
    ax.axvline(x = target[i], color = 'b')


def mixtureCDF(x, loc, scale, weight, base_dist) -> torch.Tensor:
    # x (n, d, 1000)
    # loc, scale, weight (n, d, c)
    # base_dist: D.Normal or D.LogNormal
    cdfs = base_dist(loc.unsqueeze(-2), scale.unsqueeze(-2)).cdf(x.unsqueeze(-1))
    return torch.sum(weight.unsqueeze(-2) * cdfs, dim=-1)


def findQuantiles(base_dist: Callable,
                  quantiles: Tuple[float, float, float],
                  loc: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor,
                  num_points=1000) -> torch.Tensor:
    n, d, _ = loc.shape
    if base_dist == D.Normal:
        x_range = torch.linspace(-10, 10, num_points)
    else:
        x_range = torch.linspace(0, 3, num_points)
    x_range = x_range.view(1, 1, -1).expand(n, d, num_points)
    cdf = mixtureCDF(x_range, loc, scale, weight, base_dist)
    values = []
    for q in quantiles:
        indices = torch.argmin(torch.abs(cdf-q), dim=-1).unsqueeze(-1)
        values += [x_range.gather(dim=-1, index=indices).squeeze(-1)]
    return torch.stack(values, dim=-1)
  

def plotWrapper(data: dict, prefix: str, batch_id: int, iteration: int,
                base_dist: Callable,
                quantiles: Tuple[float, float, float],
                ylims: Tuple[float, float]):
    # prepare data
    mask = (data[f'{prefix}_target'][batch_id] != 0)
    target = data[f'{prefix}_target'][batch_id, mask]
    loc = data[f'{prefix}_loc'][batch_id, :, mask]
    scale = data[f'{prefix}_scale'][batch_id, :, mask]
    weight = data[f'{prefix}_weight'][batch_id, :, mask]
    
    # find mean, (numerical) quantiles and construct df
    mean = mixMean(loc, weight)
    quantile_roots = findQuantiles(base_dist, quantiles, loc, scale, weight)
    df = multivariateDataFrame(mean, quantile_roots)
    
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plotMultivariateParams(df, target, quantiles, ylims, 'proposed', ax)
    fig.suptitle(f'{prefix} (iter={iteration})')


def ffxWrapper(data: dict, batch_id: int, iteration: int, quantiles: Tuple[float, float, float]):
    return plotWrapper(data, 'ffx', batch_id, iteration, D.Normal, quantiles, (-6., 6.))


def noiseWrapper(data: dict, batch_id: int, iteration: int, quantiles: Tuple[float, float, float]):
    return plotWrapper(data, 'noise', batch_id, iteration, D.LogNormal, quantiles, (0., 2.))


# =============================================================================

if __name__ == "__main__":
    fixed = 0.
    noise = 'variable' if fixed == 0. else fixed
    ds_type = 'ffx'
    model_id = f'mixture-8-transformer-128-256-8-3-dropout=0-loss=logprob-seed=0-fx={ds_type}-noise={noise}'
    date = '20250227-164027'
        
    # train and val loss
    plotTrain(date, model_id)
    plotVal(date, model_id)
    # if ds_type == "ffx":
    #     plotVal(date, model_id, suffix="kl")
    
    # proposal distribution
    iteration = 10
    data = preloadPredictions(date,
                              model_id,
                              iteration=iteration,
                              n_batches=45,
                              fixed=fixed,
                              ds_type=ds_type)
    max_d = 1
    quantiles = (0.025, 0.5, 0.975)
    for i in range (5):
        ffxWrapper(data, 500 * max_d + i, iteration, quantiles)
    # for i in range (5):
    #     rfxWrapper(data, 500 * max_d + i, iteration, quantiles)
    for i in range (5):
        noiseWrapper(data, 500 * max_d + i, iteration, quantiles)
    
    
    # plot validation loss over n for given iteration
    df_p = loss2df(data, source = "proposed")
    plotValN(df_p, iteration, source = "proposed")
    # df_a = loss2df(data, source = "analytical")
    # plotValN(df_a, iteration, source = "analytical")
