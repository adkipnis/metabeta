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
# basic plots






# -----------------------------------------------------------------------------------------
# data wrangling

def locScaleWeight(prefix: str, predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = [f'{prefix}{suffix}' for suffix in ['_loc', '_scale', '_weight']]
    out = [torch.cat([x[key] for x in predictions]) for key in keys]
    return dict(zip(keys, out))


def preloadPredictions(date: str, model_id: str, iteration: int = 100, n_batches: int = 45, fixed: float = 0, ds_type: str = "ffx") -> Dict[str, torch.Tensor]:
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

    # if ds_type == "ffx":
    #     means_a = torch.stack([x["mu_n"] for x in ds_val], dim=0).numpy()
    #     stds_a = [torch.diagonal(x["Sigma_n"], dim1=-2, dim2=-1).sqrt() for x in ds_val]
    #     stds_a = torch.stack(stds_a, dim=0).numpy()
    #     as_a = torch.stack([x["a_n"] for x in ds_val], dim=0).unsqueeze(-1)
    #     bs_a = torch.stack([x["b_n"] for x in ds_val], dim=0).unsqueeze(-1)
    #     abs_a = torch.cat([as_a, bs_a], dim=-1).numpy()
    #     assert means_a.shape[0] == means_p.shape[0], \
    #         "Different number of observations for analytical and trained solutions."
    #     out.update({"means_a": means_a, "stds_a": stds_a, "abs_a": abs_a,})
    # elif ds_type == "mfx":
    #     s = torch.stack([x["S"].sqrt() for x in ds_val], dim=0).numpy()
    #     s_emp = torch.stack([x["S_emp"].sqrt() for x in ds_val], dim=0).numpy()
    #     out.update({"s": s, "s_emp": s_emp})
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

    if source == 'proposed':
        loc = data[f'{target_type}_loc']
        scale = data[f'{target_type}_scale']
        weight = data[f'{target_type}_weight']
        target = data[f'{target_type}_target']
        return mixLogProb(loc, scale, weight, target, target_type)
    elif source == 'analytical':
        raise NotImplementedError
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


def loss2df(data, source = "proposed"):
    targets = torch.tensor(data["targets"])
    losses = lossFromPredictions(data, targets, source)
    b = losses.shape[0]
    batch_losses = [batchLoss(losses, targets, i) for i in range(b)]
    return pd.concat(batch_losses) 
   
    
# -----------------------------------------------------------------------------------------
# subplots

# plot multivariate params
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
     

# -----------------------------------------------------------------------------------------
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
    return plotWrapper(data, 'noise', batch_id, iteration, D.LogNormal, quantiles, (0., 3.))
    


# def plotParamsWrapper(data: dict, batch_id: int, iteration: int, paramtype = "beta"):
#
#     # plot ffx posterior
#     if paramtype == "ffx":
#
#         # proposed posterior
#         targets = data["targets"]
#         means_p = data["means_p"]
#         stds_p = data["stds_p"]
#         df_p, betas = mvnDataFrame(targets, means_p, stds_p, batch_id)
#
#         # analytical posterior
#         if "means_a" in data:
#             fig, axs = plt.subplots(2, sharex=True, figsize=(8, 6))
#             means_a = data["means_a"]
#             stds_a = data["stds_a"]
#             df_a, _ = mvnDataFrame(targets, means_a, stds_a, batch_id)
#             plotMvnParams(df_a, betas, "analytical", axs[0])
#             plotMvnParams(df_p, betas, "proposed", axs[1])
#         else:
#             fig, ax = plt.subplots(figsize=(8, 6))
#             plotMvnParams(df_p, betas, "proposed", ax)
#         fig.suptitle(f'iter={iteration}')
#
#
#     # plot rfx posterior
#     if paramtype == "rfx":
#         targets = data["s"]
#         stds_prop = data["s_p"]
#         stds_emp = data["s_emp"]
#         df_s, s = sDataFrame(targets, stds_prop, stds_emp, batch_id)
#         fig, ax = plt.subplots(figsize=(8, 6))
#         plotRfxParams(df_s, s, ax)
#
#     # plot noise posterior
#     if paramtype == "noise":
#         abs_a = data["abs_a"]
#         abs_p = data["abs_p"]
#         df_ig_a = igDataFrame(abs_a, batch_id)
#         df_ig_p = igDataFrame(abs_p, batch_id)
#         fig, ax = plt.subplots(figsize=(8, 6))
#         plotIGParams(df_ig_a, df_ig_p, ax)
#
#     # # plot noise std
#     # if paramtype == "sigma":
#     #     abs_p = data["abs_p"]
#     #     sigma_errors = data["sigma_errors"]
#     #     df_noise = noiseDataFrame(abs_p, batch_id)
#     #     fig, ax = plt.subplots(figsize=(8, 6))
#     #     plotNoise(df_noise, sigma_errors[batch_id], ax)
#
#
# plot validation loss
def plotVal2(df, iteration, source, focus: int = -1):
    unique_d = df['d'].unique().shape[0]
    df_agg = df.groupby(['d', 'n'])['loss'].agg(['mean', 'std', 'max', 'min']).reset_index()
    if focus >= 0:
        df_agg = df_agg[df_agg['d'] == focus]
    norm = colors.Normalize(vmin=0, vmax=unique_d)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for d_value, group in df_agg.groupby('d'):
        color = cmap(norm(d_value))
        ax.plot(group['n'], group['mean'], label=f'd={d_value}', color=color)
        ax.fill_between(group['n'], 
                        group['mean'] - group['std'], 
                        group['mean'] + group['std'], 
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
    iteration = 5
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
    for i in range (5):
        noiseWrapper(data, 500 * max_d + i, iteration, quantiles)
        # plotParamsWrapper(data, 500 * max_d + i, iteration, paramtype="rfx")
        # plotParamsWrapper(data, 500 * max_d + i, iteration, paramtype="noise")
    
    # # plot validation loss over n
    # df_p = loss2df(data, source = "proposed")
    # df_a = loss2df(data, source = "analytical")
    # plotVal2(df_p, iteration, source = "proposed")
    # plotVal2(df_a, iteration, source = "analytical")
