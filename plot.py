import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from dataset import LMDataset
from utils import dsFilenameVal
from train import mixMean, mixVariance
from torch import distributions as D

# -----------------------------------------------------------------------------------------
# basic plots

# Create a color map from light blue to dark blue
cmap = colors.LinearSegmentedColormap.from_list("custom_blues", ["#add8e6", "#000080"])

# plot training loss
def plotTrain(date: str, model_id: str):
    path = Path('losses', model_id, date, 'loss_train.csv')
    df = pd.read_csv(path)
    df = df.groupby(['iteration'])['loss'].agg(['mean', 'std', 'max', 'min']).reset_index()
    df = df[df['iteration'] != 1]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['iteration'], df['mean'], label='Mean Loss', color='blue')
    
    # Shade min-max area
    ax.fill_between(df['iteration'],
                    # df['mean'] - df['std'], 
                    # df['mean'] + df['std'], 
                    df['min'], 
                    df['max'], 
                    color='blue', alpha=0.3)
    
    plt.xlabel('Iteration')
    plt.ylabel('-log p(target)')
    # plt.ylim(-1, 3)
    plt.grid(True) 
    plt.show()


# plot validation loss
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
    
    # Create the plot
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


# def mixMean(locs: np.ndarray,
#             weights: np.ndarray) -> np.ndarray:
#     return np.sum(locs * weights, axis=-1)


# def mixVariance(locs: np.ndarray,
#                 scales: np.ndarray,
#                 weights: np.ndarray,
#                 mean: np.ndarray) -> np.ndarray:
#     second_moments = np.square(locs) + np.square(scales)
#     return np.sum(second_moments * weights, axis=-1) - np.square(mean)


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


# def sDataFrame(targets, rfx_param_matrix, std_matrix, batch_id: int) -> tuple:
#     mask = (targets[batch_id] != 0.)
#     target = targets[batch_id, mask]
#     data_alpha = rfx_param_matrix[batch_id, :, mask, 0].transpose(1,0)
#     data_beta = rfx_param_matrix[batch_id, :, mask, 1].transpose(1,0)
#     data_std = std_matrix[batch_id, :, mask].transpose(1,0)
#     n, d = data_std.shape
#     values_alpha = data_alpha.flatten() + 1.
#     values_beta = data_beta.flatten()
#     values_mean = values_beta/(values_alpha - 1.)
#     values_mode = values_beta/(values_alpha + 1.)
#     values_std = data_std.flatten()
#     row_indices = np.repeat(np.arange(n), d) + 1
#     column_indices = np.tile(np.arange(d), n)
#     return pd.DataFrame({
#         'n' : row_indices,
#         'alpha' : values_alpha,
#         'beta' : values_beta,
#         'std_prop' : np.sqrt(values_alpha),
#         'mean' : np.sqrt(values_mean), # as we compare with std but the loss function optimizes for variance
#         'mode' : np.sqrt(values_mode),
#         'std' : values_std,
#         'd': column_indices
#     }), target


# plot validation loss from predictions
def lossFromPredictions(data, targets, source = "proposed"):
    if source == "proposed":
        means = torch.tensor(data["means_p"])
        stds = torch.tensor(data["stds_p"])
    elif source == "analytical":
        means = torch.tensor(data["means_a"])
        stds = torch.tensor(data["stds_a"])
        mask = (stds == 0.).float()
        stds = stds + mask
    else:
        raise ValueError
    betas = targets.unsqueeze(1).expand_as(means)
    proposal = D.Normal(means, stds)
    return -proposal.log_prob(betas) # (batch, n, d)
    
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
    

if __name__ == "__main__":
    noise_type = "variable"
    fixed = 0. # equivalent to noise_type
    ds_type = "mfx"
    # nice result
    # model_id = f'transformer-256-512-8-4-dropout=0-noise={noise_type}-seed=0-loss=logprob' # server
    # date = '20250120-090048' # nice
    
    model_id = f'transformer-128-256-8-3-dropout=0-noise={noise_type}-seed=0-loss=logprob' # macbook
    date = '20250128-141519'
        
    # train and val loss
    plotTrain(date, model_id)
    plotVal(date, model_id)
    if ds_type == "ffx":
        plotVal(date, model_id, suffix="kl")
    
    # proposal distribution
    iteration = 30
    data = preloadPredictions(date,
                              model_id,
                              iteration=iteration,
                              n_batches=45,
                              fixed=fixed,
                              ds_type=ds_type)
    max_d = 1
    for i in range (5):
        plotParamsWrapper(data, 500 * max_d + i, iteration, paramtype="beta")
    
    for i in range (20):
        plotParamsWrapper(data, 500 * max_d + i, iteration, paramtype="rfx")
    
    for i in range (5):
        plotParamsWrapper(data, 500 * max_d + i, iteration, paramtype="ig")
        
    
    # plot validation loss over n
    df_p = loss2df(data, source = "proposed")
    df_a = loss2df(data, source = "analytical")
    plotVal2(df_p, iteration, source = "proposed")
    plotVal2(df_a, iteration, source = "analytical")
    
    # # misc
    # alpha, beta = 3, 8
    # dist = torch.distributions.inverse_gamma.InverseGamma(alpha, beta)
    # x = torch.arange(0.1, 4., step=(4.-.1)/500)
    # probs = dist.log_prob(x).detach().exp()
    # plt.plot(x, probs)
    # print(dist.mean)
    # print(dist.mode)
    
    # betas = torch.tensor([1, 2, 3])
    # means = torch.tensor([1.1, 2, 3])
    # stds = torch.tensor([0.1, 1, 1])
    # proposal = torch.distributions.normal.Normal(means, stds)
    # loss = -proposal.log_prob(betas)
    # print(loss)
    
    
