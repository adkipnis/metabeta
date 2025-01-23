import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from dataset import LMDataset
from utils import dsFilenameVal
from torch import distributions as D


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


def preloadPredictions(date: str, model_id: str, iteration: int = 100, n_batches: int = 45, fixed: float = 0, ds_type: str = "ffx") -> Dict[str, np.ndarray]:
    # gather predicted means and stds
    paths = [Path('predictions', model_id, date, f'predictions_i={iteration}_b={batch}.pt')
             for batch in range(n_batches)]
    predictions = [torch.load(paths[batch], weights_only=False) for batch in range(n_batches)]
    means_p = torch.cat([x["means"] for x in predictions]).numpy()
    stds_p = torch.cat([x["stds"] for x in predictions]).numpy()
    s_p = torch.cat([x["s"] for x in predictions]).numpy()
    abs_p = torch.cat([x["abs"] for x in predictions]).numpy()
    
    # gather validation data
    filename = dsFilenameVal(ds_type, 8, 50, fixed)
    ds_val_raw = torch.load(filename, weights_only=False)
    ds_val = LMDataset(**ds_val_raw, permute=False)
    targets = torch.stack([x["beta"] for x in ds_val], dim=0).numpy()
    sigma_errors = torch.stack([x["sigma_error"] for x in ds_val], dim=0).numpy()
    out = {"targets": targets, "sigma_errors": sigma_errors,
           "means_p": means_p, 
           "stds_p": stds_p,
           "s_p": s_p,
           "abs_p": abs_p,}
    if ds_type == "ffx":
        means_a = torch.stack([x["mu_n"] for x in ds_val], dim=0).numpy()
        stds_a = [torch.diagonal(x["Sigma_n"], dim1=-2, dim2=-1).sqrt() for x in ds_val]
        stds_a = torch.stack(stds_a, dim=0).numpy()
        as_a = torch.stack([x["a_n"] for x in ds_val], dim=0).unsqueeze(-1)
        bs_a = torch.stack([x["b_n"] for x in ds_val], dim=0).unsqueeze(-1)
        abs_a = torch.cat([as_a, bs_a], dim=-1).numpy()
        assert means_a.shape[0] == means_p.shape[0], \
            "Different number of observations for analytical and trained solutions."
        out.update({"means_a": means_a, "stds_a": stds_a, "abs_a": abs_a,})
    elif ds_type == "mfx":
        s = torch.stack([x["S"].sqrt() for x in ds_val], dim=0).numpy()
        out.update({"s": s})
    return out


def mvnDataFrame(targets, means_matrix, stds_matrix, batch_id: int) -> tuple:
    mask = (targets[batch_id] != 0.)
    betas = targets[batch_id, mask]
    data_m = means_matrix[batch_id, :, mask].transpose(1,0)
    data_s = stds_matrix[batch_id, :, mask].transpose(1,0)
    values_m = data_m.flatten()
    values_s = data_s.flatten()
    row_indices = np.repeat(np.arange(data_m.shape[0]), data_m.shape[1]) + 1
    column_indices = np.tile(np.arange(data_m.shape[1]), data_m.shape[0])
    return pd.DataFrame({
        'n' : row_indices,
        'mean' : values_m,
        'std' : values_s,
        'd': column_indices
    }), betas
 
    
def analytical_alpha(n: int):
    ns = np.arange(n) + 1
    alpha = 3. + ns / 2.
    return alpha


def igDataFrame(abs_matrix, batch_id: int) -> dict:
    _, n, d = abs_matrix.shape
    if d == 2:
        data_a = abs_matrix[batch_id, :, 0]
        data_b = abs_matrix[batch_id, :, 1]
    else:
        data_a = analytical_alpha(n)
        data_b = abs_matrix[batch_id, :, 0]
    values_a = data_a.flatten()
    values_b = data_b.flatten()
    row_indices = np.arange(data_a.shape[0]) + 1
    return pd.DataFrame({
        'n' : row_indices,
        'alpha' : values_a,
        'beta' : values_b,
    })


def noiseDataFrame(noise_matrix, batch_id: int) -> dict: 
    data = noise_matrix[batch_id, :, 0]
    values = data.flatten()
    row_indices = np.arange(data.shape[0]) + 1
    return pd.DataFrame({
        'n' : row_indices,
        'error_p' : values,
    })


# plot mvn params
def plotMvnParams(df, betas, est_type: str, ax):
    unique_d = df['d'].unique()
    norm = colors.Normalize(vmin=unique_d.min(), vmax=unique_d.max())
    
    # Create the plot
    for d_value, group in df.groupby('d'):
        beta = betas[d_value].item()
        color = cmap(norm(d_value))
        ax.plot(group['n'], group['mean'], label=f'd={d_value}', color=color)
        ax.fill_between(group['n'], 
                        group['mean'] - group['std'], 
                        group['mean'] + group['std'], 
                        color=color, alpha=0.1)  # Shade ± SD
        ax.axhline(y=beta, color=color, linestyle=':', linewidth=1.5)
    
    # Adding labels and title
    ax.set_xlabel('n')  # X-axis label
    ax.set_ylabel(f'{est_type}')
    if est_type == "analytical": 
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(-7, 7)
    ax.grid(True)           # Show grid
    
    
# plot ig params
def plotIGParams(df_a, df_p, ax):
    # Create the plot
    # ax.plot(df['n'], df['alpha'], label='alpha', color='green')
    ax.plot(df_a['n'], df_a['beta'], label='analytical', color='green')
    ax.plot(df_p['n'], df_p['beta'], label='proposed', color='red')
    
    # Adding labels and title
    ax.set_xlabel('n')  # X-axis label
    ax.set_ylabel(f'noise parameter')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # ax.set_ylim(0, 10)
    ax.grid(True)           # Show grid
    

def plotNoise(df, sigma_error, ax):
    ax.plot(df['n'], df['error_p'], label='proposed', color='orange')
    ax.axhline(y=sigma_error, color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('n')  # X-axis label
    ax.set_ylabel('noise SD')
    ax.set_ylim(0, 4)
    ax.grid(True)
     

def plotParamsWrapper(data: dict, batch_id: int, iteration: int, paramtype = "beta"):
    # unpack
    targets = data["targets"]
    means_a = data["means_a"]
    means_p = data["means_p"]
    stds_a = data["stds_a"]
    stds_p = data["stds_p"]
    abs_a = data["abs_a"]
    abs_p = data["abs_p"]
    sigma_errors = data["sigma_errors"]
    
    # plot MVN parameters
    if paramtype == "beta":
        df_a, betas = mvnDataFrame(targets, means_a, stds_a, batch_id)
        df_p, _ = mvnDataFrame(targets, means_p, stds_p, batch_id)
        fig, axs = plt.subplots(2, sharex=True, figsize=(8, 6))
        fig.suptitle(f'iter={iteration}')
        plotMvnParams(df_a, betas, "analytical", axs[0])
        plotMvnParams(df_p, betas, "proposed", axs[1])
    
    # plot IG parameters
    if paramtype == "ig":
        df_ig_a = igDataFrame(abs_a, batch_id)
        df_ig_p = igDataFrame(abs_p, batch_id)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotIGParams(df_ig_a, df_ig_p, ax)
    
    # plot noise std
    if paramtype == "sigma":
        df_noise = noiseDataFrame(abs_p, batch_id)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotNoise(df_noise, sigma_errors[batch_id], ax)
  

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
    # nice result
    # model_id = f'transformer-256-512-8-4-dropout=0-noise={noise_type}-seed=0-loss=logprob' # server
    # date = '20250120-090048'
    
    model_id = f'transformer-128-256-8-3-dropout=0-noise={noise_type}-seed=0-loss=logprob' # macbook
    date = '20250120-170752'
    
    
    # train and val loss
    plotTrain(date, model_id)
    plotVal(date, model_id)
    plotVal(date, model_id, suffix="kl")
    
    # proposal distribution
    iteration = 100
    data = preloadPredictions(date,
                              model_id,
                              iteration=iteration,
                              n_batches=45,
                              noise=noise_type)
    max_d = 7
    for i in range (5):
        plotParamsWrapper(data, 500 * max_d + i, iteration)
    
    for i in range (5):
        plotParamsWrapper(data, 500 * max_d + i, iteration, paramtype = "ig")
        
    
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
    
    