import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from dataset import LMDataset


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
def plotVal(date: str, model_id: str, suffix: str = "val"):
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
    # df_agg = df_agg[df_agg['d'] == 0]
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


def preloadPredictions(date: str, model_id: str, iteration: int = 100, n_batches: int = 75) -> Dict[str, torch.Tensor]:
    # gather predicted means and stds
    paths = [Path('predictions', model_id, date,
                f'predictions_i={iteration}_b={batch}.pt') for batch in range(n_batches)]
    predictions = [torch.load(paths[batch], weights_only=False) for batch in range(n_batches)]
    means_p = torch.cat([x["means"] for x in predictions]).numpy()
    stds_p = torch.cat([x["stds"] for x in predictions]).numpy()
    abs_p = torch.cat([x["abs"] for x in predictions]).numpy()
    
    # gather analytical means and stds
    path = Path('data', 'dataset-val-noise=0.1.pt')
    ds_val_raw = torch.load(path, weights_only=False)
    ds_val = LMDataset(**ds_val_raw)
    targets = torch.stack([x["beta"] for x in ds_val], dim = 0).numpy()
    means_a = torch.stack([x["mu_n"] for x in ds_val], dim = 0).numpy()
    stds_a = [torch.diagonal(x["Sigma_n"], dim1=-2, dim2=-1).sqrt() for x in ds_val]
    stds_a = torch.stack(stds_a, dim=0).numpy()
    # TODO: correct for noise variance
    as_a = torch.stack([x["a_n"] for x in ds_val], dim = 0).unsqueeze(-1)
    bs_a = torch.stack([x["b_n"] for x in ds_val], dim = 0).unsqueeze(-1)
    abs_a = torch.cat([as_a, bs_a], dim=-1).numpy()
    assert means_a.shape[0] == means_p.shape[0], \
        "Different number of observations for analytical and trained solutions."
    return {"targets": targets,
            "means_p": means_p, "means_a": means_a,
            "stds_p": stds_p, "stds_a": stds_a,
            "abs_p": abs_p, "abs_a": abs_a,}
            


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
def plotIgParams(df, est_type: str, ax):
    # Create the plot
    # ax.plot(df['n'], df['alpha'], label='alpha', color='green')
    ax.plot(df['n'], df['beta'], label='beta', color='red')
    
    # Adding labels and title
    ax.set_xlabel('n')  # X-axis label
    ax.set_ylabel(f'{est_type}')
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(1, 10)
    ax.grid(True)           # Show grid
     

def plotParamsWrapper(data: dict, batch_id: int, iteration: int):
    # unpack
    targets = data["targets"]
    means_a = data["means_a"]
    means_p = data["means_p"]
    stds_a = data["stds_a"]
    stds_p = data["stds_p"]
    abs_a = data["abs_a"]
    abs_p = data["abs_p"]
    
    # plot MVN parameters
    df_a, betas = mvnDataFrame(targets, means_a, stds_a, batch_id)
    df_p, _ = mvnDataFrame(targets, means_p, stds_p, batch_id)
    fig, axs = plt.subplots(2, sharex=True, figsize=(8, 6))
    fig.suptitle(f'iter={iteration}')
    plotMvnParams(df_a, betas, "analytical", axs[0])
    plotMvnParams(df_p, betas, "proposed", axs[1])
    
    # plot IG parameters
    
    df_ig_a = igDataFrame(abs_a, batch_id)
    df_ig_p = igDataFrame(abs_p, batch_id)
    fig, axs = plt.subplots(2, sharex=True, figsize=(8, 6))
    plotIgParams(df_ig_a, "analytical", axs[0])
    plotIgParams(df_ig_p, "proposed", axs[1])
    

if __name__ == "__main__":
    # model_id = 'transformer-256-512-8-4-dropout=0.01-noise=0.1-seed=0-loss=logprob' # server
    model_id = 'transformer-128-256-8-1-dropout=0-noise=variable-seed=0-loss=logprob' # macbook
    date = '20250113-161345'
    
    # train and val loss
    plotTrain(date, model_id)
    plotVal(date, model_id)
    plotVal(date, model_id, suffix="kl")
    
    # proposal distribution
    iteration = 20
    data = preloadPredictions(date, model_id, iteration=iteration, n_batches=45)
    max_d = 1
    for i in range (1):
        plotParamsWrapper(data, 500 * max_d + i, iteration)
    
    
