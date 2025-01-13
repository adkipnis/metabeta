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


