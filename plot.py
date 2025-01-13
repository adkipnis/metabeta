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


