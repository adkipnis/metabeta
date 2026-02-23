import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plotCorrelation(
    x: np.ndarray | pd.DataFrame,
    names: list[str] | None = None,
):
    """plot correlation matrix over columns of x"""
    # init fig
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # plot correlation matrix
    corr = np.corrcoef(x, rowvar=False)
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap='RdBu')

    # plot colorbar
    cbar = fig.colorbar(
        cax,
        ax=ax,
        fraction=0.046,  # controls height relative to axes
        pad=0.04,  # spacing from plot
    )
    cbar.set_ticks(np.linspace(-1, 1, 5).tolist())
    cbar.ax.tick_params(labelsize=16)

    # ticks and labels
    ax.tick_params(axis='x', bottom=False, top=True)
    d = len(corr)
    ticks = np.arange(0, d, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if names is None:
        names = [rf'$x_{i+1}$' for i in range(d)]
    ax.set_xticklabels(names, fontsize=20)
    ax.set_yticklabels(names, fontsize=20)

    plt.show()

