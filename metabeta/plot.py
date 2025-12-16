import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from matplotlib.axes import Axes
import seaborn as sns

def dataset(x: np.ndarray | pd.DataFrame,
            names: list[str] | None = None,
            color: str = 'green',
            alpha: float = 0.8,
            title: str = '',
            kde: bool = True):
    # adapted from https://github.com/bayesflow-org/bayesflow
    def _histplot(x, **kwargs):
        # sns histplot but with detached axes
        ax2 = plt.gca().twinx()
        sns.histplot(x, **kwargs, ax=ax2)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        ax2.set_ylabel("")
        ax2.set_yticks([])
        ax2.set_yticklabels([])

    # init
    _, d = x.shape
    g = sns.PairGrid(pd.DataFrame(x), height=2.5)

    # setup names
    _names = []
    if names is not None:
        _names = names
    else:
        if isinstance(x, pd.DataFrame):
            x = x.select_dtypes(include=['number'])
            _names = x.columns.to_list()
        else:
            mask = ~np.isnan(x).any(axis=0)
            x = x[:, mask]
            _names = [rf'$x_{i+1}$' for i in range(d)]

    # histograms along main diagonal
    g.map_diag(
        _histplot, color=color, alpha=alpha, kde=kde,
        fill=True, stat="density", common_norm=False)

    # scatter plots along upper trinagular
    g.map_upper(
        sns.scatterplot, color=color,
        alpha=0.6, s=40, edgecolor="k", lw=0)

    # KDEs along lower triangular
    if kde:
        g.map_lower(sns.kdeplot, color=color, alpha=alpha, fill=True)

    # set labels
    for i in range(d):
        g.axes[i, 0].set_ylabel(_names[i], fontsize=16)
        g.axes[d - 1, i].set_xlabel(_names[i], fontsize=16)
        for j in range(d):
            g.axes[i, j].grid(alpha=0.5)
            g.axes[i, j].set_axisbelow(True)

    # set title
    if title:
        g.figure.suptitle(title, fontsize=20, y=1.001)
    g.tight_layout()
    return g


