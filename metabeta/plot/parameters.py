import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from metabeta.utils.evaluation import Proposal, getAllNames


def plotParameters(
    proposal: Proposal,
    index: int = 0,
    names: list[str] | None = None,
    color: str = 'darkgreen',
    alpha: float = 0.75,
    title: str = '',
    kde: bool = True,
):
    """pair grid of parameter samples for a single dataset at batch {index}
    - histograms along diagonal
    - scatter in the upper triangular
    - (optional) kde plots in the lower triangular
    """

    def _kdeplot(x, **kwargs):
        # sns histplot but with detached axes
        ax2 = plt.gca().twinx()
        sns.kdeplot(x, **kwargs, ax=ax2)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        ax2.set_ylabel('')
        ax2.set_yticks([])
        ax2.set_yticklabels([])

    def _histplot(x, **kwargs):
        # sns histplot but with detached axes
        ax2 = plt.gca().twinx()
        sns.histplot(x, **kwargs, ax=ax2)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        ax2.set_ylabel('')
        ax2.set_yticks([])
        ax2.set_yticklabels([])

    # init
    x = proposal.samples_g[index].numpy()
    s, d = x.shape
    g = sns.PairGrid(pd.DataFrame(x), height=2.5)

    # setup names
    _names = []
    if names is not None:
        _names = names
    else:
        name_dict = getAllNames(proposal.d, proposal.q)
        name_dict.pop('rfx')
        _names = np.concat(list(name_dict.values()))

    # histograms along main diagonal
    if kde:
        g.map_diag(
            func=_kdeplot,
            color=color,
            alpha=alpha**2,
            fill=True,
            common_norm=False,
        )
    else:
        g.map_diag(
            func=_histplot,
            color=color,
            alpha=alpha,
            kde=False,
            fill=True,
            stat='density',
            common_norm=False,
        )

    # scatter plots along upper trinagular
    alpha_point = 1 / np.log(s)
    g.map_upper(sns.scatterplot, color=color, alpha=alpha_point, s=40, edgecolor='k', lw=0)

    # KDEs along lower triangular
    g.map_lower(sns.kdeplot, color=color, alpha=alpha, fill=True, warn_singular=False)

    # set labels
    for i in range(d):
        g.axes[i, 0].set_ylabel(_names[i], fontsize=16)
        g.axes[d - 1, i].set_xlabel(_names[i], fontsize=16)
        for j in range(d):
            g.axes[i, j].grid(alpha=0.5)
            g.axes[i, j].set_axisbelow(True)

    # move x-label on top
    for i, ax in enumerate(g.axes[0, :]):
        xlabel = g.axes[-1, i].get_xlabel()
        # ax.set_xlabel(xlabel, fontsize=16)
        g.fig.text(
            ax.get_position().x0 + ax.get_position().width / 2,
            ax.get_position().y1 + 0.03,
            xlabel,
            ha='center',
            va='bottom',
            fontsize=16,
        )
    for i in range(g.axes.shape[1]):
        g.axes[-1, i].set_xlabel('')

    # set title
    if title:
        g.figure.suptitle(title, fontsize=20, y=1.001)
    g.tight_layout()
    return g
