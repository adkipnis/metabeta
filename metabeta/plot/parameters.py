import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from metabeta.utils.evaluation import Proposal, getAllNames


def _kdeplot_on(ax, x, **kwargs):
    """KDE overlay on a given axis using a detached twin y-axis."""
    ax2 = ax.twinx()
    sns.kdeplot(x, **kwargs, ax=ax2)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.set_yticklabels([])


def _kdeplot(x, **kwargs):
    # sns kdeplot but with detached axes
    _kdeplot_on(plt.gca(), x, **kwargs)


def _histplot(x, **kwargs):
    # sns histplot but with detached axes
    ax2 = plt.gca().twinx()
    sns.histplot(x, **kwargs, ax=ax2)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.set_yticklabels([])


def plotParameters(
    proposal: Proposal,
    index: int = 0,
    names: list[str] | None = None,
    color: str = 'darkgreen',
    prior_color: str = 'steelblue',
    alpha: float = 0.75,
    title: str = '',
    kde: bool = True,
    prior: Proposal | None = None,
):
    """pair grid of parameter samples for a single dataset at batch {index}
    - histograms / KDEs along diagonal
    - scatter in the upper triangular
    - KDE contours in the lower triangular

    If prior is given, overlays prior marginal KDEs on the diagonal (independent
    twin y-axis, non-intrusive) and light unfilled prior contours on the lower
    triangle.
    """

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

    # marginal posterior
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

    # marginal prior
    if prior is not None:
        x_prior = prior.samples_g[index].numpy()
        for i in range(d):
            _kdeplot_on(
                g.axes[i, i],
                x_prior[:, i],
                color=prior_color,
                alpha=0.50,
                fill=False,
                common_norm=False,
                lw=1.5,
            )

    # 2d posterior scatter
    alpha_point = 1 / np.log(s)
    g.map_upper(sns.scatterplot, color=color, alpha=alpha_point, s=40, edgecolor='k', lw=0)

    # 2d posterior KDE contours
    g.map_lower(sns.kdeplot, color=color, alpha=alpha, fill=True, warn_singular=False)

    # 2d prior KDE contours
    if prior is not None:
        x_prior = prior.samples_g[index].numpy()
        x_prior_df = pd.DataFrame(x_prior)
        for i in range(d):
            for j in range(i):
                sns.kdeplot(
                    data=x_prior_df,
                    x=j,
                    y=i,
                    ax=g.axes[i, j],
                    color=prior_color,
                    fill=False,
                    alpha=0.30,
                    warn_singular=False,
                )

    # set labels
    for i in range(d):
        g.axes[i, 0].set_ylabel(_names[i], fontsize=16)
        g.axes[d - 1, i].set_xlabel(_names[i], fontsize=16)
        for j in range(d):
            g.axes[i, j].grid(alpha=0.5)
            g.axes[i, j].set_axisbelow(True)

    # reposition labels
    for i, ax in enumerate(g.axes[0, :]):
        xlabel = g.axes[-1, i].get_xlabel()
        g.figure.text(
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
