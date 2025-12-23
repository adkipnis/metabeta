import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from matplotlib.axes import Axes
import seaborn as sns

class Plot:
    @staticmethod
    def dataset(x: np.ndarray | pd.DataFrame,
                names: list[str] | None = None,
                color: str = 'green',
                alpha: float = 0.75,
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
        n, d = x.shape
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
            _histplot, color=color, alpha=alpha, kde=False,
            fill=True, stat="density", common_norm=False)

        # scatter plots along upper trinagular
        alpha_point = 1 / np.log(n)
        g.map_upper(
            sns.scatterplot, color=color,
            alpha=alpha_point, s=40, edgecolor="k", lw=0)

        # KDEs along lower triangular
        if kde:
            g.map_lower(sns.kdeplot, color=color, alpha=alpha, fill=True,
                        warn_singular=False)
        else:
            g.map_lower(
                sns.scatterplot, color=color,
                alpha=alpha_point, s=40, edgecolor="k", lw=0)

        # set labels
        for i in range(d):
            g.axes[i, 0].set_ylabel(_names[i], fontsize=16)
            g.axes[d-1, i].set_xlabel(_names[i], fontsize=16)
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
                xlabel, ha='center', va='bottom', fontsize=16) 
        for i in range(g.axes.shape[1]):
            g.axes[-1, i].set_xlabel('')

        # set title
        if title:
            g.figure.suptitle(title, fontsize=20, y=1.001)
        g.tight_layout()
        return g


    @staticmethod
    def correlation(x: np.ndarray | pd.DataFrame,
                    names: list[str] | None = None,
                    ):
        # init fig
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        
        # plot correlation matrix
        corr = np.corrcoef(x, rowvar=False)
        cax = ax.matshow(corr, vmin=-1, vmax=1, cmap='RdBu')
        
        # plot colorbar
        cbar = fig.colorbar(
            cax,
            ax=ax,
            fraction=0.046,   # controls height relative to axes
            pad=0.04          # spacing from plot
        )
        cbar.set_ticks(np.linspace(-1, 1, 5))
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

plot = Plot()
