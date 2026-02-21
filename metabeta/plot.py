import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator
import seaborn as sns

cmap = plt.get_cmap('tab20')
PALETTE = [mcolors.to_hex(cmap(i)) for i in range(0, cmap.N, 2)]
PALETTE += [mcolors.to_hex(cmap(i)) for i in range(1, cmap.N, 2)]


class Plot:
    @staticmethod
    def dataset(
        x: np.ndarray | pd.DataFrame,
        names: list[str] | None = None,
        color: str = 'green',
        alpha: float = 0.75,
        title: str = '',
        kde: bool = True,
    ):
        """pair grid for all dataset columns
        - histograms along diagonal
        - scatter in the upper triangular
        - (optional) kde plots in the lower triangular
        inspired by https://github.com/bayesflow-org/bayesflow
        """

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
            _histplot,
            color=color,
            alpha=alpha,
            kde=False,
            fill=True,
            stat='density',
            common_norm=False,
        )

        # scatter plots along upper trinagular
        alpha_point = 1 / np.log(n)
        g.map_upper(sns.scatterplot, color=color, alpha=alpha_point, s=40, edgecolor='k', lw=0)

        # KDEs along lower triangular
        if kde:
            g.map_lower(sns.kdeplot, color=color, alpha=alpha, fill=True, warn_singular=False)
        else:
            g.map_lower(sns.scatterplot, color=color, alpha=alpha_point, s=40, edgecolor='k', lw=0)

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

    @staticmethod
    def correlation(
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

    @staticmethod
    def _groupedRecovery(
        ax: Axes,
        targets: np.ndarray,
        estimates: np.ndarray,
        mask: np.ndarray,
        stats: dict[str, float],
        # strings
        names: list[str],
        title: str = '',
        y_name: str = 'Estimate',
        # plot details
        marker: str = 'o',
        alpha: float = 0.15,
        upper: bool = True,
        lower: bool = True,
    ) -> None:
        """scatter plot: ground truth vs. estimates"""
        # check sizes
        d = len(names)
        assert targets.shape[-1] == estimates.shape[-1] == d, 'shape mismatch'
        # if targets.dim() == 3:
        # ...

        # init figure
        ax.set_axisbelow(True)
        ax.grid(True)
        min_val = min(targets.min(), estimates.min())
        max_val = max(targets.max(), estimates.max())
        delta = max_val - min_val
        const = delta * 0.05
        limits = (min_val - const, max_val + const)
        ax.set_xlim(limits, auto=False)
        ax.set_ylim(limits, auto=False)
        ax.plot(limits, limits, '--', lw=2, zorder=1, color='grey')  # diagline

        # overlay plots
        for i in range(d):
            mask_i = mask[..., i]
            if mask_i.sum() == 0:
                continue
            targets_i = targets[mask_i, i]
            estimates_i = estimates[mask_i, i]
            color_i = 'darkviolet' if 'epsilon' in names[i] else None
            ax.scatter(
                targets_i,
                estimates_i,
                color=color_i,
                s=70,
                marker=marker,
                alpha=alpha,
                label=names[i],
            )

        # stats info
        txt = [f'{k} = {v:.3f}' for k, v in stats.items()]
        ax.text(
            0.7,
            0.1,
            '\n'.join(txt),
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=22,
            bbox=dict(
                boxstyle='round',
                facecolor=(1, 1, 1, 0.7),
                edgecolor=(0, 0, 0, alpha),
            ),
        )

        # final touches
        ml = max(np.floor(delta / 4), 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(ml))
        ax.yaxis.set_major_locator(MultipleLocator(ml))
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=22, markerscale=2.5, loc='upper left')
        if y_name:
            ax.set_ylabel(y_name, fontsize=26, labelpad=10)
        if upper and lower:
            ax.set_title(title, fontsize=30, pad=15)
            ax.set_xlabel('Ground Truth', fontsize=26, labelpad=10)
        elif upper:
            ax.set_title(title, fontsize=30, pad=15)
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelcolor='w', size=1)
        elif lower:
            ax.set_xlabel('Ground Truth', fontsize=26, labelpad=10)

    def groupedRecovery(
        self,
        axs: Axes,
        targets: list[torch.Tensor],
        estimates: list[torch.Tensor],
        masks: list[torch.Tensor],
        metrics: list[dict[str, float]],
        # string
        names: list[list[str]],
        titles: list[str] = [],
        y_name: str = 'Estiamte',
        # plot details
        marker: str = 'o',
        alpha: float = 0.15,
        upper: bool = True,
        lower: bool = True,
    ) -> None:
        first = True
        for ax, tar, est, mas, met, nam, tit in zip(
            axs, targets, estimates, masks, metrics, names, titles
        ):
            # colors = palette[i:i+len(_names)]
            self._groupedRecovery(
                ax,
                targets=tar.numpy(),
                estimates=est.numpy(),
                mask=mas.numpy(),
                stats=met,
                names=nam,
                title=tit,
                y_name=(y_name if first else ''),
                marker=marker,
                alpha=alpha,
                upper=upper,
                lower=lower,
            )
            first = False


plot = Plot()
