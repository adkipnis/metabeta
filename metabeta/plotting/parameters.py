import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from metabeta.utils.evaluation import Proposal, getAllNames, getCorrRfxNames

_GREEK_LATEX = {'σ': r'\sigma', 'ε': r'\varepsilon', 'μ': r'\mu', 'τ': r'\tau'}


def _formatName(name: str) -> str:
    # 'σ_Intercept' -> r'$\sigma_{\mathrm{Intercept}}$',  'σ_ε' -> r'$\sigma_\varepsilon$'
    if '_' not in name:
        return name
    base, sub = name.split('_', 1)
    for ch, latex in _GREEK_LATEX.items():
        base = base.replace(ch, latex)
        sub = sub.replace(ch, latex)
    if sub.startswith('\\'):  # single Greek letter — no \mathrm wrapper
        return rf'${base}_{sub}$'
    return rf'${base}_{{\mathrm{{{sub}}}}}$'


def _kdeplot_on(ax, x, **kwargs):
    """KDE overlay on a given axis using a detached twin y-axis."""
    ax2 = ax.twinx()
    sns.kdeplot(x, **kwargs, ax=ax2)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.set_yticklabels([])


def _prior_pdf_on(ax, x_grid, pdf, **kwargs):
    """Plot an analytical prior PDF on a detached twin y-axis.

    Saves and restores the shared x-axis limits so that the prior's wide domain
    (±6σ) does not expand the column's axis range beyond the posterior window.
    """
    xlim = ax.get_xlim()
    ax2 = ax.twinx()
    ax2.plot(x_grid, pdf, **kwargs)
    # Explicitly set ylim to avoid degenerate zero-range axis when pdf is constant
    # (e.g. LKJ(1) uniform prior); also clips negative y for cleanliness.
    ax2.set_ylim(0, float(np.max(pdf)) * 1.1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax.set_xlim(xlim)



def _histplot(x, **kwargs):
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
    truth_color: str = 'blueviolet',
    alpha: float = 0.75,
    title: str = '',
    kde: bool = True,
    prior_pdfs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    truth: np.ndarray | None = None,
    d_active: int | None = None,
    q_active: int | None = None,
    height: float = 2.5,
    prior_xlim: bool = True,
    show_corr: bool = True,
):
    """Pair-grid of parameter samples for a single dataset at batch {index}.

    - Diagonal: marginal posterior KDE/histogram, with analytical prior PDF overlaid if
      prior_pdfs is given (one (x_grid, density) pair per parameter, in display order).
    - Upper triangle: posterior scatter.
    - Lower triangle: posterior density contours.

    prior_xlim controls axis scaling:
      True  — axes show the prior range, capped at posterior_mean ± 40·posterior_SD
              so the posterior remains clearly visible (default).
      False — axes are set by the posterior data range (tight).

    d_active and q_active trim padded proposals to the active FFX and RFX dims.
    samples_g layout: ffx[0:proposal.d] | sigma_rfx[proposal.d:proposal.d+proposal.q] | sigma_eps
    """

    def _active_samples(prop: Proposal, idx: int) -> np.ndarray:
        x = prop.samples_g[idx].numpy()
        if d_active is None and q_active is None:
            return x
        _d = d_active if d_active is not None else prop.d
        _q = q_active if q_active is not None else prop.q
        parts = [x[:, :_d]]
        if _q > 0:
            parts.append(x[:, prop.d : prop.d + _q])
        if prop.has_sigma_eps:
            parts.append(x[:, prop.d + prop.q : prop.d + prop.q + 1])
        return np.concatenate(parts, axis=-1)

    # init
    x = _active_samples(proposal, index)

    # optionally append lower-triangle correlation columns
    _q_c = q_active if q_active is not None else proposal.q
    _n_corr = 0
    if show_corr and proposal.corr_rfx is not None and _q_c >= 2:
        corr_mat = proposal.corr_rfx[index].numpy()  # (S, q, q)
        corr_cols = np.stack(
            [corr_mat[:, i, j] for i in range(_q_c) for j in range(i)], axis=1
        )
        x = np.concatenate([x, corr_cols], axis=1)
        _n_corr = _q_c * (_q_c - 1) // 2

    s, d = x.shape
    g = sns.PairGrid(pd.DataFrame(x), height=height)

    # first column index that is a sigma parameter (strictly non-negative)
    _d_sigma = d_active if d_active is not None else proposal.d
    # first column index that is a correlation parameter (bounded [-1, 1])
    _d_corr_start = d - _n_corr

    # setup names
    _names: list[str]
    if names is not None:
        _names = [_formatName(n) for n in names]
        if _n_corr > 0:
            _names += getCorrRfxNames(_q_c)
    else:
        _d_n = d_active if d_active is not None else proposal.d
        _q_n = q_active if q_active is not None else proposal.q
        name_dict = getAllNames(_d_n, _q_n)
        name_dict.pop('rfx')
        _names = [_formatName(n) for n in np.concat(list(name_dict.values()))]
        if _n_corr > 0:
            _names += getCorrRfxNames(_q_c)  # already LaTeX, no _formatName needed

    # marginal posterior — loop instead of map_diag to allow per-column KDE clipping
    for i in range(d):
        kw: dict = dict(color=color, common_norm=False)
        if _d_sigma <= i < _d_corr_start:
            kw['clip'] = (0, None)
        elif i >= _d_corr_start:
            kw['clip'] = (-1, 1)
        if kde:
            _kdeplot_on(g.axes[i, i], x[:, i], alpha=alpha**2, fill=True, **kw)
        else:
            ax2 = g.axes[i, i].twinx()
            sns.histplot(x[:, i], alpha=alpha, kde=False, fill=True, stat='density', ax=ax2, **kw)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.set_ylabel('')
            ax2.set_yticks([])
            ax2.set_yticklabels([])

    # 2d posterior scatter
    alpha_point = 1 / np.log(s)
    g.map_upper(sns.scatterplot, color=color, alpha=alpha_point, s=40, edgecolor='k', lw=0)

    # 2d posterior KDE contours
    g.map_lower(sns.kdeplot, color=color, alpha=alpha, fill=True, warn_singular=False)

    # axis scaling — prior mode widens each parameter's axis to the prior range,
    # capped at posterior_mean ± 6·posterior_SD so the posterior stays readable
    if prior_xlim and prior_pdfs is not None:
        n_prior = len(prior_pdfs)
        for i in range(d):
            if i >= n_prior:
                continue
            prior_lo = float(prior_pdfs[i][0][0])
            prior_hi = float(prior_pdfs[i][0][-1])
            post_mean = float(x[:, i].mean())
            post_sd = float(x[:, i].std())
            lo = max(prior_lo, post_mean - 40.0 * post_sd)
            hi = min(prior_hi, post_mean + 40.0 * post_sd)
            # shared axes: setting any cell in the column/row propagates to all
            g.axes[0, i].set_xlim(lo, hi)
            g.axes[i, 0].set_ylim(lo, hi)

    # marginal prior — analytical PDF line, drawn after map_upper/lower so that the
    # posterior has already set the axis limits before we save/restore them
    if prior_pdfs is not None:
        for i, (x_grid, pdf) in enumerate(prior_pdfs[:d]):
            _prior_pdf_on(
                g.axes[i, i], x_grid, pdf,
                color=prior_color, lw=1.5, alpha=0.6,
            )

    # 2d prior analytical contours — evaluated on the axis-range grid so that levels
    # are chosen relative to the density variation *within the visible window*.
    # Skip cells where either marginal is essentially flat (e.g. uniform LKJ prior):
    # the product-of-marginals density would vary only along the other axis, producing
    # uninformative vertical/horizontal lines rather than 2D contours.
    if prior_pdfs is not None:
        n_prior = len(prior_pdfs)
        for i in range(d):
            for j in range(i):
                if i >= n_prior or j >= n_prior:
                    continue
                pi_flat = float(prior_pdfs[i][1].max()) - float(prior_pdfs[i][1].min()) < 1e-8
                pj_flat = float(prior_pdfs[j][1].max()) - float(prior_pdfs[j][1].min()) < 1e-8
                if pi_flat or pj_flat:
                    continue
                ax = g.axes[i, j]
                xlim, ylim = ax.get_xlim(), ax.get_ylim()

                xg_j, pdf_j = prior_pdfs[j]
                xg_i, pdf_i = prior_pdfs[i]
                xg = np.linspace(xlim[0], xlim[1], 100)
                yg = np.linspace(ylim[0], ylim[1], 100)
                # left/right=0 handles half-distributions that start at 0
                pj = np.interp(xg, xg_j, pdf_j, left=0.0, right=0.0)
                pi = np.interp(yg, xg_i, pdf_i, left=0.0, right=0.0)

                Z = pi[:, np.newaxis] * pj[np.newaxis, :]  # (100, 100)
                X, Y = np.meshgrid(xg, yg)

                if Z.max() > 0:
                    ax.contour(X, Y, Z, colors=[prior_color], alpha=0.30, linewidths=1.0)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

    # ground truth: vertical line on diagonal, x marker on off-diagonal
    if truth is not None:
        for i in range(d):
            g.axes[i, i].axvline(truth[i], color=truth_color, lw=1.5, ls='--', zorder=5, alpha=0.8)
        for i in range(d):
            for j in range(d):
                if i != j:
                    g.axes[i, j].scatter(
                        [truth[j]],
                        [truth[i]],
                        color=truth_color,
                        marker='x',
                        s=60,
                        lw=2.0,
                        alpha=0.8,
                        zorder=5,
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
