"""
experiments/evaluation/agreement_scatter.py — Posterior means and spread: MB+IMH vs NUTS.

Pools all active parameters of the strictly NUTS-converged datasets in the requested real-data
test sets (default: small-*-real and medium-*-real) and compares the IMH-refined MB posterior
against NUTS in two panels:

  1. Posterior means — scatter of MB+IMH vs NUTS per parameter (identity line = agreement),
     colored by parameter type (beta, sigma, rho; random effects only with --rfx, as they
     outnumber the global parameters ~10:1), rasterized.
  2. Posterior SDs — the same scatter for the marginal posterior sds, on log-log axes so the
     sd ratio reads as vertical distance from the identity line; the median ratio is annotated
     (1 = matched spread, < 1 = MB underdispersed).

Proposals are read from the sample caches written by posterior_eval (loadOrSampleMB /
loadOrRefine); on a cache miss, MB sampling and IMH refinement run live (slow for the medium
sets on CPU). Sets are processed one at a time to bound peak memory.

Usage (from within experiments/evaluation/):
    uv run python agreement_scatter.py
    uv run python agreement_scatter.py --data_ids small-n-real medium-n-real
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from agreement_marginals import CKPTS, loadProposals
from metabeta.utils.results import Proposal
from metabeta.utils.sampling import setSeed
from metabeta.utils.logger import setupLogging
from metabeta.utils.plot import DPI, niceify
from metabeta.utils.experiments import RESULTS_DIR

logger = logging.getLogger(__name__)

# one color per parameter type (beta/sigma/rho follow paramColor's hues; alpha gets its own
# hue rather than paramColor's pale-beta variant, which would blend into beta when pooled)
TYPE_COLORS = {
    r'$\beta$': '#fb73a2',
    r'$\sigma$': '#1f77b4',
    r'$\rho$': '#cba818',
    r'$\alpha$': '#2e8b57',
}

LW = 3.0
FS_LEGEND = 24
EPS = 1e-8


# ---------------------------------------------------------------------------
# CLI


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Posterior mean/spread comparison: MB+IMH vs NUTS on real datasets',
    )
    parser.add_argument('--data_ids',   type=str, nargs='+',
                        default=[f'{s}-{f}-real' for s in ('small', 'medium') for f in 'nbp'],
                        help='Real-data test sets to pool (default: small/medium x n/b/p)')
    parser.add_argument('--n_samples',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--prefix',     type=str, default='latest')
    parser.add_argument('--outdir',     type=str, default=str(RESULTS_DIR))
    parser.add_argument('--refine',     action=argparse.BooleanOptionalAction, default=True,
                        help='IMH-refine the MB posterior (default: true); --no-refine '
                             'compares the raw MB flow posterior and appends _mb to the '
                             'output filename')
    parser.add_argument('--rfx',        action=argparse.BooleanOptionalAction, default=False,
                        help='Include random effects (default: false; they outnumber the '
                             'global parameters ~10:1 and drown out the other types)')
    parser.add_argument('--transparent', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--verbosity',  type=int, default=1)
    # fmt: on
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pooled per-parameter statistics


def _newPool() -> dict[str, list]:
    return {t: [] for t in TYPE_COLORS}


def _corrStats(p: Proposal, b: int) -> tuple[float, float]:
    """(mean, sd) of the off-diagonal correlation marginal for dataset b."""
    c = p.corr_rfx[b]
    c = c.reshape(-1, c.shape[-2], c.shape[-1])[:, 1, 0]
    return float(c.mean()), float(c.std())


def collectStats(
    data_id: str,
    means: dict[str, list],
    sds: dict[str, list],
    cfg: argparse.Namespace,
) -> int:
    """Append (mean_nuts, mean_mb) and (sd_nuts, sd_mb) pairs per parameter type; returns the
    number of pooled datasets."""
    batch, p_mb, p_nuts, _ = loadProposals(
        data_id, cfg.prefix, cfg.n_samples, cfg.batch_size, cfg.seed, refine=cfg.refine
    )
    B = batch['X'].shape[0]

    # per-parameter moments over the sample axis
    mom = {}
    for tag, p in (('mb', p_mb), ('nuts', p_nuts)):
        mom[f'{tag}_ffx'] = (p.ffx.mean(-2), p.ffx.std(-2))          # (B, d)
        mom[f'{tag}_sig'] = (p.sigma_rfx.mean(-2), p.sigma_rfx.std(-2))   # (B, q)
        if cfg.rfx:
            mom[f'{tag}_rfx'] = (p.rfx.mean(-2), p.rfx.std(-2))      # (B, m, q)
        if p.has_sigma_eps:
            mom[f'{tag}_eps'] = (p.sigma_eps.mean(-1), p.sigma_eps.std(-1))  # (B,)

    def add(t: str, mean_mb, sd_mb, mean_nuts, sd_nuts) -> None:
        means[t].append(np.stack([np.atleast_1d(mean_nuts), np.atleast_1d(mean_mb)], axis=-1))
        sds[t].append(np.stack([np.atleast_1d(sd_nuts), np.atleast_1d(sd_mb)], axis=-1))

    for b in range(B):
        d_mask = batch['mask_d'][b].numpy().astype(bool)
        q_mask = batch['mask_q'][b].numpy().astype(bool)
        g_mask = batch['mask_n'][b].any(-1).numpy().astype(bool)
        q_act = int(q_mask.sum())

        add(
            r'$\beta$',
            mom['mb_ffx'][0][b].numpy()[d_mask],
            mom['mb_ffx'][1][b].numpy()[d_mask],
            mom['nuts_ffx'][0][b].numpy()[d_mask],
            mom['nuts_ffx'][1][b].numpy()[d_mask],
        )
        add(
            r'$\sigma$',
            mom['mb_sig'][0][b].numpy()[q_mask],
            mom['mb_sig'][1][b].numpy()[q_mask],
            mom['nuts_sig'][0][b].numpy()[q_mask],
            mom['nuts_sig'][1][b].numpy()[q_mask],
        )
        if 'mb_eps' in mom and 'nuts_eps' in mom:
            add(
                r'$\sigma$',
                float(mom['mb_eps'][0][b]),
                float(mom['mb_eps'][1][b]),
                float(mom['nuts_eps'][0][b]),
                float(mom['nuts_eps'][1][b]),
            )
        if q_act == 2 and p_mb.corr_rfx is not None and p_nuts.corr_rfx is not None:
            m_mb, s_mb = _corrStats(p_mb, b)
            m_nu, s_nu = _corrStats(p_nuts, b)
            add(r'$\rho$', m_mb, s_mb, m_nu, s_nu)
        if cfg.rfx:
            add(
                r'$\alpha$',
                mom['mb_rfx'][0][b].numpy()[g_mask][:, q_mask].ravel(),
                mom['mb_rfx'][1][b].numpy()[g_mask][:, q_mask].ravel(),
                mom['nuts_rfx'][0][b].numpy()[g_mask][:, q_mask].ravel(),
                mom['nuts_rfx'][1][b].numpy()[g_mask][:, q_mask].ravel(),
            )

    del batch, p_mb, p_nuts, mom
    return B


# ---------------------------------------------------------------------------
# Plot


def _scatterPanel(
    ax,
    data: dict[str, np.ndarray],
    title: str,
    stats: dict[str, float],
    method_label: str,
    scale: str = 'linear',
) -> None:
    """Identity-line scatter of MB+IMH (y) vs NUTS (x) values, colored by parameter type.

    ``scale``: 'log' for strictly positive values (sds), 'symlog' for signed values spanning
    orders of magnitude (means pooled across rescaled datasets), else linear.
    """
    pooled = np.concatenate(list(data.values()))
    if scale == 'log':
        pooled = np.clip(pooled, EPS, None)
        lims = (pooled.min() / 1.5, pooled.max() * 1.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif scale == 'symlog':
        m = 1.5 * float(np.abs(pooled).max())
        lims = (-m, m)
        ax.set_xscale('symlog', linthresh=1.0)
        ax.set_yscale('symlog', linthresh=1.0)
    else:
        lo, hi = pooled.min(), pooled.max()
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, ls='--', c='gray', lw=1.5, zorder=0)
    # draw the most numerous type first so smaller categories stay visible on top
    for t in sorted(data, key=lambda t: len(data[t]), reverse=True):
        x, y = data[t][:, 0], data[t][:, 1]
        if scale == 'log':
            x, y = np.clip(x, EPS, None), np.clip(y, EPS, None)
        ax.scatter(
            x, y, s=14, color=TYPE_COLORS[t], alpha=0.35, linewidths=0, label=t, rasterized=True
        )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_box_aspect(1)
    niceify(
        ax,
        {
            'title': title,
            'xlabel': 'NUTS',
            'ylabel': method_label,
            'show_legend': False,
            'despine': True,
            'grid_alpha': 1.0,
            'stats': stats,
        },
    )
    ax.grid(False)


def plotScatter(
    means: dict[str, np.ndarray],
    sds: dict[str, np.ndarray],
    n_datasets: int,
    cfg: argparse.Namespace,
    outdir: Path,
) -> None:
    sns.set_style('white')
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6.7, 6.7), dpi=DPI)

    method_label = 'MB+IMH' if cfg.refine else 'MB'
    pooled_means = np.concatenate(list(means.values()))
    r = float(np.corrcoef(pooled_means[:, 0], pooled_means[:, 1])[0, 1])
    _scatterPanel(axs[0], means, 'Posterior Means', {'$r$': r}, method_label, scale='symlog')

    pooled_sds = np.concatenate(list(sds.values()))
    med = float(np.median(pooled_sds[:, 1] / np.clip(pooled_sds[:, 0], EPS, None)))
    _scatterPanel(
        axs[1], sds, 'Posterior SDs', {r'$\mathrm{med.\ ratio}$': med}, method_label, scale='log'
    )
    axs[1].set_ylabel('')

    # same right-legend styling as plotComparison (used by evaluate.py): opaque dot proxies
    handles = [
        Line2D([], [], marker='o', linestyle='', markersize=10, color=TYPE_COLORS[t], label=t)
        for t in means
    ]
    fig.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(0.995, 0.5),
        fontsize=FS_LEGEND,
        markerscale=2.5,
    )
    fig.tight_layout(rect=(0, 0, 0.99, 1))
    logger.info('Pooled %d datasets', n_datasets)
    stem = 'agreement_scatter' if cfg.refine else 'agreement_scatter_mb'
    for ending in ('png', 'pdf'):
        out = outdir / f'{stem}.{ending}'
        fig.savefig(out, bbox_inches='tight', pad_inches=0.15, transparent=cfg.transparent)
        logger.info('Saved plot to %s', out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)

    means, sds = _newPool(), _newPool()
    n_datasets = 0
    for data_id in cfg.data_ids:
        if data_id not in CKPTS:
            logger.warning('Skipping %s: no reference checkpoint known', data_id)
            continue
        n = collectStats(data_id, means, sds, cfg)
        n_datasets += n
        logger.info('%s: pooled %d converged datasets', data_id, n)

    means_arr = {t: np.concatenate(v) for t, v in means.items() if v}
    sds_arr = {t: np.concatenate(v) for t, v in sds.items() if v}
    for t in means_arr:
        ratio = sds_arr[t][:, 1] / np.clip(sds_arr[t][:, 0], EPS, None)
        logger.info('%s: %d params, median sd-ratio %.3f', t, len(ratio), float(np.median(ratio)))

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plotScatter(means_arr, sds_arr, n_datasets, cfg, outdir)


if __name__ == '__main__':
    main()
