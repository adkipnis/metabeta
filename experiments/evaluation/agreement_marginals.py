"""
experiments/evaluation/agreement_marginals.py — Marginal posterior overlays: MB+IMH vs NUTS.

For chosen datasets from the pre-generated real-data test batches (outputs/data/
{size}-{fam}-real/test.fit.npz), overlays the marginal posterior densities of the IMH-refined
MB posterior (purple) and NUTS (golden) for one fixed effect, one sigma, one random-effect
correlation, and one random effect per dataset (one row each).

Within each dataset, the displayed parameter of each type is the one with the smallest
normalized 1D Wasserstein distance to the NUTS marginal (quantile-matched, scaled by the NUTS
marginal sd), unless overridden via --ffx_override. Proposals are read from the sample caches
written by posterior_eval (loadOrSampleMB / loadOrRefine), so reruns are cheap; on a cache
miss, MB sampling and IMH refinement run live.

The default datasets are the strongest-agreement examples found on the small-*-real test sets
(NUTS-converged, strict mode): cane (small-b-real, #67; best correlation agreement) and
respiratory (small-n-real, #262; best overall agreement among q=2 datasets).

Usage (from repo root):
    uv run python experiments/evaluation/agreement_marginals.py
    uv run python experiments/evaluation/agreement_marginals.py \\
        --datasets small-b-real:67 small-n-real:262 --ffx_override small-b-real:67:0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from metabeta.utils.dataloader import Collection, collateGrouped, subsetBatch
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.posterior_eval import (
    fit2proposal,
    loadModel,
    loadOrRefine,
    loadOrSampleMB,
    posthocDefaults,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.utils.device import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.plot import DPI, niceify
from metabeta.utils.results import Proposal
from metabeta.utils.experiments import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# reference checkpoints per real-data test set (as in the paper's real-data evaluation)
CKPTS = {
    'small-n-real': 'data=small-n-mixed_model=large_seed=13',
    'small-b-real': 'data=small-b-mixed_model=large_seed=6',
    'small-p-real': 'data=small-p-mixed_model=large_seed=4',
    'medium-n-real': 'data=medium-n-mixed_model=large_seed=14',
    'medium-b-real': 'data=medium-b-mixed_model=large_seed=3',
    'medium-p-real': 'data=medium-p-mixed_model=large_seed=11',
}

FAMILY = {'n': 'normal', 'b': 'bernoulli', 'p': 'poisson'}

COL_MB = '#663399'    # rebeccapurple — metabeta
COL_NUTS = '#B8860B'  # darkgoldenrod — NUTS

LW = 3.0
FS_ROW = 26
FS_LEGEND = 24

# quantile grid for the normalized 1D Wasserstein distance
QS = np.linspace(0.005, 0.995, 199)


# ---------------------------------------------------------------------------
# CLI


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Marginal posterior overlays: MB+IMH vs NUTS on real datasets',
    )
    parser.add_argument('--datasets',     type=str, nargs='+',
                        default=['small-b-real:67', 'small-n-real:262'],
                        help='Datasets to plot as data_id:index pairs (index into the full '
                             'test file); one row per dataset')
    parser.add_argument('--ffx_override', type=str, nargs='*',
                        default=['small-b-real:67:0'],
                        help='Override the auto (min-W1) fixed-effect panel as '
                             'data_id:index:ffx_column triples')
    parser.add_argument('--n_samples',    type=int, default=1000)
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--seed',         type=int, default=0)
    parser.add_argument('--prefix',       type=str, default='latest')
    parser.add_argument('--bw_adjust',    type=float, default=1.5,
                        help='KDE bandwidth widening; IMH chains repeat draws, which shows up '
                             'as spurious wiggles at the default bandwidth (default: 1.5)')
    parser.add_argument('--outdir',       type=str, default=str(RESULTS_DIR))
    parser.add_argument('--transparent',  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--verbosity',    type=int, default=1)
    # fmt: on
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Proposal loading (cached) and panel selection


def w1Normalized(a: np.ndarray, b: np.ndarray) -> float:
    """Quantile-matched Wasserstein-1 between sample sets, scaled by the sd of b (NUTS)."""
    s = b.std()
    if s < 1e-12:
        return float('nan')
    return float(np.mean(np.abs(np.quantile(a, QS) - np.quantile(b, QS))) / s)


def loadProposals(
    data_id: str,
    prefix: str,
    n_samples: int,
    batch_size: int,
    seed: int,
    refine: bool = True,
) -> tuple[dict[str, torch.Tensor], Proposal, Proposal, np.ndarray]:
    """Rebuild the NUTS-converged test batch and load the MB+IMH and NUTS proposals.

    Returns (batch, p_mb, p_nuts, idx_full) where idx_full maps positions in the converged
    subset back to indices into the full test file. With ``refine`` (default), the MB proposal
    is IMH-refined with the family preset; otherwise the raw MB flow posterior is returned.
    """
    if data_id not in CKPTS:
        raise KeyError(f'no reference checkpoint known for {data_id}')
    data_path = DATA_DIR / data_id / 'test.fit.npz'
    ckpt_dir = CHECKPOINT_DIR / CKPTS[data_id]
    device = setDevice('cpu')
    model, model_cfg = loadModel(ckpt_dir, prefix, device)

    col = Collection(
        data_path,
        permute=False,
        max_d=model_cfg.max_d,
        max_q=model_cfg.max_q,
        exclude_prefixes=('advi_', 'laplace_'),
    )
    B_total = len(col)
    batch = collateGrouped([col[i] for i in range(B_total)])

    # inject precomputed analytical stats from the sibling test.npz (as in real_posterior.py)
    if 'stats' not in batch:
        base_path = data_path.with_name('test.npz')
        if base_path.exists():
            base_col = Collection(
                base_path, permute=False, max_d=model_cfg.max_d, max_q=model_cfg.max_q
            )
            if len(base_col) == B_total:
                base_batch = collateGrouped([base_col[i] for i in range(B_total)])
                if 'stats' in base_batch:
                    batch['stats'] = base_batch['stats']
                del base_batch
            del base_col

    conv_mask = nutsConvergeMask(batch, mode='strict')
    idx_full = np.arange(B_total)
    if conv_mask is not None:
        batch = subsetBatch(batch, conv_mask)
        idx_full = idx_full[conv_mask]

    p_mb, _ = loadOrSampleMB(
        model,
        batch,
        data_path,
        ckpt_dir,
        prefix,
        n_samples,
        batch_size,
        seed,
        device,
        conv_mask,
        warmup=False,
    )
    p_nuts = fit2proposal(batch, 'nuts')
    p_mb.rescale(batch['sd_y'])
    p_nuts.rescale(batch['sd_y'])
    batch = rescaleData(batch)

    if not refine:
        return batch, p_mb, p_nuts, idx_full

    lf = model_cfg.likelihood_family
    method = posthocDefaults(lf)[0]
    p_ref, _ = loadOrRefine(
        method,
        p_mb,
        batch,
        data_path,
        ckpt_dir,
        prefix,
        n_samples,
        seed,
        lf,
        True,
        conv_mask,
        batch_size,
    )
    return batch, p_ref, p_nuts, idx_full


def corrSamples(p: Proposal, b: int) -> np.ndarray:
    """Off-diagonal correlation samples for dataset b, shape (S,)."""
    c = p.corr_rfx[b]
    c = c.reshape(-1, c.shape[-2], c.shape[-1])
    return c[:, 1, 0].numpy()


def pickPanels(
    p_mb: Proposal,
    p_nuts: Proposal,
    batch: dict[str, torch.Tensor],
    b: int,
    ffx_override: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Best-agreeing (min normalized W1) parameter of each type for dataset b.

    Returns [(panel_title, mb_samples, nuts_samples, w1), ...] for a fixed effect, a sigma,
    the correlation, and a random effect. Requires q = 2 (correlation parameter present).
    """
    d_act = int(batch['mask_d'][b].sum())
    q_act = int(batch['mask_q'][b].sum())
    if q_act != 2:
        raise ValueError(f'dataset {b} has q={q_act}; the correlation panel needs q=2')

    def panel(name: str, mb_s: np.ndarray, nuts_s: np.ndarray) -> tuple:
        return (name, mb_s, nuts_s, w1Normalized(mb_s, nuts_s))

    # fixed effect
    ffx = [
        panel(rf'$\beta_{{{j}}}$', p_mb.ffx[b, :, j].numpy(), p_nuts.ffx[b, :, j].numpy())
        for j in range(d_act)
    ]
    best_ffx = ffx[ffx_override] if ffx_override is not None else min(ffx, key=lambda p: p[3])

    # sigma (random-effect sds + residual sd if present)
    sigmas = [
        panel(
            rf'$\sigma_{{{j}}}$',
            p_mb.sigma_rfx[b, :, j].numpy(),
            p_nuts.sigma_rfx[b, :, j].numpy(),
        )
        for j in range(q_act)
    ]
    if p_mb.has_sigma_eps and p_nuts.has_sigma_eps:
        sigmas.append(
            panel(r'$\sigma_\epsilon$', p_mb.sigma_eps[b].numpy(), p_nuts.sigma_eps[b].numpy())
        )
    best_sigma = min(sigmas, key=lambda p: p[3])

    # correlation
    corr = panel(r'$\rho_{10}$', corrSamples(p_mb, b), corrSamples(p_nuts, b))

    # random effect
    group_mask = batch['mask_n'][b].any(-1).numpy().astype(bool)
    rfx = [
        panel(
            rf'$\alpha_{{{j}}}^{{({k})}}$',
            p_mb.rfx[b, k, :, j].numpy(),
            p_nuts.rfx[b, k, :, j].numpy(),
        )
        for k in np.flatnonzero(group_mask)
        for j in range(q_act)
    ]
    best_rfx = min(rfx, key=lambda p: p[3])

    return [best_ffx, best_sigma, corr, best_rfx]


# ---------------------------------------------------------------------------
# Plot


def plotAgreement(
    chosen: list[tuple[str, int]],
    ffx_overrides: dict[tuple[str, int], int],
    cfg: argparse.Namespace,
    outdir: Path,
) -> None:
    sns.set_style('white')
    nrows, ncols = len(chosen), 4
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), dpi=DPI, squeeze=False
    )

    cache: dict[str, tuple] = {}
    for i, (data_id, idx) in enumerate(chosen):
        if data_id not in cache:
            cache[data_id] = loadProposals(
                data_id, cfg.prefix, cfg.n_samples, cfg.batch_size, cfg.seed
            )
        batch, p_mb, p_nuts, idx_full = cache[data_id]
        conv_pos = np.flatnonzero(idx_full == idx)
        if len(conv_pos) == 0:
            raise ValueError(f'{data_id} #{idx} is not in the NUTS-converged subset')
        b = int(conv_pos[0])

        panels = pickPanels(p_mb, p_nuts, batch, b, ffx_overrides.get((data_id, idx)))
        with np.load(DATA_DIR / data_id / 'test.fit.npz', allow_pickle=True) as raw:
            src = str(np.asarray(raw['source'][idx]).item()).split('__')[0]
        family = FAMILY[data_id.split('-')[1]]
        logger.info(
            '%s #%d (%s, %s): %s',
            data_id,
            idx,
            src,
            family,
            '  '.join(f'{name} w1={v:.3f}' for name, _, _, v in panels),
        )

        for j, (name, mb_s, nuts_s, _) in enumerate(panels):
            ax = axs[i, j]
            sns.kdeplot(
                nuts_s,
                ax=ax,
                color=COL_NUTS,
                fill=True,
                alpha=0.35,
                linewidth=LW,
                cut=3,
                bw_adjust=cfg.bw_adjust,
            )
            sns.kdeplot(
                mb_s,
                ax=ax,
                color=COL_MB,
                fill=True,
                alpha=0.35,
                linewidth=LW,
                cut=3,
                bw_adjust=cfg.bw_adjust,
            )
            info = {
                'title': name,
                'show_legend': False,
                'despine': True,
                'ylabel': f'{src}\n({family}, #{idx})' if j == 0 else None,
                'ylabel_fs': FS_ROW,
                'grid_alpha': 1.0,  # sentinel: skip niceify's grid
            }
            niceify(ax, info)
            ax.grid(False)
            if j > 0:
                ax.set_ylabel('')
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

    # same right-legend styling as plotComparison (used by evaluate.py): opaque dot proxies
    handles = [
        Line2D([], [], marker='o', linestyle='', markersize=10, color=COL_MB, label='MB+IMH'),
        Line2D([], [], marker='o', linestyle='', markersize=10, color=COL_NUTS, label='NUTS'),
    ]
    fig.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(0.995, 0.5),
        fontsize=FS_LEGEND,
        markerscale=2.5,
    )
    fig.tight_layout(rect=(0, 0, 0.99, 1), h_pad=2.5)
    for ending in ('png', 'pdf'):
        out = outdir / f'agreement_marginals.{ending}'
        fig.savefig(out, bbox_inches='tight', pad_inches=0.15, transparent=cfg.transparent)
        logger.info('Saved plot to %s', out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)

    chosen: list[tuple[str, int]] = []
    for spec in cfg.datasets:
        data_id, idx = spec.rsplit(':', 1)
        chosen.append((data_id, int(idx)))

    ffx_overrides: dict[tuple[str, int], int] = {}
    for spec in cfg.ffx_override or []:
        data_id, idx, j = spec.rsplit(':', 2)
        ffx_overrides[(data_id, int(idx))] = int(j)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plotAgreement(chosen, ffx_overrides, cfg, outdir)


if __name__ == '__main__':
    main()
