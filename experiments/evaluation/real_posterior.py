"""
experiments/evaluation/real_posterior.py — Posterior comparison on real data: MB and ADVI vs NUTS.

Evaluates a model checkpoint on the pre-generated real-data test batch at
outputs/data/{size}-{fam}-real/test.fit.npz, comparing MB and ADVI posteriors
against NUTS as reference.  Since there are no ground-truth parameters, all
metrics are relative to NUTS; only NUTS-converged datasets are included.

Optionally layers post-hoc refinements on the raw MB flow posterior (extra
``MB+<method>`` rows), using the same samplers as experiments/posthoc/ablation.py.
The SNIS/Laplace families keep their PSIS-smoothed IS weights and the comparison
metrics are weight-aware (IMH returns an equal-weight chain); see refineProposal.
The method(s) come from ``--methods`` or, if omitted, the per-family default in
metabeta/configs/presets.yaml (imhMarginal for Normal); pass ``--methods`` with no
values for raw MB only.

Metrics (median ± MAD over datasets):
  r          — Pearson r of posterior means (method vs NUTS), pooled active params
  σ-ratio    — per-dataset median(std_method / std_NUTS) across active params
  rank-MAD   — per-dataset mean |empirical quantile − expected| for NUTS-in-MB rank fracs
  ΔNLL       — LOO-NLL(method) − LOO-NLL(NUTS), per dataset
  Δtime (s)  — tpd(method) − tpd(NUTS), seconds per dataset

Usage (from repo root):
    uv run python experiments/evaluation/real_posterior.py --checkpoint PATH
    uv run python experiments/evaluation/real_posterior.py --checkpoint PATH --prefix best --n_samples 512
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from metabeta.evaluation.point import pointEstimate
from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import Collection, collateGrouped, subsetBatch
from metabeta.utils.evaluation import nutsConvergeMask, subsetProposal
from metabeta.utils.results import Proposal
from metabeta.utils.device import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR
from metabeta.utils.posterior_eval import (
    FAM_LETTER,
    SUPPORTED_METHODS,
    fit2proposal,
    fitBatchMask,
    loadModel,
    loadOrComputeSummary,
    loadOrRefine,
    loadOrSampleMB,
    posthocDefaults,
    validMethods,
)

OUT_DIR = RESULTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Real-data posterior comparison: MB and ADVI vs NUTS',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint',       type=str, required=True)
    parser.add_argument('--prefix',           type=str, default='latest')
    parser.add_argument('--device',           type=str, default='cpu')
    parser.add_argument('--n_samples',        type=int, default=1000)
    parser.add_argument('--batch_size',       type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4,
                        help='Datasets per predictive/LOO summary chunk; lower to bound peak '
                             'memory (NUTS s=4000 tensors are large). Try 1-2 for large/huge.')
    parser.add_argument('--seed',             type=int, default=0)
    parser.add_argument('--outdir',           type=str, default=str(OUT_DIR))
    parser.add_argument('--verbosity',        type=int, default=1)
    parser.add_argument('--data_ids',         type=str, nargs='+', default=None,
                        help='Data IDs to evaluate (default: small/medium/large/huge-{fam}-real)')
    parser.add_argument('--decimals',         type=int, default=2,
                        help='Decimal places in table cells (default: 2)')
    parser.add_argument('--rescale',          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warmup',           action=argparse.BooleanOptionalAction, default=True,
                        help='Untimed 1-sample MB warm-up before timed sampling (default: true)')
    parser.add_argument('--convergence_mode', type=str, default='strict',
                        choices=['liberal', 'strict'])
    parser.add_argument('--methods',          type=str, nargs='*', default=None,
                        choices=list(SUPPORTED_METHODS),
                        help='Post-hoc refinement methods to run on top of raw MB, evaluated '
                             'as extra rows vs NUTS. Default: the family preset in presets.yaml '
                             '(isMarginal for Normal). Pass an empty list to run raw MB only.')
    # fmt: on
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-dataset posterior comparison metrics


def _masks(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Returns (mask_d, mask_q, group_mask) of shapes (B,max_d), (B,max_q), (B,max_m)."""
    return (
        batch.get('mask_d'),
        batch.get('mask_q'),
        batch['mask_n'].any(-1),
    )


def _wmean1d(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """Weighted mean of a (B, S) tensor over the sample axis (uniform if w is None)."""
    return x.mean(-1) if w is None else (x * w).sum(-1)


def _weightedMean(p: Proposal) -> dict[str, torch.Tensor]:
    """Per-param posterior means for the whole batch, honouring p.weights.

    Reuses evaluation.point.pointEstimate (softmax IS weights sum to 1; falls back to a
    plain mean when p.weights is None, so raw MB / ADVI / NUTS / IMH rows are unchanged).
    """
    w = p.weights
    out = {
        'ffx': pointEstimate(p.ffx, w, 'mean'),  # (B, d)
        'sigma_rfx': pointEstimate(p.sigma_rfx, w, 'mean'),  # (B, q)
        'rfx': pointEstimate(p.rfx, w, 'mean'),  # (B, m, q)
    }
    if p.has_sigma_eps:
        out['sigma_eps'] = _wmean1d(p.sigma_eps, w)  # (B,)
    return out


def _weightedStd(p: Proposal) -> dict[str, torch.Tensor]:
    """Per-param posterior std for the whole batch, honouring p.weights.

    Reuses evaluation.point.pointEstimate(..., 'std') (uniform → population std, so raw
    MB / ADVI / NUTS / IMH rows are unchanged).
    """
    w = p.weights
    out = {
        'ffx': pointEstimate(p.ffx, w, 'std'),
        'sigma_rfx': pointEstimate(p.sigma_rfx, w, 'std'),
        'rfx': pointEstimate(p.rfx, w, 'std'),
    }
    if p.has_sigma_eps:
        m1 = _wmean1d(p.sigma_eps, w)
        m2 = _wmean1d(p.sigma_eps.square(), w)
        out['sigma_eps'] = (m2 - m1.square()).clamp_min(0.0).sqrt()  # (B,)
    return out


def _pooledMeans(
    mean: dict[str, torch.Tensor],
    b: int,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    group_mask: torch.Tensor,
) -> np.ndarray:
    """Posterior means for all active params of dataset b as a flat numpy array."""
    d_mask = (
        mask_d[b] if mask_d is not None else torch.ones(mean['ffx'].shape[-1], dtype=torch.bool)
    )
    q_mask = (
        mask_q[b]
        if mask_q is not None
        else torch.ones(mean['sigma_rfx'].shape[-1], dtype=torch.bool)
    )

    # rfx mean[b]: (max_m, max_q) — select active groups then active columns
    mean_rfx = mean['rfx'][b][group_mask[b]][:, q_mask].ravel()

    parts = [
        mean['ffx'][b][d_mask].numpy(),
        mean['sigma_rfx'][b][q_mask].numpy(),
        mean_rfx.numpy(),
    ]
    if 'sigma_eps' in mean:
        parts.append(mean['sigma_eps'][b].reshape(1).numpy())
    return np.concatenate(parts)


def computeCorr(
    p_method: Proposal,
    p_nuts: Proposal,
    batch: dict[str, torch.Tensor],
) -> np.ndarray:
    """Pearson r between all active posterior means (method vs NUTS) per dataset.

    Returns (B,) float array; NaN when fewer than 2 active params.
    """
    B = p_method.ffx.shape[0]
    mask_d, mask_q, group_mask = _masks(batch)
    mean_m = _weightedMean(p_method)
    mean_n = _weightedMean(p_nuts)
    r_vals = np.empty(B)
    for b in range(B):
        v_m = _pooledMeans(mean_m, b, mask_d, mask_q, group_mask)
        v_n = _pooledMeans(mean_n, b, mask_d, mask_q, group_mask)
        r_vals[b] = np.corrcoef(v_m, v_n)[0, 1] if len(v_m) >= 2 else np.nan
    return r_vals


def _stdRatios(
    std_m: dict[str, torch.Tensor],
    std_n: dict[str, torch.Tensor],
    b: int,
    mask_d: torch.Tensor | None,
    mask_q: torch.Tensor | None,
    group_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-active-entry std_method / std_nuts for dataset b as a flat tensor."""
    d_mask = (
        mask_d[b] if mask_d is not None else torch.ones(std_m['ffx'].shape[-1], dtype=torch.bool)
    )
    q_mask = (
        mask_q[b]
        if mask_q is not None
        else torch.ones(std_m['sigma_rfx'].shape[-1], dtype=torch.bool)
    )

    def _ratio(a: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
        return a / b_.clamp(min=1e-8)

    # rfx std[b]: (max_m, max_q) — select active groups then active columns
    rfx_std_m = std_m['rfx'][b][group_mask[b]][:, q_mask]
    rfx_std_n = std_n['rfx'][b][group_mask[b]][:, q_mask]

    parts: list[torch.Tensor] = [
        _ratio(std_m['ffx'][b][d_mask], std_n['ffx'][b][d_mask]),
        _ratio(std_m['sigma_rfx'][b][q_mask], std_n['sigma_rfx'][b][q_mask]),
        _ratio(rfx_std_m, rfx_std_n).reshape(-1),
    ]
    if 'sigma_eps' in std_m:
        parts.append(_ratio(std_m['sigma_eps'][b].reshape(1), std_n['sigma_eps'][b].reshape(1)))
    return torch.cat(parts)


def computeSigmaRatio(
    p_method: Proposal,
    p_nuts: Proposal,
    batch: dict[str, torch.Tensor],
) -> np.ndarray:
    """Per-dataset median(std_method / std_nuts) across all active params. Returns (B,)."""
    B = p_method.ffx.shape[0]
    mask_d, mask_q, group_mask = _masks(batch)
    std_m = _weightedStd(p_method)
    std_n = _weightedStd(p_nuts)
    ratios = np.empty(B)
    for b in range(B):
        vals = _stdRatios(std_m, std_n, b, mask_d, mask_q, group_mask)
        ratios[b] = float(vals.median()) if vals.numel() > 0 else np.nan
    return ratios


def _rankFracs(mb: np.ndarray, nuts: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted rank of each NUTS sample within the MB marginal (weighted ECDF at nuts points).

    mb, nuts: (N, S) — N active entries, S samples each. w: (N, S) MB sample weights (rows
    sum to 1) or None for uniform. The unweighted path (w=None) is exactly searchsorted / S;
    the weighted path replaces the sample count with the cumulative weight below each value.
    Returns all fracs flattened.
    """
    order = np.argsort(mb, axis=1)
    mb_sorted = np.take_along_axis(mb, order, axis=1)
    if w is None:
        S = mb_sorted.shape[1]
        return np.concatenate(
            [np.searchsorted(mb_sorted[i], nuts[i]) / S for i in range(len(mb_sorted))]
        )
    w_sorted = np.take_along_axis(w, order, axis=1)
    N = mb_sorted.shape[0]
    # cdf0[i, k] = cumulative weight of the k smallest MB samples (cdf0[:, 0] = 0)
    cdf0 = np.concatenate([np.zeros((N, 1)), np.cumsum(w_sorted, axis=1)], axis=1)
    return np.concatenate(
        [cdf0[i][np.searchsorted(mb_sorted[i], nuts[i], side='left')] for i in range(N)]
    )


def _rankMADFromFracs(fracs: np.ndarray) -> float:
    expected = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    empirical = np.array([np.percentile(fracs, 100 * q) for q in expected])
    return float(np.mean(np.abs(empirical - expected)))


def computeRankMAD(
    p_method: Proposal,
    p_nuts: Proposal,
    batch: dict[str, torch.Tensor],
) -> np.ndarray:
    """Per-dataset rank calibration error for NUTS-in-method rank fracs.

    Within each dataset, rank fractions are pooled over active parameter entries.
    Returns (B,), where 0 indicates perfect shape agreement and larger values
    indicate systematic disagreement.
    """
    mask_d, mask_q, group_mask = _masks(batch)
    B = p_method.ffx.shape[0]
    mads = np.empty(B)
    # only the method's marginal carries IS weights; NUTS draws are equal-weight targets
    w_all = p_method.weights   # (B, S) or None

    def _add_global(
        all_fracs: list[np.ndarray],
        mb_v: torch.Tensor,
        nu_v: torch.Tensor,
        b: int,
        mask: torch.Tensor | None,
        w_b: np.ndarray | None,
    ) -> None:
        mb_np, nu_np = mb_v[b].numpy(), nu_v[b].numpy()   # (S, D)
        D = mb_np.shape[-1]
        for di in range(D):
            if mask is not None and not bool(mask[b, di]):
                continue
            w = None if w_b is None else w_b[None, :]
            all_fracs.append(_rankFracs(mb_np[:, di][None, :], nu_np[:, di][None, :], w))

    for b in range(B):
        all_fracs: list[np.ndarray] = []
        w_b = w_all[b].numpy() if w_all is not None else None

        _add_global(all_fracs, p_method.ffx, p_nuts.ffx, b, mask_d, w_b)
        _add_global(all_fracs, p_method.sigma_rfx, p_nuts.sigma_rfx, b, mask_q, w_b)
        if p_method.has_sigma_eps:
            _add_global(
                all_fracs,
                p_method.sigma_eps.unsqueeze(-1),
                p_nuts.sigma_eps.unsqueeze(-1),
                b,
                None,
                w_b,
            )

        q_mask = (
            mask_q[b].numpy().astype(bool)
            if mask_q is not None
            else np.ones(p_method.rfx.shape[-1], dtype=bool)
        )
        active_groups = group_mask[b].numpy().astype(bool)
        if active_groups.any() and q_mask.any():
            mb_rfx = p_method.rfx[b][active_groups].numpy()   # (m_active, S, q)
            nu_rfx = p_nuts.rfx[b][active_groups].numpy()
            m_active = mb_rfx.shape[0]
            # same per-sample weights apply to every group of this dataset
            w_rfx = None if w_b is None else np.broadcast_to(w_b[None, :], (m_active, w_b.shape[0]))
            for k in np.flatnonzero(q_mask):
                all_fracs.append(_rankFracs(mb_rfx[:, :, k], nu_rfx[:, :, k], w_rfx))

        mads[b] = _rankMADFromFracs(np.concatenate(all_fracs)) if all_fracs else np.nan

    return mads


# ---------------------------------------------------------------------------
# Row assembly


def _ms(arr: np.ndarray | None) -> tuple[float, float] | None:
    """Median and unscaled median absolute deviation, NaNs ignored."""
    if arr is None:
        return None
    a = arr[~np.isnan(arr)]
    if len(a) == 0:
        return (float('nan'), float('nan'))
    median = float(np.median(a))
    mad = float(np.median(np.abs(a - median)))
    return (median, mad)


def _buildRow(
    label: str,
    r: np.ndarray,
    sigma_ratio: np.ndarray,
    rank_mad: np.ndarray,
    delta_nll: np.ndarray,
    delta_tpd: np.ndarray | None,
) -> dict:
    return {
        'method': label,
        'r': _ms(r),
        'sigma_ratio': _ms(sigma_ratio),
        'rank_mad': _ms(rank_mad),
        'delta_nll': _ms(delta_nll),
        'delta_tpd': _ms(delta_tpd),
    }


# ---------------------------------------------------------------------------
# Main evaluation


def evaluateReal(
    model: Approximator,
    data_path: Path,
    max_d: int,
    max_q: int,
    lf: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    rescale: bool,
    convergence_mode: str,
    ckpt_dir: Path,
    prefix: str,
    seed: int,
    methods: list[str],
    summary_chunk_size: int = 4,
    warmup: bool = True,
) -> list[dict]:
    col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
    B_total = len(col)
    batch = collateGrouped([col[i] for i in range(B_total)])

    # Precomputed analytical stats (beta_est/BLUPs from precompute.py) live in the sibling
    # {partition}.npz, not the .fit.npz; inject them so MB sampling reuses the MAP statistics
    # instead of recomputing glmm() live (matching evaluate.py / oracle_posterior.py).
    if 'stats' not in batch:
        base_path = data_path.with_name(data_path.name.replace('.fit.npz', '.npz'))
        if base_path.exists() and base_path != data_path:
            base_col = Collection(base_path, permute=False, max_d=max_d, max_q=max_q)
            if len(base_col) == B_total:
                base_batch = collateGrouped([base_col[i] for i in range(B_total)])
                if 'stats' in base_batch:
                    batch['stats'] = base_batch['stats']
                del base_batch
            del base_col
        if 'stats' not in batch:
            logger.warning(
                'No precomputed stats for %s — MB sampling recomputes glmm() live (slower). '
                'Run metabeta/analytical/precompute.py for this data_id/partition.',
                data_path.parent.name,
            )

    # Restrict to NUTS-converged datasets
    conv_mask = nutsConvergeMask(batch, mode=convergence_mode)
    if conv_mask is not None:
        n_conv = int(conv_mask.sum())
        logger.info('NUTS convergence (%s): %d / %d', convergence_mode, n_conv, B_total)
        if n_conv == 0:
            logger.warning('No converged datasets; aborting.')
            return []
        batch = subsetBatch(batch, conv_mask)
    else:
        logger.warning('No NUTS convergence diagnostics found; using all %d datasets', B_total)

    B = batch['X'].shape[0]

    # Full-test-file masks used to key the per-method summary caches:
    #   MB/NUTS cover the converged subset; ADVI additionally requires a successful fit.
    conv_full = conv_mask if conv_mask is not None else np.ones(B_total, dtype=bool)

    # ADVI subset (some fits may have failed)
    advi_mask = fitBatchMask(batch, 'advi')
    n_advi = int(advi_mask.sum())
    logger.info('ADVI available: %d / %d', n_advi, B)
    advi_batch: dict | None = subsetBatch(batch, advi_mask) if n_advi > 0 else None
    advi_full = conv_full.copy()
    advi_full[conv_full] = advi_mask

    # Inference
    proposal_mb, mb_tpd_arr = loadOrSampleMB(
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
        warmup=warmup,
    )
    proposal_nuts = fit2proposal(batch, 'nuts')
    proposal_advi = fit2proposal(advi_batch, 'advi') if advi_batch is not None else None

    # Rescale all to original data space before metric computation
    if rescale:
        proposal_mb.rescale(batch['sd_y'])
        proposal_nuts.rescale(batch['sd_y'])
        if proposal_advi is not None:
            proposal_advi.rescale(advi_batch['sd_y'])
        batch = rescaleData(batch)
        if advi_batch is not None:
            advi_batch = rescaleData(advi_batch)

    # LOO-NLL via getSummary (cached); NRMSE/corr will be NaN since real data has no ground truth
    summary_mb = loadOrComputeSummary(
        proposal_mb,
        batch,
        data_path,
        'mb',
        conv_full,
        lf,
        rescale,
        ckpt_dir=ckpt_dir,
        prefix=prefix,
        n_samples=n_samples,
        seed=seed,
        summary_chunk_size=summary_chunk_size,
    )
    summary_nuts = loadOrComputeSummary(
        proposal_nuts,
        batch,
        data_path,
        'nuts',
        conv_full,
        lf,
        rescale,
        summary_chunk_size=summary_chunk_size,
    )
    summary_advi = (
        loadOrComputeSummary(
            proposal_advi,
            advi_batch,
            data_path,
            'advi',
            advi_full,
            lf,
            rescale,
            summary_chunk_size=summary_chunk_size,
        )
        if proposal_advi is not None
        else None
    )

    # Post-hoc refinements layered on the raw MB posterior (evaluated vs the full NUTS ref).
    # (label, proposal, batch, tpd_arr, summary) tuples appended between MB and ADVI.
    refined_specs: list[tuple] = []
    for method in validMethods(methods, lf):
        logger.info('Refining MB with %s', method)
        p_ref, refine_s = loadOrRefine(
            method,
            proposal_mb,
            batch,
            data_path,
            ckpt_dir,
            prefix,
            n_samples,
            seed,
            lf,
            rescale,
            conv_mask,
            batch_size,
        )
        summary_ref = loadOrComputeSummary(
            p_ref,
            batch,
            data_path,
            method,
            conv_full,
            lf,
            rescale,
            ckpt_dir=ckpt_dir,
            prefix=prefix,
            n_samples=n_samples,
            seed=seed,
            summary_chunk_size=summary_chunk_size,
        )
        # refinement runs on top of MB, so its cost adds to the MB per-dataset time
        tpd_ref = mb_tpd_arr + refine_s / B
        refined_specs.append((f'MB+{method}', p_ref, batch, tpd_ref, summary_ref))

    nuts_tpd_arr = batch.get('nuts_duration')            # (B,) tensor or None
    advi_mask_t = torch.from_numpy(advi_mask)           # bool tensor for indexing

    rows: list[dict] = []

    method_specs = (
        [('MB', proposal_mb, batch, mb_tpd_arr, summary_mb)]
        + refined_specs
        + [
            (
                'ADVI',
                proposal_advi,
                advi_batch,
                advi_batch.get('advi_duration') if advi_batch is not None else None,
                summary_advi,
            )
        ]
    )

    for label, p_method, batch_sub, tpd_arr, summary in method_specs:
        if p_method is None or summary is None:
            continue

        is_advi = label == 'ADVI'

        # NUTS references restricted to this method's subset
        p_nuts_ref = subsetProposal(proposal_nuts, advi_mask) if is_advi else proposal_nuts
        nuts_loo = summary_nuts.per_dataset.loo_nll
        nuts_nll = nuts_loo[advi_mask_t] if is_advi else nuts_loo
        nuts_tpd = (
            nuts_tpd_arr[advi_mask_t] if (nuts_tpd_arr is not None and is_advi) else nuts_tpd_arr
        )

        delta_tpd: np.ndarray | None = None
        if tpd_arr is not None and nuts_tpd is not None:
            delta_tpd = (tpd_arr.float() - nuts_tpd.float()).numpy()

        rows.append(
            _buildRow(
                label=label,
                r=computeCorr(p_method, p_nuts_ref, batch_sub),
                sigma_ratio=computeSigmaRatio(p_method, p_nuts_ref, batch_sub),
                rank_mad=computeRankMAD(p_method, p_nuts_ref, batch_sub),
                delta_nll=(summary.per_dataset.loo_nll - nuts_nll).float().numpy(),
                delta_tpd=delta_tpd,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Table output

METRICS = ['r', 'sigma_ratio', 'rank_mad', 'delta_nll', 'delta_tpd']

HEADERS_MD = ['method', 'r ↑', 'σ-ratio → 1', 'rank-MAD ↓', 'ΔLOO-NLL ↓', 'Δtime ↓']

HEADERS_TEX = [
    r'$\mathrm{model}$',
    r'$r$',
    r'$\sigma\text{-ratio}$',
    r'$\mathrm{rank\text{-}MAD}$',
    r'$\Delta\mathrm{LOO\text{-}NLL}$',
    r'$\Delta\mathrm{time}$',
]


def _fmtMd(val: tuple[float, float] | float | None, dp: int = 2) -> str:
    if val is None:
        return 'NA'
    if isinstance(val, tuple):
        m, s = val
        return 'NA' if m != m else f'{m:.{dp}f} ± {s:.{dp}f}'
    return f'{val:.{dp}f}' if val == val else 'NA'


def _fmtTex(val: tuple[float, float] | float | None, dp: int = 2) -> str:
    if val is None:
        return r'\textrm{NA}'
    if isinstance(val, tuple):
        m, s = val
        return r'\textrm{NA}' if m != m else f'${m:.{dp}f} \\pm {s:.{dp}f}$'
    return f'${val:.{dp}f}$' if val == val else r'\textrm{NA}'


def saveTables(
    rows_by_regime: dict[str, list[dict]],
    outdir: Path,
    run_name: str,
    prefix: str,
    dp: int = 2,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # include the checkpoint prefix so latest/best runs of the same checkpoint don't clobber
    stem = f'{run_name}_{prefix}'

    fmt_md = lambda v: _fmtMd(v, dp)
    fmt_tex = lambda v: _fmtTex(v, dp)

    # --- Markdown ---
    md_rows = []
    for regime, rows in rows_by_regime.items():
        for r in rows:
            md_rows.append([regime, r['method']] + [fmt_md(r[m]) for m in METRICS])
    md_table = tabulate(
        md_rows,
        headers=['regime', 'method'] + HEADERS_MD[1:],
        tablefmt='pipe',
        stralign='right',
    )
    md_path = outdir / f'real_{stem}.md'
    md_path.write_text(f'# Real-data evaluation: {stem}\n\n{md_table}\n')
    logger.info('Saved Markdown → %s', md_path)

    # --- LaTeX ---
    header_cols = ' & '.join(HEADERS_TEX)
    lines: list[str] = [
        r'\begin{tabular}{cc|ccccc}',
        r'    \toprule',
        rf'    $\mathrm{{regime}}$ & {header_cols} \\',
        r'    \midrule',
    ]
    first = True
    for regime, rows in rows_by_regime.items():
        if not first:
            lines.append(r'    \midrule')
        first = False
        for j, row in enumerate(rows):
            regime_cell = rf'\texttt{{{regime}}}' if j == 0 else ''
            cells = ' & '.join(
                [rf'\texttt{{{row["method"]}}}'] + [fmt_tex(row[m]) for m in METRICS]
            )
            lines.append(rf'      {regime_cell} & {cells} \\')
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    tex_path = outdir / f'real_{stem}.tex'
    tex_path.write_text('\n'.join(lines))
    logger.info('Saved LaTeX → %s', tex_path)


# ---------------------------------------------------------------------------
# Main

DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)

    ckpt_dir = Path(cfg.checkpoint)
    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)

    lf = model_cfg.likelihood_family
    fam = FAM_LETTER[lf]
    dp = getattr(cfg, 'decimals', 2)

    data_ids: list[str] = getattr(cfg, 'data_ids', None) or [
        f'{s}-{fam}-real' for s in DEFAULT_SIZES
    ]

    # --methods: explicit list (possibly empty for raw MB only) overrides the family preset.
    methods = cfg.methods if cfg.methods is not None else posthocDefaults(lf)

    logger.info('Checkpoint: %s  (prefix=%s)', ckpt_dir.name, cfg.prefix)
    logger.info('max_d=%d  max_q=%d  family=%d', model_cfg.max_d, model_cfg.max_q, lf)
    logger.info('Evaluating: %s', data_ids)
    logger.info('Refinement methods: %s', methods or '(none — raw MB only)')

    rows_by_regime: dict[str, list[dict]] = {}
    for data_id in data_ids:
        data_path = DATA_DIR / data_id / 'test.fit.npz'
        if not data_path.exists():
            logger.warning('Skipping %s: test.fit.npz not found', data_id)
            continue
        regime = data_id.split('-')[0]
        logger.info('\n--- Regime: %s (%s) ---', regime, data_id)
        rows = evaluateReal(
            model=model,
            data_path=data_path,
            max_d=model_cfg.max_d,
            max_q=model_cfg.max_q,
            lf=lf,
            n_samples=cfg.n_samples,
            batch_size=cfg.batch_size,
            device=device,
            rescale=cfg.rescale,
            convergence_mode=cfg.convergence_mode,
            ckpt_dir=ckpt_dir,
            prefix=cfg.prefix,
            seed=cfg.seed,
            methods=methods,
            summary_chunk_size=cfg.summary_chunk_size,
            warmup=getattr(cfg, 'warmup', True),
        )
        if rows:
            rows_by_regime[regime] = rows

    if not rows_by_regime:
        logger.error('No regimes evaluated — check data_ids and checkpoint.')
        return

    # Console summary
    md_rows = []
    for regime, rows in rows_by_regime.items():
        for r in rows:
            md_rows.append([regime, r['method']] + [_fmtMd(r[m], dp) for m in METRICS])
    print(
        '\n' + tabulate(md_rows, headers=['regime', 'method'] + HEADERS_MD[1:], tablefmt='simple')
    )

    saveTables(rows_by_regime, Path(cfg.outdir), ckpt_dir.name, cfg.prefix, dp=dp)


if __name__ == '__main__':
    main()
