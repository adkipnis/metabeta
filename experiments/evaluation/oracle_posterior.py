"""
Oracle evaluation: evaluate one model checkpoint on one sampled test set (``--data_id``).

Both the checkpoint and the single test set are required; pick a ``--data_id`` whose d/q fit
the checkpoint's capacity (datasets beyond max_d/max_q are dropped by the capacity filter, so
a small-capacity model on a larger regime yields few or no rows). To sweep sizes, launch one
process per (checkpoint, data_id) pair.

Loads NUTS/ADVI/Laplace fits from the test.fit.npz batch and produces a LaTeX + Markdown table
with mean ± std over parameter dimensions (for NRMSE/ECE/EACE/R) and over datasets (for
LOO-NLL). Unlike real_posterior.py, the sampled test sets carry ground-truth parameters, so
the metrics are absolute (vs the true values) rather than relative to NUTS.

MB posterior samples, per-method summaries, and post-hoc refinements are cached next to the
data (siblings of test.fit.npz), keyed by checkpoint/prefix/n_samples/seed and by the
capacity/convergence subset, mirroring experiments/evaluation/real_posterior.py and
metabeta/evaluation/evaluate.py.

Optionally layers post-hoc refinements on the raw MB flow posterior (extra ``MB+<method>``
rows). The method(s) come from ``--methods`` or, if omitted, the per-family default in
metabeta/configs/presets.yaml; pass ``--methods`` with no values for raw MB only.

Usage (from repo root):
    uv run python experiments/evaluation/oracle_posterior.py --checkpoint PATH --data_id small-n-sampled
    uv run python experiments/evaluation/oracle_posterior.py --checkpoint PATH --data_id small-n-sampled --n_samples 100 --batch_size 4
    uv run python experiments/evaluation/oracle_posterior.py --checkpoint PATH --data_id small-n-sampled --methods   # raw MB only
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

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
        description='Oracle evaluation of one checkpoint on one sampled test set',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_id',    type=str, required=True,
                        help='Single sampled data id to evaluate, e.g. small-n-sampled. Pick one '
                             'whose d/q fit the checkpoint capacity (datasets beyond it are '
                             'dropped by the capacity filter).')
    parser.add_argument('--prefix',     type=str, default='latest')
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--n_samples',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=1,
                        help='Datasets per chunk for posterior predictive / LOO summaries')
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--outdir',     type=str, default=str(OUT_DIR))
    parser.add_argument('--verbosity',  type=int, default=1)
    parser.add_argument('--decimals',         type=int, default=2,
                        help='Decimal places in table cells (default: 2)')
    parser.add_argument('--rescale',          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--convergence_mode', type=str, default='liberal',
                        choices=['liberal', 'strict'])
    parser.add_argument('--methods',          type=str, nargs='*', default=None,
                        choices=list(SUPPORTED_METHODS),
                        help='Post-hoc refinement methods to run on top of raw MB, evaluated '
                             'as extra rows. Default: the family preset in presets.yaml. '
                             'Pass an empty list to run raw MB only.')
    # fmt: on
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Batch helpers


def capacityMask(batch: dict[str, torch.Tensor], max_d: int, max_q: int) -> np.ndarray:
    d_active = batch['mask_d'].sum(-1).numpy()
    q_active = batch['mask_q'].sum(-1).numpy()
    return (d_active <= max_d) & (q_active <= max_q)


def trimBatch(batch: dict[str, torch.Tensor], max_d: int, max_q: int) -> dict[str, torch.Tensor]:
    """Slice all relevant tensors to model's max_d/max_q and recompute derived masks.

    Safe because permute=False ensures features are in natural (ascending) order,
    so slicing to max_d preserves exactly the active dimensions.
    """
    out = dict(batch)

    for key in ('X', 'ffx', 'nu_ffx', 'tau_ffx', 'mask_d'):
        if key in out:
            out[key] = out[key][..., :max_d]

    for key in ('Z', 'sigma_rfx', 'tau_rfx', 'mask_q'):
        if key in out:
            out[key] = out[key][..., :max_q]

    if 'rfx' in out:
        out['rfx'] = out['rfx'][..., :max_q]

    if 'corr_rfx' in out:
        out['corr_rfx'] = out['corr_rfx'][..., :max_q, :max_q]

    for method in ('nuts', 'advi', 'laplace'):
        if f'{method}_ffx' in out:
            out[f'{method}_ffx'] = out[f'{method}_ffx'][..., :max_d]
        if f'{method}_sigma_rfx' in out:
            out[f'{method}_sigma_rfx'] = out[f'{method}_sigma_rfx'][..., :max_q]
        if f'{method}_rfx' in out:
            out[f'{method}_rfx'] = out[f'{method}_rfx'][..., :max_q]
        if f'{method}_corr_rfx' in out:
            out[f'{method}_corr_rfx'] = out[f'{method}_corr_rfx'][..., :max_q, :max_q]

    # recompute masks that depend on mask_q
    B = out['mask_q'].shape[0]
    out['mask_mq'] = out['mask_m'].unsqueeze(-1) & out['mask_q'].unsqueeze(-2)
    q = max_q
    out['mask_corr'] = (
        torch.stack(
            [out['mask_q'][..., i] & out['mask_q'][..., j] for i in range(1, q) for j in range(i)],
            dim=-1,
        )
        if q >= 2
        else out['mask_q'].new_zeros(B, 0)
    )

    return out


def dropFitKeys(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a view-like dict excluding large cached NUTS/ADVI/Laplace fit tensors."""
    return {
        k: v
        for k, v in batch.items()
        if not (k.startswith('nuts_') or k.startswith('advi_') or k.startswith('laplace_'))
    }


def methodFitBatch(batch: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return only cached fit tensors for one method."""
    stem = f'{prefix}_'
    return {k: v for k, v in batch.items() if k.startswith(stem)}


def loadRegimeBatch(
    data_path: Path,
    max_d: int,
    max_q: int,
) -> tuple[dict[str, torch.Tensor], int, int, np.ndarray]:
    """Load test batch, filtering and padding/trimming to model capacity.

    When the test set fits within the model (d_file ≤ max_d, q_file ≤ max_q),
    loads with max_d/max_q so the model receives correctly-padded inputs.
    Otherwise loads natively, filters datasets by capacity, and trims to max_d/max_q.

    Returns (batch, n_total, n_kept, cap_mask) where cap_mask is a full-test-file boolean
    (length n_total) marking which datasets survive the capacity filter — folded into the
    posterior-sample / summary cache keys so subsets get distinct caches.
    """
    col = Collection(data_path, permute=False)
    d_file, q_file = col.d, col.q
    n_total = len(col)

    if d_file <= max_d and q_file <= max_q:
        col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
        batch = collateGrouped([col[i] for i in range(n_total)])
        return batch, n_total, n_total, np.ones(n_total, dtype=bool)

    # Some datasets exceed capacity: load natively, filter, trim
    batch = collateGrouped([col[i] for i in range(n_total)])
    cap_mask = capacityMask(batch, max_d, max_q)
    n_kept = int(cap_mask.sum())
    batch = subsetBatch(batch, cap_mask)
    batch = trimBatch(batch, max_d, max_q)
    return batch, n_total, n_kept, cap_mask


def _capFull(cap_mask: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Lift a boolean mask defined over the capacity-kept datasets to the full test file."""
    full = cap_mask.copy()
    full[cap_mask] = sub
    return full


# ---------------------------------------------------------------------------
# Metric helpers


def flattenActiveParams(
    metric_dict: dict[str, torch.Tensor],
    active_d: torch.Tensor,
    active_q: torch.Tensor,
    has_eps: bool,
) -> torch.Tensor:
    """Flatten per-parameter-dimension metrics to a 1-D tensor over active dims only.

    Handles ffx (d,), sigma_rfx (q,), rfx (q,), sigma_eps (scalar).
    Excludes corr_rfx.
    """
    parts: list[torch.Tensor] = []
    if 'ffx' in metric_dict:
        parts.append(metric_dict['ffx'][active_d].float())
    if 'sigma_rfx' in metric_dict:
        parts.append(metric_dict['sigma_rfx'][active_q].float())
    if 'rfx' in metric_dict:
        parts.append(metric_dict['rfx'][active_q].float())
    if has_eps and 'sigma_eps' in metric_dict:
        val = metric_dict['sigma_eps'].float()
        parts.append(val.reshape(1))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _ms(t: torch.Tensor) -> tuple[float, float]:
    """Mean and Bessel-corrected std, ignoring NaNs."""
    t = t[~torch.isnan(t)]
    if len(t) == 0:
        return float('nan'), float('nan')
    mean = t.mean().item()
    std = t.std(correction=1).item() if len(t) > 1 else 0.0
    return mean, std


def _medianMad(t: torch.Tensor) -> tuple[float, float]:
    """Median and MAD, ignoring NaNs."""
    t = t[~torch.isnan(t)].double()
    if len(t) == 0:
        return float('nan'), float('nan')
    med = t.median().item()
    mad = (t - med).abs().median().item()
    return med, mad


def buildRow(
    label: str,
    regime: str,
    corr_vals: torch.Tensor,
    nrmse_vals: torch.Tensor,
    ece_vals: torch.Tensor,
    eace_vals: torch.Tensor,
    loo_nll: torch.Tensor | None,
    tpd_arr: torch.Tensor | None,
) -> dict:
    row: dict = {'regime': regime, 'method': label}
    row['r'] = _ms(corr_vals)
    row['NRMSE'] = _ms(nrmse_vals)
    row['ECE'] = _ms(ece_vals)
    row['EACE'] = _ms(eace_vals)
    row['LOO-NLL'] = _medianMad(loo_nll) if loo_nll is not None else None
    row['time'] = _ms(tpd_arr.float()) if tpd_arr is not None else None
    return row


# ---------------------------------------------------------------------------
# Regime evaluation


def _evalGroup(
    quads: list[dict],
    regime: str,
    lf: int,
    rescale: bool,
    data_path: Path,
    ckpt_dir: Path,
    prefix: str,
    n_samples: int,
    seed: int,
    summary_chunk_size: int,
) -> list[dict]:
    """Evaluate a list of quad dicts (label/method/proposal/batch/tpd/mask/model_derived).

    Proposals and batches are assumed already rescaled (rescale happens once in
    evaluateRegime, before conv subsets are built). Summaries are cached per method/mask.
    """
    rows = []
    for q in quads:
        proposal = q['proposal']
        if proposal is None:
            continue
        proposal.to('cpu')
        batch = q['batch']
        model_derived = q['model_derived']
        summary = loadOrComputeSummary(
            proposal,
            batch,
            data_path,
            q['method'],
            q['mask'],
            lf,
            rescale,
            ckpt_dir=ckpt_dir if model_derived else None,
            prefix=prefix if model_derived else None,
            n_samples=n_samples if model_derived else None,
            seed=seed if model_derived else None,
            summary_chunk_size=summary_chunk_size,
        )
        ag = summary.aggregated
        active_d = batch['mask_d'].any(0)
        active_q = batch['mask_q'].any(0)
        has_eps = 'sigma_eps' in ag.nrmse
        rows.append(
            buildRow(
                q['label'],
                regime,
                corr_vals=flattenActiveParams(ag.corr, active_d, active_q, has_eps),
                nrmse_vals=flattenActiveParams(ag.nrmse, active_d, active_q, has_eps),
                ece_vals=flattenActiveParams(ag.ece, active_d, active_q, has_eps),
                eace_vals=flattenActiveParams(ag.eace, active_d, active_q, has_eps),
                loo_nll=summary.per_dataset.loo_nll,
                tpd_arr=q['tpd'],
            )
        )
    return rows


def _refProposalAndTpd(
    cap_batch: dict[str, torch.Tensor],
    data_batch: dict[str, torch.Tensor],
    prefix: str,
    mask: np.ndarray,
) -> tuple[Proposal | None, dict | None, torch.Tensor | None]:
    """Build a fit-based reference proposal + its data batch + per-dataset durations.

    ``mask`` is the fit-success mask over the capacity-kept datasets. Returns (None, None,
    None) when no dataset succeeded for this method.
    """
    if not mask.any():
        return None, None, None
    fit_batch = subsetBatch(methodFitBatch(cap_batch, prefix), mask)
    proposal = fit2proposal(fit_batch, prefix)
    data_sub = subsetBatch(data_batch, mask)
    tpd = fit_batch.get(f'{prefix}_duration')
    return proposal, data_sub, tpd


def evaluateRegime(
    model: Approximator,
    data_path: Path,
    max_d: int,
    max_q: int,
    lf: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    regime: str,
    ckpt_dir: Path,
    prefix: str,
    seed: int,
    methods: list[str],
    rescale: bool = True,
    convergence_mode: str = 'liberal',
    summary_chunk_size: int = 1,
) -> tuple[list[dict], list[dict] | None]:
    """Returns (rows_full, rows_conv) — rows_conv is None if no convergence data."""
    logger.info('\n--- Regime: %s ---', regime)

    cap_batch, n_total, n_kept, cap_mask = loadRegimeBatch(data_path, max_d, max_q)
    logger.info('  Capacity filter: %d / %d (d≤%d, q≤%d)', n_kept, n_total, max_d, max_q)
    if n_kept == 0:
        logger.warning('  No datasets pass capacity filter — skipping.')
        return [], None

    advi_mask = fitBatchMask(cap_batch, 'advi')
    laplace_mask = fitBatchMask(cap_batch, 'laplace')
    logger.info('  ADVI success: %d / %d', int(advi_mask.sum()), n_kept)
    logger.info('  Laplace success: %d / %d', int(laplace_mask.sum()), n_kept)

    conv_mask = nutsConvergeMask(cap_batch, mode=convergence_mode)

    data_batch = dropFitKeys(cap_batch)

    # MB samples over the full capacity-kept batch (cached, keyed by cap_mask).
    proposal_mb, mb_tpd_arr = loadOrSampleMB(
        model,
        data_batch,
        data_path,
        ckpt_dir,
        prefix,
        n_samples,
        batch_size,
        seed,
        device,
        cap_mask,
    )
    proposal_nuts = fit2proposal(cap_batch, 'nuts')
    nuts_tpd = cap_batch.get('nuts_duration')

    proposal_advi, advi_data_batch, advi_tpd = _refProposalAndTpd(
        cap_batch, data_batch, 'advi', advi_mask
    )
    proposal_laplace, laplace_data_batch, laplace_tpd = _refProposalAndTpd(
        cap_batch, data_batch, 'laplace', laplace_mask
    )

    # Fit tensors are no longer needed; free them before summaries / refinement.
    for key in list(cap_batch):
        if key.startswith(('nuts_', 'advi_', 'laplace_')):
            del cap_batch[key]
    gc.collect()

    # Rescale everything ONCE to original data space before metrics/refinement (rescale is
    # in-place on proposals, so this must precede conv subsetting).
    if rescale:
        proposal_mb.rescale(data_batch['sd_y'])
        proposal_nuts.rescale(data_batch['sd_y'])
        if proposal_advi is not None:
            proposal_advi.rescale(advi_data_batch['sd_y'])
        if proposal_laplace is not None:
            proposal_laplace.rescale(laplace_data_batch['sd_y'])
        data_batch = rescaleData(data_batch)
        if advi_data_batch is not None:
            advi_data_batch = rescaleData(advi_data_batch)
        if laplace_data_batch is not None:
            laplace_data_batch = rescaleData(laplace_data_batch)

    # Post-hoc refinements on the (rescaled) raw MB posterior (cached, keyed by cap_mask).
    refined: list[tuple[str, Proposal, float]] = []
    for method in validMethods(methods, lf):
        logger.info('  Refining MB with %s', method)
        p_ref, refine_s = loadOrRefine(
            method,
            proposal_mb,
            data_batch,
            data_path,
            ckpt_dir,
            prefix,
            n_samples,
            seed,
            lf,
            rescale,
            cap_mask,
            batch_size,
        )
        refined.append((method, p_ref, refine_s))

    def _refTpd(refine_s: float) -> torch.Tensor:
        return mb_tpd_arr + refine_s / n_kept

    # ---- Full group quads ----
    full_quads: list[dict] = [
        {
            'label': 'MB',
            'method': 'mb',
            'proposal': proposal_mb,
            'batch': data_batch,
            'tpd': mb_tpd_arr,
            'mask': cap_mask,
            'model_derived': True,
        }
    ]
    for method, p_ref, refine_s in refined:
        full_quads.append(
            {
                'label': f'MB+{method}',
                'method': method,
                'proposal': p_ref,
                'batch': data_batch,
                'tpd': _refTpd(refine_s),
                'mask': cap_mask,
                'model_derived': True,
            }
        )
    full_quads.append(
        {
            'label': 'NUTS',
            'method': 'nuts',
            'proposal': proposal_nuts,
            'batch': data_batch,
            'tpd': nuts_tpd,
            'mask': cap_mask,
            'model_derived': False,
        }
    )
    full_quads.append(
        {
            'label': 'ADVI',
            'method': 'advi',
            'proposal': proposal_advi,
            'batch': advi_data_batch,
            'tpd': advi_tpd,
            'mask': _capFull(cap_mask, advi_mask),
            'model_derived': False,
        }
    )
    full_quads.append(
        {
            'label': 'LA',
            'method': 'laplace',
            'proposal': proposal_laplace,
            'batch': laplace_data_batch,
            'tpd': laplace_tpd,
            'mask': _capFull(cap_mask, laplace_mask),
            'model_derived': False,
        }
    )

    rows = _evalGroup(
        full_quads,
        regime,
        lf,
        rescale,
        data_path,
        ckpt_dir,
        prefix,
        n_samples,
        seed,
        summary_chunk_size,
    )

    # ---- Converged-subset group quads ----
    rows_conv: list[dict] | None = None
    if conv_mask is not None:
        n_conv = int(conv_mask.sum())
        logger.info('  NUTS convergence (%s): %d / %d', convergence_mode, n_conv, n_kept)
        if 0 < n_conv < n_kept:
            conv_idx = torch.from_numpy(conv_mask)
            conv_batch = subsetBatch(data_batch, conv_mask)
            conv_full = _capFull(cap_mask, conv_mask)

            conv_quads: list[dict] = [
                {
                    'label': 'MB',
                    'method': 'mb',
                    'proposal': subsetProposal(proposal_mb, conv_mask),
                    'batch': conv_batch,
                    'tpd': mb_tpd_arr[conv_idx],
                    'mask': conv_full,
                    'model_derived': True,
                }
            ]
            for method, p_ref, refine_s in refined:
                conv_quads.append(
                    {
                        'label': f'MB+{method}',
                        'method': method,
                        'proposal': subsetProposal(p_ref, conv_mask),
                        'batch': conv_batch,
                        'tpd': _refTpd(refine_s)[conv_idx],
                        'mask': conv_full,
                        'model_derived': True,
                    }
                )
            conv_quads.append(
                {
                    'label': 'NUTS',
                    'method': 'nuts',
                    'proposal': subsetProposal(proposal_nuts, conv_mask),
                    'batch': conv_batch,
                    'tpd': nuts_tpd[conv_idx] if nuts_tpd is not None else None,
                    'mask': conv_full,
                    'model_derived': False,
                }
            )

            for label, method, proposal, mask, tpd in (
                ('ADVI', 'advi', proposal_advi, advi_mask, advi_tpd),
                ('LA', 'laplace', proposal_laplace, laplace_mask, laplace_tpd),
            ):
                if proposal is None:
                    continue
                sel = conv_mask[mask]                 # conv status among the fit-success subset
                if not sel.any():
                    continue
                sel_idx = torch.from_numpy(sel)
                conv_quads.append(
                    {
                        'label': label,
                        'method': method,
                        'proposal': subsetProposal(proposal, sel),
                        'batch': subsetBatch(data_batch, mask & conv_mask),
                        'tpd': tpd[sel_idx] if tpd is not None else None,
                        'mask': _capFull(cap_mask, mask & conv_mask),
                        'model_derived': False,
                    }
                )

            rows_conv = _evalGroup(
                conv_quads,
                regime,
                lf,
                rescale,
                data_path,
                ckpt_dir,
                prefix,
                n_samples,
                seed,
                summary_chunk_size,
            )

    return rows, rows_conv


# ---------------------------------------------------------------------------
# Table output

METRICS = ['r', 'NRMSE', 'ECE', 'EACE', 'LOO-NLL', 'time']


def _fmtMd(val: tuple[float, float] | float | None, dp: int = 2) -> str:
    if val is None:
        return 'NA'
    if isinstance(val, tuple):
        m, s = val
        if m != m:  # NaN check
            return 'NA'
        return f'{m:.{dp}f} ± {s:.{dp}f}'
    return f'{val:.{dp}f}'


def _fmtTex(val: tuple[float, float] | float | None, dp: int = 2) -> str:
    if val is None:
        return 'NA'
    if isinstance(val, tuple):
        m, s = val
        if m != m:  # NaN check
            return 'NA'
        return f'${m:.{dp}f} \\pm {s:.{dp}f}$'
    return f'${val:.{dp}f}$'


def saveTables(
    rows_by_regime: dict[str, list[dict]],
    outdir: Path,
    run_name: str,
    dp: int = 2,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fmt_md = lambda v: _fmtMd(v, dp)
    fmt_tex = lambda v: _fmtTex(v, dp)

    # --- Markdown ---
    md_rows = []
    for regime, rows in rows_by_regime.items():
        for r in rows:
            md_rows.append([regime, r['method']] + [fmt_md(r[c]) for c in METRICS])
    md_table = tabulate(
        md_rows,
        headers=['regime', 'method'] + METRICS,
        tablefmt='pipe',
        stralign='right',
    )
    md_path = outdir / f'oracle_{run_name}.md'
    md_path.write_text(f'# Oracle Evaluation: {run_name}\n\n{md_table}\n')
    logger.info('Saved Markdown → %s', md_path)

    # --- LaTeX ---
    header_cols = (
        r'$r$ & $\mathrm{NRMSE}$ & $\mathrm{ECE}$ & '
        r'$\mathrm{EACE}$ & $\mathrm{LOO\text{-}NLL}$ & $\mathrm{time}$'
    )
    lines: list[str] = [
        r'\begin{tabular}{cc|cccccc}',
        r'    \toprule',
        rf'    $\mathrm{{regime}}$ & $\mathrm{{model}}$ & {header_cols} \\',
    ]
    for regime, rows in rows_by_regime.items():
        lines.append(r'    \midrule')
        for j, row in enumerate(rows):
            regime_cell = rf'\texttt{{{regime}}}' if j == 0 else ''
            method_cell = rf'\texttt{{{row["method"]}}}'
            cells = ' & '.join(fmt_tex(row[c]) for c in METRICS)
            lines.append(rf'      {regime_cell} & {method_cell} & {cells} \\')
    lines += [r'    \bottomrule', r'\end{tabular}', '']

    tex_path = outdir / f'oracle_{run_name}.tex'
    tex_path.write_text('\n'.join(lines))
    logger.info('Saved LaTeX → %s', tex_path)


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)

    ckpt_dir = Path(cfg.checkpoint)
    model, model_cfg_ns = loadModel(ckpt_dir, cfg.prefix, device)
    max_d: int = model_cfg_ns.max_d
    max_q: int = model_cfg_ns.max_q
    lf: int = model_cfg_ns.likelihood_family

    data_id = cfg.data_id
    regime = data_id.split('-')[0]
    # stem includes the data id so evaluating one checkpoint on several test sets never clobbers
    stem = f'{ckpt_dir.name}_{data_id}'

    # --methods: explicit list (possibly empty for raw MB only) overrides the family preset.
    methods = cfg.methods if cfg.methods is not None else posthocDefaults(lf)

    logger.info('Model: %s  max_d=%d  max_q=%d  likelihood=%d', ckpt_dir.name, max_d, max_q, lf)
    logger.info('Evaluating: %s', data_id)
    logger.info('Refinement methods: %s', methods or '(none — raw MB only)')

    data_path = DATA_DIR / data_id / 'test.fit.npz'
    if not data_path.exists():
        logger.error('%s: test.fit.npz not found', data_id)
        return

    rows, rows_conv = evaluateRegime(
        model,
        data_path,
        max_d,
        max_q,
        lf,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        device=device,
        regime=regime,
        ckpt_dir=ckpt_dir,
        prefix=cfg.prefix,
        seed=cfg.seed,
        methods=methods,
        rescale=cfg.rescale,
        convergence_mode=cfg.convergence_mode,
        summary_chunk_size=cfg.summary_chunk_size,
    )
    if not rows:
        logger.error('No datasets evaluated — check that %s fits the checkpoint capacity.', data_id)
        return

    dp = getattr(cfg, 'decimals', 2)

    # Console summary
    md_rows = [[regime, r['method']] + [_fmtMd(r[c], dp) for c in METRICS] for r in rows]
    print('\n' + tabulate(md_rows, headers=['regime', 'method'] + METRICS, tablefmt='simple'))

    saveTables({regime: rows}, Path(cfg.outdir), stem, dp=dp)
    if rows_conv:
        saveTables({regime: rows_conv}, Path(cfg.outdir), f'{stem}_conv', dp=dp)


if __name__ == '__main__':
    main()
