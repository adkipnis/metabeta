"""
Oracle evaluation: evaluate one model checkpoint on one sampled test set (``--data_id``).

Both the checkpoint and the single test set are required; pick a ``--data_id`` whose d/q fit
the checkpoint's capacity (datasets beyond max_d/max_q are dropped by the capacity filter, so
a small-capacity model on a larger regime yields few or no rows). To sweep sizes, launch one
process per (checkpoint, data_id) pair.

Loads NUTS/ADVI/Laplace fits from the test.fit.npz batch (and R-INLA fits from the sibling
test.inla_matched.npz, if present) and produces a LaTeX + Markdown table
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
from metabeta.utils.results import Proposal, getMasks
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
    parser.add_argument('--warmup',           action=argparse.BooleanOptionalAction, default=True,
                        help='Untimed 1-sample MB warm-up before timed sampling (default: true)')
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


# All cached-fit prefixes; the multi-GB ``*_rfx`` sample tensors live under these. The base
# (data-only) batch and each single-method fit batch are loaded with the complementary set
# excluded, so at most one method's fit tensors are ever materialized at a time (see
# evaluateRegime). Collection.exclude_prefixes drops matching keys without decompressing them.
FIT_PREFIXES = ('nuts_', 'advi_', 'laplace_')


def fitExcludePrefixes(keep: str | None) -> tuple[str, ...]:
    """Prefixes to exclude so a Collection loads only ``keep``'s fits (all fits if keep is None)."""
    return tuple(p for p in FIT_PREFIXES if p != f'{keep}_')


# R-INLA fits live in a sibling of test.fit.npz (metabeta/simulation/inla.py): test.inla.npz
# with PC scale priors, test.inla_matched.npz with each dataset's simulator scale prior (the
# reported INLA row). Method name → file; every file stores its fits under the 'inla_' prefix.
INLA_FILES = {'inla': ('INLA', 'test.inla_matched.npz')}


def _fitAxis(a: np.ndarray, axis: int, size: int) -> np.ndarray:
    """Trim or zero-pad ``a`` along ``axis`` to ``size``."""
    if a.shape[axis] >= size:
        return np.take(a, np.arange(size), axis=axis)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (0, size - a.shape[axis])
    return np.pad(a, pad)


def loadInlaBatch(
    inla_path: Path, method: str, cap_mask: np.ndarray, max_d: int, max_q: int, max_m: int
) -> dict[str, torch.Tensor]:
    """INLA joint draws over the capacity-kept datasets, as ``{method}_*`` tensors in the
    collateFits layout, so fitBatchMask / fit2proposal treat INLA like the test.fit.npz methods.

    The file is padded to its own d/q/m maxima; draws are trimmed or zero-padded to the batch's.
    A dataset counts as failed if INLA failed, timed out, or returned non-finite draws.
    """
    with np.load(inla_path) as raw:
        f = {k: raw[k][cap_mask] for k in raw.files if k.endswith('_samples')}
        failed = raw['inla_failed'][cap_mask].astype(bool)
        wall = raw['inla_wall_s'][cap_mask]
    ffx = _fitAxis(f['inla_ffx_samples'], 1, max_d)                          # (B, d, S)
    sigma = _fitAxis(f['inla_sigma_rfx_samples'], 1, max_q)                  # (B, q, S)
    rfx = _fitAxis(_fitAxis(f.pop('inla_rfx_samples'), 1, max_q), 2, max_m)  # (B, q, m, S)
    failed |= ~(np.isfinite(ffx).all((1, 2)) & np.isfinite(sigma).all((1, 2)))
    failed |= ~np.isfinite(rfx).all((1, 2, 3))
    out = {}
    if 'inla_sigma_eps_samples' in f:
        eps = f['inla_sigma_eps_samples'][:, 0]                              # (B, S)
        failed |= ~np.isfinite(eps).all(1)
        out[f'{method}_sigma_eps'] = torch.from_numpy(np.nan_to_num(eps))
    out |= {
        f'{method}_ffx': torch.from_numpy(np.nan_to_num(ffx).transpose(0, 2, 1).copy()),
        f'{method}_sigma_rfx': torch.from_numpy(np.nan_to_num(sigma).transpose(0, 2, 1).copy()),
        f'{method}_rfx': torch.from_numpy(np.nan_to_num(rfx).transpose(0, 2, 3, 1).copy()),
        f'{method}_duration': torch.from_numpy(wall.astype(np.float32)),
        f'{method}_failed': torch.from_numpy(failed),
    }
    return out


def methodFitBatch(batch: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return only cached fit tensors for one method."""
    stem = f'{prefix}_'
    return {k: v for k, v in batch.items() if k.startswith(stem)}


def _probeShapes(data_path: Path) -> tuple[int, int, int]:
    """Read (d_file, q_file, n_total) from the npz cheaply (small top-level arrays only)."""
    with np.load(data_path, allow_pickle=True) as raw:
        d_arr = raw['d']
        q_arr = raw['q']
        return int(d_arr.max()), int(q_arr.max()), int(d_arr.shape[0])


def loadRegimeBatch(
    data_path: Path,
    max_d: int,
    max_q: int,
    exclude_prefixes: tuple[str, ...] = FIT_PREFIXES,
) -> tuple[dict[str, torch.Tensor], int, int, np.ndarray]:
    """Load a test batch, filtering/padding/trimming to model capacity.

    ``exclude_prefixes`` is forwarded to Collection; it defaults to all fit prefixes so the
    base batch stays data-only (the multi-GB fit tensors are never materialized). Pass
    ``fitExcludePrefixes('nuts')`` etc. to load exactly one method's fits.

    When the test set fits within the model (d_file ≤ max_d, q_file ≤ max_q), loads with
    max_d/max_q so the model receives correctly-padded inputs. Otherwise loads natively,
    filters datasets by capacity, and trims to max_d/max_q.

    Returns (batch, n_total, n_kept, cap_mask) where cap_mask is a full-test-file boolean
    (length n_total) marking which datasets survive the capacity filter — folded into the
    posterior-sample / summary cache keys so subsets get distinct caches.
    """
    d_file, q_file, n_total = _probeShapes(data_path)

    if d_file <= max_d and q_file <= max_q:
        col = Collection(
            data_path, permute=False, max_d=max_d, max_q=max_q, exclude_prefixes=exclude_prefixes
        )
        batch = collateGrouped([col[i] for i in range(n_total)])
        return batch, n_total, n_total, np.ones(n_total, dtype=bool)

    # Some datasets exceed capacity: load natively, filter, trim
    col = Collection(data_path, permute=False, exclude_prefixes=exclude_prefixes)
    batch = collateGrouped([col[i] for i in range(n_total)])
    cap_mask = capacityMask(batch, max_d, max_q)
    n_kept = int(cap_mask.sum())
    batch = subsetBatch(batch, cap_mask)
    batch = trimBatch(batch, max_d, max_q)
    return batch, n_total, n_kept, cap_mask


def nutsConvergeMaskFromNpz(
    data_path: Path,
    cap_mask: np.ndarray,
    mode: str,
) -> np.ndarray | None:
    """NUTS convergence mask over the capacity-kept datasets, read from the small nuts_* diagnostics.

    Loads only the tiny diagnostic arrays (divergences/rhat/ess/…) — never the multi-GB
    nuts_rfx samples — so it can run before any fit proposal is materialized.
    """
    diag_keys = (
        'nuts_divergences',
        'nuts_draws',
        'nuts_rhat',
        'nuts_ess',
        'nuts_ess_tail',
        'nuts_max_treedepth',
    )
    with np.load(data_path, allow_pickle=True) as raw:
        diag = {k: torch.as_tensor(raw[k]) for k in diag_keys if k in raw.files}
    if 'nuts_divergences' not in diag:
        return None
    idx = torch.from_numpy(cap_mask)
    diag = {
        k: (v[idx] if torch.is_tensor(v) and v.shape[:1] == (cap_mask.shape[0],) else v)
        for k, v in diag.items()
    }
    # nuts_draws is stored per-dataset (uniform); nutsConvergeMask wants a scalar draw count.
    if 'nuts_draws' in diag:
        diag['nuts_draws'] = torch.as_tensor(int(diag['nuts_draws'].reshape(-1)[0].item()))
    return nutsConvergeMask(diag, mode=mode)


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


def _medianMad(t: torch.Tensor) -> tuple[float, float]:
    """Median and MAD, ignoring NaNs."""
    t = t[~torch.isnan(t)].double()
    if len(t) == 0:
        return float('nan'), float('nan')
    med = t.median().item()
    mad = (t - med).abs().median().item()
    return med, mad


def _meanStd(t: torch.Tensor) -> tuple[float, float]:
    """Mean and Bessel-corrected std, ignoring NaNs."""
    t = t[~torch.isnan(t)].double()
    if len(t) == 0:
        return float('nan'), float('nan')
    std = t.std(correction=1).item() if len(t) > 1 else 0.0
    return t.mean().item(), std


# Statistic name → (center, spread) function.
STATS = {'mean ± std': _meanStd, 'median ± MAD': _medianMad}
# Primary statistic per column (the paper's tables). r/NRMSE/ECE/EACE are per-parameter
# aggregates over the test set, so their spread runs over parameter dimensions (fixed effects,
# rfx scales, rfx, sigma_eps): the mean is the "average over all parameters" and the median
# would pick a typical fixed effect and hide the variance components. LOO-NLL and time are
# per-dataset values with heavy right tails: median ± MAD, matching the runtime tables.
PRIMARY_STAT = {
    'r': 'mean ± std',
    'NRMSE': 'mean ± std',
    'ECE': 'mean ± std',
    'EACE': 'mean ± std',
    'LOO-NLL': 'median ± MAD',
    'time': 'median ± MAD',
}


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
    """One table row: ``row[metric]`` holds the column's primary statistic (PRIMARY_STAT),
    ``row['stats'][name][metric]`` every statistic in STATS, so the writer can also emit one
    table per statistic."""
    values = {
        'r': corr_vals,
        'NRMSE': nrmse_vals,
        'ECE': ece_vals,
        'EACE': eace_vals,
        'LOO-NLL': loo_nll,
        'time': tpd_arr.float() if tpd_arr is not None else None,
    }
    row: dict = {'regime': regime, 'method': label, 'stats': {}}
    for name, fn in STATS.items():
        row['stats'][name] = {k: (fn(v) if v is not None else None) for k, v in values.items()}
    row.update({k: row['stats'][PRIMARY_STAT[k]][k] for k in values})
    return row


# ---------------------------------------------------------------------------
# Regime evaluation


def _summaryRow(
    label: str,
    method: str,
    proposal: Proposal,
    batch: dict[str, torch.Tensor],
    mask: np.ndarray,
    tpd: torch.Tensor | None,
    model_derived: bool,
    regime: str,
    lf: int,
    rescale: bool,
    data_path: Path,
    ckpt_dir: Path,
    prefix: str,
    n_samples: int,
    seed: int,
    summary_chunk_size: int,
) -> dict:
    """Summarize one (proposal, batch) — cached per method/mask — and build its metric row.

    ``proposal``/``batch`` are assumed already rescaled. Fit methods (model_derived=False)
    cache without the checkpoint in the key; mb/refined key on checkpoint/n_samples/seed.
    """
    proposal.to('cpu')
    summary = loadOrComputeSummary(
        proposal,
        batch,
        data_path,
        method,
        mask,
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
    row = buildRow(
        label,
        regime,
        corr_vals=flattenActiveParams(ag.corr, active_d, active_q, has_eps),
        nrmse_vals=flattenActiveParams(ag.nrmse, active_d, active_q, has_eps),
        ece_vals=flattenActiveParams(ag.ece, active_d, active_q, has_eps),
        eace_vals=flattenActiveParams(ag.eace, active_d, active_q, has_eps),
        loo_nll=summary.per_dataset.loo_nll,
        tpd_arr=tpd,
    )
    # per-class (NRMSE, r) over active dims (App. B.3 layout; sigma = rfx scales + sigma_eps)
    # and per-dataset errors, for downstream tables
    sigmas = lambda m: {k: m[k] for k in ('sigma_rfx', 'sigma_eps') if k in m}
    row['by_class'] = {
        'beta': (ag.nrmse['ffx'][active_d], ag.corr['ffx'][active_d]),
        'sigma': tuple(
            flattenActiveParams(sigmas(m), active_d, active_q, has_eps) for m in (ag.nrmse, ag.corr)
        ),
        'alpha': (ag.nrmse['rfx'][active_q], ag.corr['rfx'][active_q]),
    }
    row['per_dataset'] = perDatasetErrors(ag.estimates, batch) | {
        'loo_nll': summary.per_dataset.loo_nll,
        'time': tpd if tpd is not None else torch.full((len(mask.nonzero()[0]),), float('nan')),
    }
    row['mask'] = mask
    return row


def perDatasetErrors(
    est: dict[str, torch.Tensor], data: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """RMSE of the posterior mean per dataset and parameter class, over active dimensions."""
    masks = getMasks(data)
    out = {}
    classes = {'beta': 'ffx', 'sigma': 'sigma_rfx', 'alpha': 'rfx', 'eps': 'sigma_eps'}
    for name, key in classes.items():
        if key not in est:
            continue
        se = (est[key] - data[key]).square()
        mask = masks[key]
        if mask is None:                                     # sigma_eps: (B,)
            out[f'rmse_{name}'] = se.sqrt()
            continue
        dims = tuple(range(1, se.dim()))
        out[f'rmse_{name}'] = ((se * mask).sum(dims) / mask.sum(dims).clamp_min(1)).sqrt()
    return out


def evaluateRegime(
    model: Approximator,
    data_path: Path,
    base_path: Path,
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
    warmup: bool = True,
) -> dict[str, list[dict]]:
    """Returns rows per dataset subset: '' (all), 'conv' (NUTS-converged, absent if that is
    all or none) and 'lowmq' (converged, lowest m/q quartile; absent without NUTS diagnostics).

    ``base_path`` is the base ``{partition}.npz`` (data + precomputed analytical ``stats``);
    ``data_path`` is the ``{partition}.fit.npz`` (NUTS/ADVI/Laplace fits + diagnostics).
    Loading the base data from ``base_path`` populates ``data['stats']`` so MB sampling reuses
    the precomputed MAP statistics instead of recomputing glmm() live (matching evaluate.py and
    how the model was trained). Fits/diagnostics/caches use ``data_path``.

    Memory is bounded by streaming: the base batch carries no fit tensors, and each reference
    method (NUTS/ADVI/Laplace/INLA) is loaded, summarized, and freed one at a time, so at most one
    method's multi-GB fit samples are resident at once (plus MB + refinements).
    """
    logger.info('\n--- Regime: %s ---', regime)

    # Base data batch from {partition}.npz — carries precomputed analytical stats (beta_est,
    # BLUPs), so collateGrouped populates data['stats'] and the model skips the live MAP fit.
    data_batch, n_total, n_kept, cap_mask = loadRegimeBatch(base_path, max_d, max_q)
    if 'stats' not in data_batch:
        logger.warning(
            '  No precomputed stats in %s — MB sampling will recompute glmm() live (slow). '
            'Run metabeta/analytical/precompute.py for this data_id/partition.',
            base_path.name,
        )
    logger.info('  Capacity filter: %d / %d (d≤%d, q≤%d)', n_kept, n_total, max_d, max_q)
    if n_kept == 0:
        logger.warning('  No datasets pass capacity filter — skipping.')
        return {'': []}

    # NUTS convergence from the small diagnostic arrays (no fit samples materialized).
    conv_mask = nutsConvergeMaskFromNpz(data_path, cap_mask, convergence_mode)

    # MB samples over the capacity-kept batch (cached, keyed by cap_mask).
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
        warmup=warmup,
    )

    # Rescale MB + data ONCE, before subsetting (rescale is in-place on the proposal).
    if rescale:
        proposal_mb.rescale(data_batch['sd_y'])
        data_batch = rescaleData(data_batch)

    # Named subsets of the capacity-kept datasets, one row group each: all ('') and the
    # NUTS-converged ones (Table 1), plus the converged datasets in the lowest quartile of
    # groups per random effect (m/q), where Laplace-type approximations are expected to be
    # weakest. The conv group is skipped when it equals the full set.
    subsets = {'': np.ones(n_kept, dtype=bool)}
    if conv_mask is not None:
        n_conv = int(conv_mask.sum())
        logger.info('  NUTS convergence (%s): %d / %d', convergence_mode, n_conv, n_kept)
        if 0 < n_conv < n_kept:
            subsets['conv'] = conv_mask
        mq = (data_batch['m'].float() / data_batch['mask_q'].sum(-1)).numpy()
        subsets['lowmq'] = conv_mask & (mq <= np.quantile(mq, 0.25))

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
            device=device,
        )
        refined.append((method, p_ref, refine_s))

    rows: dict[str, list[dict]] = {name: [] for name in subsets}

    def _rows(label, method, proposal, batch, success, tpd, model_derived, cache_path):
        """One row per subset for a method whose fits succeeded on ``success`` (n_kept,)."""
        for name, sub in subsets.items():
            sel = sub[success]                           # subset membership within success
            if not sel.any():
                continue
            whole = bool(sel.all())
            rows[name].append(
                _summaryRow(
                    label,
                    method,
                    proposal if whole else subsetProposal(proposal, sel),
                    batch if whole else subsetBatch(batch, sel),
                    _capFull(cap_mask, _capFull(success, sel)),
                    tpd if whole or tpd is None else tpd[torch.from_numpy(sel)],
                    model_derived,
                    regime,
                    lf,
                    rescale,
                    cache_path,
                    ckpt_dir,
                    prefix,
                    n_samples,
                    seed,
                    summary_chunk_size,
                )
            )

    # ---- MB + refined (model-derived; small, reused for all subsets) ----
    everything = np.ones(n_kept, dtype=bool)
    mb_specs = [('MB', 'mb', proposal_mb, mb_tpd_arr)]
    mb_specs += [(f'MB+{m}', m, p, mb_tpd_arr + s / n_kept) for (m, p, s) in refined]
    for label, method, proposal, tpd in mb_specs:
        _rows(label, method, proposal, data_batch, everything, tpd, True, data_path)

    del refined
    gc.collect()

    # ---- Reference methods, STREAMED one at a time (only one fit-tensor set resident) ----
    # NUTS/ADVI/LA come from test.fit.npz, INLA from its sibling file (INLA_FILES).
    references = [('NUTS', 'nuts'), ('ADVI', 'advi'), ('LA', 'laplace')]
    references += [(label, method) for method, (label, _) in INLA_FILES.items()]
    for label, method in references:
        if method in INLA_FILES:
            cache_path = data_path.parent / INLA_FILES[method][1]
            if not cache_path.exists():
                logger.info('  %s: no %s — skipping.', label, cache_path.name)
                continue
            fit_batch = loadInlaBatch(
                cache_path,
                method,
                cap_mask,
                data_batch['mask_d'].shape[1],
                data_batch['mask_q'].shape[1],
                data_batch['mask_m'].shape[1],
            )
        else:
            cache_path = data_path
            fit_batch, _, _, _ = loadRegimeBatch(
                data_path, max_d, max_q, exclude_prefixes=fitExcludePrefixes(method)
            )
            if f'{method}_ffx' not in fit_batch:        # method absent from this test file
                logger.info('  %s: no fits in file — skipping.', label)
                del fit_batch
                gc.collect()
                continue
        success = fitBatchMask(fit_batch, method)        # (n_kept,)
        logger.info('  %s success: %d / %d', label, int(success.sum()), n_kept)
        if not success.any():
            del fit_batch
            gc.collect()
            continue

        method_batch = subsetBatch(methodFitBatch(fit_batch, method), success)
        del fit_batch
        proposal = fit2proposal(method_batch, method)
        tpd = method_batch.get(f'{method}_duration')     # (n_success,)
        data_sub = subsetBatch(data_batch, success)      # already rescaled
        if rescale:
            proposal.rescale(data_sub['sd_y'])
        _rows(label, method, proposal, data_sub, success, tpd, False, cache_path)
        del proposal, method_batch, data_sub
        gc.collect()

    return rows


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

    def cell(row: dict, metric: str, stat: str):
        if stat == 'primary' or 'stats' not in row:
            return row[metric]
        return row['stats'][stat][metric]

    primary_label = 'primary: ' + ', '.join(f'{k} {v}' for k, v in PRIMARY_STAT.items())
    tables = {'primary': primary_label, **{k: f'{k} (all columns)' for k in STATS}}

    # --- Markdown: the per-column primary table first, then one table per statistic ---
    md_parts = [f'# Oracle Evaluation: {run_name}']
    for stat, label in tables.items():
        md_rows = []
        for regime, rows in rows_by_regime.items():
            for r in rows:
                md_rows.append([regime, r['method']] + [fmt_md(cell(r, c, stat)) for c in METRICS])
        md_table = tabulate(
            md_rows,
            headers=['regime', 'method'] + METRICS,
            tablefmt='pipe',
            stralign='right',
        )
        md_parts.append(f'## {label}\n\n{md_table}')
    md_path = outdir / f'oracle_{run_name}.md'
    md_path.write_text('\n\n'.join(md_parts) + '\n')
    logger.info('Saved Markdown → %s', md_path)

    # --- LaTeX: the per-column primary table under the plain name (what the paper inputs),
    # the single-statistic tables as separate files so an \input never pulls in two tabulars ---
    header_cols = (
        r'$r$ & $\mathrm{NRMSE}$ & $\mathrm{ECE}$ & '
        r'$\mathrm{EACE}$ & $\mathrm{LOO\text{-}NLL}$ & $\mathrm{time}$'
    )
    for stat, label in tables.items():
        lines: list[str] = [
            rf'% entries: {label}; r/NRMSE/ECE/EACE spread over parameter dimensions, LOO-NLL/time over datasets',
            r'\begin{tabular}{cc|cccccc}',
            r'    \toprule',
            rf'    $\mathrm{{regime}}$ & $\mathrm{{model}}$ & {header_cols} \\',
        ]
        for regime, rows in rows_by_regime.items():
            lines.append(r'    \midrule')
            for j, row in enumerate(rows):
                regime_cell = rf'\texttt{{{regime}}}' if j == 0 else ''
                method_cell = rf'\texttt{{{row["method"]}}}'
                cells = ' & '.join(fmt_tex(cell(row, c, stat)) for c in METRICS)
                lines.append(rf'      {regime_cell} & {method_cell} & {cells} \\')
        lines += [r'    \bottomrule', r'\end{tabular}', '']
        suffix = '' if stat == 'primary' else '_' + stat.split(' ')[0] + stat.split(' ')[-1]
        tex_path = outdir / f'oracle_{run_name}{suffix}.tex'
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

    # Base data (+ precomputed analytical stats from precompute.py) lives in test.npz; the
    # fits live in test.fit.npz. Fall back to the fit file if the base is absent (no stats →
    # live glmm, slower).
    base_path = DATA_DIR / data_id / 'test.npz'
    if not base_path.exists():
        logger.warning(
            '%s: test.npz not found — using test.fit.npz for base data (no stats)', data_id
        )
        base_path = data_path

    rows_by_subset = evaluateRegime(
        model,
        data_path,
        base_path,
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
        warmup=getattr(cfg, 'warmup', True),
    )
    rows = rows_by_subset['']
    if not rows:
        logger.error('No datasets evaluated — check that %s fits the checkpoint capacity.', data_id)
        return

    dp = getattr(cfg, 'decimals', 2)

    # Console summary
    md_rows = [[regime, r['method']] + [_fmtMd(r[c], dp) for c in METRICS] for r in rows]
    print('\n' + tabulate(md_rows, headers=['regime', 'method'] + METRICS, tablefmt='simple'))

    for name, sub_rows in rows_by_subset.items():
        if sub_rows:
            run_name = f'{stem}_{name}' if name else stem
            saveTables({regime: sub_rows}, Path(cfg.outdir), run_name, dp=dp)


if __name__ == '__main__':
    main()
