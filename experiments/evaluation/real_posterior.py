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
metabeta/configs/presets.yaml (isMarginal for Normal); pass ``--methods`` with no
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
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.evaluation.point import pointEstimate
from metabeta.evaluation.summary import getSummary
from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler
from metabeta.posthoc.metropolis import MetropolisSampler
from metabeta.utils.dataloader import Collection, collateGrouped, sliceBatch, subsetBatch, toDevice
from metabeta.utils.evaluation import EvaluationSummary, nutsConvergeMask, subsetProposal
from metabeta.utils.results import Proposal, concatProposalsBatch
from metabeta.utils.device import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.posterior_cache import (
    loadProposalCache,
    posteriorSampleCacheName,
    saveProposalCache,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.utils.templates import PRESETS, loadConfigFromCheckpoint
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, loadApproximator

OUT_DIR = RESULTS_DIR

logger = logging.getLogger(__name__)

_FAM_LETTER = {0: 'n', 1: 'b', 2: 'p'}

# Post-hoc refinement methods that can be layered on the raw MB flow posterior.
# The SNIS/Laplace families set PSIS-smoothed IS weights (proposal.is_results);
# the IMH family returns an equal-weight chain. The comparison metrics are
# weight-aware (uniform when proposal.weights is None), so no resampling is needed.
SNIS_METHODS = ('is', 'isFull', 'isMarginal')          # ImportanceSampler
LAPLACE_METHODS = ('isLaplace', 'rbAttach')            # LaplaceImportanceSampler (GLMM only)
IMH_METHODS = ('imhMarginal', 'imhGlobal')             # MetropolisSampler
SUPPORTED_METHODS = SNIS_METHODS + LAPLACE_METHODS + IMH_METHODS

# IMH pool geometry: n_chains × (n_samples // n_chains) proposals, mirroring
# experiments/posthoc/ablation.py's runIMH settings.
IMH_N_CHAINS = 4
IMH_BURNIN = 25


def posthocDefaults(lf: int) -> list[str]:
    """Default refinement method(s) for a likelihood family, from presets.yaml.

    Returns the ``default`` method (empty list if null / unset). Alternatives are
    documented in presets but not run automatically; pass them via --methods.
    """
    entry = PRESETS.get('posthoc', {}).get(lf, {})
    default = entry.get('default')
    return [default] if default else []


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
                        help='Data IDs to evaluate (default: tiny/small/medium/large-{fam}-real)')
    parser.add_argument('--decimals',         type=int, default=2,
                        help='Decimal places in table cells (default: 2)')
    parser.add_argument('--rescale',          action=argparse.BooleanOptionalAction, default=True)
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
# Model loading


def loadModel(
    ckpt_dir: Path,
    prefix: str,
    device: torch.device,
) -> tuple[Approximator, argparse.Namespace]:
    cfg_dict = loadConfigFromCheckpoint(ckpt_dir)
    cfg = argparse.Namespace(**cfg_dict)

    model = loadApproximator(cfg, device, ckpt_dir, prefix)
    logger.info('Loaded %s/%s.pt', ckpt_dir.name, prefix)
    return model, cfg


# ---------------------------------------------------------------------------
# Batch / proposal helpers shared with oracle_posterior.py.


def fitBatchMask(batch: dict[str, torch.Tensor], prefix: str) -> np.ndarray:
    failed_key = f'{prefix}_failed'
    if failed_key not in batch:
        return np.ones(batch['X'].shape[0], dtype=bool)
    return ~batch[failed_key].cpu().numpy().astype(bool)


def fit2proposal(batch: dict[str, torch.Tensor], prefix: str) -> Proposal:
    samples_g = [batch[f'{prefix}_ffx'], batch[f'{prefix}_sigma_rfx']]
    has_sigma_eps = False
    if f'{prefix}_sigma_eps' in batch:
        samples_g.append(batch[f'{prefix}_sigma_eps'].unsqueeze(-1))
        has_sigma_eps = True
    proposed = {
        'global': {'samples': torch.cat(samples_g, dim=-1)},
        'local': {'samples': batch[f'{prefix}_rfx']},
    }
    proposal = Proposal(
        proposed,
        has_sigma_eps=has_sigma_eps,
        corr_rfx=batch.get(f'{prefix}_corr_rfx'),
    )
    proposal.tpd = batch[f'{prefix}_duration'].mean().item()
    return proposal


@torch.inference_mode()
def sampleMB(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Proposal, torch.Tensor]:
    """Sample from model; returns (proposal, tpd_arr) with tpd_arr shape (B,)."""
    B = batch['X'].shape[0]
    proposals: list[Proposal] = []
    tpd_list: list[float] = []
    for start in tqdm(range(0, B, batch_size), desc='  MB', leave=False):
        end = min(start + batch_size, B)
        b_chunk = sliceBatch(batch, start, end)
        b_chunk = toDevice(b_chunk, device)
        t0 = time.perf_counter()
        p_chunk = model.estimate(b_chunk, n_samples=n_samples)
        tpd_list.extend([(time.perf_counter() - t0) / (end - start)] * (end - start))
        p_chunk.to('cpu')
        proposals.append(p_chunk)
    merged = concatProposalsBatch(proposals)
    tpd_arr = torch.tensor(tpd_list, dtype=torch.float64)
    merged.tpd = tpd_arr.mean().item()
    return merged, tpd_arr


def _maskTag(mask: np.ndarray | None) -> str:
    """Short hash identifying which datasets (of the full test file) survived filtering."""
    if mask is None:
        return 'all'
    packed = np.packbits(mask.astype(np.uint8)).tobytes()
    return hashlib.sha1(packed).hexdigest()[:12]


def _sampleCachePath(
    data_path: Path,
    method: str,
    ckpt_dir: Path,
    prefix: str,
    n_samples: int,
    seed: int,
    mask: np.ndarray | None,
    rescale: bool | None = None,
) -> Path:
    """Cache path for a model-derived posterior-sample set (mb or a refined method).

    ``rescale`` is folded into the key for refined methods (whose weights live in a
    specific data space); the raw ``mb`` cache passes ``None`` to keep its historical
    filename unchanged.
    """
    method_tag = method if rescale is None else f'{method}-rs{int(rescale)}'
    cache_name = posteriorSampleCacheName(
        partition=f'test-{_maskTag(mask)}',
        method=method_tag,
        checkpoint_name=ckpt_dir.name,
        checkpoint_prefix=prefix,
        n_samples=n_samples,
        seed=seed,
    )
    return data_path.parent / cache_name


def _refMtime(data_path: Path, ckpt_dir: Path | None = None, prefix: str | None = None) -> float:
    """Freshness reference: cache is stale if older than the data (and, for MB, the checkpoint)."""
    ref_mtime = data_path.stat().st_mtime if data_path.exists() else 0.0
    if ckpt_dir is not None and prefix is not None:
        ckpt_path = ckpt_dir / f'{prefix}.pt'
        if ckpt_path.exists():
            ref_mtime = max(ref_mtime, ckpt_path.stat().st_mtime)
    return ref_mtime


def loadOrSampleMB(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    data_path: Path,
    ckpt_dir: Path,
    prefix: str,
    n_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    mask: np.ndarray | None,
) -> tuple[Proposal, torch.Tensor]:
    """Cached wrapper around sampleMB; cache lives next to the data as test.fit.npz's sibling.

    ``mask`` identifies which datasets of the full test file are in ``batch`` (e.g. the
    NUTS-converged subset); it is folded into the cache key since it changes the batch contents.
    Alignment with the freshly built NUTS/ADVI proposals relies on ``Collection`` yielding
    datasets in a stable natural order (no sortish/shuffle) and on the mtime freshness check.
    """
    cache_path = _sampleCachePath(data_path, 'mb', ckpt_dir, prefix, n_samples, seed, mask)
    ref_mtime = _refMtime(data_path, ckpt_dir, prefix)
    if cache_path.exists() and cache_path.stat().st_mtime >= ref_mtime:
        try:
            proposal, metadata = loadProposalCache(cache_path)
            tpd_arr = torch.as_tensor(metadata['tpd_arr'], dtype=torch.float64)
            logger.info('Loaded cached MB posterior samples from %s', cache_path)
            return proposal, tpd_arr
        except (KeyError, ValueError) as exc:
            logger.warning('Ignoring invalid MB sample cache %s: %s', cache_path, exc)
    else:
        logger.info('No usable MB sample cache at %s; sampling.', cache_path)

    proposal, tpd_arr = sampleMB(model, batch, n_samples, batch_size, device)
    saveProposalCache(
        cache_path,
        proposal,
        metadata={
            'tpd_arr': tpd_arr.numpy(),
            'n_samples': n_samples,
            'seed': seed,
            'checkpoint_prefix': prefix,
            'checkpoint_name': ckpt_dir.name,
        },
    )
    logger.info('Saved MB posterior samples to %s', cache_path)
    return proposal, tpd_arr


# ---------------------------------------------------------------------------
# Post-hoc refinement of the raw MB posterior


def _sliceSamples(p: Proposal, s: int) -> Proposal:
    """Copy of ``p`` restricted to its first ``s`` draws (flow draws are i.i.d.).

    Used to hand IMH a pool whose sample count is exactly n_chains × n_steps.
    """
    global_data = {
        'samples': p.samples_g[:, :s].contiguous(),
        'log_prob': p.log_prob_g[:, :s].contiguous(),
    }
    local_data = {
        'samples': p.samples_l[:, :, :s].contiguous(),
        'log_prob': p.log_prob_l[:, :, :s].contiguous(),
    }
    corr = p._corr_rfx[:, :s].contiguous() if p._corr_rfx is not None else None
    out = Proposal(
        {'global': global_data, 'local': local_data},
        has_sigma_eps=p.has_sigma_eps,
        d_corr=p.d_corr,
        corr_rfx=corr,
    )
    out.reff = p.reff
    return out


def refineProposal(
    method: str,
    base: Proposal,
    batch: dict[str, torch.Tensor],
    lf: int,
) -> Proposal:
    """Refine the (rescaled) MB proposal ``base`` in the (rescaled) ``batch`` space.

    The SNIS/Laplace families return PSIS-smoothed IS weights on proposal.is_results;
    the IMH family returns an equal-weight chain. Both are handled by the weight-aware
    comparison metrics (computeCorr / computeSigmaRatio / computeRankMAD default to
    uniform weights, so raw MB / ADVI / NUTS / IMH rows are unchanged). Mirrors the
    sampler wiring in experiments/posthoc/ablation.py.
    """
    B = base.samples_g.shape[0]

    if method in SNIS_METHODS:
        if method == 'isMarginal' and lf != 0:
            raise ValueError('isMarginal requires the Normal likelihood (lf=0)')
        sampler = ImportanceSampler(
            batch,
            full=(method == 'isFull'),
            marginal=(method == 'isMarginal'),
            rb_redraw=(method == 'isMarginal'),
            corr_prior=True,
            pareto=True,
            likelihood_family=lf,
        )
        return sampler(base.slice_b(0, B))

    if method in LAPLACE_METHODS:
        if lf == 0:
            raise ValueError('isLaplace / rbAttach are for GLMMs (lf != 0); use isMarginal')
        sampler = LaplaceImportanceSampler(
            batch,
            attach_only=(method == 'rbAttach'),
            corr_prior=True,
            pareto=True,
            likelihood_family=lf,
        )
        return sampler(base.slice_b(0, B))

    if method in IMH_METHODS:
        if method == 'imhGlobal' and lf != 0:
            raise ValueError('imhGlobal is Normal-only; non-Normal imhMarginal already uses global')
        mode = 'global' if method == 'imhGlobal' else ('marginal' if lf == 0 else 'global')
        n_steps = base.samples_g.shape[1] // IMH_N_CHAINS
        if n_steps <= IMH_BURNIN:
            raise ValueError(
                f'IMH needs n_samples > {IMH_N_CHAINS * IMH_BURNIN} '
                f'(got {base.samples_g.shape[1]})'
            )
        pool = _sliceSamples(base, IMH_N_CHAINS * n_steps)
        sampler = MetropolisSampler(
            batch,
            n_chains=IMH_N_CHAINS,
            n_steps=n_steps,
            burnin=IMH_BURNIN,
            mode=mode,
            likelihood_family=lf,
        )
        p_out, _ = sampler(pool)
        return p_out

    raise ValueError(f'unknown refinement method: {method}')


def loadOrRefine(
    method: str,
    base_proposal: Proposal,
    batch: dict[str, torch.Tensor],
    data_path: Path,
    ckpt_dir: Path,
    prefix: str,
    n_samples: int,
    seed: int,
    lf: int,
    rescale: bool,
    mask: np.ndarray | None,
) -> tuple[Proposal, float]:
    """Cached wrapper around refineProposal; cache is keyed by method/checkpoint/rescale.

    Returns ``(proposal, refine_seconds)`` where refine_seconds is the total wall time
    of the refinement over the batch (0 on cache hit's stored value), used to offset
    the MB per-dataset timing for the Δtime metric.
    """
    cache_path = _sampleCachePath(
        data_path, method, ckpt_dir, prefix, n_samples, seed, mask, rescale
    )
    ref_mtime = _refMtime(data_path, ckpt_dir, prefix)
    if cache_path.exists() and cache_path.stat().st_mtime >= ref_mtime:
        try:
            proposal, metadata = loadProposalCache(cache_path)
            logger.info('Loaded cached %s posterior samples from %s', method, cache_path)
            return proposal, float(metadata.get('refine_seconds', 0.0))
        except (KeyError, ValueError) as exc:
            logger.warning('Ignoring invalid %s sample cache %s: %s', method, cache_path, exc)
    else:
        logger.info('No usable %s sample cache at %s; refining.', method, cache_path)

    t0 = time.perf_counter()
    proposal = refineProposal(method, base_proposal, batch, lf)
    refine_seconds = time.perf_counter() - t0
    saveProposalCache(
        cache_path,
        proposal,
        metadata={
            'refine_seconds': refine_seconds,
            'n_samples': n_samples,
            'seed': seed,
        },
    )
    logger.info('Saved %s posterior samples to %s', method, cache_path)
    return proposal, refine_seconds


def _summaryCachePath(
    data_path: Path,
    method: str,
    mask: np.ndarray | None,
    lf: int,
    rescale: bool,
    ckpt_dir: Path | None = None,
    prefix: str | None = None,
    n_samples: int | None = None,
    seed: int | None = None,
) -> Path:
    tag = _maskTag(mask)
    # model-derived methods (mb + refined) additionally key on checkpoint/prefix/n_samples/seed
    if ckpt_dir is not None:
        name = (
            f'summary_test_{method}_{ckpt_dir.name}_{prefix}'
            f'_s{n_samples}_seed{seed}_lf{lf}_rs{int(rescale)}_{tag}.pt'
        )
    else:
        name = f'summary_test_{method}_lf{lf}_rs{int(rescale)}_{tag}.pt'
    return data_path.parent / name


def loadOrComputeSummary(
    proposal: Proposal,
    batch: dict[str, torch.Tensor],
    data_path: Path,
    method: str,
    mask: np.ndarray | None,
    lf: int,
    rescale: bool,
    ckpt_dir: Path | None = None,
    prefix: str | None = None,
    n_samples: int | None = None,
    seed: int | None = None,
    summary_chunk_size: int = 4,
) -> EvaluationSummary:
    """Cached wrapper around getSummary; cache lives next to the data (sibling of test.fit.npz).

    ``mask`` identifies which datasets of the full test file this summary covers, so the
    NUTS-converged (and, for ADVI, additionally fit-succeeded) subset is folded into the key.
    Model-derived methods (mb + refined) additionally key on checkpoint/prefix/n_samples/seed
    and invalidate on the checkpoint. ``summary_chunk_size`` bounds peak memory of the
    predictive/LOO block (does not affect the result, so it is not part of the cache key).
    """
    is_model_derived = ckpt_dir is not None
    cache_path = _summaryCachePath(
        data_path, method, mask, lf, rescale, ckpt_dir, prefix, n_samples, seed
    )
    ref_mtime = _refMtime(
        data_path,
        ckpt_dir if is_model_derived else None,
        prefix if is_model_derived else None,
    )
    if cache_path.exists() and cache_path.stat().st_mtime >= ref_mtime:
        try:
            summary = EvaluationSummary.load(cache_path)
            logger.info('Loaded cached %s summary from %s', method, cache_path)
            return summary
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.warning('Ignoring invalid %s summary cache %s: %s', method, cache_path, exc)
    else:
        logger.info('No usable %s summary cache at %s; computing.', method, cache_path)

    summary = getSummary(
        proposal,
        batch,
        likelihood_family=lf,
        compute_pred_coverage=False,
        dataset_chunk_size=summary_chunk_size,
    )
    summary.save(cache_path)
    logger.info('Saved %s summary to %s', method, cache_path)
    return summary


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


def _validMethods(methods: list[str], lf: int) -> list[str]:
    """Drop refinement methods that are incompatible with the likelihood family (warns)."""
    valid: list[str] = []
    for m in methods:
        if m in ('isMarginal', 'imhGlobal') and lf != 0:
            logger.warning('Skipping %s: Normal-only (lf=%d)', m, lf)
        elif m in LAPLACE_METHODS and lf == 0:
            logger.warning('Skipping %s: GLMM-only, Normal uses the exact marginal', m)
        else:
            valid.append(m)
    return valid


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
) -> list[dict]:
    col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
    B_total = len(col)
    batch = collateGrouped([col[i] for i in range(B_total)])

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
        model, batch, data_path, ckpt_dir, prefix, n_samples, batch_size, seed, device, conv_mask
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
    for method in _validMethods(methods, lf):
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
    dp: int = 2,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

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
    md_path = outdir / f'real_{run_name}.md'
    md_path.write_text(f'# Real-data evaluation: {run_name}\n\n{md_table}\n')
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
    tex_path = outdir / f'real_{run_name}.tex'
    tex_path.write_text('\n'.join(lines))
    logger.info('Saved LaTeX → %s', tex_path)


# ---------------------------------------------------------------------------
# Main

DEFAULT_SIZES = ['tiny', 'small', 'medium', 'large']


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)

    ckpt_dir = Path(cfg.checkpoint)
    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)

    lf = model_cfg.likelihood_family
    fam = _FAM_LETTER[lf]
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

    saveTables(rows_by_regime, Path(cfg.outdir), ckpt_dir.name, dp=dp)


if __name__ == '__main__':
    main()
