"""
Oracle evaluation: given a model checkpoint, evaluate on tiny/small/medium/large test sets.

Filters datasets that exceed the model's d/q capacity, loads NUTS/ADVI/Laplace fits from
the test.fit.npz batch, and produces a LaTeX + Markdown table with mean ± std over
parameter dimensions (for NRMSE/ECE/EACE/R) and over datasets (for LOO-NLL). Unlike
real_posterior.py, the sampled test sets carry ground-truth parameters, so the metrics are
absolute (vs the true values) rather than relative to NUTS.

MB posterior samples, per-method summaries, and post-hoc refinements are cached next to the
data (siblings of test.fit.npz), keyed by checkpoint/prefix/n_samples/seed and by the
capacity/convergence subset, mirroring experiments/evaluation/real_posterior.py and
metabeta/evaluation/evaluate.py.

Optionally layers post-hoc refinements on the raw MB flow posterior (extra ``MB+<method>``
rows). The method(s) come from ``--methods`` or, if omitted, the per-family default in
metabeta/configs/presets.yaml; pass ``--methods`` with no values for raw MB only.

Usage (from repo root):
    uv run python experiments/evaluation/oracle_posterior.py --checkpoint PATH
    uv run python experiments/evaluation/oracle_posterior.py --checkpoint PATH --n_samples 100 --batch_size 4
    uv run python experiments/evaluation/oracle_posterior.py --checkpoint PATH --data_ids small-n-sampled large-n-sampled
    uv run python experiments/evaluation/oracle_posterior.py --checkpoint PATH --methods   # raw MB only
"""

import argparse
import gc
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

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
from metabeta.evaluation.summary import getSummary
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, loadApproximator

OUT_DIR = RESULTS_DIR

DEFAULT_DATA_IDS = [
    'tiny-n-sampled',
    'small-n-sampled',
    'medium-n-sampled',
    'large-n-sampled',
]

logger = logging.getLogger(__name__)

# Post-hoc refinement methods that can be layered on the raw MB flow posterior.
# The SNIS/Laplace families set PSIS-smoothed IS weights (proposal.is_results);
# the IMH family returns an equal-weight chain. Mirrors real_posterior.py.
SNIS_METHODS = ('is', 'isFull', 'isMarginal')          # ImportanceSampler
LAPLACE_METHODS = ('isLaplace', 'rbAttach')            # LaplaceImportanceSampler (GLMM only)
IMH_METHODS = ('imhMarginal', 'imhGlobal', 'imhLaplace')   # MetropolisSampler
SUPPORTED_METHODS = SNIS_METHODS + LAPLACE_METHODS + IMH_METHODS

# IMH pool geometry: n_chains × (n_samples // n_chains) proposals, mirroring
# experiments/posthoc/ablation.py's runIMH settings.
IMH_N_CHAINS = 4
IMH_BURNIN = 25


def posthocDefaults(lf: int) -> list[str]:
    """Default refinement method for a likelihood family, from presets.yaml.

    Returns ``[default]`` (empty list if null / unset). Override with --methods.
    """
    entry = PRESETS.get('posthoc', {}).get(lf, {})
    default = entry.get('default')
    return [default] if default else []


def _validMethods(methods: list[str], lf: int) -> list[str]:
    """Drop refinement methods that are incompatible with the likelihood family (warns)."""
    valid: list[str] = []
    for m in methods:
        if m in ('isMarginal', 'imhGlobal') and lf != 0:
            logger.warning('Skipping %s: Normal-only (lf=%d)', m, lf)
        elif m in LAPLACE_METHODS + ('imhLaplace',) and lf == 0:
            logger.warning('Skipping %s: GLMM-only, Normal uses the exact marginal', m)
        else:
            valid.append(m)
    return valid


# ---------------------------------------------------------------------------
# CLI


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Oracle cross-size evaluation', argument_default=argparse.SUPPRESS
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prefix',     type=str, default='latest')
    parser.add_argument('--device',     type=str, default='cpu')
    parser.add_argument('--n_samples',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_batch_size', type=int, default=1,
                        help='Datasets per chunk for posterior predictive / LOO summaries')
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--data_ids',   type=str, nargs='+', default=DEFAULT_DATA_IDS)
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
# Model loading


def loadModel(
    ckpt_dir: Path,
    prefix: str,
    device: torch.device,
) -> tuple[Approximator, argparse.Namespace]:
    cfg_dict = loadConfigFromCheckpoint(ckpt_dir)
    cfg = argparse.Namespace(**cfg_dict)

    model = loadApproximator(cfg, device, ckpt_dir, prefix)

    return model, cfg


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
# Inference helpers


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
    corr_rfx = batch.get(f'{prefix}_corr_rfx', None)
    proposal = Proposal(proposed, has_sigma_eps=has_sigma_eps, corr_rfx=corr_rfx)
    proposal.tpd = batch[f'{prefix}_duration'].mean().item()
    return proposal


@torch.no_grad()
def sampleMB(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Proposal, torch.Tensor]:
    """Returns (proposal, tpd_arr) where tpd_arr is per-dataset time (s), shape (B,)."""
    B = batch['X'].shape[0]
    proposals: list[Proposal] = []
    tpd_list: list[float] = []
    for start in tqdm(range(0, B, batch_size), desc='  MB', leave=False):
        end = min(start + batch_size, B)
        b_chunk = sliceBatch(batch, start, end)
        b_chunk = toDevice(b_chunk, device)
        t0_chunk = time.perf_counter()
        p_chunk = model.estimate(b_chunk, n_samples=n_samples)
        tpd_chunk = (time.perf_counter() - t0_chunk) / (end - start)
        tpd_list.extend([tpd_chunk] * (end - start))
        p_chunk.to('cpu')
        proposals.append(p_chunk)
        del b_chunk
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    merged = concatProposalsBatch(proposals)
    tpd_arr = torch.tensor(tpd_list, dtype=torch.float64)
    merged.tpd = tpd_arr.mean().item()
    return merged, tpd_arr


# ---------------------------------------------------------------------------
# Caching (posterior samples, refinements, summaries) — siblings of test.fit.npz.


def _maskTag(mask: np.ndarray | None) -> str:
    """Short hash identifying which datasets (of the full test file) survived filtering."""
    if mask is None:
        return 'all'
    packed = np.packbits(mask.astype(np.uint8)).tobytes()
    return hashlib.sha1(packed).hexdigest()[:12]


def _refMtime(data_path: Path, ckpt_dir: Path | None = None, prefix: str | None = None) -> float:
    """Freshness reference: cache is stale if older than the data (and, for MB, the checkpoint)."""
    ref_mtime = data_path.stat().st_mtime if data_path.exists() else 0.0
    if ckpt_dir is not None and prefix is not None:
        ckpt_path = ckpt_dir / f'{prefix}.pt'
        if ckpt_path.exists():
            ref_mtime = max(ref_mtime, ckpt_path.stat().st_mtime)
    return ref_mtime


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
    specific data space); the raw ``mb`` cache passes ``None`` to keep its filename simple.
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
    capacity-kept subset); it is folded into the cache key since it changes the batch contents.
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
# Post-hoc refinement of the raw MB posterior (mirrors real_posterior.py)


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


def _refineChunk(
    method: str,
    base: Proposal,
    batch: dict[str, torch.Tensor],
    lf: int,
) -> Proposal:
    """Refine a single sub-batch of datasets (see refineProposal for the chunking wrapper)."""
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
        if method == 'imhLaplace' and lf == 0:
            raise ValueError('imhLaplace is for GLMMs (lf != 0); use imhMarginal')
        if method == 'imhLaplace':
            mode = 'laplace'
        elif method == 'imhGlobal':
            mode = 'global'
        else:
            mode = 'marginal' if lf == 0 else 'global'
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


def refineProposal(
    method: str,
    base: Proposal,
    batch: dict[str, torch.Tensor],
    lf: int,
    batch_size: int,
) -> Proposal:
    """Refine the (rescaled) MB proposal ``base`` in the (rescaled) ``batch`` space.

    Sub-batched over datasets in chunks of ``batch_size``: the marginal likelihood
    materialises a (b, max_m, max_n, s) tensor, so processing the whole batch at once OOMs
    on large (m, n) regimes. Each dataset's PSIS/softmax normalisation and MH chain are
    per-dataset independent, so chunking is exact.
    """
    B = base.samples_g.shape[0]
    if batch_size >= B:
        return _refineChunk(method, base, batch, lf)

    chunks: list[Proposal] = []
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        chunks.append(
            _refineChunk(method, base.slice_b(start, end), sliceBatch(batch, start, end), lf)
        )
    return concatProposalsBatch(chunks)


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
    batch_size: int,
) -> tuple[Proposal, float]:
    """Cached wrapper around refineProposal; cache is keyed by method/checkpoint/rescale.

    Returns ``(proposal, refine_seconds)`` where refine_seconds is the total wall time of the
    refinement over the batch (from the cache on a hit), used to offset the MB per-dataset
    timing for the ``time`` metric.
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
    proposal = refineProposal(method, base_proposal, batch, lf, batch_size)
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
    summary_chunk_size: int = 1,
) -> EvaluationSummary:
    """Cached wrapper around getSummary; cache lives next to the data (sibling of test.fit.npz).

    ``mask`` identifies which datasets of the full test file this summary covers (capacity
    and/or convergence subset), so distinct subsets get distinct caches. Model-derived methods
    (mb + refined) additionally key on checkpoint/prefix/n_samples/seed and invalidate on the
    checkpoint. ``summary_chunk_size`` bounds peak memory of the predictive/LOO block (does not
    affect the result, so it is not part of the cache key).
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
    summary_batch_size: int,
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
            summary_chunk_size=summary_batch_size,
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
    summary_batch_size: int = 1,
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
    for method in _validMethods(methods, lf):
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
        summary_batch_size,
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
                summary_batch_size,
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
    run_name = ckpt_dir.name

    # --methods: explicit list (possibly empty for raw MB only) overrides the family preset.
    methods = cfg.methods if cfg.methods is not None else posthocDefaults(lf)

    logger.info('Model: %s  max_d=%d  max_q=%d  likelihood=%d', run_name, max_d, max_q, lf)
    logger.info('Refinement methods: %s', methods or '(none — raw MB only)')

    rows_by_regime: dict[str, list[dict]] = {}
    rows_by_regime_conv: dict[str, list[dict]] = {}
    for data_id in cfg.data_ids:
        data_path = DATA_DIR / data_id / 'test.fit.npz'
        if not data_path.exists():
            logger.warning('Skipping %s: test.fit.npz not found', data_id)
            continue
        regime = data_id.split('-')[0]
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
            summary_batch_size=cfg.summary_batch_size,
        )
        if rows:
            rows_by_regime[regime] = rows
        if rows_conv:
            rows_by_regime_conv[regime] = rows_conv

    if not rows_by_regime:
        logger.error('No regimes evaluated — check data_ids and checkpoint.')
        return

    dp = getattr(cfg, 'decimals', 2)

    # Console summary
    md_rows = []
    for regime, rows in rows_by_regime.items():
        for r in rows:
            md_rows.append([regime, r['method']] + [_fmtMd(r[c], dp) for c in METRICS])
    print('\n' + tabulate(md_rows, headers=['regime', 'method'] + METRICS, tablefmt='simple'))

    saveTables(rows_by_regime, Path(cfg.outdir), run_name, dp=dp)
    if rows_by_regime_conv:
        saveTables(rows_by_regime_conv, Path(cfg.outdir), f'{run_name}_conv', dp=dp)


if __name__ == '__main__':
    main()
