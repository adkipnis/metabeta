"""Shared scaffolding for the posterior-comparison experiment scripts.

Both experiments/evaluation/oracle_posterior.py (metrics vs ground truth) and
experiments/evaluation/real_posterior.py (metrics vs NUTS) load a checkpoint, sample the
MB flow posterior, optionally layer post-hoc refinements on it, load NUTS/ADVI/Laplace fit
proposals, and summarise each. This module holds the parts that are identical between them:

  * post-hoc method registry + family compatibility filtering
  * checkpoint loading and fit → Proposal conversion
  * MB sampling and the on-disk caches (posterior samples, refinements, summaries)

The per-script pieces (CLI, metric computation, table layout) stay in the scripts, since
their metrics and outputs differ. Caches live next to the data as siblings of test.fit.npz
and are keyed by checkpoint/prefix/n_samples/seed and by a hash of the dataset subset mask.
"""

import argparse
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler
from metabeta.posthoc.metropolis import MetropolisSampler
from metabeta.utils.dataloader import sliceBatch, toDevice
from metabeta.utils.evaluation import EvaluationSummary
from metabeta.utils.results import Proposal, concatProposalsBatch
from metabeta.utils.posterior_cache import (
    loadProposalCache,
    posteriorSampleCacheName,
    saveProposalCache,
)
from metabeta.utils.templates import PRESETS, loadConfigFromCheckpoint
from metabeta.utils.experiments import loadApproximator
from metabeta.evaluation.summary import getSummary

logger = logging.getLogger(__name__)

# Likelihood-family index → dataset-id letter (Normal / Bernoulli / Poisson).
FAM_LETTER = {0: 'n', 1: 'b', 2: 'p'}

# Post-hoc refinement methods that can be layered on the raw MB flow posterior.
# The SNIS/Laplace families set PSIS-smoothed IS weights (proposal.is_results);
# the IMH family returns an equal-weight chain.
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


def validMethods(methods: list[str], lf: int) -> list[str]:
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
# Model loading


def loadModel(
    ckpt_dir: Path,
    prefix: str,
    device: torch.device,
) -> tuple[Approximator, argparse.Namespace]:
    """Load an Approximator + its training-time config namespace from a checkpoint dir."""
    cfg_dict = loadConfigFromCheckpoint(ckpt_dir)
    cfg = argparse.Namespace(**cfg_dict)
    model = loadApproximator(cfg, device, ckpt_dir, prefix)
    logger.info('Loaded %s/%s.pt', ckpt_dir.name, prefix)
    return model, cfg


# ---------------------------------------------------------------------------
# Fit → Proposal conversion + MB sampling


def fitBatchMask(batch: dict[str, torch.Tensor], prefix: str) -> np.ndarray:
    """Boolean mask (shape B) of datasets whose ``{prefix}`` fit succeeded (all True if absent)."""
    failed_key = f'{prefix}_failed'
    if failed_key not in batch:
        return np.ones(batch['X'].shape[0], dtype=bool)
    return ~batch[failed_key].cpu().numpy().astype(bool)


def fit2proposal(batch: dict[str, torch.Tensor], prefix: str) -> Proposal:
    """Wrap cached ``{prefix}_*`` fit tensors (NUTS/ADVI/Laplace) as a Proposal."""
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
    """Sample from the model; returns (proposal, tpd_arr) with per-dataset time (s), shape (B,).

    ``no_grad`` rather than ``inference_mode``: the analytical MAP fit inside the model runs
    ``loss.backward()`` under ``torch.enable_grad()``, which inference-mode forbids.
    """
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
    capacity-kept or NUTS-converged subset); it is folded into the cache key since it changes
    the batch contents.
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
    timing for the Δtime / time metric.
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
    summary_chunk_size: int = 4,
) -> EvaluationSummary:
    """Cached wrapper around getSummary; cache lives next to the data (sibling of test.fit.npz).

    ``mask`` identifies which datasets of the full test file this summary covers (capacity,
    convergence, and/or fit-success subset), so distinct subsets get distinct caches.
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
