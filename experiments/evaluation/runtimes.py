"""Runtime comparison: metabeta vs NUTS, ADVI and Laplace, per size regime.

Every size uses its regime-matched checkpoint (BEST_SEEDS from scripts/build_ckpt.py) on its
own test set, the same arrangement as the misspecification studies.  The cross-product of the
old script — every model against every test set — is no longer meaningful: the size presets
now define *disjoint* d bands (small 1-4, medium 5-8, large 9-12, huge 13-16), so a smaller
model cannot represent a single dataset of a larger regime, and the cross terms are empty.

Three things the median speedup alone does not say, and that the tables below report:

  1. **The tail.** NUTS wall time is heavy-tailed (worst datasets run 10-25x its median);
     the raw flow's is essentially flat, because its cost tracks the architecture, not the
     data.  Reported as median / p95 / mean over the slowest 5% / max.
  2. **The tail is where NUTS also fails.** The slowest 5% of NUTS runs have a far lower
     convergence rate than the bulk, so the reference spends its largest wall-clock budget
     exactly where it returns an unusable posterior.  ``t/converged`` (total wall time divided
     by the number of converged datasets) prices a *usable* posterior rather than a run.
  3. **Laplace is the fast classical baseline, not NUTS.** Omitting it flatters the speedup.
     The defensible claim is Laplace-class latency at NUTS-class calibration, which the oracle
     and agreement tables support; runtime alone does not.

metabeta appears as two rows, both timed per dataset with a batch of one (the latency
comparable to the per-dataset wall times the fit backends record):

  * **MB^0** — the raw flow: one amortized forward pass and ``n_samples`` draws.
  * **MB** — the default pipeline: the same flow pass followed by the family's default IMH
    refinement (presets.yaml ``posthoc``; imhMarginal for Normal, imhLaplace for the GLMMs),
    exactly as the oracle/real benchmarks run it (rescaled space, 4 chains, burn-in 25).  The
    refinement consumes the flow draws, so both rows come from one pass over each dataset:
    MB^0 is the flow region, MB the flow region plus the refinement region.  The speedups of
    the reliability table are against MB — the posterior a user actually gets by default.

Not timed here, and deliberately so: batching several datasets through one forward pass, and
recomputing the analytical MAP/EB statistics that condition the summarizer instead of reading
the precomputed ones the batch carries.  Both are one-line appendix statements — the fit is
data-only, so its cost is checkpoint-independent, and the batched cost of the default pipeline
is what the oracle benchmark's time column reports.

Timings are cached next to test.fit.npz, keyed by checkpoint/prefix/samples/seed/device, and
invalidated when the data or the checkpoint is newer.  A cached timing is only as good as the
machine state that produced it: pass ``--refresh_cache`` for a clean campaign.

ADVI rows exclude datasets whose fit failed (``advi_failed``, up to 50/512 for bernoulli);
their stored durations time a run that produced nothing.

Usage (from repo root):
    uv run python experiments/evaluation/runtimes.py --family n --device cuda
    uv run python experiments/evaluation/runtimes.py --family p --sizes small medium
    uv run python experiments/evaluation/runtimes.py --family b --ds_type real --no_plot
    uv run python experiments/evaluation/runtimes.py --family n --max_datasets 8   # smoke test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.models.approximator import Approximator
from metabeta.plotting.runtimes import plotRuntimeRecords
from metabeta.utils.dataloader import Collection, collateGrouped, toDevice
from metabeta.utils.device import setDevice, synchronizeDevice
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, REPO_ROOT
from metabeta.utils.logger import setupLogging
from metabeta.utils.posterior_eval import (
    IMH_METHODS,
    loadModel,
    posthocDefaults,
    refineProposal,
    validMethods,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.utils.warmfit import nParams

# Reuse the checkpoint-seed mapping maintained for published joint checkpoints.
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR
FAMILY_NAMES = {'n': 'normal', 'b': 'bernoulli', 'p': 'poisson'}
DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']

# metabeta first, then the reference methods from cheap to expensive.
MB_FLOW = 'MB0'  # raw flow posterior (the paper's MB^0)
MB_DEFAULT = 'MB'  # flow + default IMH refinement (the paper's MB)
MB_METHODS = [MB_FLOW, MB_DEFAULT]
FIT_METHODS = ['LAPLACE', 'ADVI', 'NUTS']
METHOD_ORDER = MB_METHODS + FIT_METHODS
METHOD_LABELS = {MB_FLOW: 'MB^0', MB_DEFAULT: 'MB', 'LAPLACE': 'Laplace'}
TEX_LABELS = {MB_FLOW: r'\MBz{}', MB_DEFAULT: r'\texttt{MB}', 'LAPLACE': r'\texttt{Laplace}'}

# fraction of the slowest runs summarised separately; 5% of 512 datasets is 26 datasets,
# enough for a stable mean and small enough to still be a tail
TAIL_FRAC = 0.05

# diagnostics nutsConvergeMask reads; kept out of the model batch to avoid decompressing the
# multi-GB posterior sample arrays that share the nuts_ prefix
_DIAG_KEYS = (
    'nuts_divergences',
    'nuts_draws',
    'nuts_rhat',
    'nuts_ess',
    'nuts_ess_tail',
    'nuts_max_treedepth',
)


# ---------------------------------------------------------------------------
# Reference runtimes (already stored per dataset in test.fit.npz)


def loadReferences(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Per-dataset (durations, success masks, NUTS-converged mask) from a fit file.

    Only the small diagnostic arrays are read; the posterior samples stay on disk.
    """
    durations: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    diag: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=True) as raw:
        n = int(np.asarray(raw['d']).reshape(-1).shape[0])
        for method in FIT_METHODS:
            prefix = method.lower()
            key = f'{prefix}_duration'
            if key not in raw.files:
                continue
            durations[method] = np.asarray(raw[key], dtype=np.float64).reshape(-1)
            failed_key = f'{prefix}_failed'
            masks[method] = (
                ~np.asarray(raw[failed_key]).reshape(-1).astype(bool)
                if failed_key in raw.files
                else np.ones(n, dtype=bool)
            )
        for key in _DIAG_KEYS:
            if key not in raw.files:
                continue
            value = np.asarray(raw[key])
            # nuts_draws is stored per dataset but nutsConvergeMask wants the scalar chain
            # length; the campaign uses one setting throughout, so collapse it here
            diag[key] = torch.as_tensor(value.reshape(-1)[0] if key == 'nuts_draws' else value)

    conv = nutsConvergeMask(diag, mode='strict')
    conv = np.ones(n, dtype=bool) if conv is None else conv.astype(bool)
    return durations, masks, conv


def nutsDrawCount(family: str, sizes: list[str], ds_type: str) -> int:
    """Total NUTS draws per dataset, read from the first available fit file.

    metabeta's cost scales with the number of draws, so a runtime comparison is only fair at
    matched sample counts; deriving the default from the reference keeps the two in step
    without a magic constant that silently drifts from the fit campaign.
    """
    for size in sizes:
        path = DATA_DIR / f'{size}-{family}-{ds_type}' / 'test.fit.npz'
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as raw:
            if 'nuts_ffx' not in raw.files:
                continue
            return int(np.asarray(raw['nuts_ffx']).shape[-1])
    raise FileNotFoundError(
        f'no fit file with NUTS draws found for family {family!r}; pass --n_samples explicitly'
    )


# ---------------------------------------------------------------------------
# Model inputs


def modelCollection(fit_path: Path, max_d: int, max_q: int) -> Collection:
    """Collection for the timed model batches, preferring whichever file carries the stats.

    ``Approximator.summarize`` uses precomputed analytical statistics when the batch has them
    and otherwise calls ``_dataStatistics`` — a full MAP+EB fit — inline.  Whether a fit file
    happens to carry those arrays therefore decides what the timed region contains, and the
    campaign is inconsistent about it (normal has them in no ``test.fit.npz``, bernoulli lacks
    them only at ``small``, poisson only at ``huge``).  Timing against a file without them
    measures the analytical fit rather than amortized inference, and makes cells incomparable.

    ``test.npz`` carries the stats for every generated dir and is row-aligned with its fit file,
    so prefer it and fall back only when it cannot be used — loudly, since the resulting numbers
    mean something different.  Mirrors the fallback in likelihood_misspec.collectCondition.
    """
    base_path = fit_path.with_name('test.npz')
    fit_col = Collection(
        fit_path,
        permute=False,
        max_d=max_d,
        max_q=max_q,
        exclude_prefixes=('nuts_', 'advi_', 'laplace_'),
    )
    if fit_col.has_stats:
        return fit_col
    if not base_path.exists():
        logger.warning(
            '%s: no precomputed stats and no sibling test.npz — timings will include the '
            'analytical MAP fit and are NOT comparable to cells that have stats',
            fit_path.parent.name,
        )
        return fit_col

    base_col = Collection(base_path, permute=False, max_d=max_d, max_q=max_q)
    aligned = len(base_col) == len(fit_col) and all(
        np.array_equal(base_col.raw[key], fit_col.raw[key]) for key in ('d', 'q', 'm', 'n')
    )
    if not (base_col.has_stats and aligned):
        logger.warning(
            '%s: sibling test.npz has no stats or is not row-aligned — timings will include '
            'the analytical MAP fit and are NOT comparable to cells that have stats',
            fit_path.parent.name,
        )
        return fit_col
    logger.info(
        '%s: taking model inputs from test.npz (fit file has no stats)', fit_path.parent.name
    )
    return base_col


# ---------------------------------------------------------------------------
# metabeta timing


def resetRng(model: Approximator, seed: int) -> None:
    """Reset base-distribution RNGs so repeated runs draw identical samples."""
    posteriors = [model.posterior_g]
    if hasattr(model, 'posterior_l'):
        posteriors.append(model.posterior_l)
    for posterior in posteriors:
        base = posterior.base_dist
        if hasattr(base, 'base') and hasattr(base.base, 'rng'):
            base.base.rng = np.random.default_rng(seed)  # type: ignore[union-attr]


def pipelineOnce(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    method: str | None,
    lf: int,
    device: torch.device,
    rescale: bool,
) -> tuple[float, float]:
    """Run the flow and (optionally) its refinement on one device-resident batch of one.

    Returns ``(flow_seconds, refine_seconds)``, each bracketed by a device synchronisation.
    The refinement region is the whole post-flow pipeline of the oracle benchmark: rescaling
    the proposal and the data, the IMH chains on ``device`` and the copy of the refined draws
    back to the host (``refineProposal`` returns on the CPU).
    """
    synchronizeDevice(device)
    t0 = time.perf_counter()
    proposal = model.estimate(batch, n_samples=n_samples)
    synchronizeDevice(device)
    t_flow = time.perf_counter() - t0
    if method is None:
        return t_flow, float('nan')

    t1 = time.perf_counter()
    if rescale:
        proposal.rescale(batch['sd_y'])
        batch = rescaleData(batch)
    refined = refineProposal(method, proposal, batch, lf, batch['X'].shape[0], device=device)
    synchronizeDevice(device)
    t_refine = time.perf_counter() - t1
    del proposal, refined
    return t_flow, t_refine


def warmup(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    method: str | None,
    lf: int,
    device: torch.device,
    seed: int,
    rescale: bool,
) -> None:
    """Untimed pass through the full pipeline, absorbing one-time init before any timing.

    Mirrors ``posterior_eval._warmupModel`` and ``evaluate.py._warmupMbBatch`` for the flow,
    and additionally runs the refinement once so its first Cholesky/einsum kernels and lazy
    allocations do not land in the first timed dataset.  Uses the full ``n_samples`` because
    IMH needs a pool larger than chains x burn-in.  The CPU/CUDA RNG state is restored
    afterwards so the warm-up cannot shift the draws that follow it.
    """
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        resetRng(model, seed)
        pipelineOnce(model, batch, n_samples, method, lf, device, rescale)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@torch.no_grad()
def timeLatency(
    model: Approximator,
    col: Collection,
    idxs: list[int],
    n_samples: int,
    method: str | None,
    lf: int,
    device: torch.device,
    seed: int,
    rescale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dataset (flow, refinement) wall times with a batch of one.

    ``no_grad`` rather than ``inference_mode``: the analytical MAP fit inside the model runs
    ``loss.backward()`` under ``torch.enable_grad()``, which inference mode forbids.
    """
    flow = np.zeros(len(idxs))
    refine = np.full(len(idxs), np.nan)

    warm_batch = toDevice(collateGrouped([col[idxs[0]]]), device)
    warmup(model, warm_batch, n_samples, method, lf, device, seed, rescale)
    del warm_batch

    label = f'MB^0 + {method}' if method else 'MB^0'
    for i, idx in enumerate(tqdm(idxs, desc=f'  {label}', leave=False)):
        batch = toDevice(collateGrouped([col[idx]]), device)
        setSeed(seed)
        resetRng(model, seed)
        flow[i], refine[i] = pipelineOnce(model, batch, n_samples, method, lf, device, rescale)
        del batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return flow, refine


# ---------------------------------------------------------------------------
# Timing cache — a sibling of test.fit.npz, like the posterior-sample caches


def cachePath(
    data_path: Path,
    ckpt_dir: Path,
    prefix: str,
    n_samples: int,
    seed: int,
    device: torch.device,
) -> Path:
    return data_path.parent / (
        f'runtimes.{ckpt_dir.name}_{prefix}_s{n_samples}_seed{seed}_{device.type}.json'
    )


def loadCache(path: Path, data_path: Path, ckpt_dir: Path, prefix: str) -> dict[str, float]:
    """Load cached timings, dropping them when data or checkpoint is newer."""
    if not path.exists():
        return {}
    ref_mtime = data_path.stat().st_mtime if data_path.exists() else 0.0
    ckpt_file = ckpt_dir / f'{prefix}.pt'
    if ckpt_file.exists():
        ref_mtime = max(ref_mtime, ckpt_file.stat().st_mtime)
    if path.stat().st_mtime < ref_mtime:
        logger.info('Timing cache %s is older than its data/checkpoint — recomputing', path)
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {str(key): float(value) for key, value in raw.get('durations', {}).items()}


def saveCache(path: Path, cache: dict[str, float]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps({'durations': dict(sorted(cache.items()))}, indent=2) + '\n')
    tmp.replace(path)


def cachedLatencies(
    cache: dict[str, float],
    tags: tuple[str, str | None],
    idxs: list[int],
    compute,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dataset (flow, refinement) timings for ``idxs``, computing only the uncached ones.

    ``tags`` are the cache tags of the two regions (the refinement tag is None when no
    refinement runs).  A dataset is recomputed when either of its regions is missing, since
    the two come from one pass and the refinement consumes the flow draws.
    """
    flow_tag, refine_tag = tags
    needed = [flow_tag] + ([refine_tag] if refine_tag else [])
    flow = np.full(len(idxs), np.nan)
    refine = np.full(len(idxs), np.nan)
    missing = []
    for i, idx in enumerate(idxs):
        if all(f'{tag}:{idx}' in cache for tag in needed):
            flow[i] = cache[f'{flow_tag}:{idx}']
            if refine_tag:
                refine[i] = cache[f'{refine_tag}:{idx}']
        else:
            missing.append(i)
    if missing:
        flow_new, refine_new = compute([idxs[i] for i in missing])
        for i, f_val, r_val in zip(missing, flow_new, refine_new):
            flow[i] = f_val
            cache[f'{flow_tag}:{idxs[i]}'] = float(f_val)
            if refine_tag:
                refine[i] = r_val
                cache[f'{refine_tag}:{idxs[i]}'] = float(r_val)
    elif len(idxs):
        logger.info('%s: all %d timings cached', ' + '.join(needed), len(idxs))
    return flow, refine


# ---------------------------------------------------------------------------
# Per-cell collection


def refinementMethod(cfg: argparse.Namespace, lf: int, data_id: str) -> str | None:
    """The IMH method timed for the MB row: ``--method`` or the family default (presets.yaml)."""
    requested = [cfg.method] if cfg.method else posthocDefaults(lf)
    valid = validMethods(requested, lf)
    if not valid:
        logger.warning(
            '%s: no valid refinement method for lf=%d (requested %s) — MB row skipped',
            data_id,
            lf,
            requested or '(none)',
        )
        return None
    return valid[0]


def collectCell(
    cfg: argparse.Namespace,
    family: str,
    size: str,
    device: torch.device,
) -> list[dict] | None:
    """Per-dataset runtime records for one (family, size) regime, or None when unavailable."""
    data_id = f'{size}-{family}-{cfg.ds_type}'
    seed = BEST_SEEDS.get((FAMILY_NAMES[family], size))
    if seed is None:
        logger.warning(
            '%s: no BEST_SEEDS checkpoint for (%s, %s) — skipping', data_id, family, size
        )
        return None
    ckpt_dir = _ckpt_dir(FAMILY_NAMES[family], size, seed)
    data_path = DATA_DIR / data_id / 'test.fit.npz'
    if not data_path.exists() or not ckpt_dir.exists():
        logger.warning('%s: data or checkpoint missing — skipping', data_id)
        return None

    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)
    try:
        col = modelCollection(data_path, model_cfg.max_d, model_cfg.max_q)
    except ValueError as exc:
        # regimes are matched by construction; a mismatch means the checkpoint map is wrong
        logger.warning('%s: checkpoint does not cover this regime (%s) — skipping', data_id, exc)
        return None
    lf = int(model_cfg.likelihood_family)
    method = refinementMethod(cfg, lf, data_id)

    B = len(col)
    idxs = list(range(min(B, cfg.max_datasets))) if cfg.max_datasets else list(range(B))
    durations, masks, conv = loadReferences(data_path)
    logger.info(
        '%s: %d datasets (%d timed), %d NUTS-converged, d<=%d q<=%d, refinement %s',
        data_id,
        B,
        len(idxs),
        int(conv[idxs].sum()),
        model_cfg.max_d,
        model_cfg.max_q,
        method or '(none)',
    )
    if not col.has_stats:
        logger.warning(
            '%s: no precomputed stats — the MB^0 region includes the analytical MAP fit',
            data_id,
        )

    cache_path = cachePath(data_path, ckpt_dir, cfg.prefix, cfg.n_samples, cfg.seed, device)
    # a cached timing is only as good as the machine state that produced it, and nothing in the
    # key records that state — a contended node bakes its numbers in until asked to retime
    cache = {} if cfg.refresh_cache else loadCache(cache_path, data_path, ckpt_dir, cfg.prefix)

    rs = 'rs1' if cfg.rescale else 'rs0'
    tags = ('flow', f'refine_{method}_{rs}' if method else None)
    flow, refine = cachedLatencies(
        cache,
        tags,
        idxs,
        lambda missing: timeLatency(
            model, col, missing, cfg.n_samples, method, lf, device, cfg.seed, cfg.rescale
        ),
    )
    saveCache(cache_path, cache)

    timings: dict[str, np.ndarray] = {MB_FLOW: flow}
    if method is not None:
        timings[MB_DEFAULT] = flow + refine

    records = []
    for i, idx in enumerate(idxs):
        ds = {key: int(col.raw[key][idx]) for key in ('d', 'q', 'm', 'n')}
        base = {
            'family': family,
            'size': size,
            'source': data_id,
            'config': f'{size}-{family}-mixed',
            'idx': idx,
            **ds,
            'n_params': nParams(ds['d'], ds['q'], ds['m']),
            'nuts_converged': bool(conv[idx]),
            # the settings that make a wall time what it is: without them a records file
            # cannot be told apart from one measured on other hardware or with another head
            'device': device.type,
            'ds_type': cfg.ds_type,
            'prefix': cfg.prefix,
            'n_samples': cfg.n_samples,
            'refine_method': method,
            'rescale': bool(cfg.rescale),
            # False means the MB^0 region also ran the analytical MAP fit
            'precomputed_stats': bool(col.has_stats),
        }
        per_method = [(m, values[i]) for m, values in timings.items()]
        for fit_method in FIT_METHODS:
            if fit_method not in durations:
                continue
            if not masks[fit_method][idx]:
                continue  # failed fit: its duration times a run that produced nothing
            per_method.append((fit_method, durations[fit_method][idx]))
        for method_name, duration in per_method:
            records.append({**base, 'method': method_name, 'duration': float(duration)})
    return records


# ---------------------------------------------------------------------------
# Aggregation


def tailStats(durations: np.ndarray, frac: float = TAIL_FRAC) -> dict[str, float]:
    """Location and tail summaries of a runtime distribution."""
    finite = durations[np.isfinite(durations)]
    if not finite.size:
        return {key: float('nan') for key in ('median', 'p95', 'tail', 'max', 'total', 'mean')}
    n_tail = max(1, int(round(frac * finite.size)))
    tail = np.sort(finite)[-n_tail:]
    return {
        'median': float(np.median(finite)),
        'mean': float(finite.mean()),
        'p95': float(np.percentile(finite, 100 * (1 - frac))),
        'tail': float(tail.mean()),
        'max': float(finite.max()),
        'total': float(finite.sum()),
    }


def cellRows(records: list[dict]) -> list[dict]:
    """One row per (size, method) with the runtime distribution and its tail."""
    rows = []
    sizes = [s for s in DEFAULT_SIZES if any(r['size'] == s for r in records)]
    for size in sizes:
        sized = [r for r in records if r['size'] == size]
        for j, method in enumerate(
            [m for m in METHOD_ORDER if any(r['method'] == m for r in sized)]
        ):
            durations = np.array([r['duration'] for r in sized if r['method'] == method])
            rows.append(
                {
                    'size': size,
                    'method': method,
                    'first': j == 0,
                    'n': len(durations),
                    **tailStats(durations),
                }
            )
    return rows


def reliabilityRows(records: list[dict]) -> list[dict]:
    """Per size: how NUTS' wall-clock cost concentrates where it also fails to converge.

    ``t/conv`` is total NUTS wall time divided by the number of converged datasets — the cost
    of a *usable* posterior rather than of a run.  metabeta has no analogue because it does not
    fail, so its own median doubles as its cost per usable posterior.

    Speedups are against the default pipeline (MB: flow + IMH) when it was timed, since that is
    the posterior a user gets by default; pricing NUTS against the raw flow alone would flatter
    the comparison.  Falls back to MB^0 only when no refinement ran.
    """
    rows = []
    for size in [s for s in DEFAULT_SIZES if any(r['size'] == s for r in records)]:
        nuts = [r for r in records if r['size'] == size and r['method'] == 'NUTS']
        if not nuts:
            continue
        durations = np.array([r['duration'] for r in nuts])
        conv = np.array([r['nuts_converged'] for r in nuts])
        n_tail = max(1, int(round(TAIL_FRAC * len(durations))))
        slowest = np.argsort(durations)[-n_tail:]
        sized = [r for r in records if r['size'] == size]
        baseline = MB_DEFAULT if any(r['method'] == MB_DEFAULT for r in sized) else MB_FLOW
        mb = np.array([r['duration'] for r in sized if r['method'] == baseline])
        stats = tailStats(durations)
        rows.append(
            {
                'size': size,
                'n': len(durations),
                'baseline': baseline,
                'pct_conv': 100.0 * conv.mean(),
                'pct_conv_tail': 100.0 * conv[slowest].mean(),
                't_per_conv': stats['total'] / max(int(conv.sum()), 1),
                'speedup_median': stats['median'] / np.median(mb) if mb.size else float('nan'),
                'speedup_tail': stats['tail'] / np.median(mb) if mb.size else float('nan'),
                'speedup_conv': (
                    stats['total'] / max(int(conv.sum()), 1) / np.median(mb)
                    if mb.size
                    else float('nan')
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Tables


DIST_COLS = [
    ('median', 'median [s]'),
    ('p95', 'p95 [s]'),
    ('tail', 'worst 5% [s]'),
    ('max', 'max [s]'),
    ('total', 'total [s]'),
]


def _fmt(value: float, dp: int = 3) -> str:
    if value is None or value != value:
        return 'NA'
    if value >= 1000:
        return f'{value:.0f}'
    return f'{value:.{dp}f}'


def _mdLabel(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _texLabel(method: str) -> str:
    return TEX_LABELS.get(method, rf'\texttt{{{method}}}')


def renderDistMd(rows: list[dict], dp: int = 3) -> str:
    headers = ['size', 'method', 'n'] + [h for _, h in DIST_COLS]
    md = []
    for r in rows:
        # n is per row, not per size: ADVI drops the datasets whose fit failed
        size = r['size'] if r['first'] else ''
        md.append([size, _mdLabel(r['method']), r['n']] + [_fmt(r[k], dp) for k, _ in DIST_COLS])
    return tabulate(md, headers=headers, tablefmt='pipe', stralign='right')


def renderDistTex(rows: list[dict], dp: int = 3) -> str:
    header = (
        r'\mathrm{size} & \mathrm{method} & $n$ & \mathrm{median} & p_{95} & '
        r'\mathrm{worst\,5\%} & \mathrm{max} & \mathrm{total}'
    )
    lines = [r'\begin{tabular}{llr|ccccc}', r'    \toprule', f'    {header} \\\\', r'    \midrule']
    for i, r in enumerate(rows):
        if r['first'] and i != 0:
            lines.append(r'    \midrule')
        size = rf"\texttt{{{r['size']}}}" if r['first'] else ''
        cells = ' & '.join(f'${_fmt(r[k], dp)}$' for k, _ in DIST_COLS)
        lines.append(rf"    {size} & {_texLabel(r['method'])} & {r['n']} & {cells} \\")
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


RELIABILITY_COLS = [
    ('baseline', 'MB row'),
    ('pct_conv', '% conv'),
    ('pct_conv_tail', '% conv in slowest 5%'),
    ('t_per_conv', 't/conv [s]'),
    ('speedup_median', 'MB speedup (median)'),
    ('speedup_tail', 'MB speedup (worst 5%)'),
    ('speedup_conv', 'MB speedup (per conv)'),
]


def renderReliabilityMd(rows: list[dict]) -> str:
    headers = ['size', 'n'] + [h for _, h in RELIABILITY_COLS]
    md = []
    for r in rows:
        cells = []
        for key, _ in RELIABILITY_COLS:
            value = r[key]
            if key == 'baseline':
                cells.append(_mdLabel(value))
            elif key.startswith('pct'):
                cells.append(f'{value:.0f}')
            elif key.startswith('speedup'):
                cells.append(f'{value:.0f}x' if value == value else 'NA')
            else:
                cells.append(_fmt(value, 1))
        md.append([r['size'], r['n']] + cells)
    return tabulate(md, headers=headers, tablefmt='pipe', stralign='right')


def renderReliabilityTex(rows: list[dict]) -> str:
    header = (
        r'\mathrm{size} & $n$ & \%\,\mathrm{conv} & \%\,\mathrm{conv}\mid\mathrm{slowest\,5\%} & '
        r't/\mathrm{conv} & \mathrm{median} & \mathrm{worst\,5\%} & \mathrm{per\,conv}'
    )
    lines = [r'\begin{tabular}{lr|ccc|ccc}', r'    \toprule', f'    {header} \\\\', r'    \midrule']
    for r in rows:
        lines.append(
            rf"    \texttt{{{r['size']}}} & {r['n']} & {r['pct_conv']:.0f} & "
            rf"{r['pct_conv_tail']:.0f} & ${_fmt(r['t_per_conv'], 1)}$ & "
            rf"${r['speedup_median']:.0f}\times$ & ${r['speedup_tail']:.0f}\times$ & "
            rf"${r['speedup_conv']:.0f}\times$ \\"
        )
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI / main


def outputStem(cfg: argparse.Namespace, device: torch.device) -> str:
    """Output stem carrying every setting a wall time depends on.

    Runtime tables are not interchangeable across hardware, so the device and the sampling
    settings belong in the filename: a bare ``runtimes_{family}`` silently replaces a CPU
    campaign with a CUDA one, and the difference is then unrecoverable from the file.
    """
    if cfg.tag:
        return cfg.tag
    return f'runtimes_{cfg.family}_{cfg.ds_type}_{device.type}_s{cfg.n_samples}'


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Runtime comparison: metabeta (raw flow and flow + IMH) vs NUTS, ADVI and Laplace, per size regime.',
    )
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--ds_type', type=str, default='sampled', help='test-set variant (sampled | real)')
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=None, help='posterior draws (default: match the NUTS draw count in the fit file)')
    parser.add_argument('--method', type=str, default=None, choices=list(IMH_METHODS), help='IMH refinement for the MB row (default: the family preset in presets.yaml)')
    parser.add_argument('--rescale', action=argparse.BooleanOptionalAction, default=True, help='refine in the rescaled space, as the oracle/real benchmarks do')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_datasets', type=int, default=None, help='cap datasets per size (smoke tests)')
    parser.add_argument('--refresh_cache', action='store_true', help='retime MB even if cached (use after a contended run)')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR))
    parser.add_argument('--tag', type=str, default=None, help='override the output stem (default: family + settings)')
    parser.add_argument('--decimals', type=int, default=3)
    parser.add_argument('--no_plot', action='store_true', help='skip the runtime-vs-complexity figure')
    parser.add_argument('--verbosity', type=int, default=1)
    # fmt: on
    return parser.parse_args()


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    family = cfg.family
    if cfg.n_samples is None:
        cfg.n_samples = nutsDrawCount(family, cfg.sizes, cfg.ds_type)
        logger.info('Matching the NUTS draw count: n_samples=%d', cfg.n_samples)

    records: list[dict] = []
    for size in cfg.sizes:
        collected = collectCell(cfg, family, size, device)
        if collected:
            records.extend(collected)
    if not records:
        logger.error('No (family, size) cell could be evaluated for family %s.', family)
        return

    sizes = [s for s in DEFAULT_SIZES if any(r['size'] == s for r in records)]
    methods = sorted({r['refine_method'] for r in records if r['refine_method']})
    rows_dist = cellRows(records)
    dist_md = renderDistMd(rows_dist, dp=cfg.decimals)
    print('\n=== Runtime distribution per regime ===\n')
    print(dist_md)

    rows_rel = reliabilityRows(records)
    rel_md = renderReliabilityMd(rows_rel)
    print('\n=== NUTS tail vs reliability ===\n')
    print(rel_md)

    md = [
        f'# Runtimes ({FAMILY_NAMES[family]})\n',
        f'Sizes: {", ".join(sizes)} ({cfg.ds_type} test sets), each on its regime-matched '
        f'checkpoint. metabeta: {cfg.n_samples} draws, {device.type}, one dataset per forward '
        f'pass (latency). MB^0 is the raw flow; MB adds the default IMH refinement '
        f'({", ".join(methods) or "none"}) on the same draws. ADVI excludes failed fits.\n',
        '## Runtime distribution per regime\n',
        'Wall time per dataset. NUTS/ADVI/Laplace times are those recorded at fit time; '
        'metabeta is timed here. The tail columns are the point: the raw flow is flat because '
        'its cost tracks the architecture; the samplers are not.\n',
        dist_md,
        '',
        '## NUTS tail vs reliability\n',
        'Convergence is the strict `nutsConvergeMask` criterion. `% conv in slowest 5%` is the '
        'convergence rate *within* the slowest 5% of NUTS runs: the reference spends its '
        'largest wall-clock budget where it is least likely to return a usable posterior. '
        '`t/conv` is total NUTS wall time per converged dataset, and the speedups are against '
        'the median latency of the MB row named in `MB row`.\n',
        rel_md,
        '',
    ]

    stem = outputStem(cfg, device)
    (outdir / f'{stem}.md').write_text('\n'.join(md) + '\n')
    (outdir / f'{stem}.tex').write_text(
        renderDistTex(rows_dist, dp=cfg.decimals) + '\n' + renderReliabilityTex(rows_rel)
    )
    (outdir / f'{stem}_records.json').write_text(
        json.dumps(records, indent=2, sort_keys=True) + '\n'
    )
    logger.info('Saved tables to %s', outdir / f'{stem}.md')

    if not cfg.no_plot:
        fig_path = plotRuntimeRecords(records, out_dir=outdir, title=stem)
        logger.info('Saved figure to %s', fig_path)


if __name__ == '__main__':
    main()
