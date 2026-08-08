"""Runtime comparison: metabeta vs NUTS, ADVI and Laplace, per size regime.

Every size uses its regime-matched checkpoint (BEST_SEEDS from scripts/build_ckpt.py) on its
own test set, the same arrangement as the misspecification studies.  The cross-product of the
old script — every model against every test set — is no longer meaningful: the size presets
now define *disjoint* d bands (small 1-4, medium 5-8, large 9-12, huge 13-16), so a smaller
model cannot represent a single dataset of a larger regime, and the cross terms are empty.

Three things the median speedup alone does not say, and that the tables below report:

  1. **The tail.** NUTS wall time is heavy-tailed (worst datasets run 10-25x its median);
     metabeta's is essentially flat, because its cost tracks the architecture, not the data.
     Reported as median / p95 / mean over the slowest 5% / max.
  2. **The tail is where NUTS also fails.** The slowest 5% of NUTS runs have a far lower
     convergence rate than the bulk, so the reference spends its largest wall-clock budget
     exactly where it returns an unusable posterior.  ``t/converged`` (total wall time divided
     by the number of converged datasets) prices a *usable* posterior rather than a run.
  3. **Laplace is the fast classical baseline, not NUTS.** Omitting it flatters the speedup.
     The defensible claim is Laplace-class latency at NUTS-class calibration, which the oracle
     and agreement tables support; runtime alone does not.

metabeta is timed twice: per-dataset latency (batch of 1, comparable to the per-dataset wall
times the fit backends record) and batched throughput (sortish batches of --batch_size, the
deployment number).  Both are cached next to test.fit.npz, keyed by checkpoint/prefix/samples/
seed/k/device, and invalidated when the data or the checkpoint is newer.

ADVI rows exclude datasets whose fit failed (``advi_failed``, up to 50/512 for bernoulli);
their stored durations time a run that produced nothing.

Usage (from repo root):
    uv run python experiments/evaluation/runtimes.py --family n
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
from metabeta.utils.dataloader import Collection, SortishBatchSampler, collateGrouped, toDevice
from metabeta.utils.device import setDevice
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, REPO_ROOT
from metabeta.utils.logger import setupLogging
from metabeta.utils.moe import moeEstimate
from metabeta.utils.posterior_eval import loadModel
from metabeta.utils.sampling import setSeed
from metabeta.utils.warmfit import nParams

# Reuse the checkpoint-seed mapping maintained for published joint checkpoints.
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR
FAMILY_NAMES = {'n': 'normal', 'b': 'bernoulli', 'p': 'poisson'}
DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']

# metabeta first, then the reference methods from cheap to expensive
MB_LATENCY = 'MB'
MB_BATCHED = 'MB_batched'
FIT_METHODS = ['LAPLACE', 'ADVI', 'NUTS']
METHOD_ORDER = [MB_LATENCY, MB_BATCHED] + FIT_METHODS
METHOD_LABELS = {MB_LATENCY: 'MB', MB_BATCHED: 'MB (batched)', 'LAPLACE': 'Laplace'}

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


@torch.no_grad()
def timeLatency(
    model: Approximator,
    col: Collection,
    idxs: list[int],
    n_samples: int,
    k: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Per-dataset wall time with a batch of one — the latency comparable to the fit backends.

    ``no_grad`` rather than ``inference_mode``: the analytical MAP fit inside the model runs
    ``loss.backward()`` under ``torch.enable_grad()``, which inference mode forbids.
    """
    durations = np.zeros(len(idxs))

    # untimed pass so one-time init (lazy alloc, autotune) is not charged to the first dataset
    warmup = toDevice(collateGrouped([col[idxs[0]]]), device)
    resetRng(model, seed)
    moeEstimate(model, warmup, n_samples, k, rng=np.random.default_rng(0))
    del warmup

    for i, idx in enumerate(tqdm(idxs, desc='  MB (latency)', leave=False)):
        batch = toDevice(collateGrouped([col[idx]]), device)
        setSeed(seed)
        resetRng(model, seed)
        rng = np.random.default_rng(seed + idx)
        synchronize(device)
        t0 = time.perf_counter()
        proposal = moeEstimate(model, batch, n_samples, k, rng=rng)
        synchronize(device)
        durations[i] = time.perf_counter() - t0
        del proposal, batch
    return durations


@torch.no_grad()
def timeThroughput(
    model: Approximator,
    col: Collection,
    idxs: list[int],
    n_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Amortized per-dataset wall time in sortish batches — the deployment number.

    Batches are formed by the same sortish sampler the dataloader uses, so datasets of similar
    (m, n) travel together and padding does not inflate the cost of the small ones.  The chunk
    time is divided evenly over its datasets, mirroring ``posterior_eval.sampleMB``.
    """
    order = list(idxs)
    if batch_size < len(order):
        sampler = SortishBatchSampler(
            m_i=col.m_i[order],
            n_i_max=col.n_i_max[order],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        chunks = [[order[j] for j in chunk] for chunk in sampler]
    else:
        chunks = [order]

    # untimed warm-up on the first chunk, as in the latency path
    warmup = toDevice(collateGrouped([col[i] for i in chunks[0]]), device)
    resetRng(model, seed)
    model.estimate(warmup, n_samples=1)
    del warmup

    durations = np.zeros(len(order))
    position = {idx: i for i, idx in enumerate(order)}
    for chunk in tqdm(chunks, desc='  MB (batched)', leave=False):
        batch = toDevice(collateGrouped([col[i] for i in chunk]), device)
        setSeed(seed)
        resetRng(model, seed)
        synchronize(device)
        t0 = time.perf_counter()
        proposal = model.estimate(batch, n_samples=n_samples)
        synchronize(device)
        per_dataset = (time.perf_counter() - t0) / len(chunk)
        for idx in chunk:
            durations[position[idx]] = per_dataset
        del proposal, batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return durations


def synchronize(device: torch.device) -> None:
    """Block until queued device work finishes (no-op on CPU) so timings are accurate."""
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Timing cache — a sibling of test.fit.npz, like the posterior-sample caches


def cachePath(
    data_path: Path,
    ckpt_dir: Path,
    prefix: str,
    n_samples: int,
    seed: int,
    k: int,
    device: torch.device,
) -> Path:
    return data_path.parent / (
        f'runtimes.{ckpt_dir.name}_{prefix}_s{n_samples}_seed{seed}_k{k}_{device.type}.json'
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


def cachedTimings(
    cache: dict[str, float],
    tag: str,
    idxs: list[int],
    compute,
) -> np.ndarray:
    """Return per-dataset timings for ``idxs``, computing only the ones not already cached.

    ``compute`` is called with the missing indices only.  The batched path is an exception it
    handles itself: its per-dataset value depends on the batch its dataset lands in, so a
    partial recompute would mix batch compositions — the caller passes all-or-nothing there.
    """
    keys = [f'{tag}:{idx}' for idx in idxs]
    out = np.full(len(idxs), np.nan)
    missing = []
    for i, key in enumerate(keys):
        if key in cache:
            out[i] = cache[key]
        else:
            missing.append(i)
    if missing:
        values = compute([idxs[i] for i in missing])
        for i, value in zip(missing, values):
            out[i] = value
            cache[keys[i]] = float(value)
    elif len(idxs):
        logger.info('%s: all %d timings cached', tag, len(idxs))
    return out


# ---------------------------------------------------------------------------
# Per-cell collection


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
        col = Collection(
            data_path,
            permute=False,
            max_d=model_cfg.max_d,
            max_q=model_cfg.max_q,
            exclude_prefixes=('nuts_', 'advi_', 'laplace_'),
        )
    except ValueError as exc:
        # regimes are matched by construction; a mismatch means the checkpoint map is wrong
        logger.warning('%s: checkpoint does not cover this regime (%s) — skipping', data_id, exc)
        return None

    B = len(col)
    idxs = list(range(min(B, cfg.max_datasets))) if cfg.max_datasets else list(range(B))
    durations, masks, conv = loadReferences(data_path)
    logger.info(
        '%s: %d datasets (%d timed), %d NUTS-converged, d<=%d q<=%d',
        data_id,
        B,
        len(idxs),
        int(conv[idxs].sum()),
        model_cfg.max_d,
        model_cfg.max_q,
    )

    cache_path = cachePath(data_path, ckpt_dir, cfg.prefix, cfg.n_samples, cfg.seed, cfg.k, device)
    cache = loadCache(cache_path, data_path, ckpt_dir, cfg.prefix)

    mb_latency = cachedTimings(
        cache,
        'latency',
        idxs,
        lambda missing: timeLatency(model, col, missing, cfg.n_samples, cfg.k, device, cfg.seed),
    )
    mb_batched = None
    if cfg.batch_size > 1:
        tag = f'batched{cfg.batch_size}'
        # all-or-nothing: a per-dataset batched time is only meaningful together with the
        # batch composition that produced it, so a partial refill would mix regimes
        if all(f'{tag}:{idx}' in cache for idx in idxs):
            mb_batched = np.array([cache[f'{tag}:{idx}'] for idx in idxs])
            logger.info('%s: all %d timings cached', tag, len(idxs))
        else:
            mb_batched = timeThroughput(
                model, col, idxs, cfg.n_samples, cfg.batch_size, device, cfg.seed
            )
            for idx, value in zip(idxs, mb_batched):
                cache[f'{tag}:{idx}'] = float(value)
    saveCache(cache_path, cache)

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
        }
        timings = [(MB_LATENCY, mb_latency[i])]
        if mb_batched is not None:
            timings.append((MB_BATCHED, mb_batched[i]))
        for method in FIT_METHODS:
            if method not in durations:
                continue
            if not masks[method][idx]:
                continue  # failed fit: its duration times a run that produced nothing
            timings.append((method, durations[method][idx]))
        for method, duration in timings:
            records.append({**base, 'method': method, 'duration': float(duration)})
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
                    'method': METHOD_LABELS.get(method, method),
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
        mb = np.array(
            [r['duration'] for r in records if r['size'] == size and r['method'] == MB_LATENCY]
        )
        stats = tailStats(durations)
        rows.append(
            {
                'size': size,
                'n': len(durations),
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


def renderDistMd(rows: list[dict], dp: int = 3) -> str:
    headers = ['size', 'method', 'n'] + [h for _, h in DIST_COLS]
    md = []
    for r in rows:
        # n is per row, not per size: ADVI drops the datasets whose fit failed
        size = r['size'] if r['first'] else ''
        md.append([size, r['method'], r['n']] + [_fmt(r[k], dp) for k, _ in DIST_COLS])
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
        lines.append(rf"    {size} & \texttt{{{r['method']}}} & {r['n']} & {cells} \\")
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


RELIABILITY_COLS = [
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
            if key.startswith('pct'):
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


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Runtime comparison: metabeta vs NUTS, ADVI and Laplace, per size regime.',
    )
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--ds_type', type=str, default='sampled', help='test-set variant (sampled | real)')
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8, help='batched-throughput batch size (1 disables the batched row)')
    parser.add_argument('--k', type=int, default=0, help='extra pseudo-MoE permuted views (0 = off)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_datasets', type=int, default=None, help='cap datasets per size (smoke tests)')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR))
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

    records: list[dict] = []
    for size in cfg.sizes:
        collected = collectCell(cfg, family, size, device)
        if collected:
            records.extend(collected)
    if not records:
        logger.error('No (family, size) cell could be evaluated for family %s.', family)
        return

    sizes = [s for s in DEFAULT_SIZES if any(r['size'] == s for r in records)]
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
        f'checkpoint. metabeta: {cfg.n_samples} draws, k={cfg.k}, {device.type}. Latency is a '
        f'batch of one; batched is sortish batches of {cfg.batch_size}. ADVI excludes failed '
        'fits.\n',
        '## Runtime distribution per regime\n',
        'Wall time per dataset. NUTS/ADVI/Laplace times are those recorded at fit time; '
        'metabeta is timed here. The tail columns are the point: metabeta is flat because its '
        'cost tracks the architecture, the samplers are not.\n',
        dist_md,
        '',
        '## NUTS tail vs reliability\n',
        'Convergence is the strict `nutsConvergeMask` criterion. `% conv in slowest 5%` is the '
        'convergence rate *within* the slowest 5% of NUTS runs: the reference spends its '
        'largest wall-clock budget where it is least likely to return a usable posterior. '
        '`t/conv` is total NUTS wall time per converged dataset, and the speedups are against '
        'metabeta median latency.\n',
        rel_md,
        '',
    ]

    stem = f'runtimes_{family}'
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
