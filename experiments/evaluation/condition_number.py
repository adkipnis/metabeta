"""MB↔NUTS posterior agreement as a function of the design-matrix condition number κ₂(X).

κ₂(X) = σ_max/σ_min of each dataset's stacked fixed-effects design (a property of the
input) is the independent variable; datasets are binned by it. Within each bin we report
per-dataset agreement to the NUTS reference, reusing real_posterior.py's metrics: r (mean
correlation), σ-ratio (std_MB/std_NUTS → 1), rank-MAD (marginal shape → 0), and ΔLOO-NLL.
These are paired to NUTS on the same dataset, so the intrinsic posterior widening at high κ
cancels; % NUTS-converged per bin marks the reference's own operating boundary.

Run on real-n data (synthetic designs are near-orthogonal, κ≈1); each size uses its
regime-matched checkpoint to avoid confounding κ with capacity mismatch, then pairs are
pooled and binned globally.

Usage (from repo root):
    uv run python experiments/evaluation/condition_number.py
    uv run python experiments/evaluation/condition_number.py --distribution_only
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from metabeta.utils.dataloader import Collection, collateGrouped, subsetBatch
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.device import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
from metabeta.utils.posterior_eval import (
    fit2proposal,
    loadModel,
    loadOrComputeSummary,
    loadOrRefine,
    loadOrSampleMB,
    validMethods,
)

# real_posterior.py sits in this directory (on sys.path[0] at run time).
from real_posterior import computeCorr, computeSigmaRatio, computeRankMAD, _ms

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR

# Regime-matched Normal checkpoints (prefix=latest, n_samples=1000, seed=0), keyed to hit
# the existing MB-sample / summary caches rather than resampling.
SIZE_MODELS: dict[str, str] = {
    'small': 'data=small-n-mixed_model=large_seed=13',
    'medium': 'data=medium-n-mixed_model=large_seed=14',
    'large': 'data=large-n-mixed_model=large_seed=9',
    'huge': 'data=huge-n-mixed_model=large_seed=16',
}
DEFAULT_SIZES = list(SIZE_MODELS)

# κ bin edges; last bin is the effectively-singular (rank-deficient) bucket.
KAPPA_EDGES = [1.0, 3.0, 6.0, 10.0, 1e6, np.inf]

# imhMarginal is the Normal (lf=0) presets default: IMH on the marginal posterior, MB-initialised.
METHOD_LABELS = {'mb': 'MB', 'imhMarginal': 'MB+IMH', 'isMarginal': 'MB+SNIS'}

METRIC_KEYS = ('r', 'sigma_ratio', 'rank_mad', 'delta_nll')
AGREE_METRICS = [
    ('r', 'r ↑'),
    ('sigma_ratio', 'σ-ratio →1'),
    ('rank_mad', 'rank-MAD ↓'),
    ('delta_nll', 'ΔLOO-NLL ↓'),
]


# ---------------------------------------------------------------------------
# Condition number


def conditionNumbers(batch: dict[str, torch.Tensor], standardize: bool = False) -> np.ndarray:
    """κ₂(X) per dataset from the SVD of the stacked (N_obs, d_active) fixed-effects design.

    ``standardize`` centres/scales non-constant columns first (κ = pure collinearity).
    Returns (B,) float64; +inf when the smallest singular value underflows to 0.
    """
    X, mask_n, mask_d = batch['X'], batch['mask_n'], batch['mask_d']
    kappa = np.empty(X.shape[0], dtype=np.float64)
    for b in range(X.shape[0]):
        M = X[b][mask_n[b]][:, mask_d[b]].double().numpy()
        if M.shape[0] == 0 or M.shape[1] == 0:
            kappa[b] = np.nan
            continue
        if standardize:
            sd = M.std(0)
            nz = sd > 1e-12
            M = M.copy()
            M[:, nz] = (M[:, nz] - M.mean(0)[nz]) / sd[nz]
        s = np.linalg.svd(M, compute_uv=False)
        kappa[b] = s[0] / s[-1] if s[-1] > 0 else np.inf
    return kappa


# ---------------------------------------------------------------------------
# Per-size record collection


def _agreementArrays(
    proposal,
    proposal_nuts,
    batch_c,
    data_path,
    method,
    conv_mask,
    lf,
    n_samples,
    seed,
    ckpt_dir,
    prefix,
    summary_chunk_size,
    nuts_loo,
) -> dict[str, np.ndarray]:
    """Per-dataset MB↔NUTS agreement (over the converged subset) for one proposal."""
    summary = loadOrComputeSummary(
        proposal,
        batch_c,
        data_path,
        method,
        conv_mask,
        lf,
        True,
        ckpt_dir=ckpt_dir,
        prefix=prefix,
        n_samples=n_samples,
        seed=seed,
        summary_chunk_size=summary_chunk_size,
    )
    return {
        'r': computeCorr(proposal, proposal_nuts, batch_c),
        'sigma_ratio': computeSigmaRatio(proposal, proposal_nuts, batch_c),
        'rank_mad': computeRankMAD(proposal, proposal_nuts, batch_c),
        'delta_nll': (summary.per_dataset.loo_nll - nuts_loo).float().numpy(),
    }


def collectSizeRecords(
    size: str,
    device: torch.device,
    n_samples: int,
    batch_size: int,
    seed: int,
    prefix: str,
    convergence_mode: str,
    summary_chunk_size: int,
    standardize: bool,
    methods: list[str],
) -> dict[str, np.ndarray] | None:
    """Per-dataset arrays for one real-n size, aligned over the full test file.

    Keys: kappa, converged, size, and ``{method}:{metric}`` for method in ('mb', *methods).
    Agreement metrics are NaN for non-converged datasets (no trustworthy NUTS reference).
    """
    data_id = f'{size}-n-real'
    ckpt_dir = CHECKPOINT_DIR / SIZE_MODELS[size]
    data_path = DATA_DIR / data_id / 'test.fit.npz'
    if not data_path.exists() or not ckpt_dir.exists():
        logger.warning('%s: data or checkpoint missing — skipping', data_id)
        return None

    model, model_cfg = loadModel(ckpt_dir, prefix, device)
    max_d, max_q, lf = model_cfg.max_d, model_cfg.max_q, model_cfg.likelihood_family

    col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
    B_total = len(col)
    batch = collateGrouped([col[i] for i in range(B_total)])

    # Reuse precomputed analytical stats from the sibling test.npz (matches evaluate.py).
    if 'stats' not in batch:
        base_path = data_path.with_name('test.npz')
        if base_path.exists():
            base_col = Collection(base_path, permute=False, max_d=max_d, max_q=max_q)
            if len(base_col) == B_total:
                base_batch = collateGrouped([base_col[i] for i in range(B_total)])
                if 'stats' in base_batch:
                    batch['stats'] = base_batch['stats']

    kappa = conditionNumbers(batch, standardize=standardize)
    conv_mask = nutsConvergeMask(batch, mode=convergence_mode)
    if conv_mask is None:
        logger.warning('%s: no NUTS diagnostics; treating all as converged', data_id)
        conv_mask = np.ones(B_total, dtype=bool)
    n_conv = int(conv_mask.sum())
    logger.info(
        '%s: %d datasets, %d NUTS-converged (%s)', data_id, B_total, n_conv, convergence_mode
    )

    active_methods = ['mb'] + validMethods(methods, lf)
    out: dict[str, np.ndarray] = {
        'size': np.array([size] * B_total, dtype=object),
        'kappa': kappa,
        'converged': conv_mask.astype(bool),
    }
    for m in active_methods:
        for mk in METRIC_KEYS:
            out[f'{m}:{mk}'] = np.full(B_total, np.nan)
    if n_conv == 0:
        return out

    batch_c = subsetBatch(batch, conv_mask)
    proposal_mb, _ = loadOrSampleMB(
        model, batch_c, data_path, ckpt_dir, prefix, n_samples, batch_size, seed, device, conv_mask
    )
    proposal_nuts = fit2proposal(batch_c, 'nuts')
    proposal_mb.rescale(batch_c['sd_y'])
    proposal_nuts.rescale(batch_c['sd_y'])
    batch_c = rescaleData(batch_c)

    nuts_loo = loadOrComputeSummary(
        proposal_nuts,
        batch_c,
        data_path,
        'nuts',
        conv_mask,
        lf,
        True,
        summary_chunk_size=summary_chunk_size,
    ).per_dataset.loo_nll
    idx = np.flatnonzero(conv_mask)

    for m in active_methods:
        if m == 'mb':
            proposal = proposal_mb
        else:
            logger.info('%s: refining MB with %s', data_id, m)
            proposal, _ = loadOrRefine(
                m,
                proposal_mb,
                batch_c,
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
        arrs = _agreementArrays(
            proposal,
            proposal_nuts,
            batch_c,
            data_path,
            m,
            conv_mask,
            lf,
            n_samples,
            seed,
            ckpt_dir,
            prefix,
            summary_chunk_size,
            nuts_loo,
        )
        for mk in METRIC_KEYS:
            out[f'{m}:{mk}'][idx] = arrs[mk]
    return out


# ---------------------------------------------------------------------------
# Phase 1: κ distribution


def _logQuantiles(kappa: np.ndarray) -> dict:
    lk = np.log10(np.clip(kappa[np.isfinite(kappa)], 1.0, None))
    d = {f'p{p}': 10 ** np.percentile(lk, p) for p in (0, 25, 50, 75, 90, 95, 99, 100)}
    d['#singular(inf)'] = int(np.sum(~np.isfinite(kappa)))
    return d


def distributionTable(records: dict[str, np.ndarray]) -> str:
    sizes = sorted(set(records['size']), key=DEFAULT_SIZES.index) + ['pooled']
    stat_names = ['p0', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'p100', '#singular(inf)', 'n']
    cols = {}
    for s in sizes:
        k = records['kappa'] if s == 'pooled' else records['kappa'][records['size'] == s]
        cols[s] = {**_logQuantiles(k), 'n': len(k)}
    rows = [
        [name]
        + [
            f'{int(cols[s][name])}' if name in ('#singular(inf)', 'n') else f'{cols[s][name]:.3g}'
            for s in sizes
        ]
        for name in stat_names
    ]
    return tabulate(rows, headers=['statistic'] + sizes, tablefmt='pipe', stralign='right')


# ---------------------------------------------------------------------------
# Phase 2: κ-binned agreement table


def _binLabel(lo: float, hi: float) -> str:
    fmt = lambda x: '∞' if x == np.inf else (f'{x:.0e}' if x >= 1e6 else f'{x:g}')
    return f'[{fmt(lo)}, {fmt(hi)})'


def binnedTable(
    records: dict[str, np.ndarray], edges: list[float], methods: list[str]
) -> list[dict]:
    """One row per (κ bin × method); bin-level stats carried on the first method row."""
    kappa, conv = records['kappa'], records['converged']
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (kappa >= lo) & (kappa < hi)
        n_total = int(in_bin.sum())
        if n_total == 0:
            continue
        conv_in = in_bin & conv
        n_conv = int(conv_in.sum())
        k_finite = kappa[in_bin][np.isfinite(kappa[in_bin])]
        median_kappa = float(np.median(k_finite)) if len(k_finite) else np.inf
        for j, m in enumerate(methods):
            row = {
                'bin': _binLabel(lo, hi),
                'method': METHOD_LABELS.get(m, m),
                'first': j == 0,
                'n': n_total,
                'n_conv': n_conv,
                'pct_conv': 100.0 * n_conv / n_total,
                'median_kappa': median_kappa,
            }
            for key, _ in AGREE_METRICS:
                row[key] = _ms(records[f'{m}:{key}'][conv_in]) if n_conv > 0 else None
            rows.append(row)
    return rows


def _fmtMs(val, dp: int) -> str:
    if val is None:
        return 'NA'
    m, s = val
    return 'NA' if m != m else f'{m:.{dp}f} ± {s:.{dp}f}'


def renderBinnedMd(rows: list[dict], dp: int = 3) -> str:
    headers = ['κ bin', 'method', 'n', 'n_conv', '% NUTS-conv', 'median κ'] + [
        h for _, h in AGREE_METRICS
    ]
    md = []
    for r in rows:
        mk = '∞' if not np.isfinite(r['median_kappa']) else f"{r['median_kappa']:.2f}"
        lead = (
            [r['bin'], r['n'], r['n_conv'], f"{r['pct_conv']:.0f}", mk] if r['first'] else [''] * 5
        )
        md.append([lead[0], r['method'], *lead[1:]] + [_fmtMs(r[k], dp) for k, _ in AGREE_METRICS])
    return tabulate(md, headers=headers, tablefmt='pipe', stralign='right')


def renderBinnedTex(rows: list[dict], dp: int = 3) -> str:
    def cell(val):
        if val is None:
            return r'\textrm{NA}'
        m, s = val
        return r'\textrm{NA}' if m != m else f'${m:.{dp}f} \\pm {s:.{dp}f}$'

    header = (
        r'$\kappa_2(X)$ & \mathrm{model} & $n$ & $n_{\mathrm{conv}}$ & \%\,\mathrm{NUTS} & '
        r'$\tilde\kappa$ & $r$ & $\sigma\text{-ratio}$ & $\mathrm{rank\text{-}MAD}$ & '
        r'$\Delta\mathrm{LOO\text{-}NLL}$'
    )
    lines = [
        r'\begin{tabular}{llrrrr|cccc}',
        r'    \toprule',
        f'    {header} \\\\',
        r'    \midrule',
    ]
    for i, r in enumerate(rows):
        if r['first'] and i != 0:
            lines.append(r'    \midrule')
        mk = r'$\infty$' if not np.isfinite(r['median_kappa']) else f"${r['median_kappa']:.2f}$"
        lead = (
            (rf"\texttt{{{r['bin']}}}", r['n'], r['n_conv'], f"{r['pct_conv']:.0f}", mk)
            if r['first']
            else ('', '', '', '', '')
        )
        cells = ' & '.join(cell(r[k]) for k, _ in AGREE_METRICS)
        lines.append(
            rf"    {lead[0]} & \texttt{{{r['method']}}} & {lead[1]} & {lead[2]} & "
            rf'{lead[3]} & {lead[4]} & {cells} \\'
        )
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI / main


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='MB↔NUTS agreement vs design-matrix condition number on real-n data.',
    )
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES,
                        choices=DEFAULT_SIZES, help='Real-n sizes to pool (default: all).')
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--convergence_mode', type=str, default='strict',
                        choices=['liberal', 'strict'])
    parser.add_argument('--methods', type=str, nargs='*', default=['imhMarginal'],
                        choices=list(METHOD_LABELS)[1:],
                        help='Refinements added as extra rows (default: imhMarginal); MB is always included.')
    parser.add_argument('--standardize', action='store_true',
                        help='κ on column-standardized designs (pure collinearity); off by default '
                             '(continuous cols already standardized, singular tail is scale-invariant).')
    parser.add_argument('--distribution_only', action='store_true',
                        help='Only report the κ distribution (Phase 1), no MB/NUTS.')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR))
    parser.add_argument('--decimals', type=int, default=3)
    parser.add_argument('--verbosity', type=int, default=1)
    # fmt: on
    return parser.parse_args()


def _distributionOnly(size: str, standardize: bool) -> dict[str, np.ndarray] | None:
    """Phase-1-only: κ per dataset, no model / fit proposal loaded."""
    data_path = DATA_DIR / f'{size}-n-real' / 'test.npz'
    if not data_path.exists():
        logger.warning('%s: test.npz not found — skipping', size)
        return None
    col = Collection(data_path, permute=False)
    B = len(col)
    kappa = conditionNumbers(collateGrouped([col[i] for i in range(B)]), standardize=standardize)
    return {
        'size': np.array([size] * B, dtype=object),
        'kappa': kappa,
        'converged': np.zeros(B, dtype=bool),
    }


def _pool(per_size: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([r[k] for r in per_size]) for k in per_size[0]}


def main() -> None:
    cfg = setup()
    setupLogging(cfg.verbosity)
    setSeed(cfg.seed)
    device = setDevice(cfg.device)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = 'std' if cfg.standardize else 'raw'
    methods = cfg.methods or []

    per_size = []
    for size in cfg.sizes:
        rec = (
            _distributionOnly(size, cfg.standardize)
            if cfg.distribution_only
            else collectSizeRecords(
                size,
                device,
                cfg.n_samples,
                cfg.batch_size,
                cfg.seed,
                cfg.prefix,
                cfg.convergence_mode,
                cfg.summary_chunk_size,
                cfg.standardize,
                methods,
            )
        )
        if rec is not None:
            per_size.append(rec)
    if not per_size:
        logger.error('No sizes evaluated.')
        return
    records = _pool(per_size)

    dist_md = distributionTable(records)
    print('\n=== κ₂(X) distribution (log-quantiles, %s) ===\n' % tag)
    print(dist_md)
    (outdir / f'condition_number_distribution_{tag}.md').write_text(
        f'# κ₂(X) distribution ({tag})\n\n{dist_md}\n'
    )
    if cfg.distribution_only:
        return

    rows = binnedTable(records, KAPPA_EDGES, ['mb'] + methods)
    binned_md = renderBinnedMd(rows, dp=cfg.decimals)
    print('\n=== MB↔NUTS agreement by κ₂(X) bin (median ± MAD over converged datasets) ===\n')
    print(binned_md)

    stem = f'condition_number_agreement_{tag}'
    (outdir / f'{stem}.md').write_text(
        f'# MB↔NUTS agreement vs κ₂(X) ({tag})\n\n'
        f'Sizes: {", ".join(cfg.sizes)} (real-n). Reference: NUTS ({cfg.convergence_mode}).\n\n'
        f'{binned_md}\n'
    )
    (outdir / f'{stem}.tex').write_text(renderBinnedTex(rows, dp=cfg.decimals))
    logger.info('Saved tables to %s', outdir)


if __name__ == '__main__':
    main()
