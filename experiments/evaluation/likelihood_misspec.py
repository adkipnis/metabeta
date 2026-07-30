"""MB↔NUTS agreement and posterior quality under likelihood misspecification.

Companion to experiments/simulation/likelihood_misspec.py, which regenerates test outcomes
under contaminated likelihoods (Normal→Student-t, Poisson→NegBin, Bernoulli→latent logit
noise) with increasing severity, while both metabeta and NUTS fit the original — now
misspecified — model with the stored priors.  This script produces the dose–response tables:

  1. MB↔NUTS agreement per severity (r, σ-ratio, rank-MAD, ΔLOO-NLL; real_posterior.py
     metrics, median ± MAD over the NUTS-converged subset).  If MB's agreement with NUTS is
     stable across severities, MB's extra error under misspecification is attributable to the
     model, not the amortization.
  2. Quality vs the generating parameters per severity (EACE / NRMSE / cov90 for global and
     local parameters, plus LOO-NLL) for MB, MB+IMH and NUTS.  Under contamination the
     generating parameters are no longer the fitted model's pseudo-true values, so absolute
     numbers degrade by design — the question is whether MB degrades in step with NUTS.

All metrics are computed on the NUTS-converged subset (paired comparisons); % converged per
severity marks where the reference itself starts to fail.  Sizes are pooled per severity;
``--per_size`` adds per-size agreement breakdowns.  Each size uses its regime-matched
checkpoint (BEST_SEEDS from scripts/build_ckpt.py — one pretrained model per family × size).

Usage (from repo root):
    uv run python experiments/evaluation/likelihood_misspec.py --family n
    uv run python experiments/evaluation/likelihood_misspec.py --family n --sizes small --per_size
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.device import setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, REPO_ROOT
from metabeta.utils.posterior_eval import (
    fit2proposal,
    loadModel,
    loadOrComputeSummary,
    loadOrRefine,
    loadOrSampleMB,
    posthocDefaults,
    validMethods,
)

# Reuse the checkpoint-seed mapping maintained for published joint checkpoints.
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

# sibling experiment scripts (this directory is sys.path[0] at run time)
from condition_number import LF_FROM_FAM, METHOD_LABELS, _fmtMs
from real_posterior import computeCorr, computeSigmaRatio, computeRankMAD, _ms
from data_poverty import localEntries, globalEntries, _eace, _nrmse, _nrmseByType

logger = logging.getLogger(__name__)

FAMILY_NAMES = {'n': 'normal', 'b': 'bernoulli', 'p': 'poisson'}

OUT_DIR = RESULTS_DIR
DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']
NUTS_LABEL = 'NUTS'

# family letter → ordered (ds_type tag, display label); severity 0 first.
CONDITIONS: dict[str, list[tuple[str, str]]] = {
    'n': [
        ('misbase', 'baseline'),
        ('student10', 't(ν=10)'),
        ('student5', 't(ν=5)'),
        ('student3', 't(ν=3)'),
    ],
    'p': [
        ('misbase', 'baseline'),
        ('negbin10', 'NB(φ=10)'),
        ('negbin3', 'NB(φ=3)'),
        ('negbin1', 'NB(φ=1)'),
    ],
    'b': [
        ('misbase', 'baseline'),
        ('latent05', 'σ_c=0.5'),
        ('latent10', 'σ_c=1'),
        ('latent20', 'σ_c=2'),
    ],
}

AGREE_METRICS = [
    ('r', 'r ↑'),
    ('sigma_ratio', 'σ-ratio →1'),
    ('rank_mad', 'rank-MAD ↓'),
    ('delta_nll', 'ΔLOO-NLL ↓'),
]


# ---------------------------------------------------------------------------
# Per-condition collection


def collectCondition(
    cfg: argparse.Namespace,
    family: str,
    size: str,
    tag: str,
    device: torch.device,
    methods: list[str],
) -> dict | None:
    """Per-dataset agreement arrays and quality entries for one condition dir.

    Proposals are computed on the full condition batch (NUTS fits exist for every dataset);
    the converged subset is applied downstream when aggregating, so all methods stay paired.
    """
    data_id = f'{size}-{family}-{tag}'
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
    max_d, max_q, lf = model_cfg.max_d, model_cfg.max_q, model_cfg.likelihood_family

    col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
    B = len(col)
    batch = collateGrouped([col[i] for i in range(B)])

    # Reuse precomputed analytical stats from the sibling test.npz (matches condition_number).
    if 'stats' not in batch:
        base_path = data_path.with_name('test.npz')
        if base_path.exists():
            base_col = Collection(base_path, permute=False, max_d=max_d, max_q=max_q)
            if len(base_col) == B:
                base_batch = collateGrouped([base_col[i] for i in range(B)])
                if 'stats' in base_batch:
                    batch['stats'] = base_batch['stats']

    conv = nutsConvergeMask(batch, mode=cfg.convergence_mode)
    if conv is None:
        logger.warning('%s: no NUTS diagnostics; treating all as converged', data_id)
        conv = np.ones(B, dtype=bool)
    conv = conv.astype(bool)
    logger.info('%s: %d datasets, %d NUTS-converged', data_id, B, int(conv.sum()))

    proposal_mb, _ = loadOrSampleMB(
        model,
        batch,
        data_path,
        ckpt_dir,
        cfg.prefix,
        cfg.n_samples,
        cfg.batch_size,
        cfg.seed,
        device,
        None,
    )
    proposal_nuts = fit2proposal(batch, 'nuts')
    proposal_mb.rescale(batch['sd_y'])
    proposal_nuts.rescale(batch['sd_y'])
    batch = rescaleData(batch)

    nuts_loo = (
        loadOrComputeSummary(
            proposal_nuts,
            batch,
            data_path,
            'nuts',
            None,
            lf,
            True,
            summary_chunk_size=cfg.summary_chunk_size,
        )
        .per_dataset.loo_nll.float()
        .numpy()
    )

    out = {
        'size': np.array([size] * B, dtype=object),
        'conv': conv,
        'nuts:loo': nuts_loo,
    }
    entries = {
        NUTS_LABEL: {
            'local': localEntries(proposal_nuts, batch, conv),
            'global': globalEntries(proposal_nuts, batch, conv, lf),
        }
    }
    for m in methods:
        if m == 'mb':
            proposal = proposal_mb
        else:
            logger.info('%s: refining MB with %s', data_id, m)
            proposal, _ = loadOrRefine(
                m,
                proposal_mb,
                batch,
                data_path,
                ckpt_dir,
                cfg.prefix,
                cfg.n_samples,
                cfg.seed,
                lf,
                True,
                None,
                cfg.batch_size,
            )
        summary = loadOrComputeSummary(
            proposal,
            batch,
            data_path,
            m,
            None,
            lf,
            True,
            ckpt_dir=ckpt_dir,
            prefix=cfg.prefix,
            n_samples=cfg.n_samples,
            seed=cfg.seed,
            summary_chunk_size=cfg.summary_chunk_size,
        )
        out[f'{m}:r'] = computeCorr(proposal, proposal_nuts, batch)
        out[f'{m}:sigma_ratio'] = computeSigmaRatio(proposal, proposal_nuts, batch)
        out[f'{m}:rank_mad'] = computeRankMAD(proposal, proposal_nuts, batch)
        out[f'{m}:loo'] = summary.per_dataset.loo_nll.float().numpy()
        out[f'{m}:delta_nll'] = out[f'{m}:loo'] - nuts_loo
        entries[METHOD_LABELS.get(m, m)] = {
            'local': localEntries(proposal, batch, conv),
            'global': globalEntries(proposal, batch, conv, lf),
        }
    return {'agree': out, 'entries': entries}


def _poolAgree(per_size: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([r['agree'][k] for r in per_size]) for k in per_size[0]['agree']}


def _poolEntries(per_size: list[dict], label: str, kind: str) -> dict[str, np.ndarray]:
    parts = [r['entries'][label][kind] for r in per_size]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------------------
# Tables


def agreementRows(
    pooled: dict[str, dict[str, np.ndarray]],
    labels: dict[str, str],
    methods: list[str],
) -> list[dict]:
    """One row per (condition × method); condition-level stats on the first method row."""
    rows = []
    for tag, agree in pooled.items():
        conv = agree['conv']
        for j, m in enumerate(methods):
            row = {
                'condition': labels[tag],
                'method': METHOD_LABELS.get(m, m),
                'first': j == 0,
                'n': len(conv),
                'n_conv': int(conv.sum()),
                'pct_conv': 100.0 * conv.mean(),
            }
            for key, _ in AGREE_METRICS:
                row[key] = _ms(agree[f'{m}:{key}'][conv]) if conv.any() else None
            rows.append(row)
    return rows


def renderAgreementMd(rows: list[dict], dp: int = 3) -> str:
    headers = ['condition', 'method', 'n', 'n_conv', '% NUTS-conv'] + [h for _, h in AGREE_METRICS]
    md = []
    for r in rows:
        lead = (
            [r['condition'], r['n'], r['n_conv'], f"{r['pct_conv']:.0f}"]
            if r['first']
            else [''] * 4
        )
        md.append([lead[0], r['method'], *lead[1:]] + [_fmtMs(r[k], dp) for k, _ in AGREE_METRICS])
    return tabulate(md, headers=headers, tablefmt='pipe', stralign='right')


def renderAgreementTex(rows: list[dict], dp: int = 3) -> str:
    def cell(val):
        if val is None:
            return r'\textrm{NA}'
        m, s = val
        return r'\textrm{NA}' if m != m else f'${m:.{dp}f} \\pm {s:.{dp}f}$'

    header = (
        r'\mathrm{condition} & \mathrm{model} & $n$ & $n_{\mathrm{conv}}$ & \%\,\mathrm{NUTS} & '
        r'$r$ & $\sigma\text{-ratio}$ & $\mathrm{rank\text{-}MAD}$ & $\Delta\mathrm{LOO\text{-}NLL}$'
    )
    lines = [
        r'\begin{tabular}{llrrr|cccc}',
        r'    \toprule',
        f'    {header} \\\\',
        r'    \midrule',
    ]
    for i, r in enumerate(rows):
        if r['first'] and i != 0:
            lines.append(r'    \midrule')
        lead = (
            (rf"\texttt{{{r['condition']}}}", r['n'], r['n_conv'], f"{r['pct_conv']:.0f}")
            if r['first']
            else ('', '', '', '')
        )
        cells = ' & '.join(cell(r[k]) for k, _ in AGREE_METRICS)
        lines.append(
            rf"    {lead[0]} & \texttt{{{r['method']}}} & {lead[1]} & {lead[2]} & "
            rf'{lead[3]} & {cells} \\'
        )
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def qualityRows(
    entries: dict[str, dict[str, dict[str, np.ndarray]]],
    agree: dict[str, dict[str, np.ndarray]],
    labels: dict[str, str],
    methods: list[str],
) -> list[dict]:
    """Ground-truth quality per (condition × method), on the converged subset (paired)."""
    method_labels = [METHOD_LABELS.get(m, m) for m in methods] + [NUTS_LABEL]
    loo_keys = {METHOD_LABELS.get(m, m): f'{m}:loo' for m in methods}
    loo_keys[NUTS_LABEL] = 'nuts:loo'
    rows = []
    for tag, per_method in entries.items():
        conv = agree[tag]['conv']
        for j, label in enumerate(method_labels):
            loc = per_method[label]['local']
            glo = per_method[label]['global']
            sel_l = loc['conv'].astype(bool)
            sel_g = glo['conv'].astype(bool)
            loo = agree[tag][loo_keys[label]][conv]
            rows.append(
                {
                    'condition': labels[tag],
                    'method': label,
                    'first': j == 0,
                    'eace_g': _eace(glo['inside'][sel_g]),
                    'nrmse_g': _nrmseByType(glo, sel_g),
                    'cov90_g': float(np.nanmean(glo['cov90'][sel_g]))
                    if sel_g.any()
                    else float('nan'),
                    'eace_l': _eace(loc['inside'][sel_l]),
                    'nrmse_l': _nrmse(loc['se'][sel_l], loc['gt'][sel_l]),
                    'cov90_l': float(np.nanmean(loc['cov90'][sel_l]))
                    if sel_l.any()
                    else float('nan'),
                    'loo': float(np.nanmean(loo)) if len(loo) else float('nan'),
                }
            )
    return rows


QUALITY_COLS = [
    ('eace_g', 'EACE_g↓'),
    ('nrmse_g', 'NRMSE_g↓'),
    ('cov90_g', 'cov90_g'),
    ('eace_l', 'EACE_l↓'),
    ('nrmse_l', 'NRMSE_l↓'),
    ('cov90_l', 'cov90_l'),
    ('loo', 'LOO-NLL↓'),
]


def renderQualityMd(rows: list[dict], dp: int = 3) -> str:
    fmt = lambda v: 'NA' if (v is None or v != v) else f'{v:.{dp}f}'
    headers = ['condition', 'method'] + [h for _, h in QUALITY_COLS]
    md = []
    for r in rows:
        lead = r['condition'] if r['first'] else ''
        md.append([lead, r['method']] + [fmt(r[k]) for k, _ in QUALITY_COLS])
    return tabulate(md, headers=headers, tablefmt='pipe', stralign='right')


def renderQualityTex(rows: list[dict], dp: int = 3) -> str:
    fmt = lambda v: r'\textrm{NA}' if (v is None or v != v) else f'${v:.{dp}f}$'
    header = (
        r'\mathrm{condition} & \mathrm{model} & \mathrm{EACE}_g & \mathrm{NRMSE}_g & '
        r'\mathrm{cov}_{90,g} & \mathrm{EACE}_l & \mathrm{NRMSE}_l & \mathrm{cov}_{90,l} & '
        r'\mathrm{LOO\text{-}NLL}'
    )
    lines = [
        r'\begin{tabular}{ll|ccc|ccc|c}',
        r'    \toprule',
        f'    {header} \\\\',
        r'    \midrule',
    ]
    for i, r in enumerate(rows):
        if r['first'] and i != 0:
            lines.append(r'    \midrule')
        lead = rf"\texttt{{{r['condition']}}}" if r['first'] else ''
        cells = ' & '.join(fmt(r[k]) for k, _ in QUALITY_COLS)
        lines.append(rf"    {lead} & \texttt{{{r['method']}}} & {cells} \\")
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI / main


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='MB↔NUTS agreement and posterior quality under likelihood misspecification.',
    )
    parser.add_argument('--family', type=str, default='n', choices=list(FAMILY_NAMES))
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        help='ds_type tags to evaluate (default: all four per family)')
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--convergence_mode', type=str, default='strict', choices=['liberal', 'strict'])
    parser.add_argument('--methods', type=str, nargs='*', default=None,
                        help='refinements added as extra rows (default: family preset); MB always included')
    parser.add_argument('--per_size', action='store_true',
                        help='additionally print per-size agreement tables (no pooling)')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR))
    parser.add_argument('--decimals', type=int, default=3)
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
    lf = LF_FROM_FAM[family]
    posthoc = cfg.methods if cfg.methods is not None else posthocDefaults(lf)
    methods = ['mb'] + validMethods(posthoc, lf)

    conditions = CONDITIONS[family]
    if cfg.conditions is not None:
        known = dict(conditions)
        unknown = [t for t in cfg.conditions if t not in known]
        if unknown:
            raise KeyError(f'unknown condition tags for family {family}: {unknown}')
        conditions = [(t, known[t]) for t in cfg.conditions]
    labels = dict(conditions)
    sizes = [s for s in cfg.sizes if BEST_SEEDS.get((FAMILY_NAMES[family], s)) is not None]

    per_cond: dict[str, list[dict]] = {}
    for tag, _ in conditions:
        collected = [
            r
            for r in (collectCondition(cfg, family, s, tag, device, methods) for s in sizes)
            if r is not None
        ]
        if collected:
            per_cond[tag] = collected
    if not per_cond:
        logger.error('No conditions evaluated.')
        return

    pooled_agree = {tag: _poolAgree(per) for tag, per in per_cond.items()}
    pooled_entries = {
        tag: {
            label: {kind: _poolEntries(per, label, kind) for kind in ('local', 'global')}
            for label in per[0]['entries']
        }
        for tag, per in per_cond.items()
    }

    rows_agree = agreementRows(pooled_agree, labels, methods)
    agree_md = renderAgreementMd(rows_agree, dp=cfg.decimals)
    print('\n=== MB↔NUTS agreement by misspecification severity (median ± MAD, converged) ===\n')
    print(agree_md)

    rows_quality = qualityRows(pooled_entries, pooled_agree, labels, methods)
    quality_md = renderQualityMd(rows_quality, dp=cfg.decimals)
    print('\n=== Quality vs generating parameters by severity (converged subset) ===\n')
    print(quality_md)

    md = [
        f'# Likelihood misspecification ({family})\n',
        f'Sizes: {", ".join(sizes)}. Reference: NUTS ({cfg.convergence_mode}). '
        f'All metrics on the NUTS-converged subset (paired).\n',
        '## MB↔NUTS agreement by severity (median ± MAD over converged datasets)\n',
        agree_md,
        '',
        '## Quality vs generating parameters by severity\n',
        'Global (ffx, σ_rfx' + (', σ_ε' if lf == 0 else '') + ') and local (rfx) parameters; '
        'EACE→0 calibrated, cov90 nominal 0.900, LOO-NLL mean per dataset. Under contamination '
        "the generating parameters differ from the misspecified model's pseudo-true values, so "
        'absolute degradation is expected — the comparison is MB vs NUTS.\n',
        quality_md,
        '',
    ]

    if cfg.per_size:
        for size in sizes:
            sized = {
                tag: _poolAgree([r for r in per if r['agree']['size'][0] == size])
                for tag, per in per_cond.items()
                if any(r['agree']['size'][0] == size for r in per)
            }
            if not sized:
                continue
            tbl = renderAgreementMd(agreementRows(sized, labels, methods), dp=cfg.decimals)
            print(f'\n=== Agreement, {size} only ===\n')
            print(tbl)
            md += [f'## Agreement, {size} only\n', tbl, '']

    stem = f'likelihood_misspec_{family}'
    (outdir / f'{stem}.md').write_text('\n'.join(md) + '\n')
    (outdir / f'{stem}.tex').write_text(
        renderAgreementTex(rows_agree, dp=cfg.decimals)
        + '\n'
        + renderQualityTex(rows_quality, dp=cfg.decimals)
    )
    logger.info('Saved tables to %s', outdir / f'{stem}.md')


if __name__ == '__main__':
    main()
