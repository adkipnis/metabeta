"""Posterior quality in the data-poor / degenerate regime (the reviewer's n≪p and n≈p).

The reviewer asks how the amortized posterior behaves when data are scarce relative to the
number of parameters, and whether it is still useful there.  In a hierarchical model "n vs p"
is ambiguous, so we report it under the full range of defensible parameter counts:

  local ρ  = n_i / (d + q)   per group: data points per coefficient in that group's own
                             regression.  ρ < 1 IS the local n < p regime, and it governs the
                             *local* parameters (the random effects).
  global γ = N / p           per dataset, with p_all = d + m·q + q(q+1)/2 + 1(σ_ε): every
                             random effect counted as a free parameter (most adversarial).
                             Even so γ almost never drops below 1 — a structural property of
                             GLMMs (identifying the q×q RE covariance needs m ≳ q(q+1)/2
                             groups) — so the honest n<p question is per-group (local ρ).

On sampled data we have ground-truth θ, so per ρ / γ bin we score three posteriors — raw
amortized (MB), MB refined by independent Metropolis–Hastings (MB+IMH), and NUTS — against
the same ground truth, on complementary quality dimensions:

  EACE ↓    expected absolute calibration error: mean over credible levels of
            |empirical coverage − nominal|, aggregated across the ensemble.
  NRMSE ↓   normalised RMSE of the posterior mean (recovery accuracy).
  cov90     empirical coverage of the 90% credible interval (interpretable anchor).
  LOO-NLL ↓ leave-one-out predictive negative log-likelihood per dataset (γ table only).

NUTS is scored on its converged subset only (% NUTS-converged per bin marks where the
reference itself degrades).  All metrics reuse metabeta.evaluation functions verbatim; we
only condition them on ρ / γ by restricting the active-parameter masks.

Usage (from repo root):
    uv run python experiments/evaluation/data_poverty.py --family n
    uv run python experiments/evaluation/data_poverty.py --family n --no-predictive
    uv run python experiments/evaluation/data_poverty.py --family n --distribution_only
"""

import argparse
import logging
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
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
from metabeta.utils.posterior_eval import (
    fit2proposal,
    loadModel,
    loadOrComputeSummary,
    loadOrRefine,
    loadOrSampleMB,
    posthocDefaults,
)
from metabeta.utils.results import Proposal
from metabeta.evaluation.intervals import getCredibleIntervals
from metabeta.evaluation.point import getPointEstimates
from metabeta.evaluation.summary import EvaluationSummary

# condition_number.py sits in this directory (on sys.path[0] at run time).
from condition_number import SIZE_MODELS, LF_FROM_FAM

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR
DEFAULT_SIZES = ['small', 'medium', 'large', 'huge']

RHO_EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]
GAMMA_EDGES = [0.0, 2.0, 4.0, 8.0, np.inf]

# Credible levels: full grid for EACE, plus 90% as an interpretable coverage anchor.
ECE_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
COV90_ALPHA = 0.1
EPS = 1e-6

MB_LABEL, IMH_LABEL, NUTS_LABEL = 'MB', 'MB+IMH', 'NUTS'


# ---------------------------------------------------------------------------
# Data-to-parameter ratios


def perGroupRho(batch: dict[str, torch.Tensor]) -> np.ndarray:
    """ρ = n_i / (d + q) for every (dataset, group); NaN off active groups.  Returns (B, m)."""
    mask_n = batch['mask_n']
    npg = mask_n.sum(-1).float()
    gm = mask_n.any(-1)
    B = mask_n.shape[0]
    d = (
        batch['mask_d'].sum(-1).float()
        if 'mask_d' in batch
        else torch.full((B,), batch['X'].shape[-1])
    )
    q = batch['mask_q'].sum(-1).float() if 'mask_q' in batch else torch.zeros(B)
    rho = (npg / (d + q).clamp_min(1.0)[:, None]).numpy()
    rho[~gm.numpy()] = np.nan
    return rho


def datasetGamma(batch: dict[str, torch.Tensor], count: str, lf: int) -> np.ndarray:
    """γ = N / p per dataset.  count='all' (every RE free) or 'pop' (population params only)."""
    mask_n = batch['mask_n']
    B = mask_n.shape[0]
    N = mask_n.sum((-1, -2)).float()
    m = mask_n.any(-1).sum(-1).float()
    d = (
        batch['mask_d'].sum(-1).float()
        if 'mask_d' in batch
        else torch.full((B,), batch['X'].shape[-1])
    )
    q = batch['mask_q'].sum(-1).float() if 'mask_q' in batch else torch.zeros(B)
    cov = q * (q + 1) / 2
    resid = 1.0 if lf == 0 else 0.0
    if count == 'all':
        p = d + m * q + cov + resid
    elif count == 'pop':
        p = d + cov + resid
    else:
        raise ValueError(f'unknown count={count!r}')
    return (N / p.clamp_min(1.0)).numpy()


def datasetRho(batch: dict[str, torch.Tensor]) -> np.ndarray:
    """Per-dataset summary of local scarcity: the median active-group ρ.  Returns (B,)."""
    with np.errstate(invalid='ignore'):
        return np.nanmedian(perGroupRho(batch), axis=1)


# ---------------------------------------------------------------------------
# Per-entry extraction (reuses library CIs / point estimates, flattened over active params)


def _flat(t: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return t[mask].numpy()


def _insideStack(cis, key, gt, mask, unsqueeze: bool) -> np.ndarray:
    """(E, A) 0/1 inside-CI indicators over active entries for the EACE grid."""
    cols = []
    for a in ECE_ALPHAS:
        ci = cis[a][key].unsqueeze(-1) if unsqueeze else cis[a][key]
        cols.append(_flat((ci[..., 0, :] - EPS <= gt) & (gt <= ci[..., 1, :] + EPS), mask))
    return np.stack(cols, axis=1)


def _cov90(cis, key, gt, mask, unsqueeze: bool) -> np.ndarray:
    ci = cis[COV90_ALPHA][key].unsqueeze(-1) if unsqueeze else cis[COV90_ALPHA][key]
    return _flat((ci[..., 0, :] - EPS <= gt) & (gt <= ci[..., 1, :] + EPS), mask).astype(float)


def localEntries(proposal, batch, conv: np.ndarray) -> dict[str, np.ndarray]:
    """Per active random-effect entry: inside/α (E,A), sq-error, gt, ρ, conv."""
    mask_mq = batch['mask_mq'].bool()
    gt = batch['rfx']
    est = getPointEstimates(proposal, 'mean')['rfx']
    cis = getCredibleIntervals(proposal, alphas=ECE_ALPHAS + [COV90_ALPHA])
    rho = np.broadcast_to(perGroupRho(batch)[:, :, None], mask_mq.shape)[mask_mq.numpy()]
    cnv = np.broadcast_to(conv[:, None, None], mask_mq.shape)[mask_mq.numpy()]
    return {
        'inside': _insideStack(cis, 'rfx', gt, mask_mq, False),
        'cov90': _cov90(cis, 'rfx', gt, mask_mq, False),
        'se': _flat((est - gt).square(), mask_mq),
        'gt': _flat(gt, mask_mq),
        'ratio': rho,
        'conv': cnv,
    }


def globalEntries(proposal, batch, conv: np.ndarray, lf: int) -> dict[str, np.ndarray]:
    """Per active global-parameter entry (ffx, σ_rfx, σ_ε): inside/α, sq-error, gt, γ, ptype, conv."""
    est = getPointEstimates(proposal, 'mean')
    cis = getCredibleIntervals(proposal, alphas=ECE_ALPHAS + [COV90_ALPHA])
    gamma = datasetGamma(batch, 'all', lf)
    B = batch['ffx'].shape[0]
    specs = [
        ('ffx', 0, batch['ffx'], batch['mask_d'].bool(), False),
        ('sigma_rfx', 1, batch['sigma_rfx'], batch['mask_q'].bool(), False),
    ]
    if lf == 0 and proposal.has_sigma_eps:
        specs.append(
            (
                'sigma_eps',
                2,
                batch['sigma_eps'].unsqueeze(-1),
                torch.ones(B, 1, dtype=torch.bool),
                True,
            )
        )

    cols = {k: [] for k in ('inside', 'cov90', 'se', 'gt', 'ratio', 'ptype', 'conv')}
    for name, pid, gt, mask, unsq in specs:
        e = est['sigma_eps'].unsqueeze(-1) if name == 'sigma_eps' else est[name]
        cols['inside'].append(_insideStack(cis, name, gt, mask, unsq))
        cols['cov90'].append(_cov90(cis, name, gt, mask, unsq))
        cols['se'].append(_flat((e - gt).square(), mask))
        cols['gt'].append(_flat(gt, mask))
        cols['ratio'].append(np.broadcast_to(gamma[:, None], mask.shape)[mask.numpy()])
        cols['conv'].append(np.broadcast_to(conv[:, None], mask.shape)[mask.numpy()])
        cols['ptype'].append(np.full(int(mask.sum()), pid))
    return {k: np.concatenate(v, axis=0) for k, v in cols.items()}


def _sliceDraws(p: Proposal, s: int) -> Proposal:
    """Proposal restricted to its first ``s`` draws (samples only; fit proposals carry no
    log_prob, so this mirrors fit2proposal rather than posterior_eval._sliceSamples)."""
    data = {
        'global': {'samples': p.samples_g[:, :s].contiguous()},
        'local': {'samples': p.samples_l[:, :, :s].contiguous()},
    }
    corr = p._corr_rfx[:, :s].contiguous() if p._corr_rfx is not None else None
    return Proposal(data, has_sigma_eps=p.has_sigma_eps, d_corr=p.d_corr, corr_rfx=corr)


def _nutsSummary(data_path: Path, prop, batch, lf: int, chunk: int):
    """NUTS LOO is checkpoint-independent and already cached full-set as summary_test_nuts.pt;
    load it directly (25 min/size to recompute) and only fall back to computing if absent."""
    p = data_path.parent / 'summary_test_nuts.pt'
    if p.exists():
        try:
            s = EvaluationSummary.load(p)
            if s.per_dataset.loo_nll.shape[0] == batch['X'].shape[0]:
                logger.info('Loaded cached full-set NUTS summary %s', p)
                return s
            logger.warning('%s: NUTS summary length mismatch — recomputing', p)
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.warning('%s: NUTS summary load failed (%s) — recomputing', p, exc)
    return loadOrComputeSummary(
        prop, batch, data_path, 'nuts', None, lf, True, summary_chunk_size=chunk
    )


def datasetLoo(summary, batch, conv: np.ndarray, lf: int) -> dict[str, np.ndarray]:
    """Per-dataset LOO predictive NLL, with the dataset's γ and NUTS-converged flag."""
    return {
        'loo': summary.per_dataset.loo_nll.float().numpy(),
        'ratio': datasetGamma(batch, 'all', lf),
        'conv': conv.astype(bool),
    }


# ---------------------------------------------------------------------------
# Per-bin metric aggregation


def _eace(inside: np.ndarray) -> float:
    if inside.shape[0] == 0:
        return float('nan')
    cov = inside.mean(0)
    return float(np.mean(np.abs(cov - np.array([1 - a for a in ECE_ALPHAS]))))


def _nrmse(se: np.ndarray, gt: np.ndarray) -> float:
    if se.shape[0] == 0:
        return float('nan')
    sd = gt.std()
    return float(np.sqrt(se.mean()) / sd) if sd > 0 else float('nan')


def _nrmseByType(e: dict[str, np.ndarray], sel: np.ndarray) -> float:
    vals = [
        _nrmse(e['se'][sel & (e['ptype'] == pid)], e['gt'][sel & (e['ptype'] == pid)])
        for pid in np.unique(e['ptype'][sel])
    ]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float('nan')


def binnedRows(
    methods: dict[str, dict],
    loos: dict[str, dict] | None,
    edges: list[float],
    is_global: bool,
    unit_ratio: np.ndarray,
    unit_conv: np.ndarray,
) -> list[dict]:
    """One row per (ratio bin × method).  NUTS rows use its converged subset.

    Metrics aggregate over active parameter entries (methods[*]['ratio']); the reported
    counts are over *units* — groups (local) or datasets (global) — via unit_ratio/unit_conv,
    so the count column is unambiguous and never collides with the paper's n / n_i.
    """
    ref = methods[MB_LABEL]
    ratio = ref['ratio']
    loo_ratio = loos[MB_LABEL]['ratio'] if loos else None
    unit_conv = unit_conv.astype(bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = np.isfinite(ratio) & (ratio >= lo) & (ratio < hi)
        if not in_bin.any():
            continue
        unit_in = np.isfinite(unit_ratio) & (unit_ratio >= lo) & (unit_ratio < hi)
        n = int(unit_in.sum())
        n_conv = int((unit_in & unit_conv).sum())
        loo_bin = None
        if loo_ratio is not None:
            loo_bin = np.isfinite(loo_ratio) & (loo_ratio >= lo) & (loo_ratio < hi)
        for j, (label, e) in enumerate(methods.items()):
            sel = (in_bin & e['conv'].astype(bool)) if label == NUTS_LABEL else in_bin
            nrmse = _nrmseByType(e, sel) if is_global else _nrmse(e['se'][sel], e['gt'][sel])
            loo_val = float('nan')
            if loo_bin is not None and label in loos:
                ld = loos[label]
                lsel = (loo_bin & ld['conv'].astype(bool)) if label == NUTS_LABEL else loo_bin
                vals = ld['loo'][lsel]
                loo_val = float(np.nanmean(vals)) if len(vals) else float('nan')
            rows.append(
                {
                    'bin': _edgeLabel(lo, hi),
                    'first': j == 0,
                    'method': label,
                    'n': n,
                    'n_conv': n_conv,
                    'eace': _eace(e['inside'][sel]),
                    'nrmse': nrmse,
                    'cov90': float(np.nanmean(e['cov90'][sel])) if sel.any() else float('nan'),
                    'loo': loo_val,
                }
            )
    return rows


def _edgeLabel(lo: float, hi: float) -> str:
    fmt = lambda x: '∞' if x == np.inf else f'{x:g}'
    return f'[{fmt(lo)},{fmt(hi)})'


def renderMetricMd(
    rows: list[dict], ratio_name: str, count_label: str, with_loo: bool, dp: int = 3
) -> str:
    headers = [
        ratio_name,
        count_label,
        f'{count_label} (NUTS-conv)',
        'method',
        'EACE↓',
        'NRMSE↓',
        'cov90',
    ]
    if with_loo:
        headers.append('LOO-NLL↓')
    fmt = lambda v: 'NA' if (v is None or v != v) else f'{v:.{dp}f}'
    md = []
    for r in rows:
        lead = [r['bin'], r['n'], r['n_conv']] if r['first'] else ['', '', '']
        row = lead + [r['method'], fmt(r['eace']), fmt(r['nrmse']), fmt(r['cov90'])]
        if with_loo:
            row.append(fmt(r['loo']))
        md.append(row)
    return tabulate(md, headers=headers, tablefmt='pipe', stralign='right')


def renderMetricTex(
    rows: list[dict], ratio_name: str, unit_tex: str, with_loo: bool, dp: int = 3
) -> str:
    fmt = lambda v: r'\textrm{NA}' if (v is None or v != v) else f'${v:.{dp}f}$'
    metric_hdr = r'\mathrm{EACE} & \mathrm{NRMSE} & \mathrm{cov}_{90}'
    if with_loo:
        metric_hdr += r' & \mathrm{LOO\text{-}NLL}'
    lines = [
        r'\begin{tabular}{lrr l' + ('ccc' if not with_loo else 'cccc') + '}',
        r'    \toprule',
        rf'    {ratio_name} & {unit_tex} & {unit_tex}_{{\mathrm{{conv}}}} & \mathrm{{model}} & {metric_hdr} \\',
        r'    \midrule',
    ]
    for i, r in enumerate(rows):
        if r['first'] and i != 0:
            lines.append(r'    \midrule')
        lead = (rf"\texttt{{{r['bin']}}}", r['n'], r['n_conv']) if r['first'] else ('', '', '')
        cells = ' & '.join(
            [fmt(r['eace']), fmt(r['nrmse']), fmt(r['cov90'])]
            + ([fmt(r['loo'])] if with_loo else [])
        )
        lines.append(
            rf"    {lead[0]} & {lead[1]} & {lead[2]} & \texttt{{{r['method']}}} & {cells} \\"
        )
    lines += [r'    \bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Distribution tables


def distributionTable(per_size: dict[str, np.ndarray], edges: list[float], unit_label: str) -> str:
    labels = [_edgeLabel(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    sizes = [s for s in DEFAULT_SIZES if s in per_size]
    pooled = np.concatenate([per_size[s] for s in sizes]) if sizes else np.empty(0)
    rows = []
    for s, vals in [*[(s, per_size[s]) for s in sizes], ('pooled', pooled)]:
        v = vals[np.isfinite(vals)]
        counts = [int(((v >= lo) & (v < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
        total = max(1, sum(counts))
        rows.append(
            [s, len(v)]
            + [f'{100*c/total:.1f}%' for c in counts]
            + [f'{v.min():.2f}' if len(v) else 'NA']
        )
    return tabulate(
        rows, headers=['size', unit_label, *labels, 'min'], tablefmt='pipe', stralign='right'
    )


# ---------------------------------------------------------------------------
# Per-size collection


def collectSize(cfg, size: str, device: torch.device) -> dict | None:
    family = cfg.family
    data_id = f'{size}-{family}-sampled'
    if family not in SIZE_MODELS or size not in SIZE_MODELS[family]:
        logger.warning('%s: no regime-matched checkpoint — skipping', data_id)
        return None
    ckpt_dir = CHECKPOINT_DIR / SIZE_MODELS[family][size]
    data_path = DATA_DIR / data_id / 'test.npz'
    if not data_path.exists() or not ckpt_dir.exists():
        logger.warning('%s: data or checkpoint missing — skipping', data_id)
        return None

    lf = LF_FROM_FAM[family]
    if cfg.distribution_only:
        col = Collection(data_path, permute=False)
        B = len(col)
        batch = collateGrouped([col[i] for i in range(B)])
        return {
            'size': size,
            'grp_rho': perGroupRho(batch)[batch['mask_n'].any(-1).numpy()],
            'gamma_all': datasetGamma(batch, 'all', lf),
        }

    model, model_cfg = loadModel(ckpt_dir, cfg.prefix, device)
    max_d, max_q, lf = model_cfg.max_d, model_cfg.max_q, model_cfg.likelihood_family
    col = Collection(data_path, permute=False, max_d=max_d, max_q=max_q)
    B = len(col)
    batch = collateGrouped([col[i] for i in range(B)])

    fit_path = data_path.with_name('test.fit.npz')
    col_fit = Collection(fit_path, permute=False, max_d=max_d, max_q=max_q)
    if len(col_fit) != B:
        raise ValueError(f'{data_id}: test.npz ({B}) and test.fit.npz ({len(col_fit)}) misaligned')
    fit_batch = collateGrouped([col_fit[i] for i in range(B)])
    if 'stats' in fit_batch and 'stats' not in batch:
        batch['stats'] = fit_batch['stats']
    conv = nutsConvergeMask(fit_batch, mode=cfg.convergence_mode)
    conv = np.ones(B, dtype=bool) if conv is None else conv.astype(bool)

    group_active = batch['mask_n'].any(-1).numpy()  # (B, m)
    grp_rho = perGroupRho(batch)[group_active]  # per active group
    grp_conv = np.broadcast_to(conv[:, None], group_active.shape)[group_active]
    gamma_all = datasetGamma(batch, 'all', lf)  # per dataset

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
    proposal_nuts = fit2proposal(fit_batch, 'nuts')
    # NUTS stores ~4000 draws; the rfx tensor (B, m, 4000, q) OOMs at large/huge when held
    # alongside MB/IMH. Subsample to n_samples draws — ample for coverage/quantiles/means.
    if proposal_nuts.n_samples > cfg.n_samples:
        proposal_nuts = _sliceDraws(proposal_nuts, cfg.n_samples)

    # Refinement, LOO-NLL and metrics all operate in the original (rescaled) space.
    proposal_mb.rescale(batch['sd_y'])
    proposal_nuts.rescale(batch['sd_y'])
    batch = rescaleData(batch)

    imh_method = posthocDefaults(lf)[0]  # imhMarginal (Normal) / imhLaplace (Bernoulli/Poisson)
    proposal_imh, _ = loadOrRefine(
        imh_method,
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

    proposals = {MB_LABEL: proposal_mb, IMH_LABEL: proposal_imh, NUTS_LABEL: proposal_nuts}
    method_names = {MB_LABEL: 'mb', IMH_LABEL: imh_method, NUTS_LABEL: 'nuts'}
    out = {
        'size': size,
        'grp_rho': grp_rho,
        'grp_conv': grp_conv,
        'gamma_all': gamma_all,
        'ds_conv': conv,
        'methods': {},
        'loos': {},
    }
    for label, prop in proposals.items():
        out['methods'][label] = {
            'local': localEntries(prop, batch, conv),
            'global': globalEntries(prop, batch, conv, lf),
        }
        if cfg.predictive:
            if label == NUTS_LABEL:
                summary = _nutsSummary(data_path, prop, batch, lf, cfg.summary_chunk_size)
            else:
                summary = loadOrComputeSummary(
                    prop,
                    batch,
                    data_path,
                    method_names[label],
                    None,
                    lf,
                    True,
                    ckpt_dir=ckpt_dir,
                    prefix=cfg.prefix,
                    n_samples=cfg.n_samples,
                    seed=cfg.seed,
                    summary_chunk_size=cfg.summary_chunk_size,
                )
            out['loos'][label] = datasetLoo(summary, batch, conv, lf)
    return out


def _poolMethod(per: list[dict], label: str, kind: str) -> dict[str, np.ndarray]:
    parts = [r['methods'][label][kind] for r in per]
    return {sk: np.concatenate([p[sk] for p in parts]) for sk in parts[0]}


def _poolLoo(per: list[dict], label: str) -> dict[str, np.ndarray]:
    parts = [r['loos'][label] for r in per if label in r['loos']]
    return {sk: np.concatenate([p[sk] for p in parts]) for sk in parts[0]}


# ---------------------------------------------------------------------------
# CLI / main


def setup() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(
        description='Posterior quality (EACE / NRMSE / LOO-NLL; MB, MB+IMH, NUTS) in the data-poor regime.',
    )
    parser.add_argument('--sizes', type=str, nargs='+', default=DEFAULT_SIZES, choices=DEFAULT_SIZES)
    parser.add_argument('--family', type=str, default='n', choices=list(SIZE_MODELS))
    parser.add_argument('--prefix', type=str, default='latest')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--summary_chunk_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--convergence_mode', type=str, default='strict', choices=['liberal', 'strict'])
    parser.add_argument('--predictive', action=argparse.BooleanOptionalAction, default=True,
                        help='Compute per-dataset LOO-NLL (cached; --no-predictive to skip).')
    parser.add_argument('--distribution_only', action='store_true')
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
    tag = f'{cfg.family}_sampled'

    per = [r for r in (collectSize(cfg, s, device) for s in cfg.sizes) if r is not None]
    if not per:
        logger.error('No sizes evaluated.')
        return

    md = [f'# Data-poverty posterior quality ({tag})\n']

    dist_rho = distributionTable({r['size']: r['grp_rho'] for r in per}, RHO_EDGES, '# groups')
    dist_gamma = distributionTable(
        {r['size']: r['gamma_all'] for r in per}, GAMMA_EDGES, '# datasets'
    )
    legend = (
        'Notation (paper convention): n_i = observations in group i; n = total observations in a '
        'dataset (= Σ_i n_i); d = # fixed effects; q = # random effects; '
        'p_all = d + m·q + q(q+1)/2 + 1 (every random effect counted as a free parameter).\n'
    )
    print('\n=== local ρ = n_i/(d+q) distribution (% of groups) ===\n' + dist_rho)
    print(
        '\n=== global γ = n/p_all distribution (% of datasets; p_all counts every RE) ===\n'
        + dist_gamma
    )
    md += [
        legend,
        '## Local ρ = n_i/(d+q) distribution (% of groups per ρ bin)\n',
        dist_rho,
        '',
        '## Global γ = n/p_all distribution (% of datasets per γ bin, adversarial count)\n',
        dist_gamma,
        '',
    ]

    if not cfg.distribution_only:
        labels = [MB_LABEL, IMH_LABEL, NUTS_LABEL]
        local = {lab: _poolMethod(per, lab, 'local') for lab in labels}
        glob = {lab: _poolMethod(per, lab, 'global') for lab in labels}
        loos = (
            {lab: _poolLoo(per, lab) for lab in labels}
            if cfg.predictive and per[0]['loos']
            else None
        )

        grp_rho = np.concatenate([r['grp_rho'] for r in per])
        grp_conv = np.concatenate([r['grp_conv'] for r in per])
        ds_gamma = np.concatenate([r['gamma_all'] for r in per])
        ds_conv = np.concatenate([r['ds_conv'] for r in per])
        rows_local = binnedRows(local, None, RHO_EDGES, False, grp_rho, grp_conv)
        rows_global = binnedRows(glob, loos, GAMMA_EDGES, True, ds_gamma, ds_conv)
        tbl_local = renderMetricMd(
            rows_local, 'ρ = n_i/(d+q)', '# groups', with_loo=False, dp=cfg.decimals
        )
        tbl_global = renderMetricMd(
            rows_global, 'γ = n/p_all', '# datasets', with_loo=loos is not None, dp=cfg.decimals
        )

        print(
            '\n=== Random-effect quality by per-group ρ  [n<p at ρ<1; nominal cov90=0.900] ===\n'
            + tbl_local
        )
        print('\n=== Global-parameter quality by dataset γ (adversarial p_all) ===\n' + tbl_global)
        md += [
            '## Random-effect quality by per-group ρ = n_i/(d+q)  (local n<p at ρ<1)\n',
            'Counts are # groups. EACE→0 = calibrated; NRMSE = recovery; cov90 nominal 0.900. '
            'NUTS uses its converged subset.\n',
            tbl_local,
            '',
            '## Global-parameter quality by dataset γ = n/p_all  (adversarial: every RE counted free)\n',
            'Counts are # datasets.\n',
            tbl_global,
            '',
        ]
        (outdir / f'data_poverty_{tag}.tex').write_text(
            renderMetricTex(
                rows_local, r'$\rho=n_i/(d+q)$', r'\#\mathrm{grp}', with_loo=False, dp=cfg.decimals
            )
            + '\n\n'
            + renderMetricTex(
                rows_global,
                r'$\gamma=n/p$',
                r'\#\mathrm{ds}',
                with_loo=loos is not None,
                dp=cfg.decimals,
            )
        )

    (outdir / f'data_poverty_{tag}.md').write_text('\n'.join(md) + '\n')
    logger.info('Saved tables to %s', outdir / f'data_poverty_{tag}.md')


if __name__ == '__main__':
    main()
