"""
compare_warmstart.py — Cold vs warm-started NUTS on real data.

Two claims
----------
Claim 1 (divergence reduction):
    warm_2000 vs cold_std — same tune/draws budget, fewer divergences and better R-hat

Claim 2 (speed):
    warm_500 / warm_250 vs cold_gold — posterior quality matches the gold standard
    with 4–8× fewer tuning steps

Conditions
----------
cold_std   prior  ta=0.80  tune=2000  draws=1000  chains=4  max_td=10  typical user
cold_gold  prior  ta=0.95  tune=4000  draws=2000  chains=4  max_td=10  reference
warm_2000  MB     ta=0.90  tune=2000  draws=1000  chains=4  max_td=12  Claim 1
warm_1000  MB     ta=0.90  tune=1000  draws=1000  chains=4  max_td=12  Claim 2 mid
warm_500   MB     ta=0.90  tune=500   draws=1000  chains=4  max_td=12  Claim 2
warm_250   MB     ta=0.90  tune=250   draws=1000  chains=4  max_td=12  Claim 2 lb

Metrics per dataset
-------------------
n_div      total divergences across chains
max_rhat   max R-hat across all parameters
min_ess    min bulk ESS across all parameters
min_ess_t  min tail ESS across all parameters
wall_s     wall time (tune + sampling)
ess_s      min_ess / wall_s  (efficiency)
agree      mean |μ_cond − μ_gold| / σ_gold over globals vs cold_gold
agree_r    Pearson r of per-parameter posterior means vs cold_gold

Fits are cached per condition × dataset in {data_dir}/fits_warm_{ckpt_name}/.
Rerun with --refit to ignore the cache.

Usage (from repo root):
    uv run python experiments/compare_warmstart.py \\
        --ckpt metabeta/outputs/checkpoints/normal_dsmall-n-mixed_mlarge-r_s0/best.pt
    uv run python experiments/compare_warmstart.py \\
        --ckpt ... --data_ids tiny-n-sampled small-n-sampled
    uv run python experiments/compare_warmstart.py \\
        --ckpt ... --n_datasets 8
    uv run python experiments/compare_warmstart.py \\
        --ckpt ... --conditions cold_std warm_500
    uv run python experiments/compare_warmstart.py \\
        --ckpt ... --refit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import torch
from tabulate import tabulate

from metabeta.evaluation.summary import getSummary
from metabeta.models.approximator import Approximator
from metabeta.posthoc.warmnuts import WarmNuts
from metabeta.simulation.fit import buildPymc, extractAll
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.evaluation import Proposal
from metabeta.utils.padding import padToModel, unpad

DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = DIR / '..' / 'metabeta' / 'outputs'
WN_FLOW_SAMPLES = 100
MB_BENCH_SAMPLES = 1000

DEFAULT_DATA_IDS = ['tiny-n-sampled', 'small-n-sampled', 'medium-n-sampled', 'large-n-sampled']


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


@dataclass
class Condition:
    label: str
    init: str  # 'cold' | 'warm'
    target_accept: float
    tune: int
    draws: int
    chains: int
    max_treedepth: int = 10


CONDITIONS: list[Condition] = [
    Condition('cold_std', 'cold', 0.80, 2000, 1000, 4, 10),
    Condition('cold_gold', 'cold', 0.95, 4000, 2000, 4, 10),
    Condition('warm_2000', 'warm', 0.90, 2000, 1000, 4, 12),
    Condition('warm_1000', 'warm', 0.90, 1000, 1000, 4, 12),
    Condition('warm_500', 'warm', 0.90, 500, 1000, 4, 12),
    Condition('warm_250', 'warm', 0.90, 250, 1000, 4, 12),
]
COND_BY_LABEL = {c.label: c for c in CONDITIONS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Cold vs warm-started NUTS.')
    p.add_argument('--ckpt',       type=Path, required=True,
                   help='Path to best.pt checkpoint (parent dir name used as ckpt_name)')
    p.add_argument('--data_ids',   nargs='+', default=DEFAULT_DATA_IDS,
                   help='Data directory names under metabeta/outputs/data/')
    p.add_argument('--n_datasets', type=int, default=16)
    p.add_argument('--conditions', nargs='+', default=[c.label for c in CONDITIONS],
                   choices=[c.label for c in CONDITIONS])
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--refit',        action='store_true', help='ignore cache')
    p.add_argument('--benchmark_mb', action='store_true',
                   help='benchmark MB inference on test.npz and cache timings')
    p.add_argument('--out_dir',    type=Path,
                   default=Path(__file__).resolve().parent / 'results' / 'warm_start',
                   help='Directory for markdown result files')
    return p.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------


def loadModel(ckpt: Path) -> Approximator:
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = ApproximatorConfig(**payload['model_cfg'])
    model = Approximator(cfg)
    model.load_state_dict(payload['model_state'])
    model.eval()
    return model


def loadData(
    path: Path, n_limit: int, max_d: int | None = None, max_q: int | None = None
) -> tuple[dict, list[dict]]:
    """Return (tensor_batch for model.estimate, list of unpadded numpy dicts for PyMC).

    Pass max_d/max_q from the model so the tensor batch is padded to model capacity
    even when the dataset's native dimensions are smaller.
    """
    col = Collection(path, permute=False, max_d=max_d, max_q=max_q)
    n = min(n_limit, len(col))
    tensor_batch = collateGrouped([col[i] for i in range(n)])
    ds_list = []
    for i in range(n):
        ds = {k: v[i] for k, v in col.raw.items()}
        ds_list.append(unpad(ds, {k: ds[k] for k in 'dqmn'}))
    return tensor_batch, ds_list


def loadFitDatasets(
    fit_path: Path, n_limit: int, max_d: int, max_q: int
) -> tuple[dict, list[dict]]:
    """Load datasets from a .fit.npz file, padded to model capacity.

    Datasets whose d or q exceed the model's capacity are skipped.
    Returns (tensor_batch, list of unpadded dicts) — same contract as loadData.
    """
    with np.load(fit_path, allow_pickle=True) as raw:
        raw = dict(raw)
    n_use = min(len(raw['d']), n_limit)
    col_items: list[dict] = []
    ds_list: list[dict] = []
    for i in range(n_use):
        ds = {k: v[i] for k, v in raw.items()}
        if int(ds['d']) > max_d or int(ds['q']) > max_q:
            continue
        ds_unpad = unpad(ds, {k: int(ds[k]) for k in 'dqmn'})
        ds_list.append(ds_unpad)
        col_items.append(padToModel(ds_unpad, max_d, max_q))
    tensor_batch = collateGrouped(col_items)
    return tensor_batch, ds_list


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cachePath(fits_dir: Path, cond_label: str, idx: int) -> Path:
    return fits_dir / f'{cond_label}__{idx:03d}.npz'


def _save(path: Path, samples: dict[str, np.ndarray], diag: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **samples, **{k: np.array(v) for k, v in diag.items()})


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as f:
        raw = dict(f)
    sample_keys = {'ffx', 'sigma_rfx', 'sigma_eps', 'rfx'}
    diag_keys = {'n_div', 'max_rhat', 'min_ess', 'min_ess_t', 'wall_s'}
    return (
        {k: raw[k] for k in sample_keys if k in raw},
        {k: float(raw[k]) for k in diag_keys if k in raw},
    )


def _benchmarkMB(
    model: Approximator,
    tensor_batch: dict,
    ds_list: list[dict],
    fits_dir: Path,
    n_samples: int = MB_BENCH_SAMPLES,
    refit: bool = False,
) -> None:
    """Run model.estimate per dataset, saving samples + wall_s as mb__{idx:03d}.npz."""
    to_run = [i for i in range(len(ds_list)) if refit or not _cachePath(fits_dir, 'mb', i).exists()]
    if not to_run:
        return
    warmup = {k: v[to_run[0] : to_run[0] + 1] for k, v in tensor_batch.items()}
    with torch.inference_mode():
        model.estimate(warmup, n_samples=n_samples)
    for i in to_run:
        ds = ds_list[i]
        d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
        batch = {k: v[i : i + 1] for k, v in tensor_batch.items()}
        with torch.inference_mode():
            t0 = time.perf_counter()
            proposal = model.estimate(batch, n_samples=n_samples)
            wall_s = time.perf_counter() - t0
        samples: dict[str, np.ndarray] = {
            'ffx': proposal.ffx[0, :, :d].numpy(),
            'sigma_rfx': proposal.sigma_rfx[0, :, :q].numpy(),
            'rfx': proposal.samples_l[0, :m, :, :q].permute(1, 0, 2).numpy(),
        }
        if proposal.has_sigma_eps:
            samples['sigma_eps'] = proposal.sigma_eps[0].numpy()
        _save(_cachePath(fits_dir, 'mb', i), samples, {'wall_s': wall_s})
        print(f'  ds={i:02d}  wall={wall_s:.3f}s', flush=True)


# ---------------------------------------------------------------------------
# Trace → samples / diagnostics
# ---------------------------------------------------------------------------


def _traceToSamples(trace: az.InferenceData, ds: dict) -> dict[str, np.ndarray]:
    d, q = int(ds['d']), int(ds['q'])
    out = extractAll(trace, ds, d, q, '_x')
    samples = {
        'ffx': out['_x_ffx'].T,  # (n_s, d)
        'sigma_rfx': out['_x_sigma_rfx'].T,  # (n_s, q)
        'rfx': out['_x_rfx'].transpose(2, 1, 0),  # (n_s, m, q)
    }
    if '_x_sigma_eps' in out:
        samples['sigma_eps'] = out['_x_sigma_eps'].squeeze(0)  # (n_s,)
    return samples


def _traceToDiag(trace: az.InferenceData, wall_s: float) -> dict:
    n_div = int(trace.sample_stats['diverging'].values.sum())
    try:
        df = az.summary(trace, kind='diagnostics')
        max_rhat = float(df['r_hat'].max())
        min_ess = float(df['ess_bulk'].min())
        min_ess_t = float(df['ess_tail'].min())
    except Exception:
        max_rhat = min_ess = min_ess_t = float('nan')
    return dict(n_div=n_div, max_rhat=max_rhat, min_ess=min_ess, min_ess_t=min_ess_t, wall_s=wall_s)


# ---------------------------------------------------------------------------
# Run one condition on one dataset
# ---------------------------------------------------------------------------


def runCold(
    ds: dict,
    cond: Condition,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    model = buildPymc(ds)
    t0 = time.perf_counter()
    with model:
        trace = pm.sample(
            tune=cond.tune,
            draws=cond.draws,
            chains=cond.chains,
            target_accept=cond.target_accept,
            nuts_kwargs={'max_treedepth': cond.max_treedepth},
            random_seed=seed,
            return_inferencedata=True,
            progressbar=False,
        )
    wall_s = time.perf_counter() - t0
    return _traceToSamples(trace, ds), _traceToDiag(trace, wall_s)


def runWarm(
    ds: dict,
    cond: Condition,
    proposal,
    b_idx: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    wn = WarmNuts(
        ds,
        n_chains=cond.chains,
        tune=cond.tune,
        draws=cond.draws,
        seed=seed,
        target_accept=cond.target_accept,
        max_treedepth=cond.max_treedepth,
    )
    t0 = time.perf_counter()
    wn_proposal, raw_diag = wn(proposal, b_idx=b_idx)
    wall_s = time.perf_counter() - t0

    d, q = wn_proposal.d, wn_proposal.q
    m = wn_proposal.samples_l.shape[1]
    samples: dict[str, np.ndarray] = {
        'ffx': wn_proposal.ffx[0, :, :d].numpy(),
        'sigma_rfx': wn_proposal.sigma_rfx[0, :, :q].numpy(),
        'rfx': wn_proposal.samples_l[0, :m, :, :q].permute(1, 0, 2).numpy(),
    }
    if wn_proposal.has_sigma_eps:
        samples['sigma_eps'] = wn_proposal.sigma_eps[0].numpy()

    diag = dict(
        n_div=float(raw_diag['n_divergences']),
        max_rhat=raw_diag['max_rhat'],
        min_ess=raw_diag.get('min_ess', float('nan')),
        min_ess_t=raw_diag.get('min_ess_t', float('nan')),
        wall_s=wall_s,
    )
    return samples, diag


# ---------------------------------------------------------------------------
# Posterior agreement vs cold_gold
# ---------------------------------------------------------------------------


def _globalMeans(samples: dict[str, np.ndarray]) -> np.ndarray:
    parts = [samples['ffx'].mean(0), samples['sigma_rfx'].mean(0)]
    if 'sigma_eps' in samples:
        parts.append(np.atleast_1d(samples['sigma_eps'].mean()))
    return np.concatenate(parts)


def _globalStds(samples: dict[str, np.ndarray]) -> np.ndarray:
    parts = [samples['ffx'].std(0), samples['sigma_rfx'].std(0)]
    if 'sigma_eps' in samples:
        parts.append(np.atleast_1d(samples['sigma_eps'].std()))
    return np.concatenate(parts)


def posteriorAgreement(
    cond_samples: dict[str, np.ndarray],
    gold_samples: dict[str, np.ndarray],
) -> tuple[float, float]:
    """Return (agree, agree_r) vs gold_samples.

    agree   mean |μ_cond − μ_gold| / max(σ_gold, 1e-6) over global params
    agree_r Pearson r of per-parameter posterior means
    """
    mu_cond = _globalMeans(cond_samples)
    mu_gold = _globalMeans(gold_samples)
    sd_gold = np.maximum(_globalStds(gold_samples), 1e-6)
    agree = float(np.mean(np.abs(mu_cond - mu_gold) / sd_gold))
    if len(mu_cond) >= 2:
        agree_r = float(np.corrcoef(mu_cond, mu_gold)[0, 1])
    else:
        agree_r = float('nan')
    return agree, agree_r


# ---------------------------------------------------------------------------
# Posterior quality evaluation (recovery, calibration, PSIS-LOO)
# ---------------------------------------------------------------------------


def buildProposal(
    sample_list: list[dict[str, np.ndarray]],
    d_ffx: int,
    d_rfx: int,
    m_max: int,
) -> Proposal:
    """Build a batched Proposal (B, S, ...) from a list of per-dataset sample dicts."""
    has_sigma_eps = 'sigma_eps' in sample_list[0]
    n_s = sample_list[0]['ffx'].shape[0]

    samples_g_parts: list[np.ndarray] = []
    samples_l_parts: list[np.ndarray] = []

    for s in sample_list:
        ffx = np.zeros((n_s, d_ffx), dtype=np.float32)
        ffx[:, : s['ffx'].shape[1]] = s['ffx']

        srfx = np.zeros((n_s, d_rfx), dtype=np.float32)
        srfx[:, : s['sigma_rfx'].shape[1]] = s['sigma_rfx']

        g_parts = [ffx, srfx]
        if has_sigma_eps:
            seps = s.get('sigma_eps', np.zeros(n_s, dtype=np.float32))
            g_parts.append(seps[:, np.newaxis])
        samples_g_parts.append(np.concatenate(g_parts, axis=-1))  # (n_s, D)

        # rfx saved as (n_s, m, q); Proposal wants (M_max, n_s, d_rfx)
        rfx = s['rfx']  # (n_s, m, q)
        n_s_i, m_i, q_i = rfx.shape
        rfx_padded = np.zeros((m_max, n_s, d_rfx), dtype=np.float32)
        rfx_padded[:m_i, :, :q_i] = rfx.transpose(1, 0, 2)
        samples_l_parts.append(rfx_padded)  # (M_max, n_s, d_rfx)

    samples_g = torch.from_numpy(np.stack(samples_g_parts).astype(np.float32))  # (B, n_s, D)
    samples_l = torch.from_numpy(np.stack(samples_l_parts).astype(np.float32))  # (B, M_max, n_s, d_rfx)

    proposed = {
        'global': {'samples': samples_g},
        'local': {'samples': samples_l},
    }
    return Proposal(proposed, has_sigma_eps=has_sigma_eps)


def _meanActive(
    metric_dict: dict[str, torch.Tensor],
    active_d: torch.Tensor,
    active_q: torch.Tensor,
    has_eps: bool,
) -> float:
    """Mean of a per-parameter metric over active dimensions only."""
    parts: list[torch.Tensor] = []
    if 'ffx' in metric_dict:
        parts.append(metric_dict['ffx'][active_d].float())
    if 'sigma_rfx' in metric_dict:
        parts.append(metric_dict['sigma_rfx'][active_q].float())
    if 'rfx' in metric_dict:
        parts.append(metric_dict['rfx'][active_q].float())
    if has_eps and 'sigma_eps' in metric_dict:
        v = metric_dict['sigma_eps'].float()
        parts.append(v.reshape(1) if v.dim() == 0 else v)
    if not parts:
        return float('nan')
    return float(torch.cat(parts).nanmean().item())


def computeQuality(
    cond_results: list[dict],
    tensor_batch: dict,
    likelihood_family: int,
    d_ffx: int,
    d_rfx: int,
) -> dict[str, float]:
    """Build Proposal from NUTS samples and run getSummary to get quality metrics."""
    sample_list = [r['samples'] for r in sorted(cond_results, key=lambda r: r['idx'])]
    m_max = int(tensor_batch['rfx'].shape[1])

    proposal = buildProposal(sample_list, d_ffx, d_rfx, m_max)
    summary = getSummary(
        proposal, tensor_batch, likelihood_family=likelihood_family, compute_pred_coverage=False
    )

    active_d = tensor_batch['mask_d'].any(0)
    active_q = tensor_batch['mask_q'].any(0)
    has_eps = 'sigma_eps' in summary.nrmse

    return {
        'r': _meanActive(summary.corr, active_d, active_q, has_eps),
        'nrmse': _meanActive(summary.nrmse, active_d, active_q, has_eps),
        'eace': _meanActive(summary.eace, active_d, active_q, has_eps),
        'loo_nll': summary.mloonll if summary.loo_nll is not None else float('nan'),
        'loo_k': summary.mloo_k if summary.loo_pareto_k is not None else float('nan'),
    }


def printQualityTable(quality: dict[str, dict[str, float]], active_conds: list[str]) -> str:
    rows = []
    for cond in active_conds:
        q = quality.get(cond)
        if q is None:
            continue
        rows.append([
            cond,
            f'{q["r"]:.3f}',
            f'{q["nrmse"]:.3f}',
            f'{q["eace"]:.3f}',
            f'{q["loo_nll"]:.3f}' if np.isfinite(q['loo_nll']) else '—',
            f'{q["loo_k"]:.3f}' if np.isfinite(q['loo_k']) else '—',
        ])
    headers = ['Condition', 'r ↑', 'NRMSE ↓', 'EACE ↓', 'LOO-NLL ↓', 'LOO k']
    print(tabulate(rows, headers=headers, tablefmt='simple'))
    return tabulate(rows, headers=headers, tablefmt='pipe')


def printQualityDiff(quality: dict[str, dict[str, float]], active_conds: list[str]) -> str | None:
    """Quality delta vs cold_gold — positive Δr and negative Δ(others) means no quality loss."""
    if 'cold_gold' not in quality:
        return None
    gold = quality['cold_gold']
    non_gold = [c for c in active_conds if c != 'cold_gold' and c in quality]
    if not non_gold:
        return None

    print('\nQuality vs cold_gold  (Δr ≈ 0 and Δothers ≈ 0 means quality preserved)')
    rows = []
    for cond in non_gold:
        q = quality[cond]
        rows.append([
            cond,
            f'{q["r"] - gold["r"]:+.3f}',
            f'{q["nrmse"] - gold["nrmse"]:+.3f}',
            f'{q["eace"] - gold["eace"]:+.3f}',
            f'{q["loo_nll"] - gold["loo_nll"]:+.3f}'
            if np.isfinite(q['loo_nll']) and np.isfinite(gold['loo_nll'])
            else '—',
        ])
    headers = ['Condition', 'Δr ↑', 'Δ NRMSE ↓', 'Δ EACE ↓', 'Δ LOO-NLL ↓']
    print(tabulate(rows, headers=headers, tablefmt='simple'))
    return tabulate(rows, headers=headers, tablefmt='pipe')


# ---------------------------------------------------------------------------
# Summary and analysis tables
# ---------------------------------------------------------------------------


def _iqr(x: np.ndarray) -> tuple[float, float, float]:
    return float(np.nanmedian(x)), float(np.nanpercentile(x, 25)), float(np.nanpercentile(x, 75))


def _fmt(med: float, p25: float, p75: float, decimals: int = 3) -> str:
    fmt = f'{{:.{decimals}f}}'
    return f'{fmt.format(med)} [{fmt.format(p25)}–{fmt.format(p75)}]'


def printSummary(results: list[dict], active_conds: list[str]) -> str:
    metrics = [
        ('n_div', 'Divergences', 0),
        ('max_rhat', 'Max R-hat', 3),
        ('min_ess', 'Min ESS bulk', 0),
        ('min_ess_t', 'Min ESS tail', 0),
        ('wall_s', 'Wall time (s)', 1),
        ('ess_s', 'ESS / s', 2),
        ('agree', 'Agree (norm Δμ)', 3),
        ('agree_r', 'Agree (r)', 3),
    ]
    rows = []
    for cond_label in active_conds:
        vals = [r for r in results if r['cond'] == cond_label]
        if not vals:
            continue
        row = [cond_label]
        for key, _, dec in metrics:
            arr = np.array([v.get(key, float('nan')) for v in vals], dtype=float)
            med, p25, p75 = _iqr(arr)
            row.append('—' if np.isnan(med) else _fmt(med, p25, p75, dec))
        rows.append(row)
    headers = ['Condition'] + [m[1] for m in metrics]
    print(tabulate(rows, headers=headers, tablefmt='simple'))
    return tabulate(rows, headers=headers, tablefmt='pipe')


def printAnalysis(results: list[dict], active_conds: list[str]) -> str | None:
    """Print warm vs cold_std comparison: direction indicates improvement."""
    if 'cold_std' not in active_conds:
        return None
    cold_results = {r['idx']: r for r in results if r['cond'] == 'cold_std'}
    if not cold_results:
        return None

    warm_conds = [c for c in active_conds if c.startswith('warm_')]
    if not warm_conds:
        return None

    print('\nWarm vs cold_std  (positive Δ = improvement over cold baseline)')
    rows = []
    for cond_label in warm_conds:
        warm_rs = [r for r in results if r['cond'] == cond_label]
        if not warm_rs:
            continue
        paired = [(r, cold_results[r['idx']]) for r in warm_rs if r['idx'] in cold_results]
        if not paired:
            continue

        delta_div = np.array([c['n_div'] - w['n_div'] for w, c in paired])
        delta_rhat = np.array([c['max_rhat'] - w['max_rhat'] for w, c in paired])
        delta_ess = np.array([w['min_ess'] - c['min_ess'] for w, c in paired])
        speedup = np.array([c['wall_s'] / max(w['wall_s'], 1e-3) for w, c in paired])
        delta_ess_s = np.array([w.get('ess_s', float('nan')) - c.get('ess_s', float('nan'))
                                for w, c in paired])

        rows.append([
            cond_label,
            _fmt(*_iqr(delta_div), 0),
            _fmt(*_iqr(delta_rhat), 3),
            _fmt(*_iqr(delta_ess), 0),
            _fmt(*_iqr(speedup), 2),
            _fmt(*_iqr(delta_ess_s), 2),
        ])

    headers = ['Condition', 'Δ Div ↑', 'Δ R-hat ↑', 'Δ ESS ↑', 'Speedup ×', 'Δ ESS/s ↑']
    print(tabulate(rows, headers=headers, tablefmt='simple'))
    return tabulate(rows, headers=headers, tablefmt='pipe')


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    ckpt = args.ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    ckpt_name = ckpt.parent.name

    active_conds = [COND_BY_LABEL[l] for l in args.conditions]
    needs_warm = any(c.init == 'warm' for c in active_conds)

    model = loadModel(ckpt) if (needs_warm or args.benchmark_mb) else None

    for data_id in args.data_ids:
        data_dir = (OUTPUTS_DIR / 'data' / data_id).resolve()
        fits_dir = data_dir / f'fits_warm_{ckpt_name}'

        if not data_dir.exists():
            print(f'[{data_id}] data directory not found: {data_dir} — skipping')
            continue

        print(f'\n{"#" * 70}')
        print(f'#  {data_id}  ckpt={ckpt_name}')
        print(f'{"#" * 70}')

        max_d = model.d_ffx if model is not None else None
        max_q = model.d_rfx if model is not None else None
        tensor_batch, ds_list = loadData(
            data_dir / 'valid.npz', args.n_datasets, max_d=max_d, max_q=max_q
        )
        n_ds = len(ds_list)

        if args.benchmark_mb and model is not None:
            test_fit_path = data_dir / 'test.fit.npz'
            if not test_fit_path.exists():
                print(f'  [mb] test.fit.npz not found — skipping')
            else:
                test_tb, test_ds = loadFitDatasets(test_fit_path, args.n_datasets, max_d, max_q)
                print(f'  [mb] benchmarking {len(test_ds)} test datasets...')
                _benchmarkMB(model, test_tb, test_ds, fits_dir, refit=args.refit)

        proposal = None
        if needs_warm and model is not None:
            print(f'Running flow ({WN_FLOW_SAMPLES} samples)...')
            with torch.inference_mode():
                proposal = model.estimate(tensor_batch, n_samples=WN_FLOW_SAMPLES)

        all_results: list[dict] = []

        for cond in active_conds:
            print(
                f'\n--- {cond.label}: {cond.init}  ta={cond.target_accept}  '
                f'tune={cond.tune}  draws={cond.draws} ---'
            )
            for idx, ds in enumerate(ds_list):
                cache = _cachePath(fits_dir, cond.label, idx)
                if cache.exists() and not args.refit:
                    samples, diag = _load(cache)
                elif cond.init == 'cold':
                    samples, diag = runCold(ds, cond, args.seed)
                    _save(cache, samples, diag)
                else:
                    assert proposal is not None
                    samples, diag = runWarm(ds, cond, proposal, idx, args.seed)
                    _save(cache, samples, diag)

                diag['ess_s'] = diag['min_ess'] / max(diag['wall_s'], 1e-3)
                all_results.append(
                    {'cond': cond.label, 'idx': idx, 'samples': samples, **diag}
                )
                print(
                    f'  ds={idx:02d}  div={diag["n_div"]:4.0f}  '
                    f'rhat={diag["max_rhat"]:.3f}  '
                    f'ess={diag["min_ess"]:.0f}  '
                    f'wall={diag["wall_s"]:.1f}s',
                    flush=True,
                )

        gold_by_idx: dict[int, dict] = {}
        if 'cold_gold' in args.conditions:
            for r in all_results:
                if r['cond'] == 'cold_gold':
                    gold_by_idx[r['idx']] = r['samples']

        if gold_by_idx:
            for r in all_results:
                if r['cond'] == 'cold_gold':
                    r['agree'] = 0.0
                    r['agree_r'] = 1.0
                    continue
                gold = gold_by_idx.get(r['idx'])
                if gold is not None:
                    r['agree'], r['agree_r'] = posteriorAgreement(r['samples'], gold)

        print(f'\n{"=" * 70}')
        print(f'Summary — {data_id}  ({n_ds} datasets)')
        print(f'{"=" * 70}')
        md_sections: list[str] = [f'# {data_id}  ({n_ds} datasets)  ckpt={ckpt_name}\n']
        md_sections.append('## MCMC diagnostics\n')
        md_sections.append(printSummary(all_results, args.conditions))

        analysis_md = printAnalysis(all_results, args.conditions)
        if analysis_md:
            md_sections.append('\n## Warm vs cold_std\n')
            md_sections.append(analysis_md)

        # Quality evaluation: recovery, calibration, PSIS-LOO
        d_ffx = model.d_ffx if model is not None else int(tensor_batch['mask_d'].shape[-1])
        d_rfx = model.d_rfx if model is not None else int(tensor_batch['mask_q'].shape[-1])
        lf = int(tensor_batch['likelihood_family'][0].item())

        print(f'\nQuality — {data_id}  (recovery / calibration / PSIS-LOO)')
        quality: dict[str, dict[str, float]] = {}
        for cond in active_conds:
            cond_results = [r for r in all_results if r['cond'] == cond.label]
            if not cond_results:
                continue
            print(f'  evaluating {cond.label}...', end=' ', flush=True)
            quality[cond.label] = computeQuality(cond_results, tensor_batch, lf, d_ffx, d_rfx)
            print('done')

        md_sections.append('\n## Posterior quality\n')
        md_sections.append(printQualityTable(quality, args.conditions))

        diff_md = printQualityDiff(quality, args.conditions)
        if diff_md:
            md_sections.append('\n## Quality vs cold_gold\n')
            md_sections.append(diff_md)

        mb_wall_s = [
            float(_load(p)[1]['wall_s'])
            for p in sorted(fits_dir.glob('mb__*.npz'))
        ]
        if mb_wall_s:
            arr = np.array(mb_wall_s)
            print(
                f'\nMB timings (test.npz, {len(arr)} datasets): '
                f'median={np.median(arr):.3f}s  '
                f'[{np.percentile(arr, 25):.3f}–{np.percentile(arr, 75):.3f}]'
            )

        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / f'{ckpt_name}__{data_id}__{n_ds}ds.md'
        md_path.write_text('\n'.join(md_sections) + '\n')
        print(f'\nSaved → {md_path}')


if __name__ == '__main__':
    run(setup())
