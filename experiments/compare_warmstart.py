"""
compare_warmstart.py — Cold vs warm-started NUTS on simulated data.

Two claims
----------
Claim 1 (divergence reduction):
    warm_2000 vs cold_std — same tune/draws budget, fewer divergences and better R-hat

Claim 2 (speed):
    warm_500 / warm_250 vs cold_gold — posterior quality matches the gold standard
    with 4-8× fewer tuning steps

Conditions
----------
cold_std   prior  ta=0.80  tune=2000  draws=1000  chains=4  max_td=10  typical user
cold_gold  prior  ta=0.95  tune=4000  draws=2000  chains=4  max_td=10  reference
warm_2000  MB     ta=0.90  tune=2000  draws=1000  chains=4  max_td=12  Claim 1
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

Results are cached per condition × dataset in results/warm_start/.
Rerun with --refit to ignore the cache.

Usage (from repo root):
    uv run python experiments/compare_warmstart.py
    uv run python experiments/compare_warmstart.py --n_datasets 8
    uv run python experiments/compare_warmstart.py --conditions cold_std warm_500
    uv run python experiments/compare_warmstart.py --refit
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

from metabeta.models.approximator import Approximator
from metabeta.posthoc.warmnuts import WarmNuts
from metabeta.simulation.fit import buildPymc, extractAll
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.dataloader import Collection, collateGrouped
from metabeta.utils.families import hasSigmaEps
from metabeta.utils.padding import unpad

DIR = Path(__file__).resolve().parent
RESULTS_DIR = DIR / 'results' / 'warm_start'
WN_FLOW_SAMPLES = 100


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    label: str
    init: str            # 'cold' | 'warm'
    target_accept: float
    tune: int
    draws: int
    chains: int
    max_treedepth: int = 10


CONDITIONS: list[Condition] = [
    Condition('cold_std',  'cold', 0.80, 2000, 1000, 4, 10),
    Condition('cold_gold', 'cold', 0.95, 4000, 2000, 4, 10),
    Condition('warm_2000', 'warm', 0.90, 2000, 1000, 4, 12),
    Condition('warm_500',  'warm', 0.90,  500, 1000, 4, 12),
    Condition('warm_250',  'warm', 0.90,  250, 1000, 4, 12),
]
COND_BY_LABEL = {c.label: c for c in CONDITIONS}


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODELS = [
    dict(
        label='Normal',
        ckpt=DIR / '..' / 'metabeta' / 'outputs' / 'checkpoints'
             / 'normal_dsmall-n-mixed_msmall_s42' / 'best.pt',
        data_dir=DIR / '..' / 'metabeta' / 'outputs' / 'data' / 'small-n-sampled',
        likelihood_family=0,
    ),
    dict(
        label='Bernoulli',
        ckpt=DIR / '..' / 'metabeta' / 'outputs' / 'checkpoints'
             / 'bernoulli_dsmall-b-mixed_msmall_s42' / 'best.pt',
        data_dir=DIR / '..' / 'metabeta' / 'outputs' / 'data' / 'small-b-sampled',
        likelihood_family=1,
    ),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Cold vs warm-started NUTS.')
    p.add_argument('--n_datasets',  type=int, default=16)
    p.add_argument('--conditions',  nargs='+', default=[c.label for c in CONDITIONS],
                   choices=[c.label for c in CONDITIONS])
    p.add_argument('--models',      nargs='+', default=['Normal', 'Bernoulli'],
                   choices=['Normal', 'Bernoulli'])
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--refit',       action='store_true', help='ignore cache')
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


def loadData(data_dir: Path, n_limit: int) -> tuple[dict, list[dict]]:
    """Return (tensor_batch for model.estimate, list of unpadded numpy dicts for PyMC)."""
    col = Collection(data_dir / 'valid.npz', permute=False)
    n = min(n_limit, len(col))
    tensor_batch = collateGrouped([col[i] for i in range(n)])
    ds_list = []
    for i in range(n):
        ds = {k: v[i] for k, v in col.raw.items()}
        ds_list.append(unpad(ds, {k: ds[k] for k in 'dqmn'}))
    return tensor_batch, ds_list


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cachePath(data_stem: str, ckpt_stem: str, cond_label: str, idx: int) -> Path:
    return RESULTS_DIR / f'{data_stem}__{ckpt_stem}__{cond_label}__{idx:03d}.npz'


def _save(path: Path, samples: dict[str, np.ndarray], diag: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------------------
# Trace → samples / diagnostics
# ---------------------------------------------------------------------------

def _traceToSamples(trace: az.InferenceData, ds: dict) -> dict[str, np.ndarray]:
    d, q = int(ds['d']), int(ds['q'])
    out = extractAll(trace, ds, d, q, '_x')
    samples = {
        'ffx':       out['_x_ffx'].T,                      # (n_s, d)
        'sigma_rfx': out['_x_sigma_rfx'].T,                # (n_s, q)
        'rfx':       out['_x_rfx'].transpose(2, 1, 0),     # (n_s, m, q)
    }
    if '_x_sigma_eps' in out:
        samples['sigma_eps'] = out['_x_sigma_eps'].squeeze(0)  # (n_s,)
    return samples


def _traceTodiag(trace: az.InferenceData, wall_s: float) -> dict:
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
    return _traceToSamples(trace, ds), _traceTodiag(trace, wall_s)


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
        'ffx':       wn_proposal.ffx[0, :, :d].numpy(),
        'sigma_rfx': wn_proposal.sigma_rfx[0, :, :q].numpy(),
        'rfx':       wn_proposal.samples_l[0, :m, :, :q].permute(1, 0, 2).numpy(),
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
    """Concatenate per-parameter posterior means into a single vector."""
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

    agree   mean |μ_cond − μ_gold| / max(σ_gold, 1e-6)  over global params
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
# Summary table
# ---------------------------------------------------------------------------

def _iqr(x: np.ndarray) -> tuple[float, float, float]:
    return float(np.nanmedian(x)), float(np.nanpercentile(x, 25)), float(np.nanpercentile(x, 75))


def printSummary(results: list[dict], active_conds: list[str]) -> None:
    metrics = [
        ('n_div',     'Divergences'),
        ('max_rhat',  'Max R-hat'),
        ('min_ess',   'Min ESS bulk'),
        ('min_ess_t', 'Min ESS tail'),
        ('wall_s',    'Wall time (s)'),
        ('ess_s',     'ESS / s'),
        ('agree',     'Agree (norm Δμ)'),
        ('agree_r',   'Agree (r)'),
    ]
    rows = []
    for cond_label in active_conds:
        vals = [r for r in results if r['cond'] == cond_label]
        if not vals:
            continue
        row = [cond_label]
        for key, _ in metrics:
            arr = np.array([v.get(key, float('nan')) for v in vals], dtype=float)
            med, p25, p75 = _iqr(arr)
            if np.isnan(med):
                row.append('—')
            else:
                row.append(f'{med:.3f} [{p25:.3f}–{p75:.3f}]')
        rows.append(row)
    headers = ['Condition'] + [m[1] for m in metrics]
    print(tabulate(rows, headers=headers, tablefmt='simple'))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    active_conds = [COND_BY_LABEL[l] for l in args.conditions]
    needs_warm = any(c.init == 'warm' for c in active_conds)
    needs_gold = any(c.label != 'cold_gold' for c in active_conds)  # for agree metric

    model_cfgs = [m for m in MODELS if m['label'] in args.models]

    for mcfg in model_cfgs:
        ckpt = Path(mcfg['ckpt'])
        data_dir = Path(mcfg['data_dir'])
        lf = mcfg['likelihood_family']

        if not ckpt.exists():
            print(f'[{mcfg["label"]}] checkpoint not found: {ckpt} — skipping')
            continue

        print(f'\n{"#" * 70}')
        print(f'#  {mcfg["label"]}  likelihood_family={lf}')
        print(f'{"#" * 70}')

        model = loadModel(ckpt) if needs_warm else None
        tensor_batch, ds_list = loadData(data_dir, args.n_datasets)
        n_ds = len(ds_list)

        data_stem = data_dir.name
        ckpt_stem = ckpt.parent.name

        # Flow proposal: computed once for all warm conditions
        proposal = None
        if needs_warm and model is not None:
            print(f'Running flow ({WN_FLOW_SAMPLES} samples)...')
            with torch.inference_mode():
                proposal = model.estimate(tensor_batch, n_samples=WN_FLOW_SAMPLES)

        all_results: list[dict] = []

        for cond in active_conds:
            print(f'\n--- {cond.label}: {cond.init}  ta={cond.target_accept}  '
                  f'tune={cond.tune}  draws={cond.draws} ---')
            for idx, ds in enumerate(ds_list):
                cache = _cachePath(data_stem, ckpt_stem, cond.label, idx)
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
                all_results.append({'cond': cond.label, 'idx': idx, 'samples': samples, **diag})
                print(f'  ds={idx:02d}  div={diag["n_div"]:4.0f}  '
                      f'rhat={diag["max_rhat"]:.3f}  '
                      f'ess={diag["min_ess"]:.0f}  '
                      f'wall={diag["wall_s"]:.1f}s', flush=True)

        # Posterior agreement: requires cold_gold samples for each dataset
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
        print(f'Summary — {mcfg["label"]}  ({n_ds} datasets)')
        print(f'{"=" * 70}')
        printSummary(all_results, args.conditions)


if __name__ == '__main__':
    run(setup())
