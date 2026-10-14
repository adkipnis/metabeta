"""Oracle comparison of the IMH under-coverage remedy: local random-walk moves after the IMH.

For every (family, size) oracle test set, draw the flow pool (4 × 1,025 = 4,100 draws, the
Api layout: 4,000 kept IMH draws), refine it with the default IMH (imhMarginal on Normal,
the pseudo-marginal imhPM on GLMMs), optionally followed by LocalMoves
(posthoc/metropolis.py), and score each dataset's global parameters against

  truth   whether the central 50/90/95% interval holds the simulated value (and the rfx
          90% coverage, averaged over the dataset's groups),
  NUTS    the reference fit (test.nuts2.npz) where it converged: 90% interval width ratio
          and |Δmedian| / NUTS sd,

all in the standardized space the model and the fits live in (the scale cancels in every
score). Also records IMH acceptance, the IMH min bulk ESS (the gate), whether moves ran,
their acceptance and ESS, and the wall time per batch.

Variants are 'imh' (baseline) or '+'-joined LocalMoves tokens: 'gate' (moves where the IMH
min bulk ESS < IMH_ESS_WARN), 'moves' (moves on every dataset), and overrides of the defaults
'B<x>' (burn-in steps per active global dimension), 'C<n>' (chains), 'T<n>' (thinning),
'K<n>' (IS² draws per group). Example: gate+T1+B3.

One CSV row per (variant, dataset, parameter) goes to
metabeta/outputs/results/imh_moves/{family}_{size}.csv (resumable: finished variants are
skipped), and `--stage table` prints the per-set summary.

    uv run python experiments/posthoc/imh_moves.py --sizes small --families poisson --n-datasets 16
    uv run python experiments/posthoc/imh_moves.py --device cuda --variants imh gate gate+T1+B3
    uv run python experiments/posthoc/imh_moves.py --stage table
"""

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

from metabeta.models.api import IMH_BURNIN, IMH_ESS_WARN, IMH_N_CHAINS  # noqa: E402
from metabeta.models.approximator import Approximator  # noqa: E402
from metabeta.posthoc.metropolis import IS2_N_INNER, LocalMoves, MetropolisSampler  # noqa: E402
from metabeta.utils.config import ApproximatorConfig  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped, toDevice  # noqa: E402
from metabeta.utils.device import synchronizeDevice  # noqa: E402
from metabeta.utils.evaluation import referenceConverged  # noqa: E402
from metabeta.utils.fits import REFERENCE_TAG, fitPath, loadFits  # noqa: E402
from metabeta.utils.posterior_eval import REFINE_MAX_ELEMENTS  # noqa: E402
from metabeta.utils.sampling import setSeed  # noqa: E402

FAMILY_INITIAL = {'normal': 'n', 'bernoulli': 'b', 'poisson': 'p'}
LIKELIHOOD_FAMILY = {'normal': 0, 'bernoulli': 1, 'poisson': 2}
OUT_DIR = REPO_ROOT / 'metabeta' / 'outputs' / 'results' / 'imh_moves'
LEVELS = (0.5, 0.9, 0.95)


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['run', 'table'], default='run')
    p.add_argument('--sizes', nargs='+', default=['small', 'medium', 'large', 'huge'])
    p.add_argument('--families', nargs='+', default=['normal', 'bernoulli', 'poisson'])
    p.add_argument('--variants', nargs='+', default=['imh', 'gate'])
    p.add_argument('--n-samples', type=int, default=4000, help='kept IMH draws per dataset')
    p.add_argument('--n-datasets', type=int, default=None, help='cap per set (default: all)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--prefix', default='latest')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=str(OUT_DIR))
    return p.parse_args()
# fmt: on


def parseVariant(name: str, cfg: argparse.Namespace) -> LocalMoves | None:
    if name == 'imh':
        return None
    fields = {
        'B': ('burnin_per_dim', float),
        'C': ('n_chains', int),
        'T': ('thin', int),
        'K': ('n_inner', int),
    }
    moves = LocalMoves()
    for tok in name.split('+'):
        if tok in ('gate', 'moves'):
            moves = replace(moves, gate_ess=float(IMH_ESS_WARN) if tok == 'gate' else None)
        elif tok[:1] in fields and tok[1:]:
            key, cast = fields[tok[0]]
            moves = replace(moves, **{key: cast(tok[1:])})
        else:
            raise ValueError(f'unknown variant token {tok!r} in {name!r}')
    return moves


def loadModel(ckpt: Path, device: str) -> Approximator:
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = Approximator(ApproximatorConfig(**payload['model_cfg']))
    model.load_state_dict(payload['model_state'])
    return model.eval().to(device)


def loadReference(data_path: Path, n: int, lf: int) -> dict[str, np.ndarray] | None:
    """Reference NUTS globals (n, D, draws) in the model's layout + convergence mask."""
    if not fitPath(data_path, REFERENCE_TAG).exists():
        return None
    t = REFERENCE_TAG
    keys = [f'{t}_ffx', f'{t}_sigma_rfx'] + ([f'{t}_sigma_eps'] if lf == 0 else [])
    fits = loadFits(data_path, t, keys=keys)
    conv = referenceConverged(data_path)
    return {
        'ffx': fits[f'{t}_ffx'][:n].astype(np.float32),
        'sigma_rfx': fits[f'{t}_sigma_rfx'][:n].astype(np.float32),
        'sigma_eps': fits[f'{t}_sigma_eps'][:n, 0].astype(np.float32) if lf == 0 else None,
        'converged': conv[:n] if conv is not None else np.ones(n, dtype=bool),
    }


# ==============================================================================
# One variant on one batch
# ==============================================================================


@torch.no_grad()
def runBatch(model, batch: dict, lf: int, cfg, moves: LocalMoves | None):
    """Flow pool → IMH (→ moves) on ``batch`` (on cfg.device). Returns (proposal, diag, seconds)."""
    dev = torch.device(cfg.device)
    n_steps = math.ceil(cfg.n_samples / IMH_N_CHAINS) + IMH_BURNIN
    synchronizeDevice(dev)
    t0 = time.perf_counter()
    stats = model._dataStatistics(batch) if model.analytical_context else None
    pool = model.estimate(batch, n_samples=IMH_N_CHAINS * n_steps, stats=stats, local=False)
    sampler = MetropolisSampler(
        batch,
        n_chains=IMH_N_CHAINS,
        n_steps=n_steps,
        burnin=IMH_BURNIN,
        mode='marginal' if lf == 0 else 'laplace',
        likelihood_family=lf,
        n_inner=0 if lf == 0 else IS2_N_INNER,
        moves=moves,
    )
    out, diag = sampler(pool)
    synchronizeDevice(dev)
    seconds = time.perf_counter() - t0
    out.to('cpu')
    return out, diag, seconds


def scoreBatch(out, diag, batch: dict, ref: dict | None, idx: list[int], n_keep: int) -> list[dict]:
    """Per (dataset, parameter) rows: coverage vs truth, width/shift vs the NUTS reference."""
    b = batch['X'].shape[0]
    d, q = out.d, out.q
    sg = out.samples_g[:, :n_keep].float()  # (b, S, D_g)
    has_eps = out.has_sigma_eps
    truth = [batch['ffx'][:, :d], batch['sigma_rfx'][:, :q]]
    if has_eps:
        truth.append(batch['sigma_eps'].reshape(b, 1))
    truth = torch.cat(truth, dim=-1).cpu().float()
    active = [batch['mask_d'][:, :d], batch['mask_q'][:, :q]]
    if has_eps:
        active.append(torch.ones(b, 1, dtype=torch.bool))
    active = torch.cat([a.cpu().bool() for a in active], dim=-1)
    names = [f'ffx{i}' for i in range(d)] + [f'sigma_rfx{i}' for i in range(q)]
    names += ['sigma_eps'] if has_eps else []
    D = len(names)
    x = sg[..., :D].numpy()

    qs = sorted({0.5} | {a for lv in LEVELS for a in ((1 - lv) / 2, (1 + lv) / 2)})
    quant = np.quantile(x, qs, axis=1)  # (n_q, b, D)
    qi = {round(v, 4): i for i, v in enumerate(qs)}

    # rfx 90% coverage per dataset, averaged over its active (group, slot) entries
    rl = out.samples_l[:, :, :n_keep].float()
    lo, hi = torch.quantile(rl, torch.tensor([0.05, 0.95]), dim=2)  # (b, m, q)
    rfx_true = batch['rfx'][:, : rl.shape[1], :q].cpu().float()
    mask_mq = batch['mask_mq'][:, : rl.shape[1], :q].cpu().bool()
    inside = ((rfx_true >= lo) & (rfx_true <= hi) & mask_mq).sum((1, 2))
    rfx_cov = (inside / mask_mq.sum((1, 2)).clamp(min=1)).numpy()

    accept = diag['accept_rate'].mean(-1).cpu().numpy()
    rows = []
    for i in range(b):
        k = idx[i]
        base = {'dataset': k, 'accept': float(accept[i]), 'rfx_cov90': float(rfx_cov[i])}
        if 'moved' in diag:
            base['moved'] = bool(diag['moved'][i])
            base['move_accept'] = float(diag['move_accept'][i])
            base['ess_min'] = float(np.nanmin(diag['ess'][i].numpy()))
        for j, name in enumerate(names):
            if not active[i, j]:
                continue
            t = float(truth[i, j])
            row = dict(base, param=name, truth=t, median=float(quant[qi[0.5], i, j]))
            for lv in LEVELS:
                a, z = (
                    quant[qi[round((1 - lv) / 2, 4)], i, j],
                    quant[qi[round((1 + lv) / 2, 4)], i, j],
                )
                row[f'cov{int(lv * 100)}'] = float(a <= t <= z)
            row['width90'] = float(quant[qi[0.95], i, j] - quant[qi[0.05], i, j])
            if ref is not None:
                r = refDraws(ref, k, name)
                row['ref_converged'] = bool(ref['converged'][k])
                r05, r50, r95 = np.quantile(r, [0.05, 0.5, 0.95])
                row['width_ratio'] = row['width90'] / max(r95 - r05, 1e-12)
                row['shift_sd'] = abs(row['median'] - r50) / max(r.std(), 1e-12)
            rows.append(row)
    return rows


def scoreReference(items: list, ref: dict, lf: int) -> list[dict]:
    """The reference fit's own coverage of the simulated truth (variant 'nuts2')."""
    rows = []
    for k, it in enumerate(items):
        d, q = int(it['d']), int(it['q'])
        names = [f'ffx{j}' for j in range(d)] + [f'sigma_rfx{j}' for j in range(q)]
        truth = list(np.asarray(it['ffx'])[:d]) + list(np.asarray(it['sigma_rfx'])[:q])
        if lf == 0:
            names.append('sigma_eps')
            truth.append(float(np.asarray(it['sigma_eps']).reshape(-1)[0]))
        for name, t in zip(names, truth):
            r = refDraws(ref, k, name)
            row = {'dataset': k, 'param': name, 'truth': float(t), 'median': float(np.median(r))}
            for lv in LEVELS:
                a, z = np.quantile(r, [(1 - lv) / 2, (1 + lv) / 2])
                row[f'cov{int(lv * 100)}'] = float(a <= t <= z)
            row['width90'] = float(np.quantile(r, 0.95) - np.quantile(r, 0.05))
            row |= {'ref_converged': bool(ref['converged'][k]), 'width_ratio': 1.0, 'shift_sd': 0.0}
            rows.append(row)
    return rows


def refDraws(ref: dict, k: int, name: str) -> np.ndarray:
    if name == 'sigma_eps':
        return ref['sigma_eps'][k]
    key, j = ('ffx', int(name[3:])) if name.startswith('ffx') else ('sigma_rfx', int(name[9:]))
    return ref[key][k, j]


# ==============================================================================
# Stages
# ==============================================================================


def makeBatches(items: list, batch_size: int, pool: int) -> list[list[int]]:
    """Dataset indices grouped by size (less padding), at most batch_size per batch and at
    most REFINE_MAX_ELEMENTS padded (dataset × group × observation × draw) entries."""
    shape = [(int(it['m']), int(it['ns'].max())) for it in items]
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in sorted(range(len(items)), key=lambda j: shape[j][0] * shape[j][1]):
        cand = cur + [i]
        m = max(shape[j][0] for j in cand)
        n = max(shape[j][1] for j in cand)
        if cur and (len(cand) > batch_size or len(cand) * m * n * pool > REFINE_MAX_ELEMENTS):
            batches.append(cur)
            cand = [i]
        cur = cand
    return batches + [cur] if cur else batches


def runSet(family: str, size: str, cfg: argparse.Namespace) -> None:
    seed = BEST_SEEDS.get((family, size))
    if seed is None:
        print(f'[SKIP] no checkpoint seed for ({family}, {size})')
        return
    lf = LIKELIHOOD_FAMILY[family]
    data_path = DATA_DIR / f'{size}-{FAMILY_INITIAL[family]}-sampled' / 'test.npz'
    out_path = Path(cfg.out_dir) / f'{family}_{size}.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prev = pd.read_csv(out_path) if out_path.exists() else None
    done = set(prev['variant']) if prev is not None else set()
    todo = [v for v in cfg.variants if v not in done]
    if not todo:
        print(f'[{family} {size}] all variants done')
        return

    model = loadModel(_ckpt_dir(family, size, seed) / f'{cfg.prefix}.pt', cfg.device)
    col = Collection(data_path, permute=False)
    n = len(col) if cfg.n_datasets is None else min(cfg.n_datasets, len(col))
    items = [col[i] for i in range(n)]
    ref = loadReference(data_path, n, lf)
    print(f'[{family} {size}] {n} datasets, reference: {"yes" if ref else "no"}', flush=True)

    if ref is not None and REFERENCE_TAG not in done:
        df = pd.DataFrame(scoreReference(items, ref, lf)).assign(
            variant=REFERENCE_TAG, family=family, size=size
        )
        prev = df if prev is None else pd.concat([prev, df], ignore_index=True)
        prev.to_csv(out_path, index=False)

    # untimed warm-up (lazy init, first analytical fit)
    runBatch(model, toDevice(collateGrouped(items[:1]), cfg.device), lf, cfg, None)
    pool = IMH_N_CHAINS * (math.ceil(cfg.n_samples / IMH_N_CHAINS) + IMH_BURNIN)
    batches = makeBatches(items, cfg.batch_size, pool)
    for variant in todo:
        moves = parseVariant(variant, cfg)
        setSeed(cfg.seed)
        rows, wall = [], 0.0
        for idx in batches:
            batch = collateGrouped([items[i] for i in idx])
            out, diag, sec = runBatch(model, toDevice(dict(batch), cfg.device), lf, cfg, moves)
            wall += sec
            b = batch['X'].shape[0]
            for r in scoreBatch(out, diag, batch, ref, idx, cfg.n_samples):
                r['batch_seconds_per_ds'] = sec / b
                rows.append(r)
        df = pd.DataFrame(rows).assign(variant=variant, family=family, size=size)
        prev = df if prev is None else pd.concat([prev, df], ignore_index=True)
        prev.to_csv(out_path, index=False)
        print(f'  {variant}: {wall / n:.3f} s/dataset', flush=True)
        print(summarize(df).to_string(), flush=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per variant: coverage vs truth by parameter class, NUTS ratios, gate, time. With a
    reference fit, everything is restricted to the datasets where it converged."""
    df = df.assign(cls=df['param'].str.replace(r'\d+$', '', regex=True))
    for col in ('moved', 'move_accept', 'ess_min', 'accept', 'rfx_cov90', 'batch_seconds_per_ds'):
        if col not in df:
            df = df.assign(**{col: np.nan})
    if 'ref_converged' in df:
        df = df[df['ref_converged'].fillna(False).astype(bool)]
    per_ds = df.groupby(['variant', 'dataset']).agg(
        accept=('accept', 'first'),
        moved=('moved', 'first'),
        move_accept=('move_accept', 'first'),
        ess=('ess_min', 'first'),
        sec=('batch_seconds_per_ds', 'first'),
        rfx_cov90=('rfx_cov90', 'first'),
    )
    out = df.pivot_table(index='variant', columns='cls', values='cov90', aggfunc='mean')
    out.columns = [f'cov90_{c}' for c in out.columns]
    out['cov50'] = df.groupby('variant')['cov50'].mean()
    out['cov95'] = df.groupby('variant')['cov95'].mean()
    g = per_ds.groupby('variant')
    out['n_ds'] = g.size()
    out['rfx_cov90'] = g['rfx_cov90'].mean()
    out['accept'] = g['accept'].mean()
    out['moved'] = g['moved'].mean()
    out['move_acc'] = g['move_accept'].median()
    moved = per_ds[per_ds['moved'].fillna(False).astype(bool)]
    out['ess_moved_med'] = moved.groupby('variant')['ess'].median()
    out['s_per_ds'] = g['sec'].mean()
    if 'width_ratio' in df:
        gw = df.groupby('variant')
        out['wr_median'] = gw['width_ratio'].median()
        out['wr_q05'] = gw['width_ratio'].quantile(0.05)
        out['wr_lt_0.8'] = gw['width_ratio'].apply(lambda s: (s < 0.8).mean())
        out['wr_gt_1.25'] = gw['width_ratio'].apply(lambda s: (s > 1.25).mean())
        out['shift_median'] = gw['shift_sd'].median()
        out['shift_q95'] = gw['shift_sd'].quantile(0.95)
    return out.round(3)


def table(cfg: argparse.Namespace) -> None:
    for path in sorted(Path(cfg.out_dir).glob('*.csv')):
        print(f'\n## {path.stem}')
        print(summarize(pd.read_csv(path)).to_markdown())


if __name__ == '__main__':
    cfg = setup()
    if cfg.stage == 'table':
        table(cfg)
        sys.exit()
    for family in cfg.families:
        for size in cfg.sizes:
            runSet(family, size, cfg)
