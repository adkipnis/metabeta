"""Paired accuracy and cost of random-walk (RW, posthoc/metropolis.py LocalMoves) settings.

Per batch the default IMH runs once; every RW variant then starts from the same IMH states on
the datasets the gate selects (IMH min bulk ESS < 150), so the differences
between variants carry no IMH noise. Each variant's globals are scored against the reference
NUTS fit like experiments/posthoc/rw_gate.py (90% width ratio, |Δmedian| / NUTS sd, and the
Gaussian-implied 90% coverage error), and the RW's own wall time per batch is recorded.

Variants are '+'-joined overrides of the '+moves' RW (posterior_eval.MOVES_RW): C<n> (chains), K<n> (IS² draws per group),
T<n> (thinning), B<x> (burn-in steps per active global dimension). The first variant is the
reference of the paired differences. One CSV per set goes to
metabeta/outputs/results/rw_variants/ (resumable).

    uv run python experiments/posthoc/rw_variants.py --device cuda --sources oracle real
    uv run python experiments/posthoc/rw_variants.py --stage analyze
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'posthoc'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402
from imh_moves import (
    FAMILY_INITIAL,
    LIKELIHOOD_FAMILY,
    loadModel,
    loadReference,
    makeBatches,
    refDraws,
)  # noqa: E402
from rw_gate import Reference, coverageError, globalNames, poolSize, scoreBatch  # noqa: E402

from metabeta.models.api import IMH_BURNIN, IMH_N_CHAINS  # noqa: E402
from metabeta.posthoc.metropolis import IS2_N_INNER, LocalMoves, MetropolisSampler  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped, toDevice  # noqa: E402
from metabeta.utils.device import synchronizeDevice  # noqa: E402
from metabeta.utils.posterior_eval import MOVES_RW  # noqa: E402
from metabeta.utils.sampling import setSeed  # noqa: E402

OUT_DIR = REPO_ROOT / 'metabeta' / 'outputs' / 'results' / 'rw_variants'
LOW_ESS = 10  # analyze: datasets below it are reported apart
SETS = {
    'oracle': [(f, s) for f in LIKELIHOOD_FAMILY for s in ('large', 'huge')],
    'real': [('normal', 'medium'), ('normal', 'large')]
    + [('bernoulli', s) for s in ('medium', 'large', 'huge')]
    + [('poisson', s) for s in ('medium', 'large')],
}
TOKENS = {
    'C': ('n_chains', int),
    'K': ('n_inner', int),
    'T': ('thin', int),
    'B': ('burnin_per_dim', float),
}


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['run', 'analyze'], default='run')
    p.add_argument('--sources', nargs='+', default=['oracle', 'real'], choices=list(SETS))
    p.add_argument('--sets', nargs='+', default=None, help='family_size keys (default: all of the sources)')
    p.add_argument('--variants', nargs='+', default=['C500+K4', 'C250+K2', 'C250+K4', 'C250+K8', 'C500+K2'])
    p.add_argument('--n-samples', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=str(OUT_DIR))
    return p.parse_args()
# fmt: on


def parseVariant(name: str) -> LocalMoves:
    moves = MOVES_RW  # the '+moves' RW; tokens override it
    for tok in name.split('+'):
        key, cast = TOKENS[tok[0]]
        moves = replace(moves, **{key: cast(tok[1:])})
    return moves


@torch.no_grad()
def runSet(source: str, family: str, size: str, cfg: argparse.Namespace) -> None:
    out_path = Path(cfg.out_dir) / f'{source}_{family}_{size}.csv'
    if out_path.exists():
        print(f'[{out_path.stem}] done', flush=True)
        return
    lf, dev = LIKELIHOOD_FAMILY[family], torch.device(cfg.device)
    kind = 'sampled' if source == 'oracle' else 'real'
    data_path = DATA_DIR / f'{size}-{FAMILY_INITIAL[family]}-{kind}' / 'test.npz'
    model = loadModel(_ckpt_dir(family, size, BEST_SEEDS[(family, size)]) / 'latest.pt', cfg.device)
    col = Collection(data_path, permute=False)
    items = [col[i] for i in range(len(col))]
    ref = loadReference(data_path, len(items), lf)
    variants = {v: parseVariant(v) for v in cfg.variants}
    setSeed(cfg.seed)
    rows = []
    for idx in makeBatches(items, cfg.batch_size, poolSize(cfg.n_samples)):
        batch = toDevice(dict(collateGrouped([items[k] for k in idx])), cfg.device)
        stats = model._dataStatistics(batch) if model.analytical_context else None
        pool = model.estimate(batch, n_samples=poolSize(cfg.n_samples), stats=stats, local=False)
        sampler = MetropolisSampler(
            batch,
            n_chains=IMH_N_CHAINS,
            n_steps=pool.n_samples // IMH_N_CHAINS,
            burnin=IMH_BURNIN,
            mode='marginal' if lf == 0 else 'laplace',
            likelihood_family=lf,
            n_inner=0 if lf == 0 else IS2_N_INNER,
        )
        log_w = sampler._logWeights(pool)
        sg, _, pool_idx, accept = sampler._runChains(log_w, pool.samples_g, None)
        state = sampler._laplaceState(pool_idx) if lf != 0 else {}
        names = [globalNames(int(items[k]['d']), int(items[k]['q']), lf, pool) for k in idx]
        draws = [{nm: refDraws(ref, k, nm) for nm in ns if nm} for k, ns in zip(idx, names)]
        r = Reference(draws, ref['converged'][idx])
        for v, moves in variants.items():
            sampler.moves = moves
            synchronizeDevice(dev)
            t0 = time.perf_counter()
            sg_v, _, diag = sampler._applyMoves(pool, sg, state)
            synchronizeDevice(dev)
            sec = time.perf_counter() - t0
            moved = diag['moved'].numpy()
            res = {
                'imh': sg.float().cpu(),
                'rw': sg_v.float().cpu(),
                'ess': diag['ess'].numpy(),
                'accept': accept.mean(-1).cpu().numpy(),
                'move_accept': diag['move_accept'].numpy(),
                'khat': np.full(len(idx), np.nan),
            }
            for row in scoreBatch(res, names, r):
                i = row.pop('local')
                rows.append(row | {
                    'dataset': idx[i], 'variant': v, 'moved': bool(moved[i]),
                    'rw_seconds_per_moved': sec / max(moved.sum(), 1),
                })  # fmt: skip
    df = pd.DataFrame(rows).assign(source=source, family=family, size=size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'[{out_path.stem}] {df[df.moved].dataset.nunique()} moved datasets', flush=True)


def analyze(cfg: argparse.Namespace) -> None:
    """Per set, ESS band (below LOW_ESS or not) and variant, over the moved NUTS-converged
    datasets: mean coverage error of the globals, share with width ratio < 0.8, shift q95, the
    paired difference of the per-dataset mean coverage error to the reference variant (mean ± se)
    and the RW seconds per moved dataset; then the same pooled per source and family kind (sets
    weigh equally)."""
    paths = sorted(Path(cfg.out_dir).glob('*_*_*.csv'))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df['moved'] & df['ref_converged'].astype(bool)].copy()
    df['set'] = df['source'] + '_' + df['family'] + '_' + df['size']
    df['band'] = np.where(df['ess_min'] < LOW_ESS, f'ess<{LOW_ESS}', f'ess>={LOW_ESS}')
    df['err'] = coverageError(df['rw_wr'], df['rw_shift'])
    keys = ['set', 'band']
    per_ds = df.groupby(keys + ['dataset', 'variant'])['err'].mean().unstack('variant')
    diff = per_ds.sub(per_ds[cfg.variants[0]], axis=0)
    g = df.groupby(keys + ['variant'])
    t = pd.DataFrame({
        'n_ds': g['dataset'].nunique(),
        'cov_err': g['err'].mean(),
        'wr<0.8': g['rw_wr'].apply(lambda s: (s < 0.8).mean()),
        'shift_q95': g['rw_shift'].quantile(0.95),
        'd_err': diff.groupby(keys).mean().stack(),
        'd_err_se': diff.groupby(keys).sem().stack(),
        'rw_s': df.groupby(keys + ['variant', 'dataset'])['rw_seconds_per_moved'].first()
        .groupby(keys + ['variant']).mean(),
    })  # fmt: skip
    t = t[t['n_ds'] >= 5]
    lines = [t.round(4).to_markdown()]
    t = t.reset_index()
    kind = np.where(t['set'].str.contains('normal'), 'normal', 'glmm')
    t['group'] = t['set'].str.split('_').str[0] + '_' + kind
    pg = t.groupby(['group', 'band', 'variant'])
    pooled = pg[['cov_err', 'wr<0.8', 'd_err', 'rw_s']].mean()
    pooled['d_err_se'] = pg['d_err_se'].apply(lambda s: np.sqrt((s**2).sum()) / len(s))
    lines.append(pooled.round(4).to_markdown())
    text = '\n\n'.join(lines)
    print(text)
    (Path(cfg.out_dir) / 'analysis.md').write_text(text + '\n')


if __name__ == '__main__':
    cfg = setup()
    if cfg.stage == 'analyze':
        analyze(cfg)
        sys.exit()
    for source in cfg.sources:
        for family, size in SETS[source]:
            if cfg.sets is None or f'{family}_{size}' in cfg.sets:
                runSet(source, family, size, cfg)
