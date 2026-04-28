"""
ADVI / Pathfinder convergence experiment.

Compares:
  - ADVI with Adam     at learning rates [1e-2, 5e-3, 1e-3]
  - ADVI with Adagrad-window at learning rates [1e-2, 1e-3, 1e-4]
  - Pathfinder (L-BFGS, no LR / budget needed)

…on a random subset of tiny-n-toy test datasets, recording ELBO curves for
the ADVI variants and wall times for all methods.

Usage (from experiments/):
    python advi_elbo_convergence.py
    python advi_elbo_convergence.py --n_datasets 8 --viter 100000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pymc.variational.updates import adam, adagrad_window

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DIR = Path(__file__).resolve().parent
DATA_DIR = DIR / '..' / 'metabeta' / 'outputs' / 'data'
OUT_DIR = DIR / 'results'
BATCH_PATH = DATA_DIR / 'tiny-n-toy' / 'test.npz'

# ---------------------------------------------------------------------------
# Pathfinder loader (pymc-experimental, bypassing its broken __init__)
# ---------------------------------------------------------------------------

def _load_pathfinder():
    """Import fit_pathfinder without triggering pymc_experimental's broken __init__."""
    stub = type(sys)('stub')
    for mod in [
        'pymc_experimental',
        'pymc_experimental.statespace',
        'pymc_experimental.gp',
        'pymc_experimental.model',
        'pymc_experimental.utils',
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = stub
    pf_path = Path(
        '/sessions/modest-eager-bardeen/.local/lib/python3.10/'
        'site-packages/pymc_experimental/inference/pathfinder.py'
    )
    if not pf_path.exists():
        return None
    spec = importlib.util.spec_from_file_location('_pathfinder_mod', pf_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_pathfinder


# ---------------------------------------------------------------------------
# PyMC model builder
# ---------------------------------------------------------------------------

def build_model(ds: dict) -> pm.Model:
    """Build a Normal GLMM matching the paper's PyMC parametrisation."""
    n_obs = int(ds['n'])
    m     = int(ds['m'])
    d     = int(ds['d'])
    q     = int(ds['q'])
    correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2

    X = ds['X'][:n_obs].astype(np.float32).copy()
    y = ds['y'][:n_obs].astype(np.float32)
    Z = X[:, :q].copy()
    groups = ds['groups'][:n_obs].astype(int)

    if d > 1:
        means = X[:, 1:].mean(0)
        X[:, 1:] -= means
        Z[:, 1:] -= means[:q - 1]

    nu_ffx  = ds['nu_ffx'].astype(float)
    tau_ffx = ds['tau_ffx'].astype(float)
    tau_rfx = ds['tau_rfx'].astype(float)
    tau_eps = float(ds.get('tau_eps', 1.0))

    with pm.Model() as model:
        betas = [
            pm.Normal('Intercept' if j == 0 else f'x{j}',
                      mu=nu_ffx[j], sigma=tau_ffx[j])
            for j in range(d)
        ]
        if correlated:
            chol, _, _ = pm.LKJCholeskyCov(
                '_lkj_rfx', n=q, eta=float(ds['eta_rfx']),
                sd_dist=pm.HalfNormal.dist(sigma=tau_rfx),
                compute_corr=True,
            )
            z = pm.Normal('_rfx_offset', 0.0, 1.0, shape=(m, q))
            b = pm.Deterministic('_rfx', pt.dot(z, chol.T))
        else:
            cols = []
            for j in range(q):
                s = pm.HalfNormal(f'rfx_sigma_{j}', sigma=float(tau_rfx[j]))
                z = pm.Normal(f'rfx_offset_{j}', 0.0, 1.0, shape=m)
                cols.append(z * s)
            b = pt.stack(cols, axis=1)

        mu = sum(betas[j] * X[:, j] for j in range(d))
        mu = mu + (pt.as_tensor_variable(Z) * b[groups]).sum(axis=1)
        sigma_eps = pm.HalfNormal('sigma', sigma=tau_eps)
        pm.Normal('y_obs', mu=mu, sigma=sigma_eps, observed=y)

    return model


# ---------------------------------------------------------------------------
# ADVI runner
# ---------------------------------------------------------------------------

def fit_advi(
    model: pm.Model,
    n_iter: int,
    optimizer_fn,
    every: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (steps, elbo, wall_seconds)."""
    steps: list[int] = []
    elbo_vals: list[float] = []

    def _record(approx, hist, i):
        if i % every == 0:
            steps.append(i)
            elbo_vals.append(-float(hist[-1]))

    t0 = time.perf_counter()
    with model:
        pm.fit(
            n=n_iter,
            method='advi',
            obj_optimizer=optimizer_fn,
            callbacks=[_record],
            progressbar=False,
        )
    wall = time.perf_counter() - t0
    return np.array(steps, dtype=np.int64), np.array(elbo_vals, dtype=np.float64), wall


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def convergence_stats(elbo: np.ndarray, tail_frac: float = 0.2) -> dict:
    n_tail = max(2, int(len(elbo) * tail_frac))
    tail = elbo[-n_tail:]
    diffs = np.diff(tail)
    plateau_delta = float(np.mean(np.abs(diffs)))
    final_elbo = float(elbo[-1])
    max_elbo = float(np.max(elbo))
    plateau_rel = plateau_delta / max(abs(max_elbo), 1e-8)
    still_rising = float(np.mean(diffs)) > 0
    return dict(
        plateau_rel=plateau_rel,
        still_rising=still_rising,
        max_elbo=max_elbo,
        final_elbo=final_elbo,
    )


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

ADVI_CONFIGS = [
    # (label, optimizer_factory, color, linestyle)
    ('adam lr=1e-2',  lambda: adam(learning_rate=1e-2),          '#e05c47', '-'),
    ('adam lr=5e-3',  lambda: adam(learning_rate=5e-3),          '#e05c47', '--'),
    ('adam lr=1e-3',  lambda: adam(learning_rate=1e-3),          '#e05c47', ':'),
    ('agw lr=1e-2',   lambda: adagrad_window(learning_rate=1e-2), '#4a90d9', '-'),
    ('agw lr=1e-3',   lambda: adagrad_window(learning_rate=1e-3), '#4a90d9', '--'),
    ('agw lr=1e-4',   lambda: adagrad_window(learning_rate=1e-4), '#4a90d9', ':'),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_pathfinder = _load_pathfinder()
    if fit_pathfinder is None:
        print('WARNING: Pathfinder not available — skipping.')

    batch = dict(np.load(BATCH_PATH, allow_pickle=True))
    n_batch = len(batch['y'])

    # Diverse subset: balanced q=1 / q=2
    q_arr = batch['q']
    q1_idx = [i for i in range(n_batch) if q_arr[i] == 1][:args.n_datasets // 2]
    q2_idx = [i for i in range(n_batch) if q_arr[i] == 2][:args.n_datasets - len(q1_idx)]
    indices = sorted(q1_idx + q2_idx)

    print(f'Datasets: {indices}')
    print(f'ADVI budget: {args.viter} iters, recording every {args.every} steps')
    print(f'Configs: {len(ADVI_CONFIGS)} ADVI + {"Pathfinder" if fit_pathfinder else "no Pathfinder"}')
    print()

    all_results = []

    for idx in indices:
        ds = {k: v[idx] for k, v in batch.items()}
        n_obs = int(ds['n'])
        m     = int(ds['m'])
        d     = int(ds['d'])
        q     = int(ds['q'])
        corr  = float(ds.get('eta_rfx', 0)) > 0 and q >= 2
        print(f'=== Dataset {idx}: n={n_obs}, m={m}, d={d}, q={q}, corr={corr} ===')

        row = dict(idx=idx, n=n_obs, m=m, d=d, q=q, corr=corr, advi=[], pathfinder=None)

        # --- ADVI configs ---
        for label, opt_factory, color, ls in ADVI_CONFIGS:
            model = build_model(ds)
            steps, elbo, wall = fit_advi(model, args.viter, opt_factory(), args.every)
            stats = convergence_stats(elbo)
            row['advi'].append(dict(
                label=label, color=color, ls=ls,
                steps=steps, elbo=elbo, wall=wall, **stats,
            ))
            print(f'  {label:18s}  wall={wall:5.1f}s  '
                  f'final_ELBO={stats["final_elbo"]:8.1f}  '
                  f'plateau_rel={stats["plateau_rel"]:.4f}  '
                  f'rising={stats["still_rising"]}')

        # --- Pathfinder ---
        if fit_pathfinder is not None:
            model = build_model(ds)
            t0 = time.perf_counter()
            try:
                idata = fit_pathfinder(model=model, samples=4000, random_seed=42)
                pf_wall = time.perf_counter() - t0
                row['pathfinder'] = dict(wall=pf_wall, ok=True)
                print(f'  {"pathfinder":18s}  wall={pf_wall:5.1f}s  (no ELBO curve)')
            except Exception as e:
                pf_wall = time.perf_counter() - t0
                row['pathfinder'] = dict(wall=pf_wall, ok=False, error=str(e))
                print(f'  {"pathfinder":18s}  FAILED: {e}')
        print()
        all_results.append(row)

    # -----------------------------------------------------------------------
    # Plot: ELBO curves + wall-time bars
    # -----------------------------------------------------------------------
    n_ds = len(all_results)
    fig = plt.figure(figsize=(16, 3.2 * n_ds + 2))
    gs = GridSpec(n_ds + 1, 2, figure=fig,
                  height_ratios=[3] * n_ds + [2],
                  hspace=0.55, wspace=0.35)

    for row_i, row in enumerate(all_results):
        # --- ELBO curves ---
        ax_elbo = fig.add_subplot(gs[row_i, 0])
        best_final = max(r['final_elbo'] for r in row['advi'])
        for cfg in row['advi']:
            steps = cfg['steps']
            elbo  = cfg['elbo']
            k = max(1, len(elbo) // 40)
            smooth = np.convolve(elbo, np.ones(k) / k, mode='valid')
            steps_s = steps[k - 1:]
            ax_elbo.plot(steps, elbo, alpha=0.15, color=cfg['color'], linewidth=0.5)
            ax_elbo.plot(steps_s, smooth, color=cfg['color'], linestyle=cfg['ls'],
                         linewidth=1.6, label=cfg['label'])

        ax_elbo.set_title(
            f'ds={row["idx"]}  n={row["n"]}, m={row["m"]}, d={row["d"]}, q={row["q"]}',
            fontsize=8,
        )
        ax_elbo.set_xlabel('Iteration', fontsize=7)
        ax_elbo.set_ylabel('ELBO', fontsize=7)
        ax_elbo.tick_params(labelsize=7)
        ax_elbo.legend(fontsize=6, ncol=2, loc='lower right')

        # --- Convergence heatmap (plateau_rel per config) ---
        ax_heat = fig.add_subplot(gs[row_i, 1])
        labels  = [c['label'] for c in row['advi']]
        prel    = [c['plateau_rel'] for c in row['advi']]
        rising  = [c['still_rising'] for c in row['advi']]
        walls   = [c['wall'] for c in row['advi']]
        colors_bar = ['#2ca02c' if p < 0.005 and not r else
                      '#ff7f0e' if p < 0.02 else
                      '#d62728'
                      for p, r in zip(prel, rising)]
        bars = ax_heat.barh(range(len(labels)), prel, color=colors_bar, height=0.6)
        ax_heat.set_yticks(range(len(labels)))
        ax_heat.set_yticklabels(labels, fontsize=7)
        ax_heat.axvline(0.005, color='green', linestyle=':', linewidth=1, alpha=0.6,
                        label='converged (<0.005)')
        ax_heat.axvline(0.020, color='orange', linestyle=':', linewidth=1, alpha=0.6,
                        label='marginal (<0.02)')
        ax_heat.set_xlabel('plateau_rel (tail mean |ΔELBO| / |max ELBO|)', fontsize=7)
        ax_heat.set_title(f'Convergence  ds={row["idx"]}', fontsize=8)
        ax_heat.tick_params(labelsize=7)
        # annotate with wall time
        for i, (bar, w) in enumerate(zip(bars, walls)):
            ax_heat.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{w:.0f}s', va='center', fontsize=6, color='#333')
        ax_heat.legend(fontsize=6, loc='lower right')

    # --- Bottom: wall-time comparison across all datasets ---
    ax_time = fig.add_subplot(gs[n_ds, :])
    x = np.arange(len(all_results))
    width = 0.12
    config_labels = [c[0] for c in ADVI_CONFIGS]
    for ci, (label, _, color, ls) in enumerate(ADVI_CONFIGS):
        walls = [row['advi'][ci]['wall'] for row in all_results]
        offset = (ci - len(ADVI_CONFIGS) / 2 + 0.5) * width
        ax_time.bar(x + offset, walls, width, label=label,
                    color=color, alpha=0.5 + 0.15 * (ci % 3),
                    hatch=['', '//','..'][ci % 3])
    if any(row['pathfinder'] and row['pathfinder']['ok'] for row in all_results):
        pf_walls = [
            row['pathfinder']['wall'] if row['pathfinder'] and row['pathfinder']['ok'] else np.nan
            for row in all_results
        ]
        ax_time.bar(x + len(ADVI_CONFIGS) / 2 * width + width,
                    pf_walls, width, label='pathfinder', color='#9467bd', alpha=0.8)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([f'ds={r["idx"]}\nn={r["n"]},q={r["q"]}' for r in all_results],
                             fontsize=7)
    ax_time.set_ylabel('Wall time (s)', fontsize=8)
    ax_time.set_title('Wall-clock time by method and dataset', fontsize=9)
    ax_time.legend(fontsize=6, ncol=4)
    ax_time.tick_params(labelsize=7)

    fig.suptitle(
        f'ADVI convergence comparison  |  tiny-n-toy  |  {args.viter} ADVI iterations\n'
        'adam (red) vs adagrad_window / agw (blue) at 3 LRs each; Pathfinder (purple)',
        fontsize=10, y=1.005,
    )
    out_path = OUT_DIR / 'advi_elbo_convergence.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Figure saved → {out_path}')

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print('\n--- Convergence summary (plateau_rel; green < 0.005, orange < 0.02, red ≥ 0.02) ---')
    col_w = 10
    header = f'{"ds":>3}  {"n":>5}  {"q":>2}  ' + '  '.join(f'{c[0]:>{col_w}}' for c in ADVI_CONFIGS)
    if fit_pathfinder:
        header += f'  {"pathfinder":>{col_w}}'
    print(header)
    print('-' * len(header))
    for row in all_results:
        line = f'{row["idx"]:>3}  {row["n"]:>5}  {row["q"]:>2}  '
        for cfg in row['advi']:
            pr = cfg['plateau_rel']
            sr = cfg['still_rising']
            tag = '✓' if pr < 0.005 and not sr else '~' if pr < 0.02 else '✗'
            line += f'  {tag}{pr:.4f}'.rjust(col_w + 2)
        if row['pathfinder']:
            pf = row['pathfinder']
            line += f'  {"OK" if pf["ok"] else "ERR"} {pf["wall"]:.1f}s'.rjust(col_w + 2)
        print(line)

    # Overall verdict
    print('\n--- Best ADVI config per dataset ---')
    for row in all_results:
        best = min(row['advi'], key=lambda c: (c['still_rising'], c['plateau_rel']))
        print(f'  ds={row["idx"]}: {best["label"]}  '
              f'plateau_rel={best["plateau_rel"]:.4f}  '
              f'rising={best["still_rising"]}  '
              f'wall={best["wall"]:.1f}s')

    print()
    all_converged_adam = [
        c for row in all_results for c in row['advi']
        if 'adam' in c['label'] and c['plateau_rel'] < 0.005 and not c['still_rising']
    ]
    all_converged_agw = [
        c for row in all_results for c in row['advi']
        if 'agw' in c['label'] and c['plateau_rel'] < 0.005 and not c['still_rising']
    ]
    print(f'Configs reaching plateau_rel<0.005 and not rising:')
    print(f'  Adam:          {len(all_converged_adam)}/{len(all_results)*3} runs')
    print(f'  Adagrad-window:{len(all_converged_agw)}/{len(all_results)*3} runs')
    if fit_pathfinder:
        pf_ok = sum(1 for row in all_results if row['pathfinder'] and row['pathfinder']['ok'])
        pf_mean = np.mean([row['pathfinder']['wall'] for row in all_results
                           if row['pathfinder'] and row['pathfinder']['ok']])
        advi_mean = np.mean([c['wall'] for row in all_results for c in row['advi']])
        print(f'  Pathfinder:    {pf_ok}/{len(all_results)} OK, mean wall={pf_mean:.1f}s '
              f'(ADVI mean={advi_mean:.1f}s)')


# ---------------------------------------------------------------------------

def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='ADVI / Pathfinder convergence experiment')
    p.add_argument('--n_datasets', type=int, default=6)
    p.add_argument('--viter',      type=int, default=50_000)
    p.add_argument('--every',      type=int, default=200)
    return p.parse_args()


if __name__ == '__main__':
    run(setup())
