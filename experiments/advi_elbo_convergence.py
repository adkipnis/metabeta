"""
ADVI ELBO convergence experiment.

Runs ADVI on a handful of tiny-n-toy test datasets and records the ELBO
trajectory.  Purpose: assess whether the paper's iteration budget (5e5) and
learning rate (5e-3) produce a converged mean-field approximation, or whether
the observed underperformance vs HMC could partly reflect optimization failure.

We also compare lr=5e-3 (paper default) against lr=1e-3 (conservative alternative)
at a 50k-iteration budget (the CLI default; ≈10% of the paper budget).
If the ELBO plateaus before 50k, the 500k budget is definitively sufficient.

Usage (from experiments/):
    python advi_elbo_convergence.py
    python advi_elbo_convergence.py --n_datasets 10 --viter 100000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc import adam
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

DIR = Path(__file__).resolve().parent
DATA_DIR = DIR / '..' / 'metabeta' / 'outputs' / 'data'
OUT_DIR = DIR / 'results'
BATCH_PATH = DATA_DIR / 'tiny-n-toy' / 'test.npz'


# ---------------------------------------------------------------------------
# PyMC model builder (simplified version matching fit.py for Normal outcome)
# ---------------------------------------------------------------------------

def build_normal_model(ds: dict) -> pm.Model:
    """Build a Normal GLMM matching the paper's PyMC parameterisation."""
    n_obs = int(ds['n'])
    m = int(ds['m'])
    d = int(ds['d'])
    q = int(ds['q'])
    correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2

    X = ds['X'][:n_obs].astype(np.float32).copy()
    y = ds['y'][:n_obs].astype(np.float32)
    Z = X[:, :q].copy()
    groups = ds['groups'][:n_obs].astype(int)

    # centre predictors
    if d > 1:
        means = X[:, 1:].mean(0)
        X[:, 1:] -= means
        Z[:, 1:] -= means[:q - 1]

    nu_ffx = ds['nu_ffx'].astype(float)
    tau_ffx = ds['tau_ffx'].astype(float)
    tau_rfx = ds['tau_rfx'].astype(float)
    tau_eps = float(ds.get('tau_eps', 1.0))

    with pm.Model() as model:
        betas = [pm.Normal('Intercept' if j == 0 else f'x{j}',
                            mu=nu_ffx[j], sigma=tau_ffx[j])
                 for j in range(d)]
        if correlated:
            chol, _, sigma_vec = pm.LKJCholeskyCov(
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
# ELBO recording
# ---------------------------------------------------------------------------

def fit_advi_with_elbo(
    model: pm.Model,
    n_iter: int,
    lr: float,
    every: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit ADVI and return (steps, elbo_values, wall_time_seconds)."""
    steps: list[int] = []
    elbo_vals: list[float] = []

    def _record(approx, hist, i):
        if i % every == 0:
            steps.append(i)
            elbo_vals.append(-float(hist[-1]))  # negate loss → ELBO

    t0 = time.perf_counter()
    with model:
        pm.fit(
            n=n_iter,
            method='advi',
            obj_optimizer=adam(learning_rate=lr),
            callbacks=[_record],
            progressbar=False,
        )
    wall = time.perf_counter() - t0
    return np.array(steps, dtype=np.int64), np.array(elbo_vals, dtype=np.float64), wall


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def convergence_stats(elbo: np.ndarray, tail_frac: float = 0.2) -> dict:
    """
    Assess convergence from the ELBO trajectory.

    Returns:
        plateau_delta   – mean absolute change in ELBO in the last tail_frac
                          of iterations (small → converged)
        plateau_rel     – plateau_delta / |final ELBO| (scale-free)
        still_rising    – True if the tail is on average still increasing
        max_elbo        – maximum ELBO seen
        final_elbo      – last recorded ELBO
    """
    n_tail = max(2, int(len(elbo) * tail_frac))
    tail = elbo[-n_tail:]
    diffs = np.diff(tail)
    plateau_delta = float(np.mean(np.abs(diffs)))
    final_elbo = float(elbo[-1])
    max_elbo = float(np.max(elbo))
    plateau_rel = plateau_delta / max(abs(final_elbo), 1e-8)
    still_rising = float(np.mean(diffs)) > 0
    return dict(
        plateau_delta=plateau_delta,
        plateau_rel=plateau_rel,
        still_rising=still_rising,
        max_elbo=max_elbo,
        final_elbo=final_elbo,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    batch = dict(np.load(BATCH_PATH, allow_pickle=True))
    n_batch = len(batch['y'])
    indices = list(range(min(args.n_datasets, n_batch)))

    # Pick a diverse subset: vary by q (q=1 vs q=2) and m
    if args.n_datasets <= 8:
        q_arr = batch['q']
        m_arr = batch['m']
        q1_idx = [i for i in range(n_batch) if q_arr[i] == 1][:args.n_datasets // 2]
        q2_idx = [i for i in range(n_batch) if q_arr[i] == 2][:args.n_datasets - len(q1_idx)]
        indices = sorted(q1_idx + q2_idx)

    print(f'Running ADVI on {len(indices)} datasets '
          f'({args.viter} iters, lr_paper={args.lr_paper}, lr_alt={args.lr_alt})')

    results = []
    for idx in indices:
        ds = {k: v[idx] for k, v in batch.items()}
        n_obs, m, d, q = int(ds['n']), int(ds['m']), int(ds['d']), int(ds['q'])
        correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2
        print(f'  Dataset {idx}: n={n_obs}, m={m}, d={d}, q={q}, '
              f'corr={correlated}', end='', flush=True)

        row: dict = dict(idx=idx, n=n_obs, m=m, d=d, q=q, correlated=correlated)

        for label, lr in [('paper', args.lr_paper), ('alt', args.lr_alt)]:
            model = build_normal_model(ds)
            steps, elbo, wall = fit_advi_with_elbo(model, args.viter, lr, every=args.every)
            stats = convergence_stats(elbo)
            row[f'steps_{label}'] = steps
            row[f'elbo_{label}'] = elbo
            row[f'wall_{label}'] = wall
            row.update({f'{k}_{label}': v for k, v in stats.items()})
            print(f'  lr={lr}: {wall:.0f}s, '
                  f'final_ELBO={stats["final_elbo"]:.1f}, '
                  f'plateau_rel={stats["plateau_rel"]:.4f}, '
                  f'still_rising={stats["still_rising"]}', end='')

        print()
        results.append(row)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    n_ds = len(results)
    fig = plt.figure(figsize=(14, 3 * n_ds))
    gs = GridSpec(n_ds, 2, figure=fig, hspace=0.45, wspace=0.35)

    for row_i, row in enumerate(results):
        for col_i, (label, color) in enumerate([('paper', '#e05c47'), ('alt', '#4a90d9')]):
            ax = fig.add_subplot(gs[row_i, col_i])
            steps = row[f'steps_{label}']
            elbo = row[f'elbo_{label}']

            # Smooth with rolling median to tame noise
            k = max(1, len(elbo) // 50)
            elbo_smooth = np.convolve(elbo, np.ones(k) / k, mode='valid')
            steps_smooth = steps[k - 1:]

            ax.plot(steps, elbo, alpha=0.25, color=color, linewidth=0.6)
            ax.plot(steps_smooth, elbo_smooth, color=color, linewidth=1.5)

            lr = args.lr_paper if label == 'paper' else args.lr_alt
            pr = row[f'plateau_rel_{label}']
            sr = row[f'still_rising_{label}']
            converged_str = '✓ converged' if pr < 0.01 and not sr else '? oscillating' if pr < 0.05 else '✗ not converged'
            ax.set_title(
                f'ds={row["idx"]}  n={row["n"]}, m={row["m"]}, q={row["q"]}\n'
                f'lr={lr}  plateau_rel={pr:.4f}  {converged_str}',
                fontsize=8,
            )
            ax.set_xlabel('Iteration', fontsize=7)
            ax.set_ylabel('ELBO', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.axvline(x=steps[-1], color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    fig.suptitle(
        f'ADVI ELBO convergence  |  tiny-n-toy test set  |  {args.viter} iterations\n'
        f'Left: lr={args.lr_paper} (paper default)    Right: lr={args.lr_alt} (conservative)',
        fontsize=10, y=1.01,
    )
    out_path = OUT_DIR / 'advi_elbo_convergence.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'\nFigure saved to {out_path}')

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print('\n--- Convergence summary ---')
    header = f'{"idx":>4}  {"n":>5}  {"m":>4}  {"q":>2}  '
    header += f'{"lr_paper plateau_rel":>20}  {"rising_paper":>12}  '
    header += f'{"lr_alt plateau_rel":>18}  {"rising_alt":>10}'
    print(header)
    print('-' * len(header))
    for row in results:
        print(
            f'{row["idx"]:>4}  {row["n"]:>5}  {row["m"]:>4}  {row["q"]:>2}  '
            f'{row["plateau_rel_paper"]:>20.4f}  {str(row["still_rising_paper"]):>12}  '
            f'{row["plateau_rel_alt"]:>18.4f}  {str(row["still_rising_alt"]):>10}'
        )

    # Overall verdict
    not_converged_paper = sum(
        1 for r in results
        if r['plateau_rel_paper'] >= 0.01 or r['still_rising_paper']
    )
    not_converged_alt = sum(
        1 for r in results
        if r['plateau_rel_alt'] >= 0.01 or r['still_rising_alt']
    )
    print(f'\nNot converged (plateau_rel >= 0.01 or still rising):')
    print(f'  lr={args.lr_paper} (paper): {not_converged_paper}/{len(results)} datasets')
    print(f'  lr={args.lr_alt}  (alt):   {not_converged_alt}/{len(results)} datasets')
    if not_converged_paper == 0:
        print(f'\nVerdict: paper lr={args.lr_paper} converges on tiny-n-toy within {args.viter} iters.')
        print('  The 5e5 budget is likely more than sufficient; performance reflects mean-field expressivity.')
    elif not_converged_alt < not_converged_paper:
        print(f'\nVerdict: lr={args.lr_alt} converges more reliably. Consider switching.')
    else:
        print(f'\nVerdict: neither LR shows clean convergence — increase budget or reduce LR further.')


# ---------------------------------------------------------------------------

def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='ADVI ELBO convergence experiment')
    p.add_argument('--n_datasets', type=int, default=6,
                   help='Number of test datasets to evaluate (default 6)')
    p.add_argument('--viter', type=int, default=50_000,
                   help='ADVI iteration budget (default 50_000)')
    p.add_argument('--lr_paper', type=float, default=5e-3,
                   help='Learning rate: paper default (default 5e-3)')
    p.add_argument('--lr_alt', type=float, default=1e-3,
                   help='Learning rate: conservative alternative (default 1e-3)')
    p.add_argument('--every', type=int, default=100,
                   help='Record ELBO every N iterations (default 100)')
    return p.parse_args()


if __name__ == '__main__':
    args = setup()
    run(args)
