"""
Collect and visualize summary statistics for preprocessed test datasets.

Metrics per dataset:
  1. d — number of fixed effects (incl. intercept)
  2. Bambi default prior parameters (nu_ffx, tau_ffx per predictor)
  3. m — number of groups
  4. Group-size distribution: min, max, entropy

Usage (from metabeta/datasets/):
    uv run python overview.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as sp_entropy


# ---------------------------------------------------------------------------
# Bambi default priors (matching bambi's auto-scaling on observed data)
# ---------------------------------------------------------------------------


def BambiDefaultPriors(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute what bambi would use as default Normal priors for fixed effects.

    Bambi scales:
      - Intercept: Normal(mean(y), 2.5 * std(y))
      - Slopes:    Normal(0,       2.5 * std(y) / std(x_j))

    Returns (nu_ffx, tau_ffx) arrays of length d = 1 + n_predictors.
    """
    d = X.shape[1] + 1  # +1 for intercept
    sy = np.nanstd(y)

    nu_ffx = np.zeros(d)
    tau_ffx = np.zeros(d)

    # intercept
    nu_ffx[0] = np.nanmean(y)
    tau_ffx[0] = 2.5 * sy

    # slopes
    for j in range(X.shape[1]):
        sx = np.nanstd(X[:, j])
        nu_ffx[j + 1] = 0.0
        tau_ffx[j + 1] = 2.5 * sy / sx if sx > 1e-12 else 2.5 * sy

    return nu_ffx, tau_ffx


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


def Collect(test_dir: Path) -> list[dict]:
    """Load every .npz in test_dir and compute summary statistics."""
    rows = []
    for path in sorted(test_dir.glob('*.npz')):
        with np.load(path, allow_pickle=True) as f:
            data = dict(f)

        d = int(data['d'])
        m = int(data['m'])
        ns = data['ns']
        nu_ffx, tau_ffx = BambiDefaultPriors(data['X'], data['y'])

        # group-size entropy (uniform → max entropy → balanced groups)
        ns_prob = ns / ns.sum()
        h = float(sp_entropy(ns_prob))
        h_max = float(np.log(m))  # entropy of uniform distribution over m groups
        h_ratio = h / h_max if h_max > 0 else 1.0

        rows.append({
            'name': path.stem,
            'd': d,
            'm': m,
            'n': int(data['n']),
            'ns_min': int(ns.min()),
            'ns_max': int(ns.max()),
            'ns_mean': float(ns.mean()),
            'ns_std': float(ns.std()),
            'ns_entropy': h,
            'ns_entropy_ratio': h_ratio,  # 1 = perfectly uniform
            'nu_ffx': nu_ffx,
            'tau_ffx': tau_ffx,
            'columns': data.get('columns', np.array([])),
        })
    return rows


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def PrintSummary(rows: list[dict]) -> None:
    header = (
        f"{'dataset':<25} {'d':>3} {'m':>5} {'n':>6}"
        f"  {'ns_min':>6} {'ns_max':>6} {'H_ratio':>7}"
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        print(
            f"{r['name']:<25} {r['d']:>3} {r['m']:>5} {r['n']:>6}  "
            f"{r['ns_min']:>6} {r['ns_max']:>6} {r['ns_entropy_ratio']:>7.3f}"
        )

    # prior summary
    print('\n--- Bambi default prior parameters (nu_ffx, tau_ffx) ---')
    for r in rows:
        cols = ['intercept'] + list(r['columns'])
        pairs = [
            f'{c}: N({nu:.2f}, {tau:.2f})'
            for c, nu, tau in zip(cols, r['nu_ffx'], r['tau_ffx'])
        ]
        print(f"  {r['name']}: {', '.join(pairs)}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def Plot(rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    names = [r['name'] for r in rows]
    k = len(names)

    # -- 1. Fixed-effects dimensionality --
    fig, ax = plt.subplots(figsize=(8, 0.4 * k + 1))
    ds = [r['d'] for r in rows]
    ax.barh(names, ds)
    ax.set_xlabel('d (fixed effects incl. intercept)')
    ax.set_title('Number of fixed effects')
    fig.tight_layout()
    fig.savefig(outdir / 'n_fixed_effects.pdf')
    plt.close(fig)

    # -- 2. Number of groups --
    fig, ax = plt.subplots(figsize=(8, 0.4 * k + 1))
    ms = [r['m'] for r in rows]
    ax.barh(names, ms)
    ax.set_xlabel('m (groups)')
    ax.set_title('Number of groups')
    fig.tight_layout()
    fig.savefig(outdir / 'n_groups.pdf')
    plt.close(fig)

    # -- 3. Group-size range (min–max) with mean marker --
    fig, ax = plt.subplots(figsize=(8, 0.4 * k + 1))
    for i, r in enumerate(rows):
        ax.plot([r['ns_min'], r['ns_max']], [i, i], 'o-', color='C0', markersize=4)
        ax.plot(r['ns_mean'], i, 's', color='C1', markersize=5)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names)
    ax.set_xlabel('observations per group')
    ax.set_title('Group-size range (dots=min/max, square=mean)')
    fig.tight_layout()
    fig.savefig(outdir / 'group_sizes.pdf')
    plt.close(fig)

    # -- 4. Group-size entropy ratio --
    fig, ax = plt.subplots(figsize=(8, 0.4 * k + 1))
    hrs = [r['ns_entropy_ratio'] for r in rows]
    colors = ['C2' if h > 0.9 else 'C1' if h > 0.8 else 'C3' for h in hrs]
    ax.barh(names, hrs, color=colors)
    ax.axvline(1.0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('H / H_max  (1 = uniform)')
    ax.set_title('Group-size entropy ratio')
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(outdir / 'group_entropy.pdf')
    plt.close(fig)

    # -- 5. Bambi default tau_ffx (intercept + slopes) --
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * k + 1))
    # intercept sigma
    ax = axes[0]
    vals = [r['tau_ffx'][0] for r in rows]
    ax.barh(names, vals)
    ax.set_xlabel('tau (sigma)')
    ax.set_title('Default intercept prior sigma')
    # max slope sigma
    ax = axes[1]
    vals = [r['tau_ffx'][1:].max() if len(r['tau_ffx']) > 1 else 0 for r in rows]
    ax.barh(names, vals)
    ax.set_xlabel('tau (sigma)')
    ax.set_title('Max slope prior sigma')
    fig.tight_layout()
    fig.savefig(outdir / 'bambi_priors.pdf')
    plt.close(fig)

    print(f'\nPlots saved to {outdir}/')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_dir = Path('preprocessed/test')
    assert test_dir.exists(), f'{test_dir} not found — run from metabeta/datasets/'

    rows = Collect(test_dir)
    PrintSummary(rows)
    Plot(rows, outdir=Path('preprocessed/plots/overview'))
