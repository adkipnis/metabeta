"""Smoke test: verify Poisson LP scale calibration produces realistic sd(eta).

Samples datasets across a range of d/q/m values and checks that the resulting
linear predictor sd is in line with the real-p-reference NUTS posteriors.

For Poisson (log link), eta = log(lambda).  Crucially:
  - exp(POISSON_ETA_CLIP_MAX=10) = 22026, so any observation with eta > 10
    will be silently clipped and produce an unrealistically large rate.
  - We track both sd(eta) and the fraction of observations clipped.

After running real_nuts_reference.py --y-type count --data-id real-p-reference,
compute the NUTS_ETA_SDS constants with:

    uv run python experiments/simulation/poisson_lp_smoke_test.py --print-nuts-sds

Usage from repo root:
    uv run python experiments/simulation/poisson_lp_smoke_test.py
"""

import argparse
import dataclasses
import numpy as np
from scipy import stats

from metabeta.simulation import hypersample, Prior, Synthesizer, Simulator
from metabeta.simulation.simulator import linearPredictor
from metabeta.utils.families import POISSON_ETA_CLIP_MAX
from metabeta.utils.sampling import sampleCounts


# Real-p-reference NUTS posterior sd(eta_hat) values (one per dataset).
# Computed from real_nuts_reference.py --y-type count --data-id real-p-reference.
# Source datasets (sorted): 556_analcatdata_apnea2, 557_analcatdata_apnea1,
#   arabidopsis, chem97, epil, grouseticks, mmmec, owls, salamanders.
# All use q=1 (random intercept). Range: 0.48-1.57, mean≈0.82, p95≈1.53.
NUTS_ETA_SDS = np.array([
    0.57, 0.61, 0.87, 0.48, 0.91, 1.57, 0.89, 0.53, 1.47,
])

CONFIGS = [
    dict(label='small  (d=2,  q=1, m=20)',  d=2,  q=1, m=20,  n_per_group=10),
    dict(label='small  (d=4,  q=2, m=30)',  d=4,  q=2, m=30,  n_per_group=8),
    dict(label='medium (d=6,  q=1, m=60)',  d=6,  q=1, m=60,  n_per_group=12),
    dict(label='medium (d=8,  q=3, m=80)',  d=8,  q=3, m=80,  n_per_group=10),
    dict(label='large  (d=12, q=2, m=100)', d=12, q=2, m=100, n_per_group=15),
    dict(label='huge   (d=14, q=1, m=80)',  d=14, q=1, m=80,  n_per_group=12),
    dict(label='huge   (d=16, q=4, m=60)',  d=16, q=4, m=60,  n_per_group=10),
]

N_SAMPLES = 300
LIKELIHOOD_FAMILY = 2  # poisson


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--print-nuts-sds',
        action='store_true',
        help='Print sd(eta_hat) from real-p-reference NUTS fits and exit.',
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()
# fmt: on


def _loadNutsEtaSds() -> np.ndarray:
    """Compute sd(eta_hat) per dataset from the saved NUTS reference fit."""
    from pathlib import Path
    from metabeta.utils.experiments import DATA_DIR

    fit_path = DATA_DIR / 'real-p-reference' / 'test.fit.npz'
    if not fit_path.exists():
        raise FileNotFoundError(
            f'NUTS reference not found: {fit_path}\n'
            'Run first:  uv run python experiments/simulation/real_nuts_reference.py'
            ' --y-type count --data-id real-p-reference'
        )
    with np.load(fit_path, allow_pickle=True) as data:
        nuts_ffx = data['nuts_ffx']      # (n_ds, d_max, n_samples)
        nuts_rfx = data['nuts_rfx']      # (n_ds, q_max, m_max, n_samples)
        X_all = data['X']                # (n_ds, n_max, d_max)
        groups_all = data['groups']      # (n_ds, n_max)
        n_arr = data['n']
        d_arr = data['d']
        q_arr = data['q']
        m_arr = data['m']
        sources = data['source']

    eta_sds = []
    for i in range(len(n_arr)):
        n_i = int(n_arr[i])
        d_i = int(d_arr[i])
        q_i = int(q_arr[i])
        m_i = int(m_arr[i])
        ffx_pm = nuts_ffx[i, :d_i, :].mean(axis=-1)          # (d,)
        rfx_pm = nuts_rfx[i, :q_i, :m_i, :].mean(axis=-1).T  # (m, q)
        X_i = X_all[i, :n_i, :d_i]
        groups_i = groups_all[i, :n_i]
        Z_i = X_i[:, :q_i]
        eta = X_i @ ffx_pm + (Z_i * rfx_pm[groups_i]).sum(-1)
        eta_sds.append(float(np.std(eta)))

    names = [str(s.item() if hasattr(s, 'item') else s) for s in sources]
    return np.array(eta_sds), names


def _sampleStats(
    rng: np.random.Generator, d: int, q: int, m: int, n_per_group: int
) -> tuple[float, float]:
    """Return (sd(eta), clip_fraction) for one simulated Poisson dataset."""
    ns = np.full(m, n_per_group, dtype=int)
    hyperparams = hypersample(rng, d=d, q=q, likelihood_family=LIKELIHOOD_FAMILY)
    prior = Prior(rng=rng, hyperparams=hyperparams)
    design = Synthesizer(rng=rng)
    sim = Simulator(rng=rng, prior=prior, design=design, ns=ns)
    ds = sim.sample()
    q_sim = ds['rfx'].shape[1]
    eta = ds['X'] @ ds['ffx'] + (ds['X'][:, :q_sim] * ds['rfx'][ds['groups']]).sum(-1)
    eta_sd = float(np.std(eta))
    clip_frac = float(np.mean(eta > POISSON_ETA_CLIP_MAX))
    return eta_sd, clip_frac


def _quantile_str(arr: np.ndarray) -> str:
    qs = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return '  '.join(f'{q:5.2f}' for q in qs)


def main() -> None:
    args = setup()

    if args.print_nuts_sds:
        eta_sds, names = _loadNutsEtaSds()
        print('NUTS sd(eta_hat) per dataset:')
        for name, sd in zip(names, eta_sds):
            print(f'  {name:<45} sd={sd:.3f}')
        print()
        print('NUTS_ETA_SDS = np.array([')
        print('    ' + ', '.join(f'{v:.2f}' for v in eta_sds))
        print('])')
        return

    rng = np.random.default_rng(args.seed)

    print('Poisson LP scale calibration smoke test')
    print(f'N samples per config: {N_SAMPLES}')
    print(f'POISSON_ETA_CLIP_MAX: {POISSON_ETA_CLIP_MAX}')
    print()

    if NUTS_ETA_SDS is not None:
        print('Real-p-reference NUTS sd(eta_hat):')
        print(f'  {_quantile_str(NUTS_ETA_SDS)}')
        print(f'  [min   p25    p50    p75    p95    max]')
    else:
        print('NUTS reference: not loaded (run with --print-nuts-sds to get values)')
    print()

    header = (
        f'{"config":<32}  {"min":>5}  {"p25":>5}  {"p50":>5}  {"p75":>5}  {"p95":>5}  {"max":>5}'
        f'  {">clip":>6}  {"p95_clip":>8}'
    )
    print(header)
    print('-' * len(header))

    all_sds = []
    all_clips = []
    for cfg in CONFIGS:
        label = cfg['label']
        results = [
            _sampleStats(rng, cfg['d'], cfg['q'], cfg['m'], cfg['n_per_group'])
            for _ in range(N_SAMPLES)
        ]
        sds = np.array([r[0] for r in results])
        clips = np.array([r[1] for r in results])
        all_sds.append(sds)
        all_clips.append(clips)
        frac_clipped = float(np.mean(clips > 0))
        p95_clip = float(np.quantile(clips, 0.95))
        print(
            f'{label:<32}  {_quantile_str(sds)}  {frac_clipped:6.1%}  {p95_clip:8.4f}'
        )

    all_sds_flat = np.concatenate(all_sds)
    all_clips_flat = np.concatenate(all_clips)
    print()
    frac_clipped = float(np.mean(all_clips_flat > 0))
    p95_clip = float(np.quantile(all_clips_flat, 0.95))
    print(
        f'{"overall":<32}  {_quantile_str(all_sds_flat)}  {frac_clipped:6.1%}  {p95_clip:8.4f}'
    )

    print()
    print('Clip fraction summary (fraction of obs with eta > clip max):')
    clip_quantiles = np.quantile(all_clips_flat, [0.5, 0.75, 0.95, 0.99, 1.0])
    labels = ['p50', 'p75', 'p95', 'p99', 'max']
    for lbl, q in zip(labels, clip_quantiles):
        print(f'  {lbl}: {q:.4f}')

    if NUTS_ETA_SDS is not None:
        # KS test: generated vs NUTS reference
        ks_stat, ks_p = stats.ks_2samp(all_sds_flat, NUTS_ETA_SDS)
        print()
        print(f'KS test (generated vs NUTS):  D={ks_stat:.3f},  p={ks_p:.3f}')
        if ks_p > 0.05:
            print('  => distributions not significantly different (p > 0.05)')
        else:
            print('  => distributions differ significantly (p <= 0.05)')
    else:
        print()
        print('KS test skipped: NUTS_ETA_SDS not set.')
        print('Fill in NUTS_ETA_SDS from:')
        print('  uv run python experiments/simulation/poisson_lp_smoke_test.py --print-nuts-sds')

    # Hard pass/fail on clip fraction
    excess_clip = float(np.mean(all_clips_flat > 0.05))
    print()
    if excess_clip < 0.01:
        print(f'PASS: <1% of datasets have >5% observations clipped  ({excess_clip:.1%})')
    else:
        print(
            f'FAIL: {excess_clip:.1%} of datasets have >5% observations clipped'
            ' — LP scale may need calibration'
        )


if __name__ == '__main__':
    main()
