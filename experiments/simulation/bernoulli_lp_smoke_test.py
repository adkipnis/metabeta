"""Smoke test: verify Bernoulli LP scale calibration produces realistic sd(eta).

Samples datasets across a range of d/q/m values and checks that the resulting
linear predictor sd is in line with the real-b-reference NUTS posteriors
(sd(eta_hat) range: 0.91 – 3.27, mean ≈ 2.2, p95 ≈ 3.2).

Usage from repo root:
    uv run python experiments/simulation/bernoulli_lp_smoke_test.py
"""

import dataclasses
import numpy as np
from scipy import stats

from metabeta.simulation import hypersample, Prior, Synthesizer, Simulator
from metabeta.simulation.simulator import linearPredictor
from metabeta.utils.sampling import sampleCounts


# Real-b-reference NUTS posterior sd(eta_hat) values (one per dataset)
NUTS_ETA_SDS = np.array([
    1.34, 0.91, 2.03, 1.88, 2.38, 1.57, 1.39, 2.06,
    3.27, 3.00, 2.08, 3.03, 2.64, 3.10,
])

CONFIGS = [
    dict(label='small (d=2, q=1, m=20)',   d=2,  q=1, m=20,  n_per_group=10),
    dict(label='small (d=4, q=2, m=30)',   d=4,  q=2, m=30,  n_per_group=8),
    dict(label='medium (d=6, q=1, m=60)',  d=6,  q=1, m=60,  n_per_group=12),
    dict(label='medium (d=8, q=3, m=80)',  d=8,  q=3, m=80,  n_per_group=10),
    dict(label='large (d=12, q=2, m=100)', d=12, q=2, m=100, n_per_group=15),
    dict(label='huge (d=14, q=1, m=80)',   d=14, q=1, m=80,  n_per_group=12),
    dict(label='huge (d=16, q=4, m=60)',   d=16, q=4, m=60,  n_per_group=10),
]

N_SAMPLES = 300
LIKELIHOOD_FAMILY = 1  # bernoulli


def _sampleEtaSd(rng: np.random.Generator, d: int, q: int, m: int, n_per_group: int) -> float:
    ns = np.full(m, n_per_group, dtype=int)
    hyperparams = hypersample(rng, d=d, q=q, likelihood_family=LIKELIHOOD_FAMILY)
    prior = Prior(rng=rng, hyperparams=hyperparams)
    design = Synthesizer(rng=rng)
    sim = Simulator(rng=rng, prior=prior, design=design, ns=ns)
    ds = sim.sample()
    # Compute eta from stored params (already calibrated)
    eta = ds['X'] @ ds['ffx']
    rfx_q = ds['rfx'].shape[1]
    Z = ds['X'][:, :rfx_q]
    eta += (Z * ds['rfx'][ds['groups']]).sum(-1)
    return float(np.std(eta))


def _quantile_str(arr: np.ndarray) -> str:
    qs = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return '  '.join(f'{q:5.2f}' for q in qs)


def main() -> None:
    rng = np.random.default_rng(42)

    print('Bernoulli LP scale calibration smoke test')
    print(f'N samples per config: {N_SAMPLES}')
    print()
    print('Real-b-reference NUTS sd(eta_hat):')
    print(f'  {_quantile_str(NUTS_ETA_SDS)}')
    print(f'  [min   p25    p50    p75    p95    max]')
    print()

    header = f'{"config":<30}  {"min":>5}  {"p25":>5}  {"p50":>5}  {"p75":>5}  {"p95":>5}  {"max":>5}  {">4.0":>5}'
    print(header)
    print('-' * len(header))

    all_sds = []
    for cfg in CONFIGS:
        label = cfg['label']
        sds = np.array([
            _sampleEtaSd(rng, cfg['d'], cfg['q'], cfg['m'], cfg['n_per_group'])
            for _ in range(N_SAMPLES)
        ])
        all_sds.append(sds)
        frac_high = float(np.mean(sds > 4.0))
        print(f'{label:<30}  {_quantile_str(sds)}  {frac_high:5.1%}')

    all_sds_flat = np.concatenate(all_sds)
    print()
    print(f'{"overall":<30}  {_quantile_str(all_sds_flat)}  {float(np.mean(all_sds_flat > 4.0)):5.1%}')

    # KS test: generated vs NUTS reference
    ks_stat, ks_p = stats.ks_2samp(all_sds_flat, NUTS_ETA_SDS)
    print()
    print(f'KS test (generated vs NUTS):  D={ks_stat:.3f},  p={ks_p:.3f}')
    if ks_p > 0.05:
        print('  => distributions not significantly different (p > 0.05)')
    else:
        print('  => distributions differ significantly (p <= 0.05) — expected given broader prior')

    # Check: no sd(eta) exceeds 4.0 + some tolerance
    excess = float(np.mean(all_sds_flat > 4.05))
    print()
    if excess < 0.01:
        print(f'PASS: <1% of datasets have sd(eta) > 4.05  ({excess:.1%})')
    else:
        print(f'FAIL: {excess:.1%} of datasets have sd(eta) > 4.05 — calibration may not be working')


if __name__ == '__main__':
    main()
