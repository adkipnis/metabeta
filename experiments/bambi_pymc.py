"""
Experiment: verify that fit._buildPymc is equivalent to bambi's GLMM by
comparing their log-posteriors at random points in the shared unconstrained
parameter space.

Both models compile to the same PyMC log-posterior function.  Evaluating
logp at N random unconstrained points and checking that |diff| is at the
level of floating-point rounding (~5e-9) is a stronger proof of equivalence
than any posterior sample comparison, which cannot be made identical due to
different variable orderings seeding different MCMC trajectories.

Only uncorrelated datasets are tested because bambi v0.17.2 does not support
correlated random effects.

Run from the simulation/ directory:
    uv run python compare_bambi_pymc.py [--data_id tiny-n-toy] [--n_datasets 10] [--n_points 50]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import bambi as bmb

from metabeta.utils.families import bambiFamilyName, FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF
from metabeta.utils.padding import unpad
from metabeta.simulation.fit import Fitter


# ---------------------------------------------------------------------------
# Bambi reference implementation
# ---------------------------------------------------------------------------

def pandify(ds: dict) -> pd.DataFrame:
    df = pd.DataFrame({'i': ds['groups'], 'y': ds['y']})
    for j in range(1, int(ds['d'])):
        df[f'x{j}'] = ds['X'][:, j]
    return df


def formulate(ds: dict) -> str:
    d, q = int(ds['d']), int(ds['q'])
    fixed = ' + '.join(f'x{j}' for j in range(1, d))
    random = ' + '.join(['(1 | i)'] + [f'(0 + x{j} | i)' for j in range(1, q)])
    return f'y ~ 1 + {fixed} + {random}' if fixed else f'y ~ 1 + {random}'


def priorize(ds: dict) -> dict:
    d, q = int(ds['d']), int(ds['q'])
    nu_ffx, tau_ffx, tau_rfx = ds['nu_ffx'], ds['tau_ffx'], ds['tau_rfx']
    ffx_name   = FFX_FAMILIES[int(ds.get('family_ffx', 0))]
    sigma_name = SIGMA_FAMILIES[int(ds.get('family_sigma_rfx', 0))]
    priors = {}

    for j in range(d):
        key = 'Intercept' if j == 0 else f'x{j}'
        priors[key] = (
            bmb.Prior('Normal', mu=nu_ffx[j], sigma=tau_ffx[j])
            if ffx_name == 'normal'
            else bmb.Prior('StudentT', nu=STUDENT_DF, mu=nu_ffx[j], sigma=tau_ffx[j])
        )

    for j in range(q):
        key = '1|i' if j == 0 else f'x{j}|i'
        sigma_prior = {
            'halfnormal':  bmb.Prior('HalfNormal', sigma=float(tau_rfx[j])),
            'halfstudent': bmb.Prior('HalfStudentT', nu=STUDENT_DF, sigma=float(tau_rfx[j])),
            'exponential': bmb.Prior('Exponential', lam=1.0 / (float(tau_rfx[j]) + 1e-12)),
        }[sigma_name]
        priors[key] = bmb.Prior('Normal', mu=0, sigma=sigma_prior)

    if 'tau_eps' in ds:
        eps_name = SIGMA_FAMILIES[int(ds.get('family_sigma_eps', 0))]
        priors['sigma'] = {
            'halfnormal':  bmb.Prior('HalfNormal', sigma=float(ds['tau_eps'])),
            'halfstudent': bmb.Prior('HalfStudentT', nu=STUDENT_DF, sigma=float(ds['tau_eps'])),
            'exponential': bmb.Prior('Exponential', lam=1.0 / (float(ds['tau_eps']) + 1e-12)),
        }[eps_name]

    return priors


def bambify(ds: dict) -> bmb.Model:
    model = bmb.Model(
        formula=formulate(ds),
        data=pandify(ds),
        family=bambiFamilyName(int(ds.get('likelihood_family', 0))),
        categorical='i',
        priors=priorize(ds),
    )
    model.build()
    return model


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_logp(ds: dict, n_points: int = 50, seed: int = 0) -> np.ndarray:
    """Return |logp_bambi(pt) - logp_pymc(pt)| / max(1, |logp|) for n_points random points.

    Points are drawn in the shared unconstrained parameter space (same variable
    names and shapes in both models).  We use a relative error because the
    absolute diff scales with |logp|: as parameter values grow more extreme the
    likelihood sum has more float64 accumulation error, giving a constant
    relative error of ~3e-11 regardless of model size.  Any genuine model
    mismatch would show up as a relative error orders of magnitude larger.
    """
    fitter = Fitter.__new__(Fitter)
    fitter.ds = ds

    bm = bambify(ds)
    pm_bm   = bm.backend.model
    pm_ours = fitter._buildPymc(ds)

    with pm_bm:   logp_bm   = pm_bm.compile_logp()
    with pm_ours: logp_ours = pm_ours.compile_logp()
    with pm_bm:   ip        = pm_bm.initial_point()

    rng = np.random.default_rng(seed)
    rel_diffs = []
    for _ in range(n_points):
        pt = {k: rng.normal(size=np.asarray(v).shape) for k, v in ip.items()}
        lb, lo = float(logp_bm(pt)), float(logp_ours(pt))
        rel_diffs.append(abs(lb - lo) / max(1.0, abs(lb)))
    return np.array(rel_diffs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # fmt: off
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_id',    type=str, default='tiny-n-toy')
    parser.add_argument('--n_datasets', type=int, default=10, help='Uncorrelated datasets to test (default=10)')
    parser.add_argument('--n_points',   type=int, default=50, help='Random parameter points per dataset (default=50)')
    parser.add_argument('--tol',        type=float, default=1e-9, help='Pass threshold for max relative |diff| (default=1e-9)')
    # fmt: on
    args = parser.parse_args()

    _here = Path(__file__).resolve().parent
    data_path = _here / '..' / 'metabeta' / 'outputs' / 'data' / args.data_id / 'test.npz'
    assert data_path.exists(), f'data not found: {data_path}'

    with np.load(data_path, allow_pickle=True) as f:
        batch = dict(f)

    uncorr_idx = np.where(batch['eta_rfx'] == 0)[0][: args.n_datasets]
    print(f'Testing {len(uncorr_idx)} uncorrelated datasets from {args.data_id}')
    print(f"  (bambi v{bmb.__version__} does not support correlated rfx)")
    print()
    print(f'{"idx":>5}  {"q":>3}  {"m":>5}  {"max rel|diff|":>14}  {"mean rel|diff|":>15}')
    print('-' * 52)

    all_max = []
    for idx in uncorr_idx:
        ds = unpad({k: v[idx] for k, v in batch.items()}, {k: batch[k][idx] for k in 'dqmn'})
        diffs = compare_logp(ds, n_points=args.n_points, seed=int(idx))
        all_max.append(diffs.max())
        print(f'{idx:>5}  {int(ds["q"]):>3}  {int(ds["m"]):>5}  {diffs.max():>14.2e}  {diffs.mean():>15.2e}')

    overall = max(all_max)
    print('-' * 52)
    print(f'{"overall max":>28}  {overall:>14.2e}')
    print()
    if overall < args.tol:
        print(f'PASS  all rel|diff| < {args.tol:.0e}  (floating-point precision)')
    else:
        print(f'FAIL  max rel|diff| = {overall:.2e} >= {args.tol:.0e}')


if __name__ == '__main__':
    main()
