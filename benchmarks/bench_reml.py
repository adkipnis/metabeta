"""Benchmark batched REML (PyTorch) vs statsmodels.MixedLM.

Uses the actual metabeta simulation pipeline (Prior + Synthesizer + Simulator)
to generate datasets, mirroring the toy-n data config (normal likelihood).

Run from the repo root:
    uv run python benchmarks/bench_reml.py

Sections:
  1. q=1: agreement with statsmodels (exact match expected)
  2. q=2: full Σ_u vs statsmodels; estimated vs true correlation
  3. Timing: batched torch vs looped statsmodels
"""

import time
import warnings

import numpy as np
import pandas as pd
import torch
import statsmodels.formula.api as smf

from metabeta.simulation import Prior, Synthesizer, Simulator, hypersample
from metabeta.utils.reml import remlSolve

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
B = 8
D = 4        # fixed effects: intercept + 2 slopes
M = 20
N_PER_GROUP = 25
N_REPEAT = 5

rng_global = np.random.default_rng(SEED)
torch.manual_seed(SEED)

print(f'device: {DEVICE}  |  B={B}  d={D}  m={M}  n_per_group={N_PER_GROUP}')

# ---------------------------------------------------------------------------
# simulation helpers
# ---------------------------------------------------------------------------


def gen_dataset(rng: np.random.Generator, d: int, q: int, eta_rfx: float | None = None):
    """One dataset from the metabeta simulation pipeline (normal likelihood)."""
    ns = np.full(M, N_PER_GROUP, dtype=int)
    hyperparams = hypersample(rng, d, q, likelihood_family=0)
    if eta_rfx is not None:
        hyperparams['eta_rfx'] = np.array(eta_rfx)
    prior = Prior(rng, hyperparams)
    design = Synthesizer(rng, toy=True)
    sim = Simulator(rng, prior, design, ns)
    ds = sim.sample()
    ds['Z'] = ds['X'][:, :q].copy()
    return ds


def flat_to_grouped(ds: dict, max_n: int | None = None) -> dict:
    """Convert flat (N, d) simulation output to grouped (m, max_n, *) arrays."""
    y_flat = ds['y']
    X_flat = ds['X']
    Z_flat = ds['Z']
    groups = ds['groups']
    ns = ds['ns']
    m = int(ds['m'])
    d = X_flat.shape[-1]
    q = Z_flat.shape[-1]

    if max_n is None:
        max_n = int(ns.max())

    y = np.zeros((m, max_n), dtype=np.float32)
    X = np.zeros((m, max_n, d), dtype=np.float32)
    Z = np.zeros((m, max_n, q), dtype=np.float32)
    mask = np.zeros((m, max_n), dtype=np.float32)

    for i in range(m):
        idx = np.where(groups == i)[0]
        n_i = len(idx)
        y[i, :n_i] = y_flat[idx]
        X[i, :n_i] = X_flat[idx]
        Z[i, :n_i] = Z_flat[idx]
        mask[i, :n_i] = 1.0

    return dict(y=y, X=X, Z=Z, mask=mask, ns=ns)


def collate_batch(datasets: list[dict]) -> dict[str, torch.Tensor]:
    """Stack B grouped datasets into (B, m, n, *) torch tensors."""
    max_n = max(int(ds['ns'].max()) for ds in datasets)
    grouped = [flat_to_grouped(ds, max_n=max_n) for ds in datasets]

    def _t(arr):
        return torch.tensor(arr, device=DEVICE)

    y = _t(np.stack([g['y'] for g in grouped]))       # (B, m, n)
    X = _t(np.stack([g['X'] for g in grouped]))       # (B, m, n, d)
    Z = _t(np.stack([g['Z'] for g in grouped]))       # (B, m, n, q)
    mask = _t(np.stack([g['mask'] for g in grouped]))   # (B, m, n)
    ns = _t(np.stack([ds['ns'] for ds in datasets])).long()  # (B, m)
    mask_m = (ns > 0).to(X.dtype)

    Xm = X * mask.unsqueeze(-1)
    ym = y * mask
    Zm = Z * mask.unsqueeze(-1)
    return dict(Xm=Xm, ym=ym, Zm=Zm, mask=mask, mask_m=mask_m, ns=ns)


# ---------------------------------------------------------------------------
# statsmodels helpers
# ---------------------------------------------------------------------------


def to_df(ds: dict) -> pd.DataFrame:
    d = ds['X'].shape[-1]
    df = pd.DataFrame(ds['X'], columns=[f'x{j}' for j in range(d)])
    df['y'] = ds['y']
    df['group'] = ds['groups']
    return df


def _sm_fit(formula, df, groups, re_formula=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return smf.mixedlm(formula, df, groups=groups, re_formula=re_formula).fit(
            reml=True, disp=False
        )


def sm_fit(df: pd.DataFrame, q: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit MixedLM with q random effects (intercept [+ slope on x1]).

    Returns fe (d,), Sigma_u (q, q), sigma_eps.
    """
    covs = ' + '.join(f'x{j}' for j in range(1, D))
    re_formula = '~x1' if q == 2 else None
    res = _sm_fit(f'y ~ {covs}', df, df['group'], re_formula=re_formula)
    fe = res.fe_params.values
    cov_re = res.cov_re.values  # (q, q) but indexed by statsmodels names
    # ensure q×q even if statsmodels collapses a near-zero component
    Sigma_u_sm = np.zeros((q, q))
    Sigma_u_sm[: cov_re.shape[0], : cov_re.shape[1]] = cov_re
    sigma_eps = float(np.sqrt(res.scale))
    return fe, Sigma_u_sm, sigma_eps


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------


def cov_to_corr(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split covariance into SDs and correlation matrix."""
    sds = np.sqrt(np.maximum(np.diag(Sigma), 0.0))
    with np.errstate(invalid='ignore'):
        D_inv = np.where(sds > 0, 1.0 / sds, 0.0)
    corr = D_inv[:, None] * Sigma * D_inv[None, :]
    np.fill_diagonal(corr, 1.0)
    return sds, corr


def print_comparison(b, ds, beta_t, Sigma_u_t, se_t, fe_sm, Sigma_u_sm, se_sm):
    q = int(ds['q'])
    true_beta = ds['ffx']
    true_su, true_corr = cov_to_corr(
        np.diag(ds['sigma_rfx']) @ ds['corr_rfx'] @ np.diag(ds['sigma_rfx'])
    )
    t_su, t_corr = cov_to_corr(Sigma_u_t)
    sm_su, sm_corr = cov_to_corr(Sigma_u_sm)

    print(f'\n  b={b}')
    print(f'    {"param":<12}  {"true":>8}  {"torch":>8}  {"statsmodels":>12}')
    print(f'    {"-" * 46}')
    for j in range(D):
        print(f'    β[{j}]        {true_beta[j]:>8.4f}  {beta_t[j]:>8.4f}  {fe_sm[j]:>12.4f}')
    for j in range(q):
        print(f'    σ_u[{j}]      {true_su[j]:>8.4f}  {t_su[j]:>8.4f}  {sm_su[j]:>12.4f}')
    print(f'    σ_ε         {float(ds["sigma_eps"]):>8.4f}  {se_t:>8.4f}  {se_sm:>12.4f}')
    if q > 1:
        for i in range(1, q):
            for j in range(i):
                true_r = true_corr[i, j]
                t_r = t_corr[i, j]
                sm_r = sm_corr[i, j]
                print(f'    corr[{i},{j}]    {true_r:>8.4f}  {t_r:>8.4f}  {sm_r:>12.4f}')


# ---------------------------------------------------------------------------
# Section 1: q=1 — agreement with statsmodels
# ---------------------------------------------------------------------------

print()
print('=' * 65)
print('q=1: agreement with statsmodels  (no off-diagonal)')
print('=' * 65)

datasets_q1 = [gen_dataset(rng_global, d=D, q=1) for _ in range(B)]
batch_q1 = collate_batch(datasets_q1)
beta_r1, Sigma_u_r1, se_r1 = remlSolve(
    batch_q1['Xm'], batch_q1['ym'], batch_q1['Zm'], batch_q1['mask_m'], batch_q1['ns']
)

for b, ds in enumerate(datasets_q1):
    fe_sm, Su_sm, se_sm = sm_fit(to_df(ds), q=1)
    print_comparison(
        b,
        ds,
        beta_r1[b].cpu().numpy(),
        Sigma_u_r1[b].cpu().numpy(),
        se_r1[b].item(),
        fe_sm,
        Su_sm,
        se_sm,
    )

# ---------------------------------------------------------------------------
# Section 2: q=2 — full Σ_u; estimated vs true correlation
# ---------------------------------------------------------------------------

print()
print('=' * 65)
print('q=2: full Σ_u estimated by torch and statsmodels')
print('  DGP: rfx ~ MVN(0, Σ_u);  corr_rfx from LKJ prior (80% correlated)')
print('=' * 65)

datasets_q2 = [gen_dataset(rng_global, d=D, q=2) for _ in range(B)]
batch_q2 = collate_batch(datasets_q2)
beta_r2, Sigma_u_r2, se_r2 = remlSolve(
    batch_q2['Xm'], batch_q2['ym'], batch_q2['Zm'], batch_q2['mask_m'], batch_q2['ns']
)

for b, ds in enumerate(datasets_q2):
    fe_sm, Su_sm, se_sm = sm_fit(to_df(ds), q=2)
    print_comparison(
        b,
        ds,
        beta_r2[b].cpu().numpy(),
        Sigma_u_r2[b].cpu().numpy(),
        se_r2[b].item(),
        fe_sm,
        Su_sm,
        se_sm,
    )

# ---------------------------------------------------------------------------
# Section 3: timing
# ---------------------------------------------------------------------------

print()
print('=' * 65)
print('Timing: batched remlNormal vs looped statsmodels')
print(f'  B={B} datasets per call  |  {N_REPEAT} repeats')
print('=' * 65)

SIZES = [(10, 10), (20, 15), (30, 20), (50, 30), (100, 50)]


def _make_batch(m_: int, n_: int, q_: int) -> list[dict[str, np.ndarray]]:
    local_rng = np.random.default_rng(0)
    ns = np.full(m_, n_, dtype=int)
    dss = []
    for _ in range(B):
        hp = hypersample(local_rng, D, q_, likelihood_family=0)
        prior = Prior(local_rng, hp)
        design = Synthesizer(local_rng, toy=True)
        ds = Simulator(local_rng, prior, design, ns).sample()
        ds['Z'] = ds['X'][:, :q_].copy()
        dss.append(ds)
    return dss


print(
    f'\n  {"m":>6} {"n":>6}  {"q":>3} | {"torch (ms)":>12} {"statsmodels (ms)":>18} {"speedup":>9}'
)
print('  ' + '-' * 62)

for m_, n_ in SIZES:
    for q_ in (1, 2):
        dss = _make_batch(m_, n_, q_)
        bt = collate_batch(dss)
        dfs = [to_df(ds) for ds in dss]

        remlSolve(bt['Xm'], bt['ym'], bt['Zm'], bt['mask_m'], bt['ns'])  # warm-up

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_REPEAT):
            remlSolve(bt['Xm'], bt['ym'], bt['Zm'], bt['mask_m'], bt['ns'])
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / N_REPEAT * 1e3

        t0 = time.perf_counter()
        for _ in range(N_REPEAT):
            for df in dfs:
                sm_fit(df, q=q_)
        t_sm = (time.perf_counter() - t0) / N_REPEAT * 1e3

        print(
            f'  {m_:>6} {n_:>6}  q={q_} |'
            f' {t_torch:>12.1f} {t_sm:>18.1f} {t_sm / t_torch:>8.1f}x'
        )
