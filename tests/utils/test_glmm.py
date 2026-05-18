"""Tests for metabeta.analytical.glmm.glmmFull.

Tests cover shape correctness, numerical validity, trivial-case recovery,
signal-detection, and integration with Approximator._dataStatistics.
"""

import numpy as np
import pytest
import torch

from metabeta.simulation import Prior, Synthesizer, Simulator, hypersample
from metabeta.analytical.glmm import lmmBernoulli, lmmPoisson, glmm
from metabeta.analytical.linalg import _adaptiveRidge, _adaptiveRidgeBm, _eighWithJitter, _safeSolve
from metabeta.analytical.map import refineBernoulliLaplaceEb, refineNormalMapSrfx


DEVICE = torch.device('cpu')
SEED = 7


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _gen_dataset(rng, d, q, likelihood_family, m=15, n_per_group=30):
    ns = np.full(m, n_per_group, dtype=int)
    hp = hypersample(rng, d, q, likelihood_family=likelihood_family)
    pr = Prior(rng, hp)
    des = Synthesizer(rng, toy=True)
    ds = Simulator(rng, pr, des, ns).sample()
    ds['Z'] = ds['X'][:, :q].copy()
    return ds


def _collate(datasets, d, q):
    B = len(datasets)
    max_m = max(int(ds['m']) for ds in datasets)
    max_n = int(max(ds['ns'].max() for ds in datasets))

    ym = torch.zeros(B, max_m, max_n)
    Xm = torch.zeros(B, max_m, max_n, d)
    Zm = torch.zeros(B, max_m, max_n, q)
    mask_n = torch.zeros(B, max_m, max_n)
    mask_m = torch.zeros(B, max_m)
    ns = torch.zeros(B, max_m)
    n_total = torch.zeros(B)

    for b, ds in enumerate(datasets):
        m_b = int(ds['m'])
        groups = ds['groups']
        mask_m[b, :m_b] = 1.0
        ns[b, :m_b] = torch.as_tensor(ds['ns'].astype(np.float32))
        n_total[b] = int(ds['n'])
        for g in range(m_b):
            idx = np.where(groups == g)[0]
            n_g = len(idx)
            ym[b, g, :n_g] = torch.as_tensor(ds['y'][idx].astype(np.float32))
            Xm[b, g, :n_g] = torch.as_tensor(ds['X'][idx].astype(np.float32))
            Zm[b, g, :n_g] = torch.as_tensor(ds['Z'][idx].astype(np.float32))
            mask_n[b, g, :n_g] = 1.0

    return dict(Xm=Xm, ym=ym, Zm=Zm, mask_n=mask_n, mask_m=mask_m, ns=ns, n_total=n_total)


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'likelihood_family, q',
    [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ],
)
def test_glmm_output_shapes(likelihood_family, q):
    rng = np.random.default_rng(SEED)
    B, d, m = 4, 3, 10

    datasets = [_gen_dataset(rng, d, q, likelihood_family, m=m) for _ in range(B)]
    bt = _collate(datasets, d, q)

    result = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        likelihood_family=likelihood_family,
    )

    max_m = bt['Xm'].shape[1]
    assert result['beta_est'].shape == (B, d), 'beta_est shape'
    assert result['sigma_rfx_est'].shape == (B, q), 'sigma_rfx_est shape'
    assert result['blup_est'].shape == (B, max_m, q), 'blup_est shape'
    assert result['phi_pearson'].shape == (B,), 'phi_pearson shape'
    assert result['psi_0'].shape == (B,), 'psi_0 shape'
    assert result['Psi_pql'].shape == (B, q, q), 'Psi_pql shape'
    assert result['Psi_lap'].shape == (B, q, q), 'Psi_lap shape'
    assert result['mean_Hg_inv'].shape == (B, q, q), 'mean_Hg_inv shape'


# ---------------------------------------------------------------------------
# 2. Finite / validity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('likelihood_family', [1, 2])
def test_glmm_finite_and_valid(likelihood_family):
    rng = np.random.default_rng(SEED + 1)
    B, d, q, m = 8, 3, 2, 12

    datasets = [_gen_dataset(rng, d, q, likelihood_family, m=m) for _ in range(B)]
    bt = _collate(datasets, d, q)

    result = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        likelihood_family=likelihood_family,
    )

    for key, val in result.items():
        assert torch.isfinite(val).all(), f'{key} contains non-finite values'

    # sigma_rfx must be non-negative
    assert (result['sigma_rfx_est'] >= 0).all(), 'sigma_rfx_est must be >= 0'

    # Psi_lap must be PSD: all eigenvalues >= 0 (up to small numerical tolerance)
    vals = torch.linalg.eigvalsh(result['Psi_lap'])
    assert (vals >= -1e-5).all(), f'Psi_lap eigenvalues below -1e-5: {vals.min()}'


# ---------------------------------------------------------------------------
# 3. Trivial case: zero random-effects variance → psi_0 ≈ 0, Psi_lap ≈ 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('likelihood_family', [1, 2])
def test_glmm_trivial_zero_rfx(likelihood_family):
    """When all random effects are exactly 0, variance estimates should be near 0."""
    rng = np.random.default_rng(SEED + 2)
    B, d, q, m, n_per_group = 16, 2, 1, 20, 50

    # build datasets forcing sigma_rfx → 0
    datasets = []
    for _ in range(B):
        ns_arr = np.full(m, n_per_group, dtype=int)
        hp = hypersample(rng, d, q, likelihood_family=likelihood_family)
        hp['tau_rfx'] = np.zeros(q)  # force scale to zero
        pr = Prior(rng, hp)
        des = Synthesizer(rng, toy=True)
        ds = Simulator(rng, pr, des, ns_arr).sample()
        ds['Z'] = ds['X'][:, :q].copy()
        datasets.append(ds)

    bt = _collate(datasets, d, q)
    result = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        likelihood_family=likelihood_family,
    )

    # psi_0 should be near 0 (clipped)
    assert (result['psi_0'] >= 0).all(), 'psi_0 must be >= 0'
    # Psi_lap diagonal (sigma_rfx^2) should be small on average
    psi_lap_diag_mean = result['Psi_lap'].diagonal(dim1=-2, dim2=-1).mean().item()
    assert (
        psi_lap_diag_mean < 1.0
    ), f'Psi_lap diagonal too large for zero-rfx case: {psi_lap_diag_mean:.4f}'


def test_glmm_bernoulli_separation_does_not_inflate_sigma_rfx():
    """Separated binary groups should not turn non-identifiability into huge sigma_rfx."""
    B, m, n_per_group, d, q = 3, 10, 6, 2, 1
    Xm = torch.zeros(B, m, n_per_group, d)
    Zm = torch.zeros(B, m, n_per_group, q)
    ym = torch.zeros(B, m, n_per_group)
    mask_n = torch.ones(B, m, n_per_group)
    mask_m = torch.ones(B, m)
    ns = torch.full((B, m), float(n_per_group))
    n_total = torch.full((B,), m * n_per_group)

    x = torch.linspace(-1.0, 1.0, n_per_group)
    Xm[..., 0] = 1.0
    Xm[..., 1] = x
    Zm[..., 0] = 1.0

    ym[:, 1::2] = 1.0
    ym[1, :2] = 0.0
    ym[2, -2:] = 1.0

    result = glmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=1,
    )

    assert torch.isfinite(result['sigma_rfx_est']).all()
    assert torch.isfinite(result['blup_est']).all()
    assert result['sigma_rfx_est'].amax().item() < 8.0
    vals = torch.linalg.eigvalsh(result['Psi_lap'])
    assert (vals >= -1e-5).all()


def test_glmm_normal_rank_deficient_z_groups_do_not_inflate_sigma_rfx():
    B, m, n_per_group, d, q = 1, 24, 5, 4, 2
    Xm = torch.zeros(B, m, n_per_group, d)
    Zm = torch.zeros(B, m, n_per_group, q)
    mask_n = torch.ones(B, m, n_per_group)
    mask_m = torch.ones(B, m)
    ns = torch.full((B, m), float(n_per_group))
    n_total = torch.full((B,), m * n_per_group)

    x = torch.linspace(-1.0, 1.0, n_per_group)
    beta = torch.tensor([0.8, 0.4, 0.8, -0.8])
    for g in range(m):
        Xm[:, g, :, 0] = 1.0
        Xm[:, g, :, 1] = float(g % 2)
        Xm[:, g, :, 2] = x
        Xm[:, g, :, 3] = float((g % 3) - 1) * x
        Zm[:, g, :, 0] = 1.0
        Zm[:, g, :, 1] = 1.0

    ym = torch.einsum('bmnd,d->bmn', Xm, beta)
    ym = ym + 0.002 * torch.sin(torch.arange(m * n_per_group).reshape(1, m, n_per_group))

    result = glmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=0,
    )

    assert torch.isfinite(result['sigma_rfx_est']).all()
    assert torch.isfinite(result['Psi']).all()
    assert torch.isfinite(result['blup_est']).all()
    assert result['sigma_rfx_est'].amax().item() < 0.5
    assert result['blup_est'].abs().amax().item() < 0.5


def test_glmm_normal_mixed_rank_z_groups_keep_sigma_bounded():
    B, m, n_per_group, d, q = 1, 16, 8, 3, 2
    Xm = torch.zeros(B, m, n_per_group, d)
    Zm = torch.zeros(B, m, n_per_group, q)
    mask_n = torch.ones(B, m, n_per_group)
    mask_m = torch.ones(B, m)
    ns = torch.full((B, m), float(n_per_group))
    n_total = torch.full((B,), m * n_per_group)

    x = torch.linspace(-1.0, 1.0, n_per_group)
    beta = torch.tensor([0.2, -0.4, 0.1])
    rfx = torch.zeros(m, q)
    for g in range(m):
        Xm[:, g, :, 0] = 1.0
        Xm[:, g, :, 1] = x
        Xm[:, g, :, 2] = float(g - m / 2) / m
        Zm[:, g, :, 0] = 1.0
        if g < m // 2:
            Zm[:, g, :, 1] = 1.0
        else:
            Zm[:, g, :, 1] = x
        rfx[g, 0] = 0.25 * torch.sin(torch.tensor(float(g)))
        rfx[g, 1] = 0.35 * torch.cos(torch.tensor(float(g)))

    ym = torch.einsum('bmnd,d->bmn', Xm, beta)
    ym = ym + torch.einsum('bmnq,mq->bmn', Zm, rfx)
    ym = ym + 0.05 * torch.sin(torch.arange(m * n_per_group).reshape(1, m, n_per_group))

    result = glmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=0,
    )

    assert torch.isfinite(result['sigma_eps_est']).all()
    assert torch.isfinite(result['sigma_rfx_est']).all()
    assert torch.isfinite(result['blup_est']).all()
    assert result['sigma_eps_est'].item() < 0.2
    assert result['sigma_rfx_est'].amax().item() < 1.5


def test_glmm_normal_blups_use_beta_for_blup_without_changing_beta_est():
    B, m, n_per_group, d, q = 1, 20, 12, 4, 1
    Xm = torch.zeros(B, m, n_per_group, d)
    Zm = torch.zeros(B, m, n_per_group, q)
    mask_n = torch.ones(B, m, n_per_group)
    mask_m = torch.ones(B, m)
    ns = torch.full((B, m), float(n_per_group))
    n_total = torch.full((B,), m * n_per_group)

    x = torch.linspace(-1.0, 1.0, n_per_group)
    group_x = torch.linspace(-1.0, 1.0, m)
    beta = torch.tensor([0.5, 0.8, -0.4, 0.3])
    rfx = torch.zeros(B, m, q)
    for g in range(m):
        Xm[:, g, :, 0] = 1.0
        Xm[:, g, :, 1] = x
        Xm[:, g, :, 2] = group_x[g]
        Xm[:, g, :, 3] = group_x[g] * x
        Zm[:, g, :, 0] = x
        rfx[:, g, 0] = 0.7 * torch.sin(torch.tensor(float(g)))

    ym = torch.einsum('bmnd,d->bmn', Xm, beta)
    ym = ym + torch.einsum('bmnq,bmq->bmn', Zm, rfx)
    ym = ym + 0.02 * torch.sin(torch.arange(m * n_per_group).reshape(1, m, n_per_group))

    result = glmm(
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        n_total,
        likelihood_family=0,
    )

    X_masked = Xm * mask_n[..., None]
    XtX = torch.einsum('bmnd,bmnk->bdk', X_masked, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', X_masked, ym)
    beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
    beta_for_blup = 0.35 * result['beta_est'] + 0.65 * beta_ols

    def blup_for(beta_value):
        ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
        eye_q = torch.eye(q)
        vals, vecs = _eighWithJitter(
            result['Psi'] + result['sigma_eps_est'].square()[:, :, None] * 1e-4 * eye_q
        )
        psi_inv = vecs @ torch.diag_embed(1.0 / vals.clamp(min=1e-30)) @ vecs.mT
        inner = result['sigma_eps_est'].square()[:, :, None, None] * psi_inv[:, None] + ZtZ
        W_g = _safeSolve(inner + _adaptiveRidgeBm(inner), eye_q.expand(B, m, q, q))
        resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_value)) * mask_n
        ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
        return torch.einsum('bmqr,bmr->bmq', W_g, ztr).clamp(-20.0, 20.0)

    assert torch.linalg.vector_norm(beta_ols - result['beta_est']).item() > 1e-3
    assert torch.allclose(result['blup_est'], blup_for(beta_for_blup), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(
        result['blup_est'], blup_for(result['beta_est']), atol=1e-4, rtol=1e-4
    )


def test_glmm_normal_laplace_eb_default_smoke():
    rng = np.random.default_rng(SEED + 21)
    B, d, q, m, n_per_group = 3, 3, 2, 10, 16
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=0, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)

    result = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        likelihood_family=0,
        nu_ffx=torch.as_tensor(np.stack([ds['nu_ffx'] for ds in datasets]), dtype=torch.float32),
        tau_ffx=torch.as_tensor(np.stack([ds['tau_ffx'] for ds in datasets]), dtype=torch.float32),
        family_ffx=torch.as_tensor([int(ds['family_ffx']) for ds in datasets], dtype=torch.long),
        tau_rfx=torch.as_tensor(np.stack([ds['tau_rfx'] for ds in datasets]), dtype=torch.float32),
        family_sigma_rfx=torch.as_tensor(
            [int(ds['family_sigma_rfx']) for ds in datasets], dtype=torch.long
        ),
        tau_eps=torch.as_tensor([float(ds['tau_eps']) for ds in datasets], dtype=torch.float32),
        family_sigma_eps=torch.as_tensor(
            [int(ds['family_sigma_eps']) for ds in datasets], dtype=torch.long
        ),
        mask_d=torch.ones(B, d, dtype=torch.bool),
        mask_q=torch.ones(B, q, dtype=torch.bool),
        normal_laplace_eb_steps=2,
        normal_laplace_eb_sigma_grid_refine=True,
    )

    assert result['beta_est'].shape == (B, d)
    assert result['sigma_rfx_est'].shape == (B, q)
    assert result['blup_est'].shape == (B, m, q)
    assert torch.isfinite(result['sigma_rfx_est']).all()
    assert torch.isfinite(result['blup_est']).all()
    assert torch.isfinite(result['normal_laplace_eb_accept']).all()
    assert torch.isfinite(result['normal_laplace_eb_sigma_grid_accept']).all()
    assert torch.isfinite(result['normal_laplace_eb_blup_guard']).all()
    assert torch.isfinite(result['normal_laplace_eb_steps']).all()


def test_refine_normal_map_beta_sigma_grid_replaces_capped_report_only():
    B, m, n_per_group, d, q = 1, 6, 5, 6, 1
    Xm = torch.zeros(B, m, n_per_group, d)
    Xm[..., 0] = 1.0
    Zm = torch.zeros(B, m, n_per_group, q)
    ym = torch.zeros(B, m, n_per_group)
    mask_n = torch.ones(B, m, n_per_group)
    mask_m = torch.ones(B, m)
    ns = torch.full((B, m), float(n_per_group))
    beta_start = torch.full((B, d), 10.0)
    stats = {
        'beta_est': beta_start,
        'sigma_rfx_est': torch.full((B, q), 0.5),
        'sigma_eps_est': torch.full((B, 1), 1.0),
        'Psi': torch.eye(q).expand(B, q, q).clone() * 0.25,
    }

    result = refineNormalMapSrfx(
        stats,
        Xm,
        ym,
        Zm,
        mask_n,
        mask_m,
        ns,
        nu_ffx=torch.zeros(B, d),
        tau_ffx=torch.ones(B, d),
        family_ffx=torch.zeros(B, dtype=torch.long),
        tau_rfx=torch.ones(B, q),
        family_sigma_rfx=torch.zeros(B, dtype=torch.long),
        tau_eps=torch.ones(B),
        family_sigma_eps=torch.zeros(B, dtype=torch.long),
        mask_d=torch.ones(B, d, dtype=torch.bool),
        mask_q=torch.ones(B, q, dtype=torch.bool),
        n_steps=1,
        lr=0.0,
        recompute_blup=False,
        beta_prior_cap=4.0,
        beta_sigma_grid=True,
        beta_sigma_grid_scales=(0.75, 1.0, 1.3333333),
    )

    assert torch.equal(result['normal_map_beta_stabilized'], torch.ones(B))
    assert torch.all(result['beta_est'].abs() < 4.0)
    assert torch.allclose(result['normal_map_beta_for_blup'], beta_start)


# ---------------------------------------------------------------------------
# 4. Recovery test: nonzero rfx → sigma_rfx_est should be positive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('likelihood_family', [1, 2])
def test_glmm_recovers_nonzero_rfx(likelihood_family):
    """With substantial rfx variance, sigma_rfx_est should be clearly positive."""
    rng = np.random.default_rng(SEED + 3)
    B, d, q, m, n_per_group = 32, 2, 1, 20, 40

    datasets = [
        _gen_dataset(rng, d, q, likelihood_family, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    # keep only datasets where true sigma_rfx > 0.2
    selected = [ds for ds in datasets if float(ds['sigma_rfx'][0]) > 0.2]
    if len(selected) < 8:
        pytest.skip('not enough high-rfx datasets; re-run with different seed')

    bt = _collate(selected, d, q)
    result = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        likelihood_family=likelihood_family,
    )

    mean_sigma_rfx_est = result['sigma_rfx_est'].mean().item()
    assert (
        mean_sigma_rfx_est > 0.01
    ), f'sigma_rfx_est should be positive when true rfx is large, got {mean_sigma_rfx_est:.4f}'


def test_refine_bernoulli_laplace_eb_smoke_q1():
    """Bernoulli diagonal Laplace-EB path is finite and shape-compatible."""
    rng = np.random.default_rng(SEED + 13)
    B, d, q, m, n_per_group = 4, 2, 1, 8, 18
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=1, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)
    stats = lmmBernoulli(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
    )

    nu_ffx = torch.as_tensor(np.stack([ds['nu_ffx'] for ds in datasets]), dtype=torch.float32)
    tau_ffx = torch.as_tensor(np.stack([ds['tau_ffx'] for ds in datasets]), dtype=torch.float32)
    family_ffx = torch.as_tensor([int(ds['family_ffx']) for ds in datasets], dtype=torch.long)
    tau_rfx = torch.as_tensor(np.stack([ds['tau_rfx'] for ds in datasets]), dtype=torch.float32)
    family_sigma_rfx = torch.as_tensor(
        [int(ds['family_sigma_rfx']) for ds in datasets], dtype=torch.long
    )
    mask_d = torch.ones(B, d, dtype=torch.bool)
    mask_q = torch.ones(B, q, dtype=torch.bool)

    result = refineBernoulliLaplaceEb(
        stats,
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        nu_ffx=nu_ffx,
        tau_ffx=tau_ffx,
        family_ffx=family_ffx,
        tau_rfx=tau_rfx,
        family_sigma_rfx=family_sigma_rfx,
        mask_d=mask_d,
        mask_q=mask_q,
        n_steps=3,
        n_inner=2,
        n_final=2,
    )

    assert result['beta_est'].shape == (B, d)
    assert result['sigma_rfx_est'].shape == (B, q)
    assert result['blup_est'].shape == stats['blup_est'].shape
    assert result['blup_var'].shape == stats['blup_var'].shape
    assert torch.isfinite(result['beta_est']).all()
    assert torch.isfinite(result['sigma_rfx_est']).all()
    assert torch.isfinite(result['blup_est']).all()
    assert torch.isfinite(result['blup_var']).all()
    assert (result['sigma_rfx_est'] >= 0).all()
    assert torch.allclose(
        result['Psi_lap'].diagonal(dim1=-2, dim2=-1),
        result['sigma_rfx_est'].square(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_refine_bernoulli_laplace_eb_blup_fallback():
    """Bernoulli EB can keep incoming BLUPs when the β jump trips the fallback."""
    rng = np.random.default_rng(SEED + 16)
    B, d, q, m, n_per_group = 3, 3, 1, 6, 12
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=1, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)
    stats = lmmBernoulli(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
    )

    result = refineBernoulliLaplaceEb(
        stats,
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        mask_d=torch.ones(B, d, dtype=torch.bool),
        mask_q=torch.ones(B, q, dtype=torch.bool),
        n_steps=1,
        n_inner=1,
        n_final=1,
        accept_only_improved=False,
        blup_fallback_beta_jump=0.0,
        return_diagnostics=True,
    )

    assert torch.allclose(result['blup_est'], stats['blup_est'])
    assert torch.allclose(result['blup_var'], stats['blup_var'])
    assert torch.equal(result['laplace_eb_blup_fallback'], torch.ones(B))
    assert torch.isfinite(result['laplace_eb_beta_jump']).all()


def test_refine_bernoulli_laplace_eb_beta_output_cap_trigger():
    """Bernoulli EB caps only separation-scale β summaries while leaving ordinary β untouched."""
    rng = np.random.default_rng(SEED + 17)
    B, d, q, m, n_per_group = 2, 2, 1, 5, 10
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=1, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)
    stats = lmmBernoulli(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
    )
    stats = dict(stats)
    stats['beta_est'] = torch.tensor([[12.0, -2.0], [5.0, -5.0]])

    result = refineBernoulliLaplaceEb(
        stats,
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        mask_d=torch.ones(B, d, dtype=torch.bool),
        mask_q=torch.ones(B, q, dtype=torch.bool),
        n_steps=1,
        n_inner=1,
        n_final=1,
        lr=0.0,
        accept_only_improved=False,
        beta_output_cap=3.0,
        beta_output_cap_trigger=8.0,
    )

    assert torch.all(result['beta_est'][0].abs() <= 3.0)
    assert torch.allclose(result['beta_est'][1], stats['beta_est'][1])


def test_refine_bernoulli_laplace_eb_sigma_prior_cap_recomputes_blup():
    """Bernoulli EB can cap sigma against the prior scale and keep BLUP outputs finite."""
    rng = np.random.default_rng(SEED + 18)
    B, d, q, m, n_per_group = 2, 2, 1, 5, 10
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=1, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)
    stats = lmmBernoulli(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
    )
    stats = dict(stats)
    stats['sigma_rfx_est'] = torch.full((B, q), 10.0)
    stats['Psi_lap'] = torch.diag_embed(stats['sigma_rfx_est'].square())
    tau_rfx = torch.ones(B, q)

    result = refineBernoulliLaplaceEb(
        stats,
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        tau_rfx=tau_rfx,
        family_sigma_rfx=torch.as_tensor(
            [int(ds['family_sigma_rfx']) for ds in datasets], dtype=torch.long
        ),
        mask_d=torch.ones(B, d, dtype=torch.bool),
        mask_q=torch.ones(B, q, dtype=torch.bool),
        n_steps=1,
        n_inner=1,
        n_final=1,
        lr=0.0,
        accept_only_improved=False,
        sigma_prior_cap=2.0,
        recompute_blup_after_calibration=True,
        return_diagnostics=True,
    )

    assert torch.all(result['sigma_rfx_est'] <= 2.0)
    assert torch.isfinite(result['blup_est']).all()
    assert torch.equal(result['laplace_eb_sigma_prior_capped'], torch.ones(B))


def test_glmm_bernoulli_laplace_eb_flag_smoke():
    """Bernoulli EB is available through glmm() behind an explicit flag."""
    rng = np.random.default_rng(SEED + 14)
    B, d, q, m, n_per_group = 2, 2, 1, 6, 12
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=1, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)

    result = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        likelihood_family=1,
        nu_ffx=torch.as_tensor(np.stack([ds['nu_ffx'] for ds in datasets]), dtype=torch.float32),
        tau_ffx=torch.as_tensor(np.stack([ds['tau_ffx'] for ds in datasets]), dtype=torch.float32),
        family_ffx=torch.as_tensor([int(ds['family_ffx']) for ds in datasets], dtype=torch.long),
        tau_rfx=torch.as_tensor(np.stack([ds['tau_rfx'] for ds in datasets]), dtype=torch.float32),
        family_sigma_rfx=torch.as_tensor(
            [int(ds['family_sigma_rfx']) for ds in datasets], dtype=torch.long
        ),
        mask_d=torch.ones(B, d, dtype=torch.bool),
        mask_q=torch.ones(B, q, dtype=torch.bool),
        bernoulli_laplace_eb=True,
        bernoulli_laplace_eb_diagnostics=True,
    )

    assert result['beta_est'].shape == (B, d)
    assert result['sigma_rfx_est'].shape == (B, q)
    assert result['blup_est'].shape == (B, m, q)
    assert torch.isfinite(result['beta_est']).all()
    assert torch.isfinite(result['sigma_rfx_est']).all()
    assert torch.isfinite(result['blup_est']).all()
    assert torch.isfinite(result['laplace_eb_accept']).all()
    assert torch.isfinite(result['laplace_eb_steps']).all()
    assert torch.isfinite(result['laplace_eb_blup_fallback']).all()
    assert torch.isfinite(result['laplace_eb_beta_jump']).all()
    assert torch.isfinite(result['laplace_eb_beta_output_capped']).all()
    assert torch.isfinite(result['laplace_eb_sigma_prior_capped']).all()
    assert result['laplace_eb_steps'].min().item() >= 1.0


def test_glmm_bernoulli_laplace_eb_default_preset_matches_explicit_kwargs():
    """The retained Bernoulli EB preset and Bernoulli default match explicit kwargs."""
    rng = np.random.default_rng(SEED + 19)
    B, d, q, m, n_per_group = 1, 2, 1, 4, 8
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=1, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)
    common = dict(
        likelihood_family=1,
        nu_ffx=torch.as_tensor(np.stack([ds['nu_ffx'] for ds in datasets]), dtype=torch.float32),
        tau_ffx=torch.as_tensor(np.stack([ds['tau_ffx'] for ds in datasets]), dtype=torch.float32),
        family_ffx=torch.as_tensor([int(ds['family_ffx']) for ds in datasets], dtype=torch.long),
        tau_rfx=torch.as_tensor(np.stack([ds['tau_rfx'] for ds in datasets]), dtype=torch.float32),
        family_sigma_rfx=torch.as_tensor(
            [int(ds['family_sigma_rfx']) for ds in datasets], dtype=torch.long
        ),
        mask_d=torch.ones(B, d, dtype=torch.bool),
        mask_q=torch.ones(B, q, dtype=torch.bool),
        bernoulli_laplace_eb_diagnostics=True,
    )

    default = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        **common,
    )
    preset = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        bernoulli_laplace_eb='bernoulli_eb',
        **common,
    )
    explicit = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        bernoulli_laplace_eb=True,
        bernoulli_laplace_eb_steps=24,
        bernoulli_laplace_eb_inner=4,
        bernoulli_laplace_eb_final=8,
        bernoulli_laplace_eb_lr=0.05,
        bernoulli_laplace_eb_beta_output_cap=3.0,
        bernoulli_laplace_eb_beta_output_cap_trigger=8.0,
        bernoulli_laplace_eb_sigma_prior_cap=2.5,
        bernoulli_laplace_eb_sigma_prior_cap_min_d=5,
        **common,
    )

    for result in (default, preset):
        assert torch.allclose(result['beta_est'], explicit['beta_est'])
        assert torch.allclose(result['sigma_rfx_est'], explicit['sigma_rfx_est'])
        assert torch.allclose(result['blup_est'], explicit['blup_est'])
        assert torch.equal(result['laplace_eb_steps'], explicit['laplace_eb_steps'])


def test_glmm_bernoulli_laplace_eb_auto_gate_smoke():
    """Bernoulli EB auto mode routes only gated datasets through the EB refinement."""
    rng = np.random.default_rng(SEED + 15)
    B, d, q, m, n_per_group = 2, 4, 1, 5, 10
    datasets = [
        _gen_dataset(rng, d, q, likelihood_family=1, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)
    mask_d = torch.tensor([[True, True, False, False], [True, True, True, True]])

    result = glmm(
        bt['Xm'],
        bt['ym'],
        bt['Zm'],
        bt['mask_n'],
        bt['mask_m'],
        bt['ns'],
        bt['n_total'],
        likelihood_family=1,
        nu_ffx=torch.as_tensor(np.stack([ds['nu_ffx'] for ds in datasets]), dtype=torch.float32),
        tau_ffx=torch.as_tensor(np.stack([ds['tau_ffx'] for ds in datasets]), dtype=torch.float32),
        family_ffx=torch.as_tensor([int(ds['family_ffx']) for ds in datasets], dtype=torch.long),
        tau_rfx=torch.as_tensor(np.stack([ds['tau_rfx'] for ds in datasets]), dtype=torch.float32),
        family_sigma_rfx=torch.as_tensor(
            [int(ds['family_sigma_rfx']) for ds in datasets], dtype=torch.long
        ),
        mask_d=mask_d,
        mask_q=torch.ones(B, q, dtype=torch.bool),
        bernoulli_laplace_eb='auto',
        bernoulli_laplace_eb_diagnostics=True,
        bernoulli_laplace_eb_gate_min_d=4,
        bernoulli_laplace_eb_gate_min_sigma=None,
        bernoulli_laplace_eb_gate_eta_abs=None,
    )

    assert result['beta_est'].shape == (B, d)
    assert result['sigma_rfx_est'].shape == (B, q)
    assert torch.equal(result['laplace_eb_gate'], torch.tensor([0.0, 1.0]))
    assert result['laplace_eb_steps'][0].item() == 0.0
    assert result['laplace_eb_steps'][1].item() >= 1.0
    assert torch.isfinite(result['beta_est']).all()
    assert torch.isfinite(result['sigma_rfx_est']).all()
    assert torch.isfinite(result['blup_est']).all()


# ---------------------------------------------------------------------------
# 5. Smoke test: Approximator._dataStatistics with non-normal families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('likelihood_family', [1, 2])
def test_approximator_data_statistics_smoke(likelihood_family):
    """glmmFull integrates correctly via Approximator._dataStatistics."""
    from metabeta.models.approximator import Approximator
    from metabeta.utils.config import ApproximatorConfig, SummarizerConfig, PosteriorConfig

    rng = np.random.default_rng(SEED + 4)
    B, d, q, m, n_per_group = 4, 2, 1, 8, 20

    datasets = [
        _gen_dataset(rng, d, q, likelihood_family, m=m, n_per_group=n_per_group) for _ in range(B)
    ]
    bt = _collate(datasets, d, q)

    # build a minimal Approximator config
    s_cfg = SummarizerConfig(
        d_model=16,
        d_ff=16,
        d_output=16,
        n_blocks=1,
    )
    p_cfg = PosteriorConfig(
        n_blocks=2,
    )
    app_cfg = ApproximatorConfig(
        d_ffx=d,
        d_rfx=q,
        likelihood_family=likelihood_family,
        summarizer_l=s_cfg,
        summarizer_g=s_cfg,
        posterior_l=p_cfg,
        posterior_g=p_cfg,
    )
    model = Approximator(app_cfg)

    # build a fake data dict that _dataStatistics expects
    # include all keys needed by _dataStatistics
    data = {
        'X': bt['Xm'],
        'y': bt['ym'],
        'Z': bt['Zm'],
        'mask_n': bt['mask_n'],
        'mask_m': bt['mask_m'],
        'ns': bt['ns'],
        'n': bt['n_total'].long(),
        'm': bt['mask_m'].sum(dim=1).long(),
    }

    stats = model._dataStatistics(data)

    assert 'beta_est' in stats
    assert 'sigma_rfx_est' in stats
    assert 'blup_est' in stats
    assert stats['beta_est'].shape == (B, d)
    assert stats['sigma_rfx_est'].shape == (B, q)
    assert stats['blup_est'].shape == (B, bt['Xm'].shape[1], q)
    assert torch.isfinite(stats['beta_est']).all()
    assert torch.isfinite(stats['sigma_rfx_est']).all()
    assert torch.isfinite(stats['blup_est']).all()
