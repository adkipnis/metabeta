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
    B, m, n_per_group, d, q = 1, 14, 10, 3, 1
    Xm = torch.zeros(B, m, n_per_group, d)
    Zm = torch.ones(B, m, n_per_group, q)
    mask_n = torch.ones(B, m, n_per_group)
    mask_m = torch.ones(B, m)
    ns = torch.full((B, m), float(n_per_group))
    n_total = torch.full((B,), m * n_per_group)

    x = torch.linspace(-1.0, 1.0, n_per_group)
    group_x = torch.linspace(-1.0, 1.0, m)
    beta = torch.tensor([0.5, 0.8, -0.4])
    rfx = 0.7 * group_x.reshape(1, m, 1)
    for g in range(m):
        Xm[:, g, :, 0] = 1.0
        Xm[:, g, :, 1] = x
        Xm[:, g, :, 2] = group_x[g]

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
    beta_for_blup = 0.5 * result['beta_est'] + 0.5 * beta_ols

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
