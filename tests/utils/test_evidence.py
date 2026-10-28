"""Tests for the t-proposal log-evidence in posthoc/evidence.py."""

import math

import numpy as np
import pytest
import torch
from torch import distributions as D

from metabeta.posthoc.evidence import tEvidence
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler
from metabeta.utils.dataloader import sliceBatch
from metabeta.utils.results import Proposal


def _batch(structures, m, n, lf=0):
    """Datasets of (d, q, eta) simulated from the model, padded to the batch's d and q (<= 2);
    priors N(0, 1) on β and half-N(1) on the SDs."""
    b = len(structures)
    d_max, q_max = max(s[0] for s in structures), max(s[1] for s in structures)
    X, Z, y = torch.zeros(b, m, n, d_max), torch.zeros(b, m, n, q_max), torch.zeros(b, m, n)
    mask_d = torch.zeros(b, d_max, dtype=torch.bool)
    mask_q = torch.zeros(b, q_max, dtype=torch.bool)
    for i, (d, q, _) in enumerate(structures):
        X[i, ..., 0] = 1.0
        X[i, ..., 1:d] = torch.randn(m, n, d - 1)
        Z[i, ..., :q] = X[i, ..., :q]
        fixed = X[i, ..., :d] @ (0.5 * torch.randn(d))
        eta = fixed + (Z[i, ..., :q] * torch.randn(m, 1, q)).sum(-1)
        y[i] = eta + 0.5 * torch.randn(m, n) if lf == 0 else torch.bernoulli(torch.sigmoid(eta))
        mask_d[i, :d], mask_q[i, :q] = True, True
    return {
        'X': X,
        'Z': Z,
        'y': y,
        'nu_ffx': torch.zeros(b, d_max),
        'tau_ffx': torch.ones(b, d_max),
        'tau_rfx': torch.ones(b, q_max),
        'tau_eps': torch.ones(b),
        'family_ffx': torch.zeros(b, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(b, dtype=torch.long),
        'family_sigma_eps': torch.zeros(b, dtype=torch.long),
        'eta_rfx': torch.tensor([s[2] for s in structures]),
        'mask_d': mask_d,
        'mask_q': mask_q,
        'mask_mq': mask_q.unsqueeze(1).expand(b, m, q_max).clone(),
        'mask_m': torch.ones(b, m, dtype=torch.bool),
        'mask_n': torch.ones(b, m, n, dtype=torch.bool),
        'n': torch.full((b,), m * n),
        'sd_y': torch.ones(b),
    }


def _posterior(data, lf, n_pool, n_post=4000):
    """Approximate posterior draws: sampling-importance-resampling from a prior pool by the
    rfx-marginal likelihood (Laplace for GLMMs), in the batch's padded layout."""
    data64 = {k: v.double() if v.is_floating_point() else v for k, v in data.items()}
    b, m = data['X'].shape[:2]
    mask_d, mask_q = data['mask_d'].unsqueeze(1), data['mask_q'].unsqueeze(1)
    q_max = mask_q.shape[-1]
    parts = [
        torch.randn(b, n_pool, mask_d.shape[-1]) * mask_d,
        torch.randn(b, n_pool, q_max).abs() * mask_q,
    ]
    if lf == 0:
        parts.append(torch.randn(b, n_pool, 1).abs())
    if q_max == 2:  # LKJ(eta) for q = 2: r = 2 Beta(eta, eta) - 1; r = 0 without the LKJ switch
        corr = (data['eta_rfx'] > 0) & (data['mask_q'].sum(-1) == 2)
        eta = data['eta_rfx'].clamp_min(1.0).view(b, 1)
        r = 2 * D.Beta(eta, eta).sample((n_pool,)).movedim(0, 1) - 1  # (b, n_pool, 1)
        parts.append(r * corr.view(b, 1, 1))
    g = torch.cat(parts, -1).double()
    pool = Proposal(
        {
            'global': {'samples': g, 'log_prob': g.new_zeros(b, n_pool)},
            'local': {
                'samples': g.new_zeros(b, m, n_pool, q_max),
                'log_prob': g.new_zeros(b, m, n_pool),
            },
        },
        has_sigma_eps=lf == 0,
        d_corr=q_max * (q_max - 1) // 2,
    )
    if lf == 0:
        sampler = ImportanceSampler(data64, marginal=True)
    else:
        sampler = LaplaceImportanceSampler(data64, likelihood_family=lf)
    ll, _ = sampler.unnormalizedPosterior(pool)
    pool.subset(torch.multinomial(torch.softmax(ll, -1), n_post, replacement=True))
    pool.data['global']['samples'] = pool.samples_g.float()
    return pool


def _mcse(res):
    """Monte Carlo SE of the IS log-evidence from the relative ESS: Var(log Ẑ) ≈ (1/eff − 1)/S."""
    return np.sqrt((1 / res['eff'] - 1) / res['S'])


def _logEvidenceQuadrature(data) -> float:
    """log p(y) of one Normal dataset with q = 1: β and the random intercepts integrate out in
    closed form, y | σ ~ N(Xν, X diag(τ²) Xᵀ + σ_rfx² Z Zᵀ + σ_eps² I); the two SDs on a
    log-scale grid (trapezoid; the integrand vanishes at both ends)."""
    m, n, d = data['X'].shape[1:]
    X = data['X'][0].double().reshape(m * n, d)
    Z = torch.block_diag(*data['Z'][0].double())  # (m n, m)
    y = data['y'][0].double().reshape(-1)
    base = X @ torch.diag(data['tau_ffx'][0].double() ** 2) @ X.T
    u = torch.linspace(-12.0, 3.0, 301, dtype=torch.double)
    log_terms = []
    for u_rfx in u:
        cov = base + (2 * u_rfx).exp() * Z @ Z.T + (2 * u).exp().view(-1, 1, 1) * torch.eye(m * n)
        ll = D.MultivariateNormal(X @ data['nu_ffx'][0].double(), cov).log_prob(y)
        lp = D.HalfNormal(data['tau_rfx'][0, 0].double()).log_prob(u_rfx.exp()) + u_rfx
        lp = lp + D.HalfNormal(data['tau_eps'][0].double()).log_prob(u.exp()) + u
        log_terms.append(ll + lp)
    h = (u[1] - u[0]).item()
    return (torch.logsumexp(torch.cat(log_terms), 0) + 2 * math.log(h)).item()


def _tEvidence(data, post, lf, n_draws, seed):
    torch.manual_seed(seed)  # the IS² inner draws
    res = tEvidence(data, post, lf, n_draws, torch.Generator().manual_seed(seed))
    return res | {'S': n_draws}


def test_t_evidence_matches_closed_form_gaussian():
    torch.manual_seed(0)
    data = _batch([(2, 1, 0.0)], m=4, n=5)
    post = _posterior(data, lf=0, n_pool=200_000)
    res = _tEvidence(data, post, 0, 20_000, seed=1)
    ref = _logEvidenceQuadrature(data)
    assert res['k'][0] < 0.5
    assert abs(res['log_ev'][0] - ref) < 4 * _mcse(res)[0] + 1e-3, (res['log_ev'][0], ref)


@pytest.mark.parametrize(
    'lf, structures, m, n, n_pool',
    [
        (0, [(2, 1, 0.0), (3, 2, 2.0), (3, 2, 0.0), (2, 1, 0.0)], 5, 6, 200_000),
        (1, [(2, 1, 0.0), (2, 2, 2.0), (2, 1, 0.0)], 8, 8, 50_000),
    ],
)
def test_mixed_batch_matches_single_dataset_calls(lf, structures, m, n, n_pool):
    """Datasets of different d, q and LKJ switch in one batch, one structure twice: the same
    per-dataset log-evidence as one call per dataset, within Monte Carlo error."""
    torch.manual_seed(0)
    data = _batch(structures, m, n, lf)
    post = _posterior(data, lf, n_pool)
    mixed = _tEvidence(data, post, lf, 8000, seed=1)
    for j in range(len(structures)):
        single = _tEvidence(sliceBatch(data, j, j + 1), post.slice_b(j, j + 1), lf, 8000, seed=2)
        assert max(mixed['k'][j], single['k'][0]) < 0.7
        tol = 5 * math.hypot(_mcse(mixed)[j], _mcse(single)[0])
        assert abs(mixed['log_ev'][j] - single['log_ev'][0]) < tol, (j, mixed, single)


@pytest.mark.parametrize('lf', [0, 1])
def test_raw_scale_jacobian(lf):
    """log p(y_raw) = log p(y_std) − n log sd_y for Normal outcomes; GLMM responses are not
    standardized."""
    torch.manual_seed(0)
    data = _batch([(2, 1, 0.0)], m=5, n=6, lf=lf)
    post = _posterior(data, lf, n_pool=20_000)
    res = {}
    for sd_y in (1.0, 2.5):
        data['sd_y'] = torch.full((1,), sd_y)
        res[sd_y] = _tEvidence(data, post, lf, 2000, seed=1)
    shift = -30 * math.log(2.5) if lf == 0 else 0.0
    assert res[2.5]['log_ev'][0] - res[1.0]['log_ev'][0] == pytest.approx(shift, abs=1e-9)
    assert np.array_equal(res[2.5]['k'], res[1.0]['k'])
