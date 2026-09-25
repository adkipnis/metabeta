"""Tests for the importance-sampling-squared (IS²) marginal likelihood in posthoc/laplace_glmm.py:
the per-θ estimate of p(y | θ_g) must be unbiased (Tran et al., arXiv:1309.3339), so the
evidence built on it is unbiased too."""

import math

import pytest
import torch
from torch import distributions as D
from torch.nn import functional as F

from metabeta.posthoc.laplace_glmm import (
    LaplaceImportanceSampler,
    logMarginalLikelihoodAGQ,
    logMarginalLikelihoodIS2,
)
from metabeta.posthoc.metropolis import MetropolisSampler
from metabeta.utils.families import logMarginalLikelihoodNormal
from metabeta.utils.results import Proposal


def _gridLogMarginal(X, Z, y, ffx, sigma_rfx, sigma_eps, mask_m, likelihood_family):
    """log p(y | θ) by dense-grid integration over a scalar random intercept, one θ (float64).

    X, Z, y: per-group lists of (n_j, d), (n_j, 1), (n_j,); ffx (d,), sigma_rfx / sigma_eps scalars.
    """
    grid = torch.linspace(-12.0, 12.0, 8001, dtype=torch.float64)
    log_dx = math.log(float(grid[1] - grid[0]))
    log_prior = -0.5 * (grid / sigma_rfx) ** 2 - math.log(sigma_rfx) - 0.5 * math.log(2 * math.pi)
    out = 0.0
    for j in range(len(X)):
        if not mask_m[j]:
            continue
        eta = (X[j] @ ffx)[None, :] + grid[:, None] * Z[j][:, 0][None, :]  # (G, n)
        yj = y[j][None, :]
        if likelihood_family == 0:
            ll = (
                -0.5 * ((yj - eta) / sigma_eps) ** 2
                - math.log(sigma_eps)
                - 0.5 * math.log(2 * math.pi)
            )
        elif likelihood_family == 1:
            ll = yj * eta - F.softplus(eta)
        else:
            ll = yj * eta - torch.exp(eta) - torch.lgamma(yj + 1.0)
        out += float(torch.logsumexp(ll.sum(-1) + log_prior, 0)) + log_dx
    return out


def _gridPosteriorMean(X, Z, y, ffx, sigma_rfx, likelihood_family):
    """E[b | θ, y_j] of one group's scalar random intercept by dense-grid integration."""
    grid = torch.linspace(-12.0, 12.0, 8001, dtype=torch.float64)
    eta = (X @ ffx)[None, :] + grid[:, None] * Z[:, 0][None, :]
    if likelihood_family == 1:
        ll = y[None, :] * eta - F.softplus(eta)
    else:
        ll = y[None, :] * eta - torch.exp(eta)
    w = torch.softmax(ll.sum(-1) - 0.5 * (grid / sigma_rfx) ** 2, 0)
    return float((w * grid).sum())


def _problem(likelihood_family, seed=0):
    """Two active groups (one tiny, all-success for Bernoulli) + one padded group, q=1 active
    rfx dim + one padded dim; returns batched tensors (b=1) and the per-θ values."""
    g = torch.Generator().manual_seed(seed)
    m, n, d = 3, 6, 2
    X = torch.cat([torch.ones(m, n, 1), torch.randn(m, n, 1, generator=g)], -1).double()
    Z = torch.cat([torch.ones(m, n, 1), torch.zeros(m, n, 1)], -1).double()
    mask_n = torch.ones(m, n, 1, dtype=torch.float64)
    mask_n[0, 3:] = 0.0  # group 0 keeps 3 observations
    if likelihood_family == 0:
        y = torch.randn(m, n, generator=g).double()
    elif likelihood_family == 1:
        y = torch.bernoulli(torch.full((m, n), 0.6), generator=g).double()
        y[0] = 1.0  # all successes: likelihood flat as b → ∞, the heavy-tail case
    else:
        y = torch.poisson(torch.full((m, n), 3.0), generator=g).double()
    y = y * mask_n[..., 0]
    mask_m = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    ffx = torch.tensor([0.3, -0.5], dtype=torch.float64)
    sigma_rfx, sigma_eps = 1.5, 0.8
    return X, Z, y, mask_n, mask_m, ffx, sigma_rfx, sigma_eps


def _estimate(problem, likelihood_family, n_rep, n_inner, **kwargs):
    """n_rep independent IS² estimates of log p(y | θ) at one θ (replicates on the s axis)."""
    X, Z, y, mask_n, mask_m, ffx, sigma_rfx, sigma_eps = problem
    ll, rfx = logMarginalLikelihoodIS2(
        ffx.expand(1, n_rep, -1),
        torch.tensor([sigma_rfx, 0.0], dtype=torch.float64).expand(1, n_rep, -1),
        torch.full((1, n_rep), sigma_eps, dtype=torch.float64),
        y[None, ..., None],
        X[None],
        Z[None],
        mask_n[None],
        mask_m[None, :, None],
        likelihood_family,
        n_inner=n_inner,
        **kwargs,
    )
    return ll[0], rfx[0]  # (n_rep,), (m, n_rep, q)


@pytest.mark.parametrize('likelihood_family', [0, 1, 2])
def test_is2_marginal_is_unbiased(likelihood_family):
    """mean_r exp(log p̂_r) matches the dense-grid p(y | θ) within Monte Carlo error."""
    torch.manual_seed(1)
    problem = _problem(likelihood_family)
    X, Z, y, mask_n, mask_m, ffx, sigma_rfx, sigma_eps = problem
    # the grid sees only each group's observed rows
    keep = [mask_n[j, :, 0].bool() for j in range(3)]
    want = _gridLogMarginal(
        [X[j, keep[j]] for j in range(3)],
        [Z[j, keep[j]] for j in range(3)],
        [y[j, keep[j]] for j in range(3)],
        ffx,
        sigma_rfx,
        sigma_eps,
        mask_m,
        likelihood_family,
    )
    ll, _ = _estimate(problem, likelihood_family, n_rep=20_000, n_inner=2)
    ratio = torch.exp(ll - want)  # (n_rep,), unbiased for 1
    se = ratio.std() / math.sqrt(ratio.numel())
    assert torch.isfinite(ratio).all()
    assert abs(ratio.mean().item() - 1.0) < 4 * se.item() + 1e-3, (ratio.mean(), se)


@pytest.mark.parametrize('likelihood_family', [1, 2])
def test_is2_rfx_weighted_by_estimate_recover_conditional_mean(likelihood_family):
    """Σ_r p̂_r b_r / Σ_r p̂_r → E[b_j | θ, y_j]: the returned draw is the one the pseudo-marginal
    extended target pairs with p̂, so p̂-weighted draws are exact conditional posterior draws."""
    torch.manual_seed(2)
    problem = _problem(likelihood_family)
    X, Z, y, mask_n, mask_m, ffx, sigma_rfx, _ = problem
    ll, rfx = _estimate(problem, likelihood_family, n_rep=20_000, n_inner=2)
    w = torch.softmax(ll, 0)  # (n_rep,)
    for j in range(2):
        keep = mask_n[j, :, 0].bool()
        want = _gridPosteriorMean(
            X[j, keep], Z[j, keep], y[j, keep], ffx, sigma_rfx, likelihood_family
        )
        b = rfx[j, :, 0]
        got = float((w * b).sum())
        se = float(((w * (b - got)) ** 2).sum().sqrt())
        assert abs(got - want) < 4 * se + 1e-3, (j, got, want, se)
    assert (rfx[2] == 0).all()  # padded group
    assert (rfx[..., 1] == 0).all() or rfx[..., 1].abs().max() < 1e-4  # padded rfx dim


# ---------------------------------------------------------------------------
# Evidence and posterior of a tiny random-intercept Bernoulli model (d = q = 1)
# ---------------------------------------------------------------------------

TAU_FFX, TAU_RFX = 1.5, 1.0  # β₀ ~ N(0, 1.5²), σ ~ HalfNormal(1)


def _tinyBernoulli(m=4, n=5, seed=3):
    g = torch.Generator().manual_seed(seed)
    b_true = torch.randn(m, 1, generator=g) * 1.2
    y = torch.bernoulli(torch.sigmoid(0.4 + b_true).expand(m, n), generator=g).double()
    y[0] = 1.0  # one all-success group
    ones = lambda *shape: torch.ones(*shape, dtype=torch.float64)
    return {
        'X': ones(1, m, n, 1),
        'Z': ones(1, m, n, 1),
        'y': y[None],
        'nu_ffx': torch.zeros(1, 1, dtype=torch.float64),
        'tau_ffx': torch.full((1, 1), TAU_FFX, dtype=torch.float64),
        'tau_rfx': torch.full((1, 1), TAU_RFX, dtype=torch.float64),
        'family_ffx': torch.zeros(1, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(1, dtype=torch.long),
        'eta_rfx': torch.zeros(1, dtype=torch.float64),
        'mask_d': torch.ones(1, 1, dtype=torch.bool),
        'mask_q': torch.ones(1, 1, dtype=torch.bool),
        'mask_mq': torch.ones(1, m, 1, dtype=torch.bool),
        'mask_m': torch.ones(1, m, dtype=torch.bool),
        'mask_n': torch.ones(1, m, n, dtype=torch.bool),
    }


def _gridPosterior(data):
    """(log p(D), E[β₀ | D], E[σ | D], E[b_0 | D]) by nested grid integration."""
    y = data['y'][0]  # (m, n)
    beta = torch.linspace(-7.0, 7.0, 281, dtype=torch.float64)
    sigma = torch.linspace(1e-3, 6.0, 300, dtype=torch.float64)
    u = torch.linspace(-9.0, 9.0, 721, dtype=torch.float64)  # b = σ u, u ~ N(0, 1)
    log_du = math.log(float(u[1] - u[0]))
    log_phi = -0.5 * u**2 - 0.5 * math.log(2 * math.pi)
    eta = beta[:, None, None] + sigma[None, :, None] * u[None, None, :]  # (B, S, U)
    ll_u = 0.0  # Σ_j log ∫ p(y_j | β + σ u) φ(u) du, (B, S)
    for j in range(y.shape[0]):
        k = y[j].sum()
        ll_j = k * eta - y.shape[1] * F.softplus(eta)  # (B, S, U)
        lse = torch.logsumexp(ll_j + log_phi, -1) + log_du
        ll_u = ll_u + lse
        if j == 0:
            # E[b_0 | β, σ, y_0] for the posterior mean of the first group's intercept
            w0 = torch.softmax(ll_j + log_phi, -1)
            b0 = (w0 * sigma[None, :, None] * u).sum(-1)
    log_prior = (
        D.Normal(0.0, TAU_FFX).log_prob(beta)[:, None]
        + D.HalfNormal(TAU_RFX).log_prob(sigma)[None, :]
    )
    log_joint = ll_u + log_prior  # (B, S)
    log_cell = math.log(float(beta[1] - beta[0])) + math.log(float(sigma[1] - sigma[0]))
    log_z = float(torch.logsumexp(log_joint.flatten(), 0)) + log_cell
    post = torch.softmax(log_joint.flatten(), 0).view_as(log_joint)
    return (
        log_z,
        float((post * beta[:, None]).sum()),
        float((post * sigma[None, :]).sum()),
        float((post * b0).sum()),
    )


def _gaussianProposal(s, m, loc_beta, scale_beta, scale_sigma):
    """β₀ ~ N, σ ~ HalfNormal (covers σ → 0, where the posterior keeps mass); the rfx are
    zeros, they only seed the Newton search."""
    q_beta = D.Normal(torch.tensor(loc_beta, dtype=torch.float64), scale_beta)
    q_sigma = D.HalfNormal(torch.tensor(scale_sigma, dtype=torch.float64))
    beta, sigma = q_beta.sample((1, s, 1)), q_sigma.sample((1, s, 1))
    log_q = (q_beta.log_prob(beta) + q_sigma.log_prob(sigma)).sum(-1)
    proposed = {
        'global': {'samples': torch.cat([beta, sigma], -1), 'log_prob': log_q},
        'local': {
            'samples': torch.zeros(1, m, s, 1, dtype=torch.float64),
            'log_prob': torch.zeros(1, m, s, dtype=torch.float64),
        },
    }
    return Proposal(proposed, has_sigma_eps=False)


def test_is2_log_evidence_matches_quadrature_under_two_proposals():
    torch.manual_seed(4)
    data = _tinyBernoulli()
    want = _gridPosterior(data)[0]
    m = data['X'].shape[1]
    # the Laplace evidence is off by ≈ −0.05 nats here (tiny groups, one all-success group)
    for params in ((0.6, 1.2, 1.5), (1.0, 1.0, 2.0)):
        sampler = LaplaceImportanceSampler(
            data, n_inner=4, likelihood_family=1, pareto=False, constrain=False
        )
        out = sampler(_gaussianProposal(40_000, m, *params))
        got = out.log_evidence.item()
        assert abs(got - want) < 0.02, (params, got, want)


def test_pseudo_marginal_imh_targets_exact_posterior():
    """With IS² weights the IMH is pseudo-marginal: posterior means of the globals and of an
    all-success group's intercept match quadrature (the Laplace redraw gives E[b_0] ≈ 0.83)."""
    torch.manual_seed(5)
    data = _tinyBernoulli()
    _, want_beta, want_sigma, want_b0 = _gridPosterior(data)
    n_chains, n_steps = 100, 200
    sampler = MetropolisSampler(
        data,
        n_chains=n_chains,
        n_steps=n_steps,
        burnin=20,
        mode='laplace',
        likelihood_family=1,
        n_eff_target=None,
        n_inner=4,
    )
    out, _ = sampler(_gaussianProposal(n_chains * n_steps, data['X'].shape[1], 0.6, 1.2, 1.5))
    assert abs(out.ffx.mean().item() - want_beta) < 0.04
    assert abs(out.sigma_rfx.mean().item() - want_sigma) < 0.03
    assert abs(out.rfx[0, 0].mean().item() - want_b0) < 0.04


# ---------------------------------------------------------------------------
# Adaptive Gauss-Hermite quadrature (AGQ): the deterministic evidence reference
# ---------------------------------------------------------------------------


def _slopeProblem(likelihood_family, seed=6):
    """Two active groups with random intercept + slope (q = 2), one padded group, one padded
    rfx dim (q_max = 3); b = 1, s = 2 global draws."""
    g = torch.Generator().manual_seed(seed)
    m, n, s = 3, 8, 2
    x = torch.randn(m, n, 1, generator=g)
    X = torch.cat([torch.ones(m, n, 1), x], -1).double()[None]
    Z = torch.cat([torch.ones(m, n, 1), x, torch.zeros(m, n, 1)], -1).double()[None]
    ffx = torch.tensor([[[0.2, -0.4], [-0.3, 0.6]]], dtype=torch.float64)
    sigma_rfx = torch.tensor([[[0.9, 0.6, 0.0], [1.4, 0.3, 0.0]]], dtype=torch.float64)
    sigma_eps = torch.tensor([[0.7, 1.1]], dtype=torch.float64)
    if likelihood_family == 0:
        y = torch.randn(1, m, n, 1, generator=g).double()
    elif likelihood_family == 1:
        y = torch.bernoulli(torch.full((1, m, n, 1), 0.5), generator=g).double()
    else:
        y = torch.poisson(torch.full((1, m, n, 1), 2.0), generator=g).double()
    mask_n = torch.ones(1, m, n, 1, dtype=torch.float64)
    mask_m = torch.tensor([[[1.0], [1.0], [0.0]]], dtype=torch.float64)
    return ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m


def _grid2dLogMarginal(ffx, sigma_rfx, y, X, Z, mask_m, likelihood_family):
    """Σ_j log ∫∫ p(y_j | b) N(b; 0, diag σ²) db over a dense 2D grid, per global draw (b, s)."""
    u = torch.linspace(-7.0, 7.0, 561, dtype=torch.float64)  # b_k = σ_k u
    log_du = math.log(float(u[1] - u[0]))
    u1, u2 = torch.meshgrid(u, u, indexing='ij')
    log_phi = -0.5 * (u1**2 + u2**2) - math.log(2 * math.pi)
    out = torch.zeros(ffx.shape[:2], dtype=torch.float64)
    for k in range(ffx.shape[1]):
        s1, s2 = sigma_rfx[0, k, 0], sigma_rfx[0, k, 1]
        for j in range(X.shape[1]):
            if not mask_m[0, j, 0]:
                continue
            eta = (
                (X[0, j] @ ffx[0, k])[None, None, :]
                + (s1 * u1)[..., None] * Z[0, j, :, 0]
                + (s2 * u2)[..., None] * Z[0, j, :, 1]
            )
            yj = y[0, j, :, 0]
            if likelihood_family == 1:
                ll = yj * eta - F.softplus(eta)
            else:
                ll = yj * eta - torch.exp(eta) - torch.lgamma(yj + 1.0)
            out[0, k] += torch.logsumexp((ll.sum(-1) + log_phi).flatten(), 0) + 2 * log_du
    return out


@pytest.mark.parametrize('likelihood_family', [0, 1, 2])
def test_agq_marginal_matches_exact_integral(likelihood_family):
    ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m = _slopeProblem(likelihood_family)
    got = logMarginalLikelihoodAGQ(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, likelihood_family, n_nodes=12
    )
    if likelihood_family == 0:
        want = logMarginalLikelihoodNormal(ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m)
        atol = 1e-6
    else:
        want = _grid2dLogMarginal(ffx, sigma_rfx, y, X, Z, mask_m, likelihood_family)
        atol = 1e-5  # Laplace (n_nodes=1) is off by 0.02-0.04 nats here
    assert got.shape == (1, 2)
    assert torch.allclose(got, want, atol=atol), (got, want)
