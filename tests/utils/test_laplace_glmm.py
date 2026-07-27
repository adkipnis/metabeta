"""Tests for posthoc/laplace_glmm.py: the Laplace marginal must be exact for
Normal likelihoods and match numerical integration for Bernoulli/Poisson."""

import math

import torch

from metabeta.posthoc.laplace_glmm import (
    laplaceRfxModes,
    logMarginalLikelihoodLaplace,
    sampleRfxLaplace,
)
from metabeta.utils.families import logMarginalLikelihoodNormal
from metabeta.utils.regularization import unconstrainedToCholesky


def _makeProblem(b=2, m=3, n=8, d=2, q=2, s=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(b, m, n, d, generator=g)
    Z = torch.randn(b, m, n, q, generator=g)
    ffx = torch.randn(b, s, d, generator=g) * 0.5
    sigma_rfx = torch.rand(b, s, q, generator=g) * 0.9 + 0.1
    sigma_eps = torch.rand(b, s, generator=g) * 0.9 + 0.1
    mask_n = torch.ones(b, m, n, 1)
    mask_m = torch.ones(b, m, 1)
    return X, Z, ffx, sigma_rfx, sigma_eps, mask_n, mask_m


def test_laplace_marginal_exact_for_normal_diag():
    X, Z, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem()
    g = torch.Generator().manual_seed(1)
    y = torch.randn(*X.shape[:3], 1, generator=g)
    want = logMarginalLikelihoodNormal(ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m)
    got, _, _ = logMarginalLikelihoodLaplace(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, likelihood_family=0, n_newton=3
    )
    assert torch.allclose(got, want, atol=1e-2)


def test_laplace_marginal_exact_for_normal_correlated():
    X, Z, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem(seed=2)
    b, s, q = sigma_rfx.shape
    g = torch.Generator().manual_seed(3)
    y = torch.randn(*X.shape[:3], 1, generator=g)
    z_corr = torch.randn(b, s, q * (q - 1) // 2, generator=g)
    L_corr = unconstrainedToCholesky(z_corr, q)
    want = logMarginalLikelihoodNormal(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, L_corr=L_corr
    )
    got, _, _ = logMarginalLikelihoodLaplace(
        ffx,
        sigma_rfx,
        sigma_eps,
        y,
        X,
        Z,
        mask_n,
        mask_m,
        likelihood_family=0,
        L_corr=L_corr,
        n_newton=3,
    )
    assert torch.allclose(got, want, atol=1e-2)


def _bruteForceMarginalGlmm(X, Z, y, ffx, sigma_rfx, likelihood_family):
    """Numerical integration over a fine 1-d rfx grid (q=1 only)."""
    b, m, n, _ = X.shape
    s = ffx.shape[1]
    grid = torch.linspace(-8.0, 8.0, 4001)
    dx = grid[1] - grid[0]
    out = torch.zeros(b, s)
    for i in range(b):
        for k in range(s):
            sig = sigma_rfx[i, k, 0].clamp(min=1e-6)
            log_prior = -0.5 * (grid / sig).pow(2) - sig.log() - 0.5 * math.log(2 * math.pi)
            for j in range(m):
                eta = X[i, j] @ ffx[i, k]  # (n,)
                eta_g = eta.unsqueeze(0) + grid.unsqueeze(1) * Z[i, j, :, 0].unsqueeze(0)
                yj = y[i, j, :, 0].unsqueeze(0)
                if likelihood_family == 1:
                    ll = yj * eta_g - torch.nn.functional.softplus(eta_g)
                else:
                    ll = yj * eta_g - torch.exp(eta_g) - torch.lgamma(yj + 1.0)
                log_int = torch.logsumexp(ll.sum(-1) + log_prior, dim=0) + dx.log()
                out[i, k] += log_int
    return out


def test_laplace_marginal_bernoulli_vs_quadrature():
    torch.manual_seed(4)
    b, m, n, d, q, s = 1, 3, 30, 2, 1, 3
    X = torch.randn(b, m, n, d)
    Z = torch.ones(b, m, n, q)  # random intercept
    ffx = torch.randn(b, s, d) * 0.5
    sigma_rfx = torch.rand(b, s, q) * 0.8 + 0.3
    sigma_eps = torch.zeros(b, s)
    eta_true = (X @ torch.randn(d)).squeeze(-1) if False else X[..., 0]
    y = torch.bernoulli(torch.sigmoid(eta_true)).unsqueeze(-1)
    mask_n = torch.ones(b, m, n, 1)
    mask_m = torch.ones(b, m, 1)

    got, _, _ = logMarginalLikelihoodLaplace(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, likelihood_family=1, n_newton=8
    )
    want = _bruteForceMarginalGlmm(X, Z, y, ffx, sigma_rfx, likelihood_family=1)
    # Laplace is approximate for Bernoulli — n=30 per group keeps the error small
    assert torch.allclose(got, want, atol=0.15 * m)


def test_laplace_marginal_poisson_vs_quadrature():
    torch.manual_seed(5)
    b, m, n, d, q, s = 1, 3, 25, 2, 1, 3
    X = torch.randn(b, m, n, d) * 0.5
    Z = torch.ones(b, m, n, q)
    ffx = torch.randn(b, s, d) * 0.3
    sigma_rfx = torch.rand(b, s, q) * 0.6 + 0.2
    sigma_eps = torch.zeros(b, s)
    y = torch.poisson(torch.exp(X[..., 0].clamp(max=3.0))).unsqueeze(-1)
    mask_n = torch.ones(b, m, n, 1)
    mask_m = torch.ones(b, m, 1)

    got, _, _ = logMarginalLikelihoodLaplace(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, likelihood_family=2, n_newton=8
    )
    want = _bruteForceMarginalGlmm(X, Z, y, ffx, sigma_rfx, likelihood_family=2)
    assert torch.allclose(got, want, atol=0.1 * m)


def test_laplace_modes_masked_groups_zero():
    X, Z, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem(seed=6)
    y = torch.bernoulli(torch.sigmoid(X[..., 0])).unsqueeze(-1)
    mask_m = mask_m.clone()
    mask_m[:, -1] = 0.0
    modes, chol_H, _, _ = laplaceRfxModes(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, likelihood_family=1
    )
    assert (modes[:, -1] == 0).all()
    rfx = sampleRfxLaplace(modes, chol_H, mask_m)
    assert (rfx[:, -1] == 0).all()
    assert (rfx[:, :-1] != 0).any()


def test_laplace_padded_qdim_invariance():
    """A padded rfx dim (zero Z column, zero sigma) must not change the marginal."""
    torch.manual_seed(7)
    b, m, n, d, q, s = 2, 3, 20, 2, 1, 3
    X = torch.randn(b, m, n, d) * 0.5
    Z = torch.ones(b, m, n, q)
    ffx = torch.randn(b, s, d) * 0.3
    sigma_rfx = torch.rand(b, s, q) * 0.6 + 0.2
    sigma_eps = torch.zeros(b, s)
    y = torch.bernoulli(torch.sigmoid(X[..., 0])).unsqueeze(-1)
    mask_n = torch.ones(b, m, n, 1)
    mask_m = torch.ones(b, m, 1)

    base, _, _ = logMarginalLikelihoodLaplace(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, likelihood_family=1, n_newton=6
    )
    Z_pad = torch.cat([Z, torch.zeros(b, m, n, 1)], dim=-1)
    sigma_pad = torch.cat([sigma_rfx, torch.zeros(b, s, 1)], dim=-1)
    padded, _, _ = logMarginalLikelihoodLaplace(
        ffx, sigma_pad, sigma_eps, y, X, Z_pad, mask_n, mask_m, likelihood_family=1, n_newton=6
    )
    assert torch.allclose(base, padded, atol=1e-3)
