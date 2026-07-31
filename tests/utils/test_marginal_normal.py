"""Tests for logMarginalLikelihoodNormal (incl. correlated Σ_rfx) and its
conjugate dual sampleRfxConditionalNormal."""

import math

import torch
from torch import distributions as D

from metabeta.utils.families import (
    logMarginalLikelihoodNormal,
    sampleRfxConditionalNormal,
)
from metabeta.utils.regularization import unconstrainedToCholesky


def _makeProblem(b=2, m=3, n=6, d=2, q=2, s=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(b, m, n, d, generator=g)
    Z = torch.randn(b, m, n, q, generator=g)
    y = torch.randn(b, m, n, 1, generator=g)
    ffx = torch.randn(b, s, d, generator=g)
    sigma_rfx = torch.rand(b, s, q, generator=g) * 0.9 + 0.1
    sigma_eps = torch.rand(b, s, generator=g) * 0.9 + 0.1
    mask_n = torch.ones(b, m, n, 1)
    mask_m = torch.ones(b, m, 1)
    return X, Z, y, ffx, sigma_rfx, sigma_eps, mask_n, mask_m


def _bruteForceMarginal(X, Z, y, ffx, sigma_rfx, sigma_eps, L_corr=None):
    """Direct MVN evaluation of V_i = Z_i Σ Z_iᵀ + σ² I per group."""
    b, m, n, _ = X.shape
    s = ffx.shape[1]
    q = Z.shape[-1]
    out = torch.zeros(b, s)
    for i in range(b):
        for k in range(s):
            if L_corr is None:
                Sigma = torch.diag(sigma_rfx[i, k].pow(2))
            else:
                L = sigma_rfx[i, k].unsqueeze(-1) * L_corr[i, k]
                Sigma = L @ L.mT
            for j in range(m):
                V = Z[i, j] @ Sigma @ Z[i, j].mT + sigma_eps[i, k].pow(2) * torch.eye(n)
                loc = X[i, j] @ ffx[i, k]
                out[i, k] += D.MultivariateNormal(loc, V).log_prob(y[i, j, :, 0])
    return out


def test_marginal_diagonal_matches_mvn():
    X, Z, y, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem()
    got = logMarginalLikelihoodNormal(ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m)
    want = _bruteForceMarginal(X, Z, y, ffx, sigma_rfx, sigma_eps)
    assert torch.allclose(got, want, atol=1e-3)


def test_marginal_correlated_matches_mvn():
    X, Z, y, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem(seed=1)
    b, s, q = sigma_rfx.shape
    g = torch.Generator().manual_seed(2)
    z_corr = torch.randn(b, s, q * (q - 1) // 2, generator=g)
    L_corr = unconstrainedToCholesky(z_corr, q)
    got = logMarginalLikelihoodNormal(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, L_corr=L_corr
    )
    want = _bruteForceMarginal(X, Z, y, ffx, sigma_rfx, sigma_eps, L_corr=L_corr)
    assert torch.allclose(got, want, atol=1e-3)


def test_marginal_identity_corr_reduces_to_diagonal():
    X, Z, y, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem(seed=3)
    b, s, q = sigma_rfx.shape
    L_eye = torch.eye(q).expand(b, s, q, q)
    diag = logMarginalLikelihoodNormal(ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m)
    corr = logMarginalLikelihoodNormal(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, L_corr=L_eye
    )
    assert torch.allclose(diag, corr, atol=1e-4)


def test_marginal_padded_dim_invariance():
    """Appending a masked q-dim (zero Z column, zero sigma, identity L_corr row)
    must not change the marginal likelihood."""
    X, Z, y, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem(seed=4)
    b, s, q = sigma_rfx.shape
    g = torch.Generator().manual_seed(5)
    z_corr = torch.randn(b, s, q * (q - 1) // 2, generator=g)
    L_corr = unconstrainedToCholesky(z_corr, q)
    base = logMarginalLikelihoodNormal(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, L_corr=L_corr
    )

    # pad q -> q+1: zero Z column, zero sigma (clamped internally), identity L row
    Z_pad = torch.cat([Z, torch.zeros(b, Z.shape[1], Z.shape[2], 1)], dim=-1)
    sigma_pad = torch.cat([sigma_rfx, torch.zeros(b, s, 1)], dim=-1)
    # z_corr entries involving the padded dim are 0 -> unconstrainedToCholesky
    # yields an identity row/col for it
    z_pad = torch.cat([z_corr, torch.zeros(b, s, q)], dim=-1)
    L_pad = unconstrainedToCholesky(z_pad, q + 1)
    padded = logMarginalLikelihoodNormal(
        ffx, sigma_pad, sigma_eps, y, X, Z_pad, mask_n, mask_m, L_corr=L_pad
    )
    assert torch.allclose(base, padded, atol=1e-3)


def test_conditional_moments_match_conjugate():
    """Empirical mean/cov of sampleRfxConditionalNormal draws must match the
    analytic Normal-Normal conditional moments."""
    torch.manual_seed(6)
    b, m, n, d, q, s = 1, 2, 40, 2, 2, 4000
    X = torch.randn(b, m, n, d)
    Z = torch.randn(b, m, n, q)
    y = torch.randn(b, m, n, 1)
    mask_n = torch.ones(b, m, n, 1)
    mask_m = torch.ones(b, m, 1)

    # one global sample, tiled s times so we get s conditional draws
    ffx1 = torch.randn(b, 1, d)
    sigma_rfx1 = torch.rand(b, 1, q) * 0.9 + 0.3
    sigma_eps1 = torch.rand(b, 1) * 0.5 + 0.5
    z_corr1 = torch.randn(b, 1, q * (q - 1) // 2)
    L_corr1 = unconstrainedToCholesky(z_corr1, q)

    rfx = sampleRfxConditionalNormal(
        ffx1.expand(b, s, d),
        sigma_rfx1.expand(b, s, q),
        sigma_eps1.expand(b, s),
        y,
        X,
        Z,
        mask_n,
        mask_m,
        L_corr=L_corr1.expand(b, s, q, q),
    )  # (b, m, s, q)

    # analytic conditional moments per group
    L = sigma_rfx1[0, 0].unsqueeze(-1) * L_corr1[0, 0]
    Sigma = L @ L.mT
    s2e = sigma_eps1[0, 0].pow(2)
    for j in range(m):
        Zj = Z[0, j]
        rj = y[0, j, :, 0] - X[0, j] @ ffx1[0, 0]
        V = torch.linalg.inv(torch.linalg.inv(Sigma) + Zj.mT @ Zj / s2e)
        mu = V @ (Zj.mT @ rj) / s2e
        emp_mean = rfx[0, j].mean(0)
        emp_cov = torch.cov(rfx[0, j].mT)
        assert torch.allclose(emp_mean, mu, atol=6.0 * V.diagonal().sqrt().max() / math.sqrt(s))
        assert torch.allclose(emp_cov, V, atol=0.15 * V.diagonal().max())


def test_conditional_masked_groups_zero():
    X, Z, y, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem(seed=7)
    mask_m = mask_m.clone()
    mask_m[:, -1] = 0.0
    rfx = sampleRfxConditionalNormal(ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m)
    assert (rfx[:, -1] == 0).all()
    assert (rfx[:, :-1] != 0).any()
