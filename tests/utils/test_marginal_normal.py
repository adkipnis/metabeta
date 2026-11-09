"""Tests for logMarginalLikelihoodNormal (incl. correlated Σ_rfx) and its
conjugate dual sampleRfxConditionalNormal."""

import math

import pytest

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


def test_conditional_mean_at_tiny_sigma_eps_matches_float64():
    """At sigma_eps 2e-19 (an IMH chain stuck on such a flow draw, issue #57) Zᵀr / σ²_eps
    overflows float32 although the conditional is ordinary: the draw is the float64
    conditional mean, which is close to the per-group least-squares fit."""
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    Z = torch.stack([torch.ones(5), x], -1).reshape(1, 1, 5, 2)
    X = torch.ones(1, 1, 5, 1)
    y = (1000.0 + 50.0 * x + torch.tensor([0.3, -0.2, 0.1, 0.0, -0.4])).reshape(1, 1, 5, 1)
    ffx, sigma_rfx = torch.zeros(1, 2, 1), torch.full((1, 2, 2), 3.0)
    sigma_eps = torch.tensor([[0.5, 2e-19]])  # σ²_eps 4e-38, a normal float32
    mask_n, mask_m = torch.ones(1, 1, 5, 1), torch.ones(1, 1, 1)
    rfx = sampleRfxConditionalNormal(ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m)
    assert torch.isfinite(rfx).all()
    Zd, yd, s2e = Z[0, 0].double(), y[0, 0, :, 0].double(), 4e-38
    precision = torch.eye(2, dtype=torch.float64) / 9.0 + Zd.T @ Zd / s2e
    mean = torch.linalg.solve(precision, Zd.T @ yd / s2e)
    torch.testing.assert_close(rfx[0, 0, 1].double(), mean, rtol=1e-4, atol=1e-3)


def test_conditional_masked_groups_zero():
    X, Z, y, ffx, sigma_rfx, sigma_eps, mask_n, mask_m = _makeProblem(seed=7)
    mask_m = mask_m.clone()
    mask_m[:, -1] = 0.0
    rfx = sampleRfxConditionalNormal(ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m)
    assert (rfx[:, -1] == 0).all()
    assert (rfx[:, :-1] != 0).any()


# ---------------------------------------------------------------------------
# sampleFfxConditionalNormal: ffx | Sigma_rfx, sigma_eps, y with the rfx integrated out
# ---------------------------------------------------------------------------


def _glsPosterior(X, Z, y, sigma_rfx, sigma_eps, nu, tau, L_corr=None):
    """Brute force: precision Σ_j X_jᵀ V_j⁻¹ X_j + diag(1/τ²), mean from the matching rhs, for
    the one state (sigma_rfx (q,), sigma_eps scalar) of dataset 0."""
    m, n, d = X.shape
    q = Z.shape[-1]
    Sigma = (
        torch.diag(sigma_rfx**2)
        if L_corr is None
        else (sigma_rfx[:, None] * L_corr) @ (sigma_rfx[:, None] * L_corr).mT
    )
    P = torch.diag(1.0 / tau**2)
    rhs = nu / tau**2
    for j in range(m):
        V = Z[j] @ Sigma @ Z[j].mT + sigma_eps**2 * torch.eye(n, dtype=X.dtype)
        Vi = torch.linalg.inv(V)
        P = P + X[j].mT @ Vi @ X[j]
        rhs = rhs + X[j].mT @ Vi @ y[j, :, 0]
    cov = torch.linalg.inv(P)
    return cov @ rhs, cov


def _ffxProblem(d=2, q=2, m=3, n=6, s=200000, seed=5, corr=False):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(1, m, n, d, generator=g, dtype=torch.float64)
    Z = torch.randn(1, m, n, q, generator=g, dtype=torch.float64)
    y = torch.randn(1, m, n, 1, generator=g, dtype=torch.float64)
    sigma_rfx = (torch.rand(1, 1, q, generator=g, dtype=torch.float64) * 0.9 + 0.1).expand(1, s, q)
    sigma_eps = (torch.rand(1, 1, generator=g, dtype=torch.float64) * 0.9 + 0.1).expand(1, s)
    ffx = torch.randn(1, s, d, generator=g, dtype=torch.float64)
    nu = torch.randn(1, 1, d, generator=g, dtype=torch.float64)
    tau = torch.rand(1, 1, d, generator=g, dtype=torch.float64) + 0.5
    L_corr = None
    if corr:
        z_corr = torch.randn(1, 1, q * (q - 1) // 2, generator=g, dtype=torch.float64)
        L_corr = unconstrainedToCholesky(z_corr, q).expand(1, s, q, q)
    masks = (torch.ones(1, m, n, 1, dtype=torch.float64), torch.ones(1, m, 1, dtype=torch.float64))
    return X, Z, y, ffx, sigma_rfx, sigma_eps, nu, tau, L_corr, masks


@pytest.mark.parametrize('corr', [False, True])
def test_ffx_conditional_normal_prior_matches_gls_posterior(corr):
    from metabeta.utils.families import sampleFfxConditionalNormal

    X, Z, y, ffx, sigma_rfx, sigma_eps, nu, tau, L_corr, (mask_n, mask_m) = _ffxProblem(corr=corr)
    torch.manual_seed(0)
    family = torch.zeros(1, dtype=torch.long)
    mask_d = torch.ones(1, 1, 2, dtype=torch.bool)
    draw, accepted = sampleFfxConditionalNormal(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, nu, tau, family, mask_d, L_corr=L_corr
    )
    assert bool(accepted.all())  # the normal-prior draw is exact: no rejection
    want_mean, want_cov = _glsPosterior(
        X[0], Z[0], y[0], sigma_rfx[0, 0], sigma_eps[0, 0], nu[0, 0], tau[0, 0],
        L_corr=None if L_corr is None else L_corr[0, 0],
    )  # fmt: skip
    s = draw.shape[1]
    sd = want_cov.diagonal().sqrt()
    assert torch.allclose(draw[0].mean(0), want_mean, atol=4 * float(sd.max()) / s**0.5)
    assert torch.allclose(
        torch.cov(draw[0].mT), want_cov, rtol=0.03, atol=1e-3 * float(sd.max()) ** 2
    )


def test_ffx_conditional_keeps_padded_dims():
    from metabeta.utils.families import sampleFfxConditionalNormal

    X, Z, y, ffx, sigma_rfx, sigma_eps, nu, tau, L_corr, (mask_n, mask_m) = _ffxProblem(s=50)
    X = X.clone()
    X[..., 1] = 0.0  # padded ffx column
    mask_d = torch.tensor([[[True, False]]])
    torch.manual_seed(1)
    draw, _ = sampleFfxConditionalNormal(
        ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, nu, tau,
        torch.zeros(1, dtype=torch.long), mask_d,
    )  # fmt: skip
    assert torch.equal(draw[..., 1], ffx[..., 1])
    assert torch.isfinite(draw).all() and not torch.equal(draw[..., 0], ffx[..., 0])


def test_ffx_conditional_keeps_states_whose_precision_does_not_factor():
    """At sigma_eps near zero the Woodbury terms of the precision cancel and rounding leaves it
    indefinite; those states keep their draw, the others are drawn (issue #57: the random-walk
    moves' start refresh raised under priors far outside the trained range)."""
    from metabeta.utils.families import sampleFfxConditionalNormal

    g = torch.Generator().manual_seed(0)
    b, m, n, d, s = 1, 20, 8, 3, 6
    X = torch.randn(b, m, n, d, generator=g)
    X[..., 0] = 1.0
    Z, y = X.clone(), torch.randn(b, m, n, 1, generator=g)
    ffx = torch.randn(b, s, d, generator=g)
    sigma_rfx = torch.tensor([2.5e3, 0.0, 1e3]).expand(b, s, d)
    sigma_eps = torch.full((b, s), 1e-8)
    sigma_eps[:, 0] = 0.5  # state 0 ordinary
    torch.manual_seed(2)
    draw, accepted = sampleFfxConditionalNormal(
        ffx, sigma_rfx, sigma_eps, y, X, Z, torch.ones(b, m, n, 1), torch.ones(b, m, 1),
        torch.zeros(b, 1, d), torch.full((b, 1, d), 88.0), torch.zeros(b, dtype=torch.long),
        torch.ones(b, 1, d, dtype=torch.bool),
    )  # fmt: skip
    assert torch.isfinite(draw).all()
    assert bool(accepted[0, 0]) and not torch.equal(draw[0, 0], ffx[0, 0])
    kept = ~accepted[0]
    assert kept.any() and torch.equal(draw[0, kept], ffx[0, kept])


def test_ffx_conditional_student_prior_targets_the_t_posterior():
    """Repeated draws form an independence-MH chain whose stationary law is the posterior under
    the Student-t prior, not the normal-prior GLS law (a tight prior at 0 against data at 2:
    the grid means are 1.94 and 1.73, the step accepts about a third)."""
    from metabeta.utils.constants import STUDENT_DF
    from metabeta.utils.families import sampleFfxConditionalNormal

    g = torch.Generator().manual_seed(8)
    m, n, s = 4, 8, 4000
    X = torch.ones(1, m, n, 1, dtype=torch.float64)
    Z = torch.ones(1, m, n, 1, dtype=torch.float64)
    y = 2.0 + 0.3 * torch.randn(1, m, n, 1, generator=g, dtype=torch.float64)
    sigma_rfx = torch.full((1, s, 1), 0.3, dtype=torch.float64)
    sigma_eps = torch.full((1, s), 0.3, dtype=torch.float64)
    nu, tau = torch.zeros(1, 1, 1, dtype=torch.float64), torch.full(
        (1, 1, 1), 0.4, dtype=torch.float64
    )
    mask_n, mask_m = torch.ones(1, m, n, 1, dtype=torch.float64), torch.ones(
        1, m, 1, dtype=torch.float64
    )
    # grid posterior under the t prior
    beta = torch.linspace(-1.0, 4.0, 2001, dtype=torch.float64)
    ll = logMarginalLikelihoodNormal(
        beta.view(1, -1, 1), sigma_rfx[:, :1].expand(1, beta.numel(), 1),
        sigma_eps[:, :1].expand(1, beta.numel()), y, X, Z, mask_n, mask_m,
    )[0]  # fmt: skip
    lp = D.StudentT(STUDENT_DF, 0.0, 0.4).log_prob(beta)
    post = torch.softmax(ll + lp, 0)
    want_mean = float((post * beta).sum())
    want_sd = float(((post * beta**2).sum() - want_mean**2).sqrt())
    torch.manual_seed(2)
    ffx = torch.zeros(1, s, 1, dtype=torch.float64)
    family = torch.ones(1, dtype=torch.long)
    mask_d = torch.ones(1, 1, 1, dtype=torch.bool)
    acc = []
    for _ in range(300):
        ffx, accepted = sampleFfxConditionalNormal(
            ffx, sigma_rfx, sigma_eps, y, X, Z, mask_n, mask_m, nu, tau, family, mask_d
        )
        acc.append(accepted.double().mean().item())
    assert 0.1 < sum(acc[-50:]) / 50 < 0.9  # the correction rejects some draws
    assert abs(ffx.mean().item() - want_mean) < 0.1 * want_sd
    assert abs(ffx.std().item() / want_sd - 1) < 0.1
