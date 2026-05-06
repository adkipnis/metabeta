from __future__ import annotations

import torch

from metabeta.posthoc.gaussian_local import analyticalRFX, _correlationPrecision


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    b, m, n, d, q, s = 2, 3, 5, 2, 2, 4
    y = torch.randn(b, m, n)
    x = torch.randn(b, m, n, d)
    x[..., 0] = 1.0
    z = torch.randn(b, m, n, q)
    z[..., 0] = 1.0
    beta = torch.randn(b, s, d)
    sigma_rfx = torch.rand(b, s, q) + 0.5
    sigma_eps = torch.rand(b, s) + 0.5
    mask_n = torch.ones(b, m, n, dtype=torch.bool)
    return y, x, z, beta, sigma_rfx, sigma_eps, mask_n


def test_analytical_rfx_identity_corr_matches_diagonal_path():
    y, x, z, beta, sigma_rfx, sigma_eps, mask_n = _inputs()
    b, s, q = sigma_rfx.shape
    eye = torch.eye(q).expand(b, s, q, q)
    sigma_inv = _correlationPrecision(sigma_rfx, eye)

    torch.manual_seed(123)
    samples_diag, log_prob_diag = analyticalRFX(y, x, z, beta, sigma_rfx, sigma_eps, mask_n)
    torch.manual_seed(123)
    samples_corr, log_prob_corr = analyticalRFX(
        y, x, z, beta, sigma_rfx, sigma_eps, mask_n, Sigma_rfx_inv=sigma_inv
    )

    assert torch.allclose(samples_corr, samples_diag, atol=1e-5)
    assert torch.allclose(log_prob_corr, log_prob_diag, atol=1e-5)


def test_analytical_rfx_uses_nonidentity_corr():
    y, x, z, beta, sigma_rfx, sigma_eps, mask_n = _inputs()
    b, s, q = sigma_rfx.shape
    corr = torch.tensor([[1.0, 0.6], [0.6, 1.0]]).expand(b, s, q, q)
    sigma_inv = _correlationPrecision(sigma_rfx, corr)

    torch.manual_seed(123)
    samples_diag, log_prob_diag = analyticalRFX(y, x, z, beta, sigma_rfx, sigma_eps, mask_n)
    torch.manual_seed(123)
    samples_corr, log_prob_corr = analyticalRFX(
        y, x, z, beta, sigma_rfx, sigma_eps, mask_n, Sigma_rfx_inv=sigma_inv
    )

    assert not torch.allclose(samples_corr, samples_diag)
    assert not torch.allclose(log_prob_corr, log_prob_diag)


def test_correlation_precision_eta_zero_forces_identity():
    y, x, z, beta, sigma_rfx, sigma_eps, mask_n = _inputs()
    b, s, q = sigma_rfx.shape
    corr = torch.tensor([[1.0, 0.6], [0.6, 1.0]]).expand(b, s, q, q)
    eta = torch.zeros(b)
    sigma_inv = _correlationPrecision(sigma_rfx, corr, eta_rfx=eta)

    torch.manual_seed(123)
    samples_diag, log_prob_diag = analyticalRFX(y, x, z, beta, sigma_rfx, sigma_eps, mask_n)
    torch.manual_seed(123)
    samples_corr, log_prob_corr = analyticalRFX(
        y, x, z, beta, sigma_rfx, sigma_eps, mask_n, Sigma_rfx_inv=sigma_inv
    )

    assert torch.allclose(samples_corr, samples_diag, atol=1e-5)
    assert torch.allclose(log_prob_corr, log_prob_diag, atol=1e-5)


def test_analytical_rfx_handles_degenerate_precision():
    y, x, z, beta, sigma_rfx, sigma_eps, mask_n = _inputs()
    b, s, q = sigma_rfx.shape
    sigma_inv = torch.zeros(b, s, q, q)
    mask_n = torch.zeros_like(mask_n)

    samples, log_prob = analyticalRFX(
        y, x, z, beta, sigma_rfx, sigma_eps, mask_n, Sigma_rfx_inv=sigma_inv
    )

    assert samples.shape == (b, x.shape[1], s, q)
    assert log_prob.shape == (b, x.shape[1], s)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(log_prob).all()
