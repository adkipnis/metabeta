"""Tests for corrToUnconstrained / unconstrainedToCholesky round-trips."""

import torch
import pytest

from metabeta.utils.regularization import corrToUnconstrained, unconstrainedToCholesky


def _lkj_corr(q: int, eta: float = 1.0, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    L = torch.distributions.LKJCholesky(q, concentration=eta).sample()
    return L @ L.mT


# ---------------------------------------------------------------------------
# round-trip: corr -> z -> L -> L@L.T == corr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('q', [2, 3, 4])
def test_round_trip_single(q: int):
    corr = _lkj_corr(q)
    z = corrToUnconstrained(corr.unsqueeze(0))  # (1, d_corr)
    L = unconstrainedToCholesky(z, q)  # (1, q, q)
    corr_back = (L @ L.mT).squeeze(0)
    assert torch.allclose(corr_back, corr, atol=1e-4), f'round-trip failed for q={q}'


@pytest.mark.parametrize('q', [2, 3])
def test_round_trip_batch(q: int):
    B = 8
    corr = torch.stack([_lkj_corr(q, seed=i) for i in range(B)])  # (B, q, q)
    z = corrToUnconstrained(corr)  # (B, d_corr)
    L = unconstrainedToCholesky(z, q)  # (B, q, q)
    corr_back = L @ L.mT
    assert torch.allclose(corr_back, corr, atol=1e-4)


def test_round_trip_with_sample_dim():
    """Test that (..., q, q) -> (..., d_corr) -> (..., q, q) works for (B, S, q, q) input."""
    B, S, q = 3, 5, 2
    corr = torch.stack(
        [_lkj_corr(q, seed=i) for i in range(B * S)]
    ).reshape(B, S, q, q)
    z = corrToUnconstrained(corr)  # (B, S, d_corr)
    assert z.shape == (B, S, q * (q - 1) // 2)
    L = unconstrainedToCholesky(z, q)  # (B, S, q, q)
    corr_back = L @ L.mT
    assert torch.allclose(corr_back, corr, atol=1e-4)


# ---------------------------------------------------------------------------
# special cases
# ---------------------------------------------------------------------------


def test_identity_maps_to_zero():
    """Identity correlation <-> z = 0."""
    for q in [2, 3, 4]:
        d_corr = q * (q - 1) // 2
        corr = torch.eye(q).unsqueeze(0)
        z = corrToUnconstrained(corr)
        assert z.shape == (1, d_corr)
        assert torch.allclose(z, torch.zeros(1, d_corr), atol=1e-5), (
            f'identity should map to z=0 for q={q}, got {z}'
        )


def test_q1_returns_empty():
    corr = torch.ones(1, 1)
    z = corrToUnconstrained(corr.unsqueeze(0))
    assert z.shape == (1, 0)


def test_q2_known_rho():
    """For q=2, corr = [[1, rho],[rho,1]] -> z = atanh(rho)."""
    rho = 0.6
    corr = torch.tensor([[1.0, rho], [rho, 1.0]]).unsqueeze(0)
    z = corrToUnconstrained(corr)
    assert z.shape == (1, 1)
    expected = torch.atanh(torch.tensor(rho))
    assert abs(float(z[0, 0]) - float(expected)) < 1e-4


# ---------------------------------------------------------------------------
# output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('q', [2, 3, 4])
def test_output_shapes(q: int):
    d_corr = q * (q - 1) // 2
    B = 4
    corr = torch.stack([_lkj_corr(q, seed=i) for i in range(B)])
    z = corrToUnconstrained(corr)
    assert z.shape == (B, d_corr)
    L = unconstrainedToCholesky(z, q)
    assert L.shape == (B, q, q)
    # L must be lower-triangular with positive diagonal
    assert torch.allclose(torch.triu(L, diagonal=1), torch.zeros_like(L), atol=1e-6)
    assert (L.diagonal(dim1=-2, dim2=-1) > 0).all()
    # corr = L @ L.T must have unit diagonal
    corr_out = L @ L.mT
    diag = corr_out.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)
