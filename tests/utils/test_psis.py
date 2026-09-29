from __future__ import annotations

import numpy as np
import pytest
from arviz_stats.base import array_stats
from scipy.special import logsumexp

from metabeta.utils.psis import psislw


@pytest.fixture
def log_w() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((3, 400))


def test_matches_arviz_on_fittable_rows(log_w):
    lw, k = psislw(log_w)
    lw_ref, k_ref = array_stats.psislw(-log_w, r_eff=1.0, axis=-1)
    np.testing.assert_allclose(lw, lw_ref)
    np.testing.assert_allclose(k, k_ref)
    np.testing.assert_allclose(logsumexp(lw, axis=-1), 0.0, atol=1e-12)


def test_degenerate_tail_returns_inf_k(log_w):
    log_w[1] = -2.0  # every draw equally likely: nothing to fit
    lw, k = psislw(log_w)
    assert np.isinf(k[1]) and np.isfinite(k[[0, 2]]).all()
    np.testing.assert_allclose(lw[1], np.log(1 / 400))
    lw_ref, _ = array_stats.psislw(-log_w[[0, 2]], r_eff=1.0, axis=-1)
    np.testing.assert_allclose(lw[[0, 2]], lw_ref)


def test_one_dimensional_input(log_w):
    lw, k = psislw(log_w[0])
    lw_ref, k_ref = array_stats.psislw(-log_w[0], r_eff=1.0, axis=-1)
    np.testing.assert_allclose(lw, lw_ref)
    assert k.shape == () and np.isclose(k, k_ref)
    lw_flat, k_flat = psislw(np.zeros(400))
    assert k_flat.shape == () and np.isinf(k_flat)
    np.testing.assert_allclose(lw_flat, np.log(1 / 400))


def test_too_few_draws_returns_inf_k():
    lw, k = psislw(np.random.default_rng(1).standard_normal((2, 20)))
    assert np.isinf(k).all()
    np.testing.assert_allclose(logsumexp(lw, axis=-1), 0.0, atol=1e-12)
