"""Unit tests for INLA helper functions in metabeta.simulation.inla.

These tests cover new code added for sigma_mode diagnostic extraction;
they mock the rpy2 interface so no R installation is required.
"""

import numpy as np
import pytest

from metabeta.simulation.inla import _sigma_mode_from_marginal


class _MockMarg:
    """Mimics the rpy2 column-selection result for a marginal table."""

    def __init__(self, tau: np.ndarray, den: np.ndarray) -> None:
        self._tau = np.asarray(tau, dtype=float)
        self._den = np.asarray(den, dtype=float)

    def rx(self, flag: bool, col: int) -> np.ndarray:
        return self._tau if col == 1 else self._den


class _MockMargHyper:
    """Mimics the rpy2 named-list object returned by INLA for hyperparameters."""

    def __init__(self, mapping: dict[str, _MockMarg]) -> None:
        self._mapping = mapping

    def rx2(self, name: str) -> _MockMarg:
        return self._mapping[name]


def _make_hyper(tau: np.ndarray, den: np.ndarray, name: str = 'Precision') -> _MockMargHyper:
    return _MockMargHyper({name: _MockMarg(tau, den)})


# ---------------------------------------------------------------------------
# basic correctness
# ---------------------------------------------------------------------------


def test_sigma_mode_returns_float():
    tau = np.array([1.0, 4.0, 9.0])
    den = np.array([0.1, 1.0, 0.1])
    mode = _sigma_mode_from_marginal(_make_hyper(tau, den), 'Precision')
    assert isinstance(mode, float)
    assert np.isfinite(mode)


def test_sigma_mode_peak_at_known_tau():
    """With density sharply peaked at tau=4, mode in sigma space should be near 0.5."""
    tau = np.array([1.0, 4.0, 9.0])
    # Peak of den at tau=4; sigma = 1/sqrt(4) = 0.5
    den = np.array([0.1, 1.0, 0.1])
    # sigma_den[0] = 0.1 * 2 / 1^3 = 0.2
    # sigma_den[1] = 1.0 * 2 / 0.5^3 = 16.0
    # sigma_den[2] = 0.1 * 2 / (1/3)^3 ≈ 5.4
    # argmax → index 1 → sigma = 0.5
    mode = _sigma_mode_from_marginal(_make_hyper(tau, den), 'Precision')
    assert mode == pytest.approx(0.5, rel=1e-6)


def test_sigma_mode_differs_from_naive_tau_argmax():
    """Change-of-variables must be applied: mode in sigma space ≠ 1/sqrt(tau_argmax_of_den)."""
    tau = np.array([1.0, 4.0])
    # tau peak at index 0 (tau=1, sigma=1)
    # sigma_den[0] = 2.0 * 2 / 1^3 = 4.0
    # sigma_den[1] = 1.0 * 2 / 0.5^3 = 16.0  ← larger despite den being smaller
    den = np.array([2.0, 1.0])
    mode = _sigma_mode_from_marginal(_make_hyper(tau, den), 'Precision')
    # Naive argmax of den → sigma = 1.0; correct CoV argmax → sigma = 0.5
    assert mode == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# edge / robustness cases
# ---------------------------------------------------------------------------


def test_sigma_mode_small_tau_no_division_by_zero():
    """Very small tau values (near-zero σ) should not raise; np.maximum guards apply."""
    tau = np.array([1e-15, 0.25, 1.0])
    den = np.array([0.1, 1.0, 0.1])
    mode = _sigma_mode_from_marginal(_make_hyper(tau, den), 'Precision')
    assert np.isfinite(mode)
    assert mode > 0.0


def test_sigma_mode_single_point():
    """Single-point marginal returns sigma = 1/sqrt(tau)."""
    tau = np.array([9.0])
    den = np.array([1.0])
    mode = _sigma_mode_from_marginal(_make_hyper(tau, den), 'Precision')
    assert mode == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_sigma_mode_uses_correct_marginal_name():
    """rx2 key lookup must select the right marginal."""
    tau = np.array([1.0, 4.0])
    den_a = np.array([1.0, 0.1])  # peak at index 0 → sigma=1.0 (after CoV check: see below)
    den_b = np.array([0.1, 1.0])  # peak at index 1 → sigma=0.5
    hyper = _MockMargHyper({'comp_A': _MockMarg(tau, den_a), 'comp_B': _MockMarg(tau, den_b)})

    # comp_A: sigma_den[0]=2/1=2, sigma_den[1]=0.1*2/0.125=1.6 → mode at index 0 → sigma=1.0
    # comp_B: sigma_den[0]=0.2,   sigma_den[1]=16.0             → mode at index 1 → sigma=0.5
    mode_a = _sigma_mode_from_marginal(hyper, 'comp_A')
    mode_b = _sigma_mode_from_marginal(hyper, 'comp_B')
    assert mode_a == pytest.approx(1.0, rel=1e-6)
    assert mode_b == pytest.approx(0.5, rel=1e-6)
