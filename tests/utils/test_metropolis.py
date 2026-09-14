"""Tests for posthoc/metropolis.py: the advisory pool-size suggestion."""

import torch

from metabeta.posthoc.metropolis import N_EFF_TARGET, SUGGEST_MAX, SUGGEST_MIN, suggestPoolSize


def test_suggest_pool_size_hits_target():
    """Suggested s must reach the effective-draw target at the measured acceptance."""
    accept = torch.tensor([[0.8, 0.8, 0.8, 0.8], [0.17, 0.17, 0.17, 0.17]])
    s = suggestPoolSize(accept)
    assert s.dtype == torch.long
    # easy dataset: 700/0.8 = 875 → rounds to 1000 (also the floor)
    assert int(s[0]) == SUGGEST_MIN
    # huge-regime acceptance: 700/0.17 ≈ 4118 → rounds up to 4500
    assert int(s[1]) == 4500
    assert (s[1] * accept[1].mean()) >= N_EFF_TARGET


def test_suggest_pool_size_monotone_in_acceptance():
    accept = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
    s = suggestPoolSize(accept)
    assert (s[:-1] >= s[1:]).all()  # lower acceptance → larger suggestion


def test_suggest_pool_size_clamps():
    accept = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    s = suggestPoolSize(accept)
    assert int(s[0]) == SUGGEST_MAX  # dead chains saturate instead of exploding
    assert int(s[1]) == SUGGEST_MIN


def test_suggest_pool_size_rounds_to_500():
    accept = torch.rand(32, 4).clamp(min=0.03)
    s = suggestPoolSize(accept)
    assert (s % 500 == 0).all()
