"""Tests for posthoc/metropolis.py: the advisory pool-size suggestion and the bulk ESS."""

import torch

from metabeta.posthoc.metropolis import (
    N_EFF_TARGET,
    SUGGEST_MAX,
    SUGGEST_MIN,
    bulkEss,
    suggestPoolSize,
)


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


def _ar1(phi: float, n_chains: int, T: int, gen: torch.Generator) -> torch.Tensor:
    """(n_chains * T,) chain-major AR(1) draws with stationary N(0, 1) marginals."""
    x = torch.empty(n_chains, T, dtype=torch.float64)
    x[:, 0] = torch.randn(n_chains, generator=gen, dtype=torch.float64)
    eps = torch.randn(n_chains, T, generator=gen, dtype=torch.float64) * (1 - phi**2) ** 0.5
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x.reshape(-1)


def test_bulk_ess_iid_and_ar1():
    """i.i.d. draws give ESS ≈ S; AR(1) gives ESS ≈ S (1 − φ) / (1 + φ)."""
    gen = torch.Generator().manual_seed(0)
    C, T = 4, 2000
    S = C * T
    x = torch.stack([_ar1(0.0, C, T, gen), _ar1(0.8, C, T, gen)], dim=-1)[None]  # (1, S, 2)
    ess = bulkEss(x, C)[0]
    assert 0.8 * S < ess[0] < 1.2 * S
    target = S * 0.2 / 1.8
    assert 0.7 * target < ess[1] < 1.3 * target


def test_bulk_ess_sticky_chain_and_padding():
    """Repeated IMH states lower the ESS; a constant (padded) dimension is NaN."""
    gen = torch.Generator().manual_seed(1)
    C, T = 4, 1000
    fresh = torch.randn(C, T // 20, generator=gen, dtype=torch.float64)
    sticky = fresh.repeat_interleave(20, dim=1).reshape(-1)  # each state held 20 steps
    x = torch.stack([sticky, torch.zeros(C * T, dtype=torch.float64)], dim=-1)[None]
    ess = bulkEss(x, C)[0]
    assert ess[0] < C * T / 10
    assert torch.isnan(ess[1])


# ---------------------------------------------------------------------------
# Local moves: coordinates and Jacobian
# ---------------------------------------------------------------------------


def _globalsLayout():
    """d = 3 (last ffx padded), q = 2, sigma_eps, one corr dim; one dataset."""
    import types

    from metabeta.posthoc.metropolis import MetropolisSampler

    sampler = types.SimpleNamespace(has_sigma_eps=True)
    for name in ('_unconstrain', '_constrain', '_activeGlobals'):
        setattr(sampler, name, getattr(MetropolisSampler, name).__get__(sampler))
    data = {
        'mask_d': torch.tensor([[True, True, False]]),
        'mask_q': torch.tensor([[True, True]]),
        'mask_corr': torch.tensor([[True]]),
        'eta_rfx': torch.tensor([1.0]),
    }
    return sampler, data


def test_local_move_coordinates_round_trip_with_exact_jacobian():
    sampler, data = _globalsLayout()
    d, q, d_corr = 3, 2, 1
    active = sampler._activeGlobals(data, d, q, d_corr)
    assert active.tolist() == [[True, True, False, True, True, True, True]]
    gen = torch.Generator().manual_seed(0)
    sg = torch.cat(
        [
            torch.randn(1, 5, 2, generator=gen, dtype=torch.float64),
            torch.zeros(1, 5, 1, dtype=torch.float64),  # padded ffx
            torch.rand(1, 5, 3, generator=gen, dtype=torch.float64) * 2 + 0.05,  # sigmas
            torch.rand(1, 5, 1, generator=gen, dtype=torch.float64) * 1.8 - 0.9,  # corr
        ],
        dim=-1,
    )
    u = sampler._unconstrain(sg, d, q, d_corr)
    back, ldj = sampler._constrain(u, sg[:, :1], active, d, q, d_corr)
    assert torch.allclose(back, sg, atol=1e-10)
    # log |dθ/du| over the active dims, by autograd
    idx = active[0].nonzero().squeeze(-1)
    for k in range(sg.shape[1]):

        def f(x, k=k):
            full = u[:, k : k + 1].clone()
            full[0, 0, idx] = x
            return sampler._constrain(full, sg[:, :1], active, d, q, d_corr)[0][0, 0, idx]

        J = torch.autograd.functional.jacobian(f, u[0, k, idx])
        assert abs(torch.linalg.slogdet(J)[1].item() - ldj[0, k].item()) < 1e-8
