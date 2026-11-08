"""Tests for posthoc/metropolis.py: the advisory pool-size suggestion and the bulk ESS."""

import math

import pytest
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
    # easy dataset: 700/0.8 = 875 -> rounds to 1000 (also the floor)
    assert int(s[0]) == SUGGEST_MIN
    # huge-regime acceptance: 700/0.17 ≈ 4118 -> rounds up to 4500
    assert int(s[1]) == 4500
    assert (s[1] * accept[1].mean()) >= N_EFF_TARGET


def test_suggest_pool_size_monotone_in_acceptance():
    accept = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
    s = suggestPoolSize(accept)
    assert (s[:-1] >= s[1:]).all()  # lower acceptance -> larger suggestion


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


@pytest.mark.parametrize('T', [1, 2, 3])
def test_bulk_ess_is_nan_for_chains_too_short_to_split(T):
    """Fewer than 4 draws per chain leave no autocorrelation estimate: NaN, not a crash (every
    IMH run reports the ESS)."""
    assert torch.isnan(bulkEss(torch.randn(2, 4 * T, 3), 4)).all()


# ---------------------------------------------------------------------------
# Local moves: coordinates and Jacobian
# ---------------------------------------------------------------------------


def _globalsLayout():
    """d = 3 (last ffx padded), q = 2, sigma_eps, one corr dim; one dataset."""
    return {
        'mask_d': torch.tensor([[True, True, False]]),
        'mask_q': torch.tensor([[True, True]]),
        'mask_corr': torch.tensor([[True]]),
        'eta_rfx': torch.tensor([1.0]),
    }


def test_local_move_coordinates_round_trip_with_exact_jacobian():
    from metabeta.utils.regularization import constrainGlobals, globalMask, unconstrainGlobals

    data = _globalsLayout()
    d, q, d_corr = 3, 2, 1
    active = globalMask(data, d, q, True, d_corr)
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
    u = unconstrainGlobals(sg, d, q, True, d_corr)
    back, ldj = constrainGlobals(u, active, d, q, True, d_corr)
    assert torch.allclose(back, sg, atol=1e-10)
    # log |dθ/du| over the active dims, by autograd
    idx = active[0].nonzero().squeeze(-1)
    for k in range(sg.shape[1]):

        def f(x, k=k):
            full = u[:, k : k + 1].clone()
            full[0, 0, idx] = x
            return constrainGlobals(full, active, d, q, True, d_corr)[0][0, 0, idx]

        J = torch.autograd.functional.jacobian(f, u[0, k, idx])
        assert abs(torch.linalg.slogdet(J)[1].item() - ldj[0, k].item()) < 1e-8


# ---------------------------------------------------------------------------
# sigma_eps refresh at the moves' chain starts (Normal data)
# ---------------------------------------------------------------------------


def test_slice_coordinate_leaves_its_target_invariant():
    """Iterations on coordinate 1 of a 3-D state, target N(0, 0.05²) on that coordinate, from
    draws of that target: the draws stay distributed like it (moments, Kolmogorov-Smirnov); the
    other coordinates and the inactive dataset stay untouched."""
    from scipy.stats import kstest

    from metabeta.posthoc.metropolis import sliceCoordinate

    torch.manual_seed(0)
    b, s = 2, 4000
    u = torch.randn(b, s, 3, dtype=torch.float64)
    u[..., 1] *= 0.05
    active = torch.tensor([True, False])
    log_target = lambda x: -0.5 * (x[..., 1] / 0.05) ** 2
    out = sliceCoordinate(log_target, u, 1, active, n_iter=3, width=0.5)
    assert torch.equal(out[..., [0, 2]], u[..., [0, 2]]) and torch.equal(out[1], u[1])
    x = out[0, :, 1]
    assert not torch.equal(x, u[0, :, 1])
    assert abs(x.mean().item()) < 0.003 and abs(x.std().item() / 0.05 - 1) < 0.04
    assert kstest(x.numpy() / 0.05, 'norm').pvalue > 0.01


def test_slice_coordinate_brings_far_starts_into_the_bulk():
    """From 40 sds out, each iteration lands uniformly on the slice and so halves the distance
    on average: after 8 iterations at least 95% of the states sit within 3 sds."""
    from metabeta.posthoc.metropolis import sliceCoordinate

    torch.manual_seed(1)
    u = torch.full((1, 2000, 2), 2.0, dtype=torch.float64)  # coordinate 0: 40 sds out
    log_target = lambda x: -0.5 * (x[..., 0] / 0.05) ** 2
    out = sliceCoordinate(log_target, u, 0, torch.tensor([True]), n_iter=8, width=0.5)
    x = out[0, :, 0]
    assert float((x.abs() < 0.15).double().mean()) > 0.95 and float(x.abs().median()) < 0.05


def test_mode_coordinate_finds_the_maximum_from_far_out():
    """The uphill walk and golden-section search put coordinate 0 at the maximum of a narrow
    target 40 widths away, in the inactive dataset and the other coordinate nothing moves."""
    from metabeta.posthoc.metropolis import modeCoordinate

    u = torch.full((2, 50, 2), -20.0, dtype=torch.float64)
    log_target = lambda x: -0.5 * ((x[..., 0] - 0.3) / 0.05) ** 2 + x[..., 1]
    out = modeCoordinate(log_target, u, 0, torch.tensor([True, False]), width=0.5)
    assert torch.equal(out[..., 1], u[..., 1]) and torch.equal(out[1], u[1])
    # ten golden-section steps on a bracket of two widths: 1.0 × 0.618^10 ≈ 0.008
    assert (out[0, :, 0] - 0.3).abs().max() < 0.01


def _normalProblem(m=12, n=15, sigma_eps=0.05, seed=11):
    """Random-intercept Normal data with a high R²: y = 1.5 + b_j + eps, tau on the priors."""
    g = torch.Generator().manual_seed(seed)
    b_true = 0.4 * torch.randn(m, 1, generator=g, dtype=torch.float64)
    y = 1.5 + b_true + sigma_eps * torch.randn(m, n, generator=g, dtype=torch.float64)
    ones = lambda *shape: torch.ones(*shape, dtype=torch.float64)
    return {
        'X': ones(1, m, n, 1),
        'Z': ones(1, m, n, 1),
        'y': y[None],
        'nu_ffx': torch.zeros(1, 1, dtype=torch.float64),
        'tau_ffx': torch.full((1, 1), 2.0, dtype=torch.float64),
        'tau_rfx': ones(1, 1),
        'tau_eps': ones(1),
        'family_ffx': torch.zeros(1, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(1, dtype=torch.long),
        'family_sigma_eps': torch.zeros(1, dtype=torch.long),
        'eta_rfx': torch.zeros(1, dtype=torch.float64),
        'mask_d': torch.ones(1, 1, dtype=torch.bool),
        'mask_q': torch.ones(1, 1, dtype=torch.bool),
        'mask_mq': torch.ones(1, m, 1, dtype=torch.bool),
        'mask_m': torch.ones(1, m, dtype=torch.bool),
        'mask_n': torch.ones(1, m, n, dtype=torch.bool),
    }


def _gridNormalPosterior(data):
    """Posterior mean and sd of (beta, sigma_rfx, sigma_eps) by a 3-D grid with the rfx
    integrated out (logMarginalLikelihoodNormal)."""
    from torch import distributions as D

    from metabeta.utils.families import logMarginalLikelihoodNormal

    beta = torch.linspace(0.5, 2.5, 81, dtype=torch.float64)
    s_rfx = torch.linspace(0.05, 1.5, 80, dtype=torch.float64)
    s_eps = torch.linspace(0.02, 0.12, 80, dtype=torch.float64)
    B, R, E = torch.meshgrid(beta, s_rfx, s_eps, indexing='ij')
    flat = lambda t: t.reshape(1, -1)
    ll = logMarginalLikelihoodNormal(
        flat(B)[..., None], flat(R)[..., None], flat(E), data['y'][..., None], data['X'],
        data['Z'], data['mask_n'][..., None].double(), data['mask_m'][..., None].double(),
    )[0]  # fmt: skip
    lp = (
        D.Normal(0.0, 2.0).log_prob(flat(B)[0])
        + D.HalfNormal(1.0).log_prob(flat(R)[0])
        + D.HalfNormal(1.0).log_prob(flat(E)[0])
    )
    post = torch.softmax(ll + lp, 0)
    moments = []
    for grid in (flat(B)[0], flat(R)[0], flat(E)[0]):
        mean = float((post * grid).sum())
        moments.append((mean, float(((post * grid**2).sum() - mean**2).sqrt())))
    return moments


def _wrongPool(s, m, sigma_eps_loc):
    """A flow-like proposal: beta and sigma_rfx about right but wide, sigma_eps lognormal
    around sigma_eps_loc (the training cap), far above the truth."""
    from torch import distributions as D

    from metabeta.utils.results import Proposal

    q_beta = D.Normal(torch.tensor(1.5, dtype=torch.float64), 0.6)
    q_rfx = D.LogNormal(torch.tensor(math.log(0.4), dtype=torch.float64), 0.5)
    q_eps = D.LogNormal(torch.tensor(math.log(sigma_eps_loc), dtype=torch.float64), 0.15)
    beta, s_rfx, s_eps = q_beta.sample((1, s, 1)), q_rfx.sample((1, s, 1)), q_eps.sample((1, s, 1))
    log_q = q_beta.log_prob(beta) + q_rfx.log_prob(s_rfx) + q_eps.log_prob(s_eps)
    proposed = {
        'global': {'samples': torch.cat([beta, s_rfx, s_eps], -1), 'log_prob': log_q.sum(-1)},
        'local': {
            'samples': torch.zeros(1, m, s, 1, dtype=torch.float64),
            'log_prob': torch.zeros(1, m, s, dtype=torch.float64),
        },
    }
    return Proposal(proposed, has_sigma_eps=True)


def test_normal_moves_land_sigma_eps_and_beta_from_a_pool_beyond_its_support():
    """A pool whose sigma_eps sits far above the posterior leaves the IMH there (it can only
    pick pool members); the moves' start refresh (sigma_eps to its conditional mode and sliced,
    exact beta draw) and the random walk recover the grid posterior of all three globals."""
    from metabeta.posthoc.metropolis import LocalMoves, MetropolisSampler

    data = _normalProblem()
    want = _gridNormalPosterior(data)
    m = data['X'].shape[1]
    n_steps = 1025
    got = {}
    for label, mv in (('imh', None), ('moves', LocalMoves())):
        torch.manual_seed(5)
        sampler = MetropolisSampler(
            data, n_chains=4, n_steps=n_steps, mode='marginal', n_eff_target=None, moves=mv
        )
        out, diag = sampler(_wrongPool(4 * n_steps, m, 0.3))
        draws = (out.ffx[0, :, 0], out.sigma_rfx[0, :, 0], out.sigma_eps[0])
        got[label] = [(t.mean().item(), t.std().item()) for t in draws]
        if mv is not None:
            assert bool(diag['moved'].all()) and float(diag['move_accept'][0]) > 0.1
    # the test is meaningful: the IMH alone misses sigma_eps by many posterior sds
    assert got['imh'][2][0] - want[2][0] > 5 * want[2][1]
    for (mean, sd), (want_mean, want_sd) in zip(got['moves'], want):
        assert abs(mean - want_mean) < 0.2 * want_sd
        assert abs(sd / want_sd - 1) < 0.2
