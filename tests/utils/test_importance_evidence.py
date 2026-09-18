"""Tests for the IS log-evidence and the LKJ-prior coordinates in posthoc/importance.py."""

import math

import torch
from torch import distributions as D

from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.families import logProbCorrRfx
from metabeta.utils.preprocessing import logJacobianStandardization
from metabeta.utils.regularization import corrLowerToUnconstrained, logDetJacobianCorr
from metabeta.utils.results import Proposal


def _batch(b=1, m=2, n=4, d=1, q=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.cat([torch.ones(b, m, n, 1), torch.randn(b, m, n, d - 1, generator=g)], -1)
    Z = X[..., :q].clone()
    y = torch.randn(b, m, n, generator=g)
    return {
        'X': X,
        'Z': Z,
        'y': y,
        'nu_ffx': torch.zeros(b, d),
        'tau_ffx': torch.ones(b, d),
        'tau_rfx': torch.ones(b, q),
        'tau_eps': torch.ones(b),
        'family_ffx': torch.zeros(b, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(b, dtype=torch.long),
        'family_sigma_eps': torch.zeros(b, dtype=torch.long),
        'eta_rfx': torch.full((b,), 2.0) if q >= 2 else torch.zeros(b),
        'mask_d': torch.ones(b, d, dtype=torch.bool),
        'mask_q': torch.ones(b, q, dtype=torch.bool),
        'mask_mq': torch.ones(b, m, q, dtype=torch.bool),
        'mask_m': torch.ones(b, m, dtype=torch.bool),
        'mask_n': torch.ones(b, m, n, dtype=torch.bool),
        'n': torch.full((b,), m * n),
        'sd_y': torch.full((b,), 2.0),
    }


def _proposal(ffx, sigma_rfx, sigma_eps, log_q, r_corr=None):
    parts = [ffx, sigma_rfx, sigma_eps.unsqueeze(-1)]
    d_corr = 0
    if r_corr is not None:
        parts.append(r_corr)
        d_corr = r_corr.shape[-1]
    samples_g = torch.cat(parts, -1)
    b, s = samples_g.shape[:2]
    q = sigma_rfx.shape[-1]
    proposed = {
        'global': {'samples': samples_g, 'log_prob': log_q},
        'local': {'samples': torch.zeros(b, 2, s, q), 'log_prob': torch.zeros(b, 2, s)},
    }
    return Proposal(proposed, has_sigma_eps=True, d_corr=d_corr)


def test_log_evidence_is_proposal_invariant():
    """logsumexp(log_w) - log S estimates the same log p(D) under two different proposals."""
    data = _batch()
    data = {k: v.double() if v.is_floating_point() else v for k, v in data.items()}
    sampler = ImportanceSampler(data, marginal=True, corr_prior=True, pareto=False, constrain=False)
    s = 200_000
    g = torch.Generator().manual_seed(1)
    estimates = []
    for shift, scale in ((0.0, 1.0), (0.4, 1.6)):
        p_ffx = D.Normal(torch.tensor(shift, dtype=torch.double), scale)
        p_sig = D.HalfNormal(torch.tensor(scale, dtype=torch.double))
        ffx = p_ffx.sample((1, s, 1))
        sigma_rfx = p_sig.sample((1, s, 1))
        sigma_eps = p_sig.sample((1, s))
        log_q = (
            p_ffx.log_prob(ffx).sum(-1)
            + p_sig.log_prob(sigma_rfx).sum(-1)
            + p_sig.log_prob(sigma_eps)
        )
        out = sampler(_proposal(ffx, sigma_rfx, sigma_eps, log_q))
        lw = out.is_results['log_w_raw']
        assert lw.shape == (1, s)
        assert torch.allclose(out.log_evidence, torch.logsumexp(lw, -1) - math.log(s), atol=1e-10)
        estimates.append(out.log_evidence.item())
    assert abs(estimates[0] - estimates[1]) < 0.05, estimates


def test_log_jacobian_standardization():
    data = _batch()
    lj = logJacobianStandardization(data)
    assert torch.allclose(lj, torch.tensor([-8.0 * math.log(2.0)]))


def test_corr_prior_is_a_density_over_stored_r():
    """The LKJ term in the weights is LKJ(z) in the dataset's own q minus the padded z→r Jacobian.

    For q_i = q = 2 the Jacobian is exactly log(1 - rho^2), i.e. the weight differs from the
    legacy density-over-z convention by -log(1 - rho^2).
    """
    data = _batch(q=2, d=2)
    s = 16
    g = torch.Generator().manual_seed(2)
    rho = torch.rand(1, s, 1, generator=g) * 1.8 - 0.9
    ffx = torch.randn(1, s, 2, generator=g)
    sigma_rfx = torch.rand(1, s, 2, generator=g) + 0.2
    sigma_eps = torch.rand(1, s, generator=g) + 0.2
    log_q = torch.zeros(1, s)
    lw = {}
    for corr_prior in (False, True):
        sampler = ImportanceSampler(
            data, marginal=True, corr_prior=corr_prior, pareto=False, constrain=False
        )
        out = sampler(_proposal(ffx, sigma_rfx, sigma_eps, log_q, r_corr=rho))
        lw[corr_prior] = out.is_results['log_w_raw']
    z = corrLowerToUnconstrained(rho, 2)
    expected = logProbCorrRfx(z, 2, data['eta_rfx']) - logDetJacobianCorr(z, 2)
    assert torch.allclose(lw[True] - lw[False], expected, atol=1e-5)
    assert torch.allclose(-logDetJacobianCorr(z, 2), -torch.log1p(-rho.squeeze(-1) ** 2), atol=1e-4)


def test_logProbCorrRfx_q_active_matches_own_dimension():
    """A q_i=2 dataset padded into q=3 must get the 2-dim LKJ density, not the 3-dim one."""
    g = torch.Generator().manual_seed(3)
    z2 = torch.randn(1, 8, 1, generator=g)
    z3 = torch.cat([z2, torch.zeros(1, 8, 2)], -1)
    eta = torch.tensor([2.5])
    lp2 = logProbCorrRfx(z2, 2, eta)
    lp3_padded = logProbCorrRfx(z3, 3, eta)
    lp3_active = logProbCorrRfx(z3, 3, eta, q_active=torch.tensor([2]))
    assert torch.allclose(lp3_active, lp2, atol=1e-5)
    # the padded evaluation differs by a sample-dependent amount, not a constant
    delta = lp3_padded - lp2
    assert delta.std() > 1e-3
    # and a full-dimension dataset is unchanged by q_active
    zf = torch.randn(1, 8, 3, generator=g)
    assert torch.allclose(
        logProbCorrRfx(zf, 3, eta), logProbCorrRfx(zf, 3, eta, q_active=torch.tensor([3]))
    )
