"""Tests for Approximator with posterior_correlation=True.

All tests use synthetic data (no fixture files needed).
"""

from __future__ import annotations

import torch
import pytest

from metabeta.models.approximator import Approximator
from metabeta.utils.config import ApproximatorConfig, SummarizerConfig, PosteriorConfig
from metabeta.utils.evaluation import Proposal, joinProposals, concatProposalsBatch
from metabeta.utils.families import hasSigmaEps


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_cfg(
    d_ffx: int = 2,
    d_rfx: int = 2,
    posterior_correlation: bool = True,
    likelihood_family: int = 0,
) -> ApproximatorConfig:
    s_cfg = SummarizerConfig(d_model=16, d_ff=16, d_output=16, n_blocks=1)
    p_cfg = PosteriorConfig(n_blocks=2)
    return ApproximatorConfig(
        d_ffx=d_ffx,
        d_rfx=d_rfx,
        likelihood_family=likelihood_family,
        posterior_correlation=posterior_correlation,
        summarizer_l=s_cfg,
        summarizer_g=s_cfg,
        posterior_l=p_cfg,
        posterior_g=p_cfg,
    )


def make_batch(
    b: int = 3,
    m: int = 5,
    n: int = 12,
    d: int = 2,
    q: int = 2,
    likelihood_family: int = 0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Construct a minimal valid batch dict (all groups/obs present, uniform ns)."""
    torch.manual_seed(seed)
    has_eps = hasSigmaEps(likelihood_family)

    # core observations
    y = torch.randn(b, m, n)
    X = torch.randn(b, m, n, d)
    X[..., 0] = 1.0  # intercept
    Z = torch.randn(b, m, n, q)
    Z[..., 0] = 1.0  # intercept

    # true parameters
    ffx = torch.randn(b, d)
    sigma_rfx = torch.rand(b, q) * 0.5 + 0.5
    rfx = torch.randn(b, m, q) * sigma_rfx.unsqueeze(1)

    # LKJ correlation matrix (identity for q=1 since LKJCholesky requires q>=2)
    if q >= 2:
        lkj = torch.distributions.LKJCholesky(q, concentration=1.5)
        L_list = [lkj.sample() for _ in range(b)]
        corr_rfx = torch.stack([L @ L.mT for L in L_list])  # (b, q, q)
    else:
        corr_rfx = torch.ones(b, 1, 1)
    eta_rfx = torch.ones(b) * 1.5

    # masks (all active)
    mask_d = torch.ones(b, d, dtype=torch.bool)
    mask_q = torch.ones(b, q, dtype=torch.bool)
    mask_m = torch.ones(b, m, dtype=torch.bool)
    mask_n = torch.ones(b, m, n, dtype=torch.bool)
    mask_mq = torch.ones(b, m, q, dtype=torch.bool)
    d_corr = q * (q - 1) // 2
    mask_corr = torch.ones(b, d_corr, dtype=torch.bool)

    # counts
    ns = torch.full((b, m), float(n))
    total_n = torch.full((b,), float(m * n))
    total_m = torch.full((b,), float(m))

    # prior hyperparameters
    nu_ffx = torch.zeros(b, d)
    tau_ffx = torch.ones(b, d)
    tau_rfx = torch.ones(b, q)

    # family indices (0 = Normal)
    family_ffx = torch.zeros(b, dtype=torch.long)
    family_sigma_rfx = torch.zeros(b, dtype=torch.long)

    batch: dict[str, torch.Tensor] = dict(
        y=y,
        X=X,
        Z=Z,
        ffx=ffx,
        rfx=rfx,
        sigma_rfx=sigma_rfx,
        corr_rfx=corr_rfx,
        eta_rfx=eta_rfx,
        mask_d=mask_d,
        mask_q=mask_q,
        mask_m=mask_m,
        mask_n=mask_n,
        mask_mq=mask_mq,
        mask_corr=mask_corr,
        ns=ns,
        n=total_n,
        m=total_m,
        nu_ffx=nu_ffx,
        tau_ffx=tau_ffx,
        tau_rfx=tau_rfx,
        family_ffx=family_ffx,
        family_sigma_rfx=family_sigma_rfx,
    )

    if has_eps:
        sigma_eps = torch.rand(b) * 0.5 + 0.5
        tau_eps = torch.ones(b)
        family_sigma_eps = torch.zeros(b, dtype=torch.long)
        batch['sigma_eps'] = sigma_eps
        batch['tau_eps'] = tau_eps
        batch['family_sigma_eps'] = family_sigma_eps

    return batch


# ---------------------------------------------------------------------------
# config tests
# ---------------------------------------------------------------------------


def test_d_corr_zero_when_disabled():
    model = Approximator(make_cfg(posterior_correlation=False))
    assert model.d_corr == 0


def test_d_corr_on_config():
    cfg = make_cfg(d_rfx=2, posterior_correlation=True)
    assert cfg.d_corr == 1
    assert Approximator(cfg).d_corr == cfg.d_corr


def test_d_corr_q2():
    model = Approximator(make_cfg(d_rfx=2, posterior_correlation=True))
    assert model.d_corr == 1  # q*(q-1)//2 = 1


def test_d_corr_q3():
    model = Approximator(make_cfg(d_rfx=3, posterior_correlation=True))
    assert model.d_corr == 3


def test_d_corr_q1_stays_zero():
    """q=1 has no correlation even when flag is True."""
    model = Approximator(make_cfg(d_rfx=1, posterior_correlation=True))
    assert model.d_corr == 0


def test_global_flow_d_target():
    """Global flow target dim must include d_corr."""
    d_ffx, d_rfx = 2, 2
    model = Approximator(make_cfg(d_ffx=d_ffx, d_rfx=d_rfx, posterior_correlation=True))
    d_corr = model.d_corr  # = 1 for q=2
    has_eps = model.has_sigma_eps
    expected = d_ffx + d_rfx + (1 if has_eps else 0) + d_corr
    assert model.posterior_g.d_target == expected, (
        f'Expected d_target={expected}, got {model.posterior_g.d_target}'
    )


# ---------------------------------------------------------------------------
# forward pass
# ---------------------------------------------------------------------------


def test_forward_log_probs_finite():
    model = Approximator(make_cfg())
    batch = make_batch()
    log_probs = model.forward(batch)
    for key in ('global', 'local'):
        assert key in log_probs
        assert torch.isfinite(log_probs[key]).all(), f'{key} contains non-finite values'


def test_forward_without_corr_flag_unchanged():
    """Disabling the flag should produce the same shapes as before."""
    model_off = Approximator(make_cfg(posterior_correlation=False))
    batch = make_batch()
    log_probs = model_off.forward(batch)
    assert torch.isfinite(log_probs['global']).all()
    assert torch.isfinite(log_probs['local']).all()


def test_forward_backward_gradients():
    model = Approximator(make_cfg())
    batch = make_batch()
    log_probs = model.forward(batch)
    loss = log_probs['global'].mean() + log_probs['local'].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# backward pass / sampling
# ---------------------------------------------------------------------------


def test_backward_proposal_shapes():
    b, m, n, d, q = 2, 4, 8, 2, 2
    n_samples = 7
    model = Approximator(make_cfg(d_ffx=d, d_rfx=q))
    batch = make_batch(b=b, m=m, n=n, d=d, q=q)

    proposal = model.estimate(batch, n_samples=n_samples)

    # global samples: [ffx | sigma_rfx | sigma_eps | corr_z]
    d_corr = model.d_corr
    has_eps = model.has_sigma_eps
    expected_D = d + q + (1 if has_eps else 0) + d_corr
    assert proposal.samples_g.shape == (b, n_samples, expected_D)
    assert proposal.log_prob_g.shape == (b, n_samples)
    assert proposal.samples_l.shape == (b, m, n_samples, q)
    assert proposal.log_prob_l.shape == (b, m, n_samples)


def test_proposal_d_and_q():
    model = Approximator(make_cfg(d_ffx=3, d_rfx=2))
    batch = make_batch(d=3, q=2)
    proposal = model.estimate(batch, n_samples=5)
    assert proposal.d == 3
    assert proposal.q == 2
    assert proposal.d_corr == model.d_corr


def test_sigmas_positive():
    """Sigma samples must be positive (softplus output)."""
    model = Approximator(make_cfg())
    batch = make_batch()
    proposal = model.estimate(batch, n_samples=10)
    assert (proposal.sigma_rfx > 0).all()
    if model.has_sigma_eps:
        assert (proposal.sigma_eps > 0).all()


def test_sigma_rfx_shape():
    b, q, n_s = 3, 2, 6
    model = Approximator(make_cfg(d_rfx=q))
    batch = make_batch(b=b, q=q)
    proposal = model.estimate(batch, n_samples=n_s)
    assert proposal.sigma_rfx.shape == (b, n_s, q)


def test_corr_rfx_shape_and_symmetry():
    b, q, n_s = 3, 2, 8
    model = Approximator(make_cfg(d_rfx=q))
    batch = make_batch(b=b, q=q)
    proposal = model.estimate(batch, n_samples=n_s)

    corr = proposal.corr_rfx
    assert corr is not None
    assert corr.shape == (b, n_s, q, q)

    # must be symmetric
    assert torch.allclose(corr, corr.mT, atol=1e-5)

    # diagonal must be 1
    diag = corr.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)

    # off-diagonal must be in (-1, 1)
    off_diag_mask = ~torch.eye(q, dtype=torch.bool)
    off = corr[..., off_diag_mask]
    assert (off.abs() < 1.0).all()


def test_corr_rfx_none_when_flag_off():
    model = Approximator(make_cfg(posterior_correlation=False))
    batch = make_batch()
    proposal = model.estimate(batch, n_samples=4)
    assert proposal.corr_rfx is None


def test_corr_rfx_none_for_q1():
    model = Approximator(make_cfg(d_rfx=1, posterior_correlation=True))
    # q=1 has no correlation; use a q=1 batch
    batch = make_batch(q=1, d=2)
    # patch mask_q and rfx for q=1 batch
    b, m, n = 3, 5, 12
    batch['mask_q'] = torch.ones(b, 1, dtype=torch.bool)
    batch['mask_mq'] = torch.ones(b, m, 1, dtype=torch.bool)
    batch['rfx'] = batch['rfx'][..., :1]
    batch['sigma_rfx'] = batch['sigma_rfx'][..., :1]
    batch['corr_rfx'] = torch.eye(1).unsqueeze(0).expand(b, -1, -1)
    batch['tau_rfx'] = batch['tau_rfx'][..., :1]
    proposal = model.estimate(batch, n_samples=4)
    assert proposal.corr_rfx is None


# ---------------------------------------------------------------------------
# Proposal API consistency (no model needed)
# ---------------------------------------------------------------------------


def _make_fake_proposal(b=2, s=5, d=2, q=2, has_eps=True, d_corr=1):
    D = d + q + (1 if has_eps else 0) + d_corr
    samples_g = torch.randn(b, s, D)
    # make sigma block positive by squaring
    sig_start, sig_end = d, d + q + (1 if has_eps else 0)
    samples_g[..., sig_start:sig_end] = samples_g[..., sig_start:sig_end].abs() + 0.1
    log_prob_g = torch.randn(b, s)
    m = 3
    samples_l = torch.randn(b, m, s, q)
    log_prob_l = torch.randn(b, m, s)
    proposed = {
        'global': {'samples': samples_g, 'log_prob': log_prob_g},
        'local': {'samples': samples_l, 'log_prob': log_prob_l},
    }
    return Proposal(proposed, has_sigma_eps=has_eps, d_corr=d_corr)


def test_proposal_d_correct():
    p = _make_fake_proposal(d=3, q=2, has_eps=True, d_corr=1)
    assert p.d == 3


def test_proposal_sigma_rfx_correct_slice():
    p = _make_fake_proposal(d=2, q=2, has_eps=True, d_corr=1)
    # sigma_rfx should be samples_g[..., 2:4]
    assert torch.equal(p.sigma_rfx, p.samples_g[..., 2:4])


def test_proposal_sigma_eps_correct_slice():
    p = _make_fake_proposal(d=2, q=2, has_eps=True, d_corr=1)
    # sigma_eps should be samples_g[..., 4] (index, not slice)
    assert torch.equal(p.sigma_eps, p.samples_g[..., 4])


def test_proposal_corr_rfx_shape():
    p = _make_fake_proposal(b=2, s=5, d=2, q=2, has_eps=True, d_corr=1)
    corr = p.corr_rfx
    assert corr is not None
    assert corr.shape == (2, 5, 2, 2)


def test_proposal_no_corr_when_d_corr_zero():
    p = _make_fake_proposal(d_corr=0)
    assert p.corr_rfx is None


def test_proposal_partition_excludes_corr_z():
    """partition() covers only the named model params; corr_z is accessed via corr_rfx."""
    p = _make_fake_proposal(d=2, q=2, has_eps=True, d_corr=1)
    parts = p.partition(p.samples_g)
    assert set(parts.keys()) == {'ffx', 'sigma_rfx', 'sigma_eps'}


def test_join_proposals_preserves_d_corr():
    p1 = _make_fake_proposal(b=2, s=3, d=2, q=2, d_corr=1)
    p2 = _make_fake_proposal(b=2, s=4, d=2, q=2, d_corr=1)
    joined = joinProposals([p1, p2])
    assert joined.d_corr == 1
    assert joined.samples_g.shape[-2] == 7  # 3+4 samples


def test_concat_proposals_batch_preserves_d_corr():
    p1 = _make_fake_proposal(b=2, s=5, d=2, q=2, d_corr=1)
    p2 = _make_fake_proposal(b=3, s=5, d=2, q=2, d_corr=1)
    merged = concatProposalsBatch([p1, p2])
    assert merged.d_corr == 1
    assert merged.samples_g.shape[0] == 5  # 2+3 batch
