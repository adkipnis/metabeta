import numpy as np
import pytest
import torch
from torch import distributions as D

from metabeta.utils.families import (
    FFX_FAMILIES,
    LIKELIHOOD_FAMILIES,
    LIKELIHOOD_HAS_SIGMA_EPS,
    SIGMA_FAMILIES,
    STUDENT_DF,
    FamilyEncoder,
    hasSigmaEps,
    logLikelihood,
    logProbFfx,
    logProbSigma,
    oneHotFamily,
    posteriorPredictiveDist,
    sampleFfxNp,
    sampleFfxTorch,
    sampleSigmaNp,
    sampleSigmaTorch,
    simulateBernoulliNp,
    simulateNormalNp,
    simulatePoissonNp,
    simulateYNp,
)


SIGMA_HALFNORMAL = SIGMA_FAMILIES.index('halfnormal')
SIGMA_HALFSTUDENT = SIGMA_FAMILIES.index('halfstudent')
SIGMA_EXPONENTIAL = SIGMA_FAMILIES.index('exponential')


# ---------------------------------------------------------------------------
# NumPy sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('family', range(len(FFX_FAMILIES)))
def test_sampleFfxNp_shape_and_finite(family):
    rng = np.random.default_rng(0)
    loc = np.array([0.0, 1.0, -0.5])
    scale = np.array([1.0, 2.0, 0.5])
    x = sampleFfxNp(family, loc, scale, rng, size=loc.shape)
    assert x.shape == loc.shape
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize('family', range(len(SIGMA_FAMILIES)))
def test_sampleSigmaNp_positive_and_finite(family):
    rng = np.random.default_rng(0)
    scale = np.array([1.0, 2.0])
    x = sampleSigmaNp(family, scale, rng, size=scale.shape)
    assert x.shape == scale.shape
    assert np.all(x >= 0)
    assert np.all(np.isfinite(x))


def test_sampleFfxNp_normal_matches_scipy():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    loc = np.array([1.0, 2.0])
    scale = np.array([0.5, 1.0])
    x = sampleFfxNp(0, loc, scale, rng1, size=loc.shape)
    from scipy.stats import norm

    expected = norm(loc, scale).rvs(size=loc.shape, random_state=rng2)
    np.testing.assert_allclose(x, expected)


def test_sampleSigmaNp_halfnormal_matches_scipy():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    scale = np.array([1.0, 2.0])
    x = sampleSigmaNp(SIGMA_HALFNORMAL, scale, rng1, size=scale.shape)
    from scipy.stats import norm

    expected = np.abs(norm(0, scale).rvs(size=scale.shape, random_state=rng2))
    np.testing.assert_allclose(x, expected)


def test_sampleSigmaNp_exponential_matches_scipy():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    scale = np.array([1.0, 2.0])
    x = sampleSigmaNp(SIGMA_EXPONENTIAL, scale, rng1, size=scale.shape)
    from scipy.stats import expon

    expected = expon(scale=scale).rvs(size=scale.shape, random_state=rng2)
    np.testing.assert_allclose(x, expected)


# ---------------------------------------------------------------------------
# Torch log-prob
# ---------------------------------------------------------------------------


def test_logProbFfx_normal_matches_torch():
    b, s, d = 4, 10, 3
    x = torch.randn(b, s, d)
    loc = torch.randn(b, 1, d)
    scale = torch.rand(b, 1, d) + 0.1
    family = torch.zeros(b, dtype=torch.long)
    lp = logProbFfx(x, loc, scale, family)
    expected = D.Normal(loc, scale).log_prob(x).sum(-1)
    assert lp.shape == (b, s)
    torch.testing.assert_close(lp, expected)


def test_logProbFfx_student_matches_torch():
    b, s, d = 4, 10, 3
    x = torch.randn(b, s, d)
    loc = torch.randn(b, 1, d)
    scale = torch.rand(b, 1, d) + 0.1
    family = torch.ones(b, dtype=torch.long)
    lp = logProbFfx(x, loc, scale, family)
    expected = D.StudentT(df=STUDENT_DF, loc=loc, scale=scale).log_prob(x).sum(-1)
    assert lp.shape == (b, s)
    torch.testing.assert_close(lp, expected)


def test_logProbFfx_mixed_batch():
    """Each batch element uses a different family; verify independently."""
    b, s, d = 4, 5, 2
    x = torch.randn(b, s, d)
    loc = torch.randn(b, 1, d)
    scale = torch.rand(b, 1, d) + 0.1
    family = torch.tensor([0, 1, 0, 1])
    lp = logProbFfx(x, loc, scale, family)

    # check element 0 (normal)
    lp0 = D.Normal(loc[0], scale[0]).log_prob(x[0]).sum(-1)
    torch.testing.assert_close(lp[0], lp0)

    # check element 1 (student)
    lp1 = D.StudentT(df=STUDENT_DF, loc=loc[1], scale=scale[1]).log_prob(x[1]).sum(-1)
    torch.testing.assert_close(lp[1], lp1)


def test_logProbFfx_with_mask():
    b, s, d = 2, 5, 3
    x = torch.randn(b, s, d)
    loc = torch.randn(b, 1, d)
    scale = torch.rand(b, 1, d) + 0.1
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float).unsqueeze(1)
    family = torch.zeros(b, dtype=torch.long)
    lp = logProbFfx(x, loc, scale, family, mask=mask)
    expected = (D.Normal(loc, scale).log_prob(x) * mask).sum(-1)
    torch.testing.assert_close(lp, expected)


def test_logProbSigma_halfnormal_matches_torch():
    b, s, q = 4, 10, 2
    x = torch.rand(b, s, q) + 0.01
    scale = torch.rand(b, 1, q) + 0.1
    family = torch.full((b,), SIGMA_HALFNORMAL, dtype=torch.long)
    lp = logProbSigma(x, scale, family)
    expected = D.HalfNormal(scale=scale).log_prob(x).sum(-1)
    assert lp.shape == (b, s)
    torch.testing.assert_close(lp, expected)


def test_logProbSigma_halfstudent_matches_torch():
    import math

    b, s, q = 4, 10, 2
    x = torch.rand(b, s, q) + 0.01
    scale = torch.rand(b, 1, q) + 0.1
    family = torch.full((b,), SIGMA_HALFSTUDENT, dtype=torch.long)
    lp = logProbSigma(x, scale, family)
    expected = D.StudentT(df=STUDENT_DF, loc=0, scale=scale).log_prob(x) + math.log(2.0)
    expected = expected.sum(-1)
    assert lp.shape == (b, s)
    torch.testing.assert_close(lp, expected)


def test_logProbSigma_exponential_matches_torch():
    b, s, q = 4, 10, 2
    x = torch.rand(b, s, q) + 0.01
    scale = torch.rand(b, 1, q) + 0.1
    family = torch.full((b,), SIGMA_EXPONENTIAL, dtype=torch.long)
    lp = logProbSigma(x, scale, family)
    expected = D.Exponential(rate=1.0 / scale).log_prob(x).sum(-1)
    assert lp.shape == (b, s)
    torch.testing.assert_close(lp, expected)


def test_logProbSigma_scalar():
    """sigma_eps is (b, s) not (b, s, q)."""
    b, s = 4, 10
    x = torch.rand(b, s) + 0.01
    scale = torch.rand(b, 1) + 0.1
    family = torch.full((b,), SIGMA_HALFNORMAL, dtype=torch.long)
    lp = logProbSigma(x, scale, family)
    expected = D.HalfNormal(scale=scale).log_prob(x)
    assert lp.shape == (b, s)
    torch.testing.assert_close(lp, expected)


def test_logProbSigma_mixed_batch():
    b, s, q = 4, 5, 2
    x = torch.rand(b, s, q) + 0.01
    scale = torch.rand(b, 1, q) + 0.1
    family = torch.tensor(
        [SIGMA_HALFNORMAL, SIGMA_HALFSTUDENT, SIGMA_HALFSTUDENT, SIGMA_HALFNORMAL]
    )
    lp = logProbSigma(x, scale, family)

    lp0 = D.HalfNormal(scale=scale[0]).log_prob(x[0]).sum(-1)
    torch.testing.assert_close(lp[0], lp0)

    import math

    lp1 = (D.StudentT(df=STUDENT_DF, loc=0, scale=scale[1]).log_prob(x[1]) + math.log(2.0)).sum(-1)
    torch.testing.assert_close(lp[1], lp1)


# ---------------------------------------------------------------------------
# Torch sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('family_idx', range(len(FFX_FAMILIES)))
def test_sampleFfxTorch_shape(family_idx):
    b, d, n_samples = 4, 3, 50
    loc = torch.randn(b, d)
    scale = torch.rand(b, d) + 0.1
    family = torch.full((b,), family_idx, dtype=torch.long)
    x = sampleFfxTorch(loc, scale, family, (n_samples,))
    assert x.shape == (n_samples, b, d)
    assert torch.all(torch.isfinite(x))


@pytest.mark.parametrize('family_idx', range(len(SIGMA_FAMILIES)))
def test_sampleSigmaTorch_shape_and_positive(family_idx):
    b, q, n_samples = 4, 2, 50
    scale = torch.rand(b, q) + 0.1
    family = torch.full((b,), family_idx, dtype=torch.long)
    x = sampleSigmaTorch(scale, family, (n_samples,))
    assert x.shape == (n_samples, b, q)
    assert torch.all(x >= 0)


def test_sampleSigmaTorch_scalar():
    """sigma_eps case: scale is (b,)."""
    b, n_samples = 4, 50
    scale = torch.rand(b) + 0.1
    family = torch.full((b,), SIGMA_HALFNORMAL, dtype=torch.long)
    x = sampleSigmaTorch(scale, family, (n_samples,))
    assert x.shape == (n_samples, b)
    assert torch.all(x >= 0)


def test_sampleFfxTorch_mixed_families():
    """Mixed batch: first two Normal, last two StudentT."""
    b, d = 4, 3
    loc = torch.zeros(b, d)
    scale = torch.ones(b, d)
    family = torch.tensor([0, 0, 1, 1])
    x = sampleFfxTorch(loc, scale, family, (100,))
    assert x.shape == (100, b, d)
    assert torch.all(torch.isfinite(x))


def test_sampleFfxTorch_empty_shape():
    """shape=() should produce (b, d)."""
    b, d = 3, 2
    loc = torch.randn(b, d)
    scale = torch.rand(b, d) + 0.1
    family = torch.zeros(b, dtype=torch.long)
    x = sampleFfxTorch(loc, scale, family, ())
    assert x.shape == (b, d)


# ---------------------------------------------------------------------------
# Round-trip: sample → log_prob consistency
# ---------------------------------------------------------------------------


def test_roundtrip_ffx_normal():
    """Samples from Normal should have finite log-prob under Normal."""
    b, d = 8, 3
    loc = torch.randn(b, d)
    scale = torch.rand(b, d) + 0.1
    family = torch.zeros(b, dtype=torch.long)
    x = sampleFfxTorch(loc, scale, family, (20,))  # (20, b, d)
    x = x.permute(1, 0, 2)  # (b, s, d)
    lp = logProbFfx(x, loc.unsqueeze(1), scale.unsqueeze(1), family)
    assert lp.shape == (b, 20)
    assert torch.all(torch.isfinite(lp))


def test_roundtrip_ffx_student():
    b, d = 8, 3
    loc = torch.randn(b, d)
    scale = torch.rand(b, d) + 0.1
    family = torch.ones(b, dtype=torch.long)
    x = sampleFfxTorch(loc, scale, family, (20,)).permute(1, 0, 2)
    lp = logProbFfx(x, loc.unsqueeze(1), scale.unsqueeze(1), family)
    assert torch.all(torch.isfinite(lp))


def test_roundtrip_sigma_halfnormal():
    b, q = 8, 2
    scale = torch.rand(b, q) + 0.1
    family = torch.zeros(b, dtype=torch.long)
    x = sampleSigmaTorch(scale, family, (20,)).permute(1, 0, 2)
    lp = logProbSigma(x, scale.unsqueeze(1), family)
    assert torch.all(torch.isfinite(lp))


def test_roundtrip_sigma_halfstudent():
    b, q = 8, 2
    scale = torch.rand(b, q) + 0.1
    family = torch.full((b,), SIGMA_HALFSTUDENT, dtype=torch.long)
    x = sampleSigmaTorch(scale, family, (20,)).permute(1, 0, 2)
    lp = logProbSigma(x, scale.unsqueeze(1), family)
    assert torch.all(torch.isfinite(lp))


def test_roundtrip_sigma_exponential():
    b, q = 8, 2
    scale = torch.rand(b, q) + 0.1
    family = torch.full((b,), SIGMA_EXPONENTIAL, dtype=torch.long)
    x = sampleSigmaTorch(scale, family, (20,)).permute(1, 0, 2)
    lp = logProbSigma(x, scale.unsqueeze(1), family)
    assert torch.all(torch.isfinite(lp))


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_oneHotFamily():
    family = torch.tensor([0, 1, 0, 1])
    oh = oneHotFamily(family, 2)
    assert oh.shape == (4, 2)
    expected = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float)
    torch.testing.assert_close(oh, expected)


def test_familyEncoder_onehot():
    enc = FamilyEncoder(n_families=(2, 2, 2), embed_dim=None)
    assert enc.d_output == 6
    families = [torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0, 0])]
    out = enc(families)
    assert out.shape == (2, 6)
    # first row: [1,0, 0,1, 1,0]
    expected_0 = torch.tensor([1, 0, 0, 1, 1, 0], dtype=torch.float)
    torch.testing.assert_close(out[0], expected_0)


def test_familyEncoder_embedding():
    enc = FamilyEncoder(n_families=(2, 2, 2), embed_dim=4)
    assert enc.d_output == 12
    families = [torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0, 0])]
    out = enc(families)
    assert out.shape == (2, 12)
    assert torch.all(torch.isfinite(out))


def test_familyEncoder_embedding_gradients():
    enc = FamilyEncoder(n_families=(2, 2), embed_dim=3)
    families = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])]
    out = enc(families)
    loss = out.sum()
    loss.backward()
    for emb in enc.embeddings:
        assert emb.weight.grad is not None


# ---------------------------------------------------------------------------
# Invalid family index
# ---------------------------------------------------------------------------


def test_sampleFfxNp_invalid_family_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(IndexError):
        sampleFfxNp(99, np.array([0.0]), np.array([1.0]), rng, (1,))


def test_sampleSigmaNp_invalid_family_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(IndexError):
        sampleSigmaNp(99, np.array([1.0]), rng, (1,))


# ---------------------------------------------------------------------------
# Likelihood constants
# ---------------------------------------------------------------------------


def test_hasSigmaEps():
    assert hasSigmaEps(0) is True  # normal
    assert hasSigmaEps(1) is False  # bernoulli
    assert hasSigmaEps(2) is False  # poisson


def test_likelihood_families_and_sigma_eps_aligned():
    assert len(LIKELIHOOD_FAMILIES) == len(LIKELIHOOD_HAS_SIGMA_EPS)


# ---------------------------------------------------------------------------
# NumPy likelihood simulation
# ---------------------------------------------------------------------------


def test_simulateNormalNp_shape_and_finite():
    rng = np.random.default_rng(0)
    eta = np.array([0.0, 1.0, -0.5, 2.0])
    y = simulateNormalNp(rng, eta, sigma_eps=0.5)
    assert y.shape == eta.shape
    assert np.all(np.isfinite(y))


def test_simulateNormalNp_mean_close_to_eta():
    rng = np.random.default_rng(0)
    eta = np.full(10_000, 3.0)
    y = simulateNormalNp(rng, eta, sigma_eps=0.1)
    assert np.isclose(y.mean(), 3.0, atol=0.01)


def test_simulateBernoulliNp_binary():
    rng = np.random.default_rng(0)
    eta = np.linspace(-3, 3, 100)
    y = simulateBernoulliNp(rng, eta, sigma_eps=0.0)
    assert set(np.unique(y)).issubset({0, 1})
    assert y.shape == eta.shape


def test_simulateBernoulliNp_high_logit_mostly_ones():
    rng = np.random.default_rng(0)
    eta = np.full(1000, 5.0)  # sigmoid(5) ≈ 0.993
    y = simulateBernoulliNp(rng, eta, sigma_eps=0.0)
    assert y.mean() > 0.95


def test_simulateBernoulliNp_low_logit_mostly_zeros():
    rng = np.random.default_rng(0)
    eta = np.full(1000, -5.0)  # sigmoid(-5) ≈ 0.007
    y = simulateBernoulliNp(rng, eta, sigma_eps=0.0)
    assert y.mean() < 0.05


def test_simulateYNp_dispatches_normal():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    eta = np.array([1.0, 2.0, 3.0])
    y1 = simulateYNp(rng1, eta, sigma_eps=0.5, likelihood_family=0)
    y2 = simulateNormalNp(rng2, eta, sigma_eps=0.5)
    np.testing.assert_allclose(y1, y2)


def test_simulateYNp_dispatches_bernoulli():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    eta = np.array([1.0, -1.0, 0.0])
    y1 = simulateYNp(rng1, eta, sigma_eps=0.0, likelihood_family=1)
    y2 = simulateBernoulliNp(rng2, eta, sigma_eps=0.0)
    np.testing.assert_allclose(y1, y2)


def test_simulateYNp_invalid_family_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(IndexError):
        simulateYNp(rng, np.array([0.0]), sigma_eps=1.0, likelihood_family=99)


# ---------------------------------------------------------------------------
# Torch log-likelihood
# ---------------------------------------------------------------------------


def _make_ll_tensors(b=4, m=3, n=5, s=10, d=2, q=1):
    """Helper to create tensors for logLikelihood tests."""
    ffx = torch.randn(b, s, d)
    sigma_eps = torch.rand(b, s) + 0.1
    rfx = torch.randn(b, m, s, q)
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    mask = torch.ones(b, m, n, 1)
    return ffx, sigma_eps, rfx, X, Z, mask


def test_logLikelihood_normal_default():
    """Default (likelihood_family=0) should match D.Normal."""
    ffx, sigma_eps, rfx, X, Z, mask = _make_ll_tensors()
    ll = logLikelihood(ffx, sigma_eps, rfx, torch.randn(4, 3, 5, 1), X, Z, mask)
    assert ll.shape == (4, 10)
    assert torch.all(torch.isfinite(ll))


def test_logLikelihood_normal_matches_manual():
    b, m, n, s, d, q = 2, 2, 3, 5, 2, 1
    torch.manual_seed(0)
    ffx = torch.randn(b, s, d)
    sigma_eps = torch.rand(b, s) + 0.1
    rfx = torch.randn(b, m, s, q)
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    y = torch.randn(b, m, n, 1)

    ll = logLikelihood(ffx, sigma_eps, rfx, y, X, Z, likelihood_family=0)

    # manual
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    eta = mu_g + mu_l
    scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
    expected = D.Normal(loc=eta, scale=scale).log_prob(y).sum(dim=(1, 2))
    torch.testing.assert_close(ll, expected)


def test_logLikelihood_bernoulli_binary_y():
    b, m, n, s, d, q = 2, 2, 3, 5, 2, 1
    torch.manual_seed(0)
    ffx = torch.randn(b, s, d)
    sigma_eps = torch.rand(b, s)  # unused
    rfx = torch.randn(b, m, s, q)
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    y = torch.randint(0, 2, (b, m, n, 1)).float()

    ll = logLikelihood(ffx, sigma_eps, rfx, y, X, Z, likelihood_family=1)

    # manual
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    eta = mu_g + mu_l
    expected = D.Bernoulli(logits=eta).log_prob(y).sum(dim=(1, 2))
    torch.testing.assert_close(ll, expected)


def test_logLikelihood_bernoulli_with_mask():
    b, m, n, s, d, q = 2, 2, 4, 5, 2, 1
    torch.manual_seed(0)
    ffx = torch.randn(b, s, d)
    sigma_eps = torch.rand(b, s)
    rfx = torch.randn(b, m, s, q)
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    y = torch.randint(0, 2, (b, m, n, 1)).float()
    mask = torch.ones(b, m, n, 1)
    mask[:, :, -1:, :] = 0  # mask out last observation

    ll = logLikelihood(ffx, sigma_eps, rfx, y, X, Z, mask=mask, likelihood_family=1)

    # manual
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    eta = mu_g + mu_l
    expected = (D.Bernoulli(logits=eta).log_prob(y) * mask).sum(dim=(1, 2))
    torch.testing.assert_close(ll, expected)


def test_logLikelihood_bernoulli_log_prob_range():
    """Bernoulli log-prob should be in [-log(2), 0]."""
    ffx, sigma_eps, rfx, X, Z, mask = _make_ll_tensors(b=2, s=5)
    y = torch.randint(0, 2, (2, 3, 5, 1)).float()
    ll = logLikelihood(ffx, sigma_eps, rfx, y, X, Z, mask=mask, likelihood_family=1)
    assert torch.all(torch.isfinite(ll))
    assert torch.all(ll <= 0)


# ---------------------------------------------------------------------------
# Torch posterior predictive distribution
# ---------------------------------------------------------------------------


def test_posteriorPredictiveDist_normal():
    b, m, n, s, d, q = 2, 3, 5, 10, 2, 1
    ffx = torch.randn(b, s, d)
    sigma_eps = torch.rand(b, s) + 0.1
    rfx = torch.randn(b, m, s, q)
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    pp = posteriorPredictiveDist(ffx, sigma_eps, rfx, X, Z, likelihood_family=0)
    assert isinstance(pp, D.Normal)
    y = pp.sample()
    assert y.shape == (b, m, n, s)
    assert torch.all(torch.isfinite(y))


def test_posteriorPredictiveDist_bernoulli():
    b, m, n, s, d, q = 2, 3, 5, 10, 2, 1
    ffx = torch.randn(b, s, d)
    sigma_eps = torch.rand(b, s)
    rfx = torch.randn(b, m, s, q)
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    pp = posteriorPredictiveDist(ffx, sigma_eps, rfx, X, Z, likelihood_family=1)
    assert isinstance(pp, D.Bernoulli)
    y = pp.sample()
    assert y.shape == (b, m, n, s)
    assert set(y.unique().tolist()).issubset({0.0, 1.0})


def test_posteriorPredictiveDist_bernoulli_log_prob():
    b, m, n, s, d, q = 2, 3, 5, 10, 2, 1
    ffx = torch.randn(b, s, d)
    sigma_eps = torch.rand(b, s)
    rfx = torch.randn(b, m, s, q)
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    pp = posteriorPredictiveDist(ffx, sigma_eps, rfx, X, Z, likelihood_family=1)
    y = torch.randint(0, 2, (b, m, n, s)).float()
    lp = pp.log_prob(y)
    assert lp.shape == (b, m, n, s)
    assert torch.all(torch.isfinite(lp))
    assert torch.all(lp <= 0)


def test_posteriorPredictiveDist_invalid_family_raises():
    ffx = torch.randn(2, 5, 2)
    sigma_eps = torch.rand(2, 5)
    rfx = torch.randn(2, 3, 5, 1)
    X = torch.randn(2, 3, 4, 2)
    Z = X[..., :1]
    with pytest.raises(ValueError, match='unknown likelihood family'):
        posteriorPredictiveDist(ffx, sigma_eps, rfx, X, Z, likelihood_family=99)


# ---------------------------------------------------------------------------
# Poisson likelihood
# ---------------------------------------------------------------------------


def test_simulatePoissonNp_nonnegative_integers():
    rng = np.random.default_rng(0)
    eta = np.array([0.0, 1.0, -1.0, 2.0])
    y = simulatePoissonNp(rng, eta)
    assert y.shape == eta.shape
    assert np.all(y >= 0)
    assert np.all(y == np.floor(y))


def test_simulatePoissonNp_mean_close_to_rate():
    rng = np.random.default_rng(0)
    eta = np.full(10_000, 1.5)  # rate = exp(1.5) ≈ 4.48
    y = simulatePoissonNp(rng, eta)
    assert np.isclose(y.mean(), np.exp(1.5), rtol=0.05)


def test_simulatePoissonNp_zero_eta():
    rng = np.random.default_rng(0)
    eta = np.full(10_000, 0.0)  # rate = 1
    y = simulatePoissonNp(rng, eta)
    assert np.isclose(y.mean(), 1.0, atol=0.05)


def test_simulateYNp_dispatches_poisson():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    eta = np.array([0.5, 1.0, -0.5])
    y1 = simulateYNp(rng1, eta, sigma_eps=0.0, likelihood_family=2)
    y2 = simulatePoissonNp(rng2, eta)
    np.testing.assert_allclose(y1, y2)


def test_logLikelihood_poisson_matches_manual():
    b, m, n, s, d, q = 2, 2, 3, 5, 2, 1
    torch.manual_seed(0)
    ffx = torch.randn(b, s, d) * 0.3  # small coefficients
    sigma_eps = torch.rand(b, s)  # unused
    rfx = torch.randn(b, m, s, q) * 0.2
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    y = torch.poisson(torch.ones(b, m, n, 1) * 3)

    ll = logLikelihood(ffx, sigma_eps, rfx, y, X, Z, likelihood_family=2)

    # manual
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    eta = mu_g + mu_l
    rate = torch.exp(eta.clamp(max=30))
    expected = D.Poisson(rate=rate).log_prob(y).sum(dim=(1, 2))
    torch.testing.assert_close(ll, expected)


def test_logLikelihood_poisson_with_mask():
    b, m, n, s, d, q = 2, 2, 4, 5, 2, 1
    torch.manual_seed(0)
    ffx = torch.randn(b, s, d) * 0.3
    sigma_eps = torch.rand(b, s)
    rfx = torch.randn(b, m, s, q) * 0.2
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    y = torch.poisson(torch.ones(b, m, n, 1) * 2)
    mask = torch.ones(b, m, n, 1)
    mask[:, :, -1:, :] = 0

    ll = logLikelihood(ffx, sigma_eps, rfx, y, X, Z, mask=mask, likelihood_family=2)

    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    eta = mu_g + mu_l
    rate = torch.exp(eta.clamp(max=30))
    expected = (D.Poisson(rate=rate).log_prob(y) * mask).sum(dim=(1, 2))
    torch.testing.assert_close(ll, expected)


def test_posteriorPredictiveDist_poisson():
    b, m, n, s, d, q = 2, 3, 5, 10, 2, 1
    ffx = torch.randn(b, s, d) * 0.3
    sigma_eps = torch.rand(b, s)
    rfx = torch.randn(b, m, s, q) * 0.2
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    pp = posteriorPredictiveDist(ffx, sigma_eps, rfx, X, Z, likelihood_family=2)
    assert isinstance(pp, D.Poisson)
    y = pp.sample()
    assert y.shape == (b, m, n, s)
    assert torch.all(y >= 0)
    assert torch.all(y == y.floor())


def test_posteriorPredictiveDist_poisson_log_prob():
    b, m, n, s, d, q = 2, 3, 5, 10, 2, 1
    ffx = torch.randn(b, s, d) * 0.3
    sigma_eps = torch.rand(b, s)
    rfx = torch.randn(b, m, s, q) * 0.2
    X = torch.randn(b, m, n, d)
    Z = X[..., :q]
    pp = posteriorPredictiveDist(ffx, sigma_eps, rfx, X, Z, likelihood_family=2)
    y = torch.poisson(torch.ones(b, m, n, s) * 2)
    lp = pp.log_prob(y)
    assert lp.shape == (b, m, n, s)
    assert torch.all(torch.isfinite(lp))
    assert torch.all(lp <= 0)
