"""Tests for psisLooNLL and posteriorPredictiveCoverage in metabeta.evaluation.predictive."""
import pytest
import torch
from torch import distributions as D

from metabeta.evaluation.predictive import (
    posteriorPredictiveCoverage,
    posteriorPredictiveNLL,
    psisLooNLL,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_pp(b: int, m: int, n: int, s: int, seed: int = 0) -> D.Distribution:
    """Normal pp with loc that varies per sample (so PSIS has non-trivial weights)."""
    torch.manual_seed(seed)
    loc = 0.5 * torch.randn(b, m, n, s)       # data-independent but varies with s
    scale = torch.ones(b, m, n, s) * 0.8
    return D.Normal(loc, scale)


def _make_data(b: int, m: int, n: int, seed: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        'y': torch.randn(b, m, n),
        'mask_n': torch.ones(b, m, n, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# shape and finiteness
# ---------------------------------------------------------------------------


def test_psisLooNLL_shape_and_finite():
    b, m, n, s = 3, 4, 6, 256
    pp = _make_pp(b, m, n, s)
    data = _make_data(b, m, n)
    loo_nll, pareto_k = psisLooNLL(pp, data)
    assert loo_nll.shape == (b,)
    assert pareto_k.shape == (b,)
    assert torch.isfinite(loo_nll).all(), f'loo_nll: {loo_nll}'
    # pareto_k may be nan for degenerate obs but should not be inf
    assert not pareto_k.isinf().any(), f'pareto_k contains inf: {pareto_k}'


def test_psisLooNLL_with_is_weights_shape_and_finite():
    b, m, n, s = 3, 4, 6, 256
    pp = _make_pp(b, m, n, s)
    data = _make_data(b, m, n)
    torch.manual_seed(3)
    w = torch.softmax(torch.randn(b, s), dim=-1)
    loo_nll, pareto_k = psisLooNLL(pp, data, w=w)
    assert loo_nll.shape == (b,)
    assert pareto_k.shape == (b,)
    assert torch.isfinite(loo_nll).all(), f'loo_nll: {loo_nll}'
    assert not pareto_k.isinf().any(), f'pareto_k contains inf: {pareto_k}'


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------


def test_psisLooNLL_respects_mask():
    """LOO NLL should be finite for variable-length groups."""
    b, m, n_max, s = 2, 3, 8, 256
    ns = [8, 5, 3]

    torch.manual_seed(2)
    y = torch.randn(b, m, n_max)
    mask_n = torch.zeros(b, m, n_max, dtype=torch.bool)
    for i in range(b):
        for j in range(m):
            mask_n[i, j, : ns[j]] = True

    data = {'y': y, 'mask_n': mask_n}
    pp = _make_pp(b, m, n_max, s)

    loo_nll, _ = psisLooNLL(pp, data)
    assert loo_nll.shape == (b,)
    assert torch.isfinite(loo_nll).all(), f'loo_nll: {loo_nll}'


# ---------------------------------------------------------------------------
# LOO NLL >= in-sample NLL for an overfitting-prone model
# ---------------------------------------------------------------------------


def test_psisLooNLL_optimism_on_average():
    """
    LOO NLL should exceed in-sample ppNLL on average when the posterior mean
    tracks the observed data (i.e. the model is 'over-tuned' to the data).
    """
    torch.manual_seed(42)
    b, m, n, s = 10, 4, 8, 512

    y = torch.randn(b, m, n)
    data = {'y': y, 'mask_n': torch.ones(b, m, n, dtype=torch.bool)}

    # pp strongly centered on y — leaving one out shifts the distribution away
    loc = y.unsqueeze(-1) + 0.1 * torch.randn(b, m, n, s)
    scale = torch.full((b, m, n, s), 0.3)
    pp = D.Normal(loc, scale)

    in_sample = posteriorPredictiveNLL(pp, data)   # (b,)
    loo_nll, _ = psisLooNLL(pp, data)              # (b,)

    mean_diff = (loo_nll - in_sample).mean().item()
    assert (
        mean_diff > 0
    ), f'Expected LOO NLL > in-sample NLL on average, got mean diff = {mean_diff:.4f}'


# ---------------------------------------------------------------------------
# Pareto k diagnostic
# ---------------------------------------------------------------------------


def test_psisLooNLL_pareto_k_benign():
    """
    For a diffuse model with many obs per group, each observation is low-
    influence and mean Pareto k should be well below 0.7.
    """
    torch.manual_seed(7)
    b, m, n, s = 4, 5, 15, 512

    y = torch.randn(b, m, n)
    data = {'y': y, 'mask_n': torch.ones(b, m, n, dtype=torch.bool)}

    # pp with varying loc (so weights differ across s) but large scale (diffuse)
    loc = 0.3 * torch.randn(b, m, n, s)
    scale = torch.full((b, m, n, s), 2.0)
    pp = D.Normal(loc, scale)

    _, pareto_k = psisLooNLL(pp, data)

    # use nanmean in case some obs have degenerate k
    k_mean = pareto_k.nanmean().item()
    assert k_mean < 0.7, f'Mean Pareto k = {k_mean:.3f}, expected < 0.7 for benign model'


# ---------------------------------------------------------------------------
# posteriorPredictiveCoverage
# ---------------------------------------------------------------------------


def test_ppc_shape_and_finite():
    b, m, n, s = 3, 4, 6, 256
    pp = _make_pp(b, m, n, s)
    data = _make_data(b, m, n)
    alphas = [0.05, 0.1, 0.5]
    cov, wid = posteriorPredictiveCoverage(pp, data, alphas=alphas)
    assert cov.shape == (len(alphas), b)
    assert wid.shape == (len(alphas), b)
    assert torch.isfinite(cov).all(), f'coverage not finite: {cov}'
    assert torch.isfinite(wid).all(), f'width not finite: {wid}'
    assert (cov >= 0).all() and (cov <= 1 + 1e-6).all()
    assert (wid >= 0).all()


def test_ppc_width_monotone_in_alpha():
    """90% intervals (alpha=0.1) must be wider than 50% intervals (alpha=0.5)."""
    torch.manual_seed(10)
    b, m, n, s = 4, 3, 8, 512
    pp = _make_pp(b, m, n, s)
    data = _make_data(b, m, n)
    cov, wid = posteriorPredictiveCoverage(pp, data, alphas=[0.5, 0.1])
    # wid[0] = 50% interval, wid[1] = 90% interval
    assert (wid[1] >= wid[0] - 1e-6).all(), f'90% width not >= 50% width: {wid}'


def test_ppc_calibrated_coverage():
    """When y is drawn from the true predictive, coverage should match nominal."""
    torch.manual_seed(42)
    b, m, n, s = 30, 3, 10, 1024

    loc = torch.zeros(b, m, n, s)
    scale = torch.ones(b, m, n, s)
    pp = D.Normal(loc, scale)
    y = torch.randn(b, m, n)   # draw from same distribution (marginal)
    data = {'y': y, 'mask_n': torch.ones(b, m, n, dtype=torch.bool)}

    alphas = [0.1, 0.5]   # 90% and 50% intervals
    cov, _ = posteriorPredictiveCoverage(pp, data, alphas=alphas)

    mean_cov = cov.mean(dim=-1)   # average over datasets
    assert abs(mean_cov[0].item() - 0.9) < 0.05, f'90% coverage = {mean_cov[0]:.3f}'
    assert abs(mean_cov[1].item() - 0.5) < 0.08, f'50% coverage = {mean_cov[1]:.3f}'


def test_ppc_diffuse_overcoverage():
    """Very diffuse predictive → near 100% coverage even for tight intervals."""
    torch.manual_seed(7)
    b, m, n, s = 4, 3, 8, 256
    y = torch.randn(b, m, n)
    data = {'y': y, 'mask_n': torch.ones(b, m, n, dtype=torch.bool)}
    loc = torch.zeros(b, m, n, s)
    scale = torch.full((b, m, n, s), 100.0)
    pp = D.Normal(loc, scale)
    cov, _ = posteriorPredictiveCoverage(pp, data, alphas=[0.05])   # 95% CI
    assert cov.mean() > 0.9, f'expected near-100% coverage with scale=100, got {cov.mean():.3f}'


def test_ppc_with_is_weights_shape_and_finite():
    b, m, n, s = 3, 4, 6, 256
    pp = _make_pp(b, m, n, s)
    data = _make_data(b, m, n)
    torch.manual_seed(5)
    w = torch.softmax(torch.randn(b, s), dim=-1)
    cov, wid = posteriorPredictiveCoverage(pp, data, w=w, alphas=[0.05, 0.5])
    assert cov.shape == (2, b)
    assert wid.shape == (2, b)
    assert torch.isfinite(cov).all()
    assert (wid >= 0).all()


def test_ppc_respects_mask():
    """Masked observations should not affect coverage or width."""
    torch.manual_seed(3)
    b, m, n_max, s = 2, 3, 8, 256
    ns = [8, 5, 3]

    y = torch.randn(b, m, n_max)
    mask_n = torch.zeros(b, m, n_max, dtype=torch.bool)
    for j in range(m):
        mask_n[:, j, : ns[j]] = True

    data = {'y': y, 'mask_n': mask_n}
    pp = _make_pp(b, m, n_max, s)

    cov, wid = posteriorPredictiveCoverage(pp, data, alphas=[0.1])
    assert cov.shape == (1, b)
    assert torch.isfinite(cov).all()
    assert torch.isfinite(wid).all()


def test_ppc_normal_fast_path_matches_sampling():
    """Normal fast path (CDF) should agree with the sampling path on coverage."""
    from metabeta.evaluation.predictive import _predQuantile

    torch.manual_seed(99)
    b, m, n, s = 8, 4, 12, 4096   # large s for accurate sampling reference

    loc = torch.randn(b, m, n, s)
    scale = torch.rand(b, m, n, s) + 0.5
    pp_normal = D.Normal(loc, scale)

    y = torch.randn(b, m, n)
    data = {'y': y, 'mask_n': torch.ones(b, m, n, dtype=torch.bool)}
    alphas = [0.1, 0.5]

    # Normal fast path
    cov_fast, wid_fast = posteriorPredictiveCoverage(pp_normal, data, alphas=alphas)

    # Sampling reference (force the slow path via a wrapper distribution)
    class _Wrap(D.Distribution):
        arg_constraints = {}
        has_rsample = False

        def __init__(self, base):
            super().__init__(base.batch_shape, base.event_shape, validate_args=False)
            self._base = base

        def sample(self, sample_shape=torch.Size()):
            return self._base.sample(sample_shape)

        def log_prob(self, x):
            return self._base.log_prob(x)

    cov_slow, _ = posteriorPredictiveCoverage(_Wrap(pp_normal), data, alphas=alphas)

    # Coverage should agree to within Monte Carlo noise (< 3pp on 4096 samples)
    diff = (cov_fast - cov_slow).abs().mean().item()
    assert diff < 0.03, f'Normal fast path vs sampling path mean coverage diff = {diff:.4f}'
