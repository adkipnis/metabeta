import numpy as np
import pytest

from metabeta.simulation.distributions import (
    Normal,
    Student,
    LogNormal,
    Uniform,
    ScaledBeta,
    Bernoulli,
    NegativeBinomial,
)


CONTINUOUS = [Normal, Student, LogNormal, Uniform, ScaledBeta]
DISCRETE = [Bernoulli, NegativeBinomial]
ALL = CONTINUOUS + DISCRETE


@pytest.mark.parametrize("Dist", ALL)
def test_sample_shape_and_finite(Dist):
    rng = np.random.default_rng(123)
    d = Dist(rng=rng, truncate=True)
    n = 500
    x = d.sample(n)

    assert isinstance(x, np.ndarray)
    assert x.shape == (n, 1)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("Dist", DISCRETE)
def test_discrete_not_truncated(Dist):
    rng = np.random.default_rng(0)
    d = Dist(rng=rng, truncate=True)
    assert d.truncate is False
    assert d.infinite_borders is True  # per your implementation


def test_bernoulli_values_are_0_or_1():
    rng = np.random.default_rng(42)
    d = Bernoulli(rng=rng, truncate=True)
    x = d.sample(1000)
    uniq = np.unique(x)
    assert set(uniq.tolist()).issubset({0, 1})


def test_negative_binomial_values_are_nonnegative_integers():
    rng = np.random.default_rng(7)
    d = NegativeBinomial(rng=rng, truncate=True)
    x = d.sample(2000)
    assert np.all(x >= 0)
    # integers (allow float dtype coming from scipy)
    assert np.all(np.isclose(x, np.round(x)))


def test_continuous_truncation_respects_borders_when_finite():
    """
    If truncation is enabled and borders are finite (i.e. at least one side finite),
    all samples must lie strictly between left and right due to (left < z) & (z < right).
    """
    rng = np.random.default_rng(2026)
    n = 2000

    for Dist in CONTINUOUS:
        d = Dist(rng=rng, truncate=True)
        x = d.sample(n)

        if d.truncate and not d.infinite_borders:
            left, right = d.borders
            if np.isfinite(left):
                assert np.all(x > left)
            if np.isfinite(right):
                assert np.all(x < right)


def test_no_truncation_matches_unbounded_sampling_path():
    rng = np.random.default_rng(99)
    n = 1000
    d = Normal(rng=rng, truncate=False)

    assert d.truncate is False
    assert d.infinite_borders is True

    x = d.sample(n)
    assert x.shape == (n, 1)
    assert np.all(np.isfinite(x))


def test_reproducibility_same_seed_same_samples():
    """
    Given identical seeds, the whole object construction + sampling should match
    because initParams, initBorders, and rvs all use the same RNG stream.
    """
    seed = 11
    n = 500

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    for Dist in ALL:
        d1 = Dist(rng=rng1, truncate=True)
        d2 = Dist(rng=rng2, truncate=True)

        # parameters and truncation-related state should match
        assert d1.truncate == d2.truncate
        assert d1.params == d2.params
        if d1.truncate:
            assert d1.borders == d2.borders

        x1 = d1.sample(n)
        x2 = d2.sample(n)
        assert np.array_equal(x1, x2)


def test_truncated_sampling_failure_raises_runtimeerror():
    """
    Force a near-impossible truncation interval by overriding borders.
    This should trigger the RuntimeError after max_iters.

    We use a Normal distribution for determinism.
    """
    rng = np.random.default_rng(0)
    d = Normal(rng=rng, truncate=True)

    # Force extremely narrow interval around an absurd location (very low acceptance)
    d.borders = (1e12, 1e12 + 1e-9)

    with pytest.raises(RuntimeError):
        _ = d.sample(10)


def test_repr_returns_string():
    rng = np.random.default_rng(0)
    for Dist in ALL:
        s = repr(Dist(rng=rng, truncate=True))
        assert isinstance(s, str)
        assert len(s) > 0


def test_uniform_support_matches_params():
    rng = np.random.default_rng(1234)
    d = Uniform(rng=rng, truncate=False)

    # scipy.stats.uniform: support is [loc, loc+scale]
    loc = d.params["loc"]
    scale = d.params["scale"]
    lb, ub = d.dist.support()

    assert np.isclose(lb, loc)
    assert np.isclose(ub, loc + scale)


def test_scaledbeta_support_matches_scale():
    rng = np.random.default_rng(1234)
    d = ScaledBeta(rng=rng, truncate=False)

    scale = d.params["scale"]
    lb, ub = d.dist.support()

    assert np.isclose(lb, 0.0)
    assert np.isclose(ub, scale) 
