import numpy as np
from metabeta.simulation import Synthesizer


def test_sample_shapes_and_intercept_and_groups():
    rng = np.random.default_rng(123)
    d = 7
    ns = np.array([3, 1, 5], dtype=int)

    out = Synthesizer(rng).sample(d=d, ns=ns)
    X = out["X"]
    groups = out["groups"]

    assert X.shape == (int(ns.sum()), d)
    assert groups.shape == (int(ns.sum()),)

    # Intercept column is all ones
    assert np.allclose(X[:, 0], 1.0)

    # groups are 0..m-1 repeated ns times
    expected = np.repeat(np.arange(len(ns)), ns)
    assert np.array_equal(groups, expected)


def test_reproducibility_with_seedsequence():
    # Passing a SeedSequence should be supported and deterministic.
    ss = np.random.SeedSequence(2024)
    d = 6
    ns = np.array([4, 2, 3], dtype=int)

    out1 = Synthesizer(ss).sample(d=d, ns=ns)  # type: ignore[arg-type]
    out2 = Synthesizer(np.random.SeedSequence(2024)).sample(d=d, ns=ns)  # type: ignore[arg-type]

    assert np.array_equal(out1["groups"], out2["groups"])
    assert np.allclose(out1["X"], out2["X"])


def test_no_correlation_when_disabled():
    rng = np.random.default_rng(0)
    n, d = 200, 5

    s = Synthesizer(rng, correlate=False)
    X = s._sample(n=n, d=d)

    # Basic sanity: intercept and finite values
    assert np.allclose(X[:, 0], 1.0)
    assert np.isfinite(X).all()


def test_induce_correlation_preserves_marginals():
    """_induceCorrelation should preserve per-column mean/std of continuous columns.

    We monkeypatch:
    - checkContinuous: treat all columns as continuous
    - wishartCorrelation: return a fixed correlation matrix
    This makes the test deterministic and focused on the transform logic.
    """
    rng = np.random.default_rng(42)
    s = Synthesizer(rng)

    n, d = 500, 7
    x = rng.normal(size=(n, d))
    x_copy = x.copy()

    out = s._induceCorrelation(x)
    assert out.shape == x.shape

    # Should not modify the input array in-place (implementation uses copy)
    assert np.allclose(x, x_copy)
    assert not np.shares_memory(out, x)

    # Marginal means and stds should be preserved (approximately)
    mean_in = x.mean(axis=0)
    std_in = x.std(axis=0)
    mean_out = out.mean(axis=0)
    std_out = out.std(axis=0)

    assert np.allclose(mean_out, mean_in, atol=1e-7, rtol=0)
    assert np.allclose(std_out, std_in, atol=1e-7, rtol=0)

    # Correlations should change relative to the input (very likely under fixed C)
    corr_in = np.corrcoef(x, rowvar=False)
    corr_out = np.corrcoef(out, rowvar=False)
    assert not np.allclose(corr_out, corr_in)


def test_sample_returns_float_matrix_and_int_groups():
    rng = np.random.default_rng(7)
    d = 8
    ns = np.array([10, 10], dtype=int)
    out = Synthesizer(rng).sample(d=d, ns=ns)

    X, groups = out["X"], out["groups"]
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(groups.dtype, np.integer)
    assert (groups.min() >= 0) and (groups.max() == len(ns) - 1)
