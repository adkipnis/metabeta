import numpy as np
import pytest

from metabeta.simulation import (
    Prior,
    Synthesizer,
    Simulator,
    simulate,
    addNonlinearity,
    adjustParamsForNonlinearity,
    NONLINEAR_KINDS,
    hypersample,
)
from metabeta.utils.sampling import sampleCounts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_X(rng: np.random.Generator, n: int = 60, d: int = 5) -> np.ndarray:
    """Standardized design matrix with intercept in column 0."""
    X = np.ones((n, d))
    X[:, 1:] = rng.standard_normal((n, d - 1))
    return X


def make_X_intercept_only(n: int = 60) -> np.ndarray:
    """Design matrix with only an intercept column (d=1)."""
    return np.ones((n, 1))


# ---------------------------------------------------------------------------
# Basic contract: shape, dtype, finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', NONLINEAR_KINDS)
@pytest.mark.parametrize('n,d', [(30, 2), (60, 5), (100, 10)])
def test_output_shape(kind, n, d):
    rng = np.random.default_rng(1)
    X = make_X(rng, n=n, d=d)
    term = addNonlinearity(rng, X, kind, scale=1.0)
    assert term.shape == (n,)
    assert np.isfinite(term).all(), f'{kind}: non-finite output'


def test_invalid_kind_raises():
    rng = np.random.default_rng(0)
    X = make_X(rng)
    with pytest.raises(ValueError, match='Unknown nonlinear kind'):
        addNonlinearity(rng, X, 'cubic', scale=1.0)


def test_invalid_X_shape_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        addNonlinearity(rng, np.ones(10), 'polynomial', scale=1.0)


# ---------------------------------------------------------------------------
# Scale=0 gives zero output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', NONLINEAR_KINDS)
def test_scale_zero_gives_zeros(kind):
    rng = np.random.default_rng(2)
    X = make_X(rng)
    term = addNonlinearity(rng, X, kind, scale=0.0)
    assert np.allclose(term, 0.0), f'{kind}: expected zeros for scale=0'


# ---------------------------------------------------------------------------
# Scale linearity: doubling scale doubles the output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', NONLINEAR_KINDS)
def test_scale_linearity(kind):
    X = make_X(np.random.default_rng(3))
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)  # identical state
    t1 = addNonlinearity(rng1, X, kind, scale=1.0)
    t2 = addNonlinearity(rng2, X, kind, scale=2.0)
    assert np.allclose(2.0 * t1, t2), f'{kind}: output not linear in scale'


# ---------------------------------------------------------------------------
# Sign / bound properties
# ---------------------------------------------------------------------------


def test_polynomial_nonneg():
    rng = np.random.default_rng(4)
    X = make_X(rng)
    term = addNonlinearity(rng, X, 'polynomial', scale=1.0)
    assert (term >= 0).all(), 'polynomial must be non-negative'


def test_smooth_bounded():
    rng = np.random.default_rng(5)
    X = make_X(rng)
    scale = 3.0
    term = addNonlinearity(rng, X, 'smooth', scale=scale)
    assert (term > -scale).all() and (
        term < scale
    ).all(), 'smooth (tanh) output must lie strictly in (−scale, scale)'


def test_step_nonneg():
    rng = np.random.default_rng(6)
    X = make_X(rng)
    term = addNonlinearity(rng, X, 'step', scale=1.0)
    assert (term >= 0).all(), 'step (ReLU) must be non-negative'


# ---------------------------------------------------------------------------
# Intercept-only X (d=1): all kinds must return zeros
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', NONLINEAR_KINDS)
def test_intercept_only_returns_zeros(kind):
    rng = np.random.default_rng(7)
    X = make_X_intercept_only(n=50)
    term = addNonlinearity(rng, X, kind, scale=2.0)
    assert np.allclose(term, 0.0), f'{kind}: expected zeros for intercept-only X, got {term[:5]}'


# ---------------------------------------------------------------------------
# Interaction with a single predictor (d=2) must also return zeros
# ---------------------------------------------------------------------------


def test_interaction_single_predictor_returns_zeros():
    rng = np.random.default_rng(8)
    X = make_X(rng, n=50, d=2)  # one predictor + intercept
    term = addNonlinearity(rng, X, 'interaction', scale=1.0)
    assert np.allclose(term, 0.0), 'interaction needs ≥2 predictors; d=2 should return zeros'


# ---------------------------------------------------------------------------
# Reproducibility: same RNG seed → identical output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', NONLINEAR_KINDS)
def test_reproducibility(kind):
    X = make_X(np.random.default_rng(9))
    t1 = addNonlinearity(np.random.default_rng(42), X, kind, scale=1.0)
    t2 = addNonlinearity(np.random.default_rng(42), X, kind, scale=1.0)
    assert np.allclose(t1, t2), f'{kind}: different output for same seed'


# ---------------------------------------------------------------------------
# All four kinds produce distinct outputs on the same (X, scale, seed)
# ---------------------------------------------------------------------------


def test_all_kinds_produce_distinct_outputs():
    X = make_X(np.random.default_rng(10), n=100, d=6)
    terms = {
        kind: addNonlinearity(np.random.default_rng(42), X, kind, scale=1.0)
        for kind in NONLINEAR_KINDS
    }

    for kind, term in terms.items():
        assert np.isfinite(term).all(), f'{kind}: non-finite values'

    kinds = list(terms)
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            a, b = terms[kinds[i]], terms[kinds[j]]
            assert not np.allclose(a, b), f'{kinds[i]} and {kinds[j]} gave identical outputs'


# ---------------------------------------------------------------------------
# Polynomial: mean of term is positive and grows with scale
# ---------------------------------------------------------------------------


def test_polynomial_positive_mean():
    rng = np.random.default_rng(11)
    X = make_X(rng, n=200, d=5)
    term = addNonlinearity(rng, X, 'polynomial', scale=1.0)
    assert term.mean() > 0, 'polynomial term should have positive mean'


# ---------------------------------------------------------------------------
# Smooth: output should not be constant (non-degenerate)
# ---------------------------------------------------------------------------


def test_smooth_not_constant():
    rng = np.random.default_rng(12)
    X = make_X(rng, n=100, d=5)
    term = addNonlinearity(rng, X, 'smooth', scale=1.0)
    assert term.std() > 0.01, 'smooth term should not be near-constant'


# ---------------------------------------------------------------------------
# adjustParamsForNonlinearity
# ---------------------------------------------------------------------------


def make_params(d: int, m: int = 5, q: int = 2, sigma_eps: float = 0.5) -> dict:
    rng = np.random.default_rng(0)
    return {
        'ffx': rng.standard_normal(d),
        'rfx': rng.standard_normal((m, q)),
        'sigma_eps': np.array(sigma_eps),
    }


def test_adjust_purely_linear_f_resid_is_zero():
    """f_X in span(X) → residual ≈ 0, ffx absorbs Δβ exactly."""
    rng = np.random.default_rng(10)
    X = make_X(rng, n=100, d=5)
    delta = rng.standard_normal(5)
    f_X = X @ delta  # exactly linear

    params = make_params(5)
    params_adj, f_resid = adjustParamsForNonlinearity(params, X, f_X)

    assert np.allclose(f_resid, 0.0, atol=1e-10), 'residual should vanish for linear f_X'
    assert np.allclose(params_adj['ffx'], params['ffx'] + delta, atol=1e-10)
    # sigma_eps unchanged when residual is zero
    assert np.isclose(float(params_adj['sigma_eps']), float(params['sigma_eps']), atol=1e-10)


def test_adjust_sigma_eps_inflated_by_residual():
    """sigma_eps must increase when f_X has a component orthogonal to X."""
    rng = np.random.default_rng(11)
    X = make_X(rng, n=200, d=5)
    f_X = addNonlinearity(rng, X, 'polynomial', scale=1.0)

    params = make_params(5, sigma_eps=0.5)
    params_adj, f_resid = adjustParamsForNonlinearity(params, X, f_X)

    sigma_orig = float(params['sigma_eps'])
    sigma_adj = float(params_adj['sigma_eps'])

    assert sigma_adj >= sigma_orig, 'sigma_eps must not decrease'
    if f_resid.var() > 1e-12:
        assert sigma_adj > sigma_orig, 'sigma_eps must strictly increase when f_resid != 0'
    # consistency check: adjusted value matches the formula
    expected = np.sqrt(sigma_orig**2 + f_resid.var())
    assert np.isclose(sigma_adj, expected, rtol=1e-6)


def test_adjust_eta_algebraically_unchanged():
    """X @ ffx + f_X must equal X @ ffx_adj + f_resid (the linear predictor is invariant).

    adjustParamsForNonlinearity intentionally inflates sigma_eps, so the noise term in y
    differs. The algebraic identity applies to eta (before noise), not y.
    """
    rng = np.random.default_rng(12)
    X = make_X(rng, n=80, d=4)
    f_X = addNonlinearity(rng, X, 'polynomial', scale=1.0)

    params = make_params(4)
    params_adj, f_resid = adjustParamsForNonlinearity(params, X, f_X)

    eta_orig = X @ params['ffx'] + f_X
    eta_adj = X @ params_adj['ffx'] + f_resid

    assert np.allclose(
        eta_orig, eta_adj, atol=1e-10
    ), 'eta must be identical regardless of decomposition'


def test_adjust_does_not_mutate_original():
    """Original params dict and its arrays must be unchanged after the call."""
    rng = np.random.default_rng(13)
    X = make_X(rng, n=80, d=4)
    f_X = addNonlinearity(rng, X, 'polynomial', scale=1.0)

    params = make_params(4)
    ffx_before = params['ffx'].copy()
    sigma_before = float(params['sigma_eps'])

    adjustParamsForNonlinearity(params, X, f_X)

    assert np.allclose(params['ffx'], ffx_before), 'original ffx was mutated'
    assert np.isclose(float(params['sigma_eps']), sigma_before), 'original sigma_eps was mutated'


def test_adjust_no_sigma_eps_key():
    """Works without sigma_eps (e.g. Bernoulli/Poisson likelihoods)."""
    rng = np.random.default_rng(14)
    X = make_X(rng, n=80, d=4)
    f_X = addNonlinearity(rng, X, 'polynomial', scale=1.0)

    params = {'ffx': rng.standard_normal(4), 'rfx': rng.standard_normal((5, 2))}
    params_adj, _ = adjustParamsForNonlinearity(params, X, f_X)

    assert 'sigma_eps' not in params_adj
    assert 'ffx' in params_adj
    assert not np.allclose(params_adj['ffx'], params['ffx'])  # ffx was adjusted


# ---------------------------------------------------------------------------
# Simulator integration: nonlinear_kind produces a valid, normalized dataset
# ---------------------------------------------------------------------------


def _assert_valid_dataset(ds: dict, d: int, q: int) -> None:
    for key in ('X', 'y', 'groups', 'm', 'n', 'ns', 'd', 'q', 'ffx', 'rfx'):
        assert key in ds, f'missing key {key!r}'
    n = int(ds['n'])
    assert ds['X'].shape == (n, d)
    assert ds['y'].shape == (n,)
    assert np.isfinite(ds['y']).all()
    # y should be normalized to unit SD by the simulator
    assert np.isclose(
        ds['y'].std(), 1.0, atol=0.02
    ), f'expected y.std() ≈ 1.0, got {ds["y"].std():.4f}'


@pytest.fixture
def base_dims():
    return dict(n=80, m=5, d=4, q=2)


@pytest.fixture
def base_prior(base_dims):
    rng = np.random.default_rng(20)
    hyp = hypersample(rng, base_dims['d'], base_dims['q'])
    return Prior(rng, hyp)


@pytest.fixture
def base_ns(base_dims):
    rng = np.random.default_rng(21)
    return sampleCounts(rng, base_dims['n'], base_dims['m'])


@pytest.mark.parametrize('kind', NONLINEAR_KINDS)
def test_simulator_nonlinear_valid_dataset(kind, base_dims, base_prior, base_ns):
    rng = np.random.default_rng(30)
    design = Synthesizer(rng)
    sim = Simulator(
        rng=rng,
        prior=base_prior,
        design=design,
        ns=base_ns,
        nonlinear_kind=kind,
        nonlinear_scale=0.5,
    )
    ds = sim.sample()
    _assert_valid_dataset(ds, d=base_dims['d'], q=base_dims['q'])


# ---------------------------------------------------------------------------
# Simulator: nonlinear_kind=None is identical to not setting it (default)
# ---------------------------------------------------------------------------


def test_simulator_none_kind_matches_default(base_dims, base_ns):
    d, q = base_dims['d'], base_dims['q']

    def _run(seed):
        rng = np.random.default_rng(seed)
        hyp = hypersample(rng, d, q)
        prior = Prior(rng, hyp)
        design = Synthesizer(rng)
        sim = Simulator(rng=rng, prior=prior, design=design, ns=base_ns)
        return sim.sample()

    def _run_none(seed):
        rng = np.random.default_rng(seed)
        hyp = hypersample(rng, d, q)
        prior = Prior(rng, hyp)
        design = Synthesizer(rng)
        sim = Simulator(rng=rng, prior=prior, design=design, ns=base_ns, nonlinear_kind=None)
        return sim.sample()

    ds1 = _run(99)
    ds2 = _run_none(99)
    assert np.allclose(
        ds1['y'], ds2['y']
    ), 'nonlinear_kind=None should produce identical output to the default'


# ---------------------------------------------------------------------------
# Simulator: nonlinear_scale=0 leaves y unchanged relative to kind=None
# ---------------------------------------------------------------------------


def test_simulator_scale_zero_leaves_y_unchanged(base_dims, base_ns):
    d, q = base_dims['d'], base_dims['q']
    seed = 77

    def _run(kind, scale):
        rng = np.random.default_rng(seed)
        hyp = hypersample(rng, d, q)
        prior = Prior(rng, hyp)
        design = Synthesizer(rng)
        sim = Simulator(
            rng=rng,
            prior=prior,
            design=design,
            ns=base_ns,
            nonlinear_kind=kind,
            nonlinear_scale=scale,
        )
        return sim.sample()['y']

    # polynomial does not consume RNG state, so scale=0 → same y as kind=None
    y_none = _run(None, 0.0)
    y_poly_zero = _run('polynomial', 0.0)
    assert np.allclose(
        y_none, y_poly_zero
    ), 'polynomial with scale=0 should produce the same y as kind=None'


# ---------------------------------------------------------------------------
# Simulator: nonlinear output actually differs from linear (scale > 0)
# ---------------------------------------------------------------------------


def test_simulator_nonlinear_changes_y(base_dims, base_ns):
    """A nonlinear perturbation with nonzero scale should change y."""
    d, q = base_dims['d'], base_dims['q']

    rng_lin = np.random.default_rng(55)
    hyp_lin = hypersample(rng_lin, d, q)
    prior_lin = Prior(rng_lin, hyp_lin)
    design_lin = Synthesizer(rng_lin)
    sim_lin = Simulator(rng=rng_lin, prior=prior_lin, design=design_lin, ns=base_ns)
    y_lin = sim_lin.sample()['y']

    # polynomial is deterministic given X, so any nonzero scale will differ
    rng_nl = np.random.default_rng(55)
    hyp_nl = hypersample(rng_nl, d, q)
    prior_nl = Prior(rng_nl, hyp_nl)
    design_nl = Synthesizer(rng_nl)
    sim_nl = Simulator(
        rng=rng_nl,
        prior=prior_nl,
        design=design_nl,
        ns=base_ns,
        nonlinear_kind='polynomial',
        nonlinear_scale=1.0,
    )
    y_nl = sim_nl.sample()['y']

    assert not np.allclose(
        y_lin, y_nl
    ), 'nonlinear perturbation with scale=1 should change y relative to linear'
