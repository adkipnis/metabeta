import numpy as np
import pytest
from metabeta.simulation import Prior, Synthesizer, Emulator, hypersample, simulate, Simulator
from metabeta.utils.sampling import sampleCounts


@pytest.fixture(scope="function")
def rng():
    # fresh RNG per test for reproducibility
    return np.random.default_rng(1)


@pytest.fixture(scope="function")
def dims():
    # small but nontrivial
    n = 80
    m = 5
    d = 4
    q = 2
    return n, m, d, q


@pytest.fixture(scope="function")
def ns(rng, dims):
    n, m, _, _ = dims
    ns = sampleCounts(rng, n, m)
    return np.asarray(ns, dtype=int)


@pytest.fixture(scope="function")
def prior(rng, dims):
    _, _, d, q = dims
    hyperparams = hypersample(rng, d, q)
    return Prior(rng, hyperparams)


def _basic_dataset_assertions(dataset: dict, d: int, q: int):
    # required keys
    for k in ["X", "groups", "y", "m", "n", "ns", "d", "q", "sigma_eps", "ffx", "rfx"]:
        assert k in dataset, f"missing key {k}"

    X = dataset["X"]
    y = dataset["y"]
    groups = dataset["groups"]
    ns = dataset["ns"]

    assert isinstance(dataset["m"], (int, np.integer))
    assert isinstance(dataset["n"], (int, np.integer))
    assert isinstance(dataset["d"], (int, np.integer))
    assert isinstance(dataset["q"], (int, np.integer))

    m = int(dataset["m"])
    n = int(dataset["n"])

    assert X.shape == (n, d)
    assert y.shape == (n,)
    assert groups.shape == (n,)
    assert len(ns) == m

    # groups must be valid indices
    assert groups.min() >= 0
    assert groups.max() < m

    # rfx should match m and q
    assert dataset["rfx"].shape == (m, q)
    assert dataset["ffx"].shape == (d,)

    # y is normalized to unit-ish std (guarded by eps)
    ystd = float(np.std(y))
    assert np.isfinite(ystd)
    assert ystd > 0.0
    # allow small tolerance since eps floor and finite sample
    assert np.isclose(ystd, 1.0, atol=1e-2, rtol=0.0)


def test_simulate_shapes(rng, prior, ns, dims):
    _, m, d, _ = dims

    # parameters
    params = prior.sample(m)

    # observations
    design = Synthesizer(rng)
    obs = design.sample(d, ns)

    y = simulate(rng, params, obs)
    assert y.shape == (int(np.sum(ns)),)
    assert np.isfinite(y).all()


def test_simulator_synthesizer_end_to_end(rng, prior, ns, dims):
    _, _, d, q = dims

    design = Synthesizer(rng)
    sim = Simulator(rng=rng, prior=prior, design=design, ns=ns, plot=False)
    dataset = sim.sample()

    _basic_dataset_assertions(dataset, d=d, q=q)

    # population R^2 should be in [0, 1]
    assert 0.0 <= float(dataset["r_squared"]) <= 1.0
    assert np.isfinite(float(dataset["cov_sum"]))


def test_simulator_reproducible_given_seed(dims, ns):
    # two identical pipelines -> identical outputs
    _, _, d, q = dims

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    hyper1 = hypersample(rng1, d, q)
    hyper2 = hypersample(rng2, d, q)

    prior1 = Prior(rng1, hyper1)
    prior2 = Prior(rng2, hyper2)

    design1 = Synthesizer(rng1)
    design2 = Synthesizer(rng2)

    sim1 = Simulator(rng=rng1, prior=prior1, design=design1, ns=ns, plot=False)
    sim2 = Simulator(rng=rng2, prior=prior2, design=design2, ns=ns, plot=False)

    ds1 = sim1.sample()
    ds2 = sim2.sample()

    # exact equality should hold for deterministic RNG use
    assert np.allclose(ds1["X"], ds2["X"])
    assert np.allclose(ds1["y"], ds2["y"])
    assert np.array_equal(ds1["groups"], ds2["groups"])
    assert np.allclose(ds1["ffx"], ds2["ffx"])
    assert np.allclose(ds1["rfx"], ds2["rfx"])
    assert np.isclose(float(ds1["sigma_eps"]), float(ds2["sigma_eps"]))


def test_simulator_emulator_updates_ns_if_present(rng, prior, ns, dims):
    """
    This test checks that your Simulator respects an updated 'ns' returned by Emulator.

    If your Emulator requires external datasets that are not available in the test
    environment, we skip the test rather than failing the suite.
    """
    _, _, d, q = dims

    try:
        design = Emulator(rng, "math")
    except Exception as e:
        pytest.skip(f"Emulator not available in test env: {e}")

    sim = Simulator(rng=rng, prior=prior, design=design, ns=ns, plot=False)
    ds = sim.sample()

    _basic_dataset_assertions(ds, d=d, q=q)

    # If emulator returns ns, Simulator should store it consistently in the output
    assert "ns" in ds
    assert len(ds["ns"]) == ds["m"] # type: ignore
    assert int(ds["n"]) == int(np.sum(ds["ns"]))
