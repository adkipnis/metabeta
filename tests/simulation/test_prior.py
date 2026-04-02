import numpy as np
import pytest
from metabeta.simulation import hypersample, Prior
from metabeta.utils.families import (
    FFX_FAMILY_PROBS,
    SIGMA_EPS_FAMILY_PROBS,
    SIGMA_RFX_FAMILY_PROBS,
)


REQUIRED_HYPER_KEYS = {'nu_ffx', 'tau_ffx', 'tau_rfx', 'tau_eps', 'eta_rfx'}
REQUIRED_PARAM_KEYS = {'ffx', 'sigma_rfx', 'sigma_eps', 'rfx', 'corr_rfx'}


def test_hypersample_keys_and_shapes():
    rng = np.random.default_rng(123)
    d, q = 5, 2

    h = hypersample(d=d, q=q, rng=rng)

    assert REQUIRED_HYPER_KEYS <= set(h.keys())
    assert 'correlated_rfx' not in h
    assert isinstance(h['tau_eps'], (float, np.floating))

    assert h['nu_ffx'].shape == (d,)
    assert h['tau_ffx'].shape == (d,)
    assert h['tau_rfx'].shape == (q,)

    # All taus should be strictly positive (per your sampling bounds)
    assert np.all(h['tau_ffx'] > 0)
    assert np.all(h['tau_rfx'] > 0)
    assert h['tau_eps'] > 0


def test_hypersample_q1_always_uncorrelated():
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = hypersample(d=3, q=1, rng=rng)
        assert float(h['eta_rfx']) == 0.0


def test_hypersample_q2_can_be_correlated():
    rng = np.random.default_rng(42)
    etas = [float(hypersample(d=3, q=2, rng=rng)['eta_rfx']) for _ in range(50)]
    assert any(e > 0 for e in etas), 'q=2 should sometimes produce correlated rfx'
    assert any(e == 0 for e in etas), 'q=2 should sometimes produce uncorrelated rfx'


def test_hypersample_eta_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        h = hypersample(d=4, q=3, rng=rng)
        eta = float(h['eta_rfx'])
        assert eta == 0.0 or 1.0 <= eta <= 2.0


def test_prior_post_init_sets_dimensions():
    rng = np.random.default_rng(0)
    d, q = 3, 4
    h = hypersample(d=d, q=q, rng=rng)

    prior = Prior(rng, h)
    assert prior.d == d
    assert prior.q == q
    assert prior.correlated_rfx == (float(h['eta_rfx']) > 0)


@pytest.mark.parametrize('d,q,m', [(1, 1, 1), (3, 1, 10), (7, 5, 2)])
def test_prior_sample_shapes_and_validity(d, q, m):
    rng = np.random.default_rng(999)
    h = hypersample(d=d, q=q, rng=rng)
    prior = Prior(rng, h)

    params = prior.sample(m=m)

    assert set(params.keys()) == REQUIRED_PARAM_KEYS

    # Shapes
    assert params['ffx'].shape == (d,)
    assert params['sigma_rfx'].shape == (q,)
    assert np.ndim(params['sigma_eps']) == 0  # scalar
    assert params['rfx'].shape == (m, q)

    # Positivity of scales
    assert np.all(params['sigma_rfx'] >= 0)
    assert float(params['sigma_eps']) >= 0

    # Finite values
    assert np.all(np.isfinite(params['ffx']))
    assert np.all(np.isfinite(params['sigma_rfx']))
    assert np.isfinite(float(params['sigma_eps']))
    assert np.all(np.isfinite(params['rfx']))


def test_corr_rfx_valid_when_correlated():
    # force correlated by retrying until eta > 0
    rng = np.random.default_rng(42)
    q = 3
    for _ in range(100):
        h = hypersample(d=4, q=q, rng=rng)
        if float(h['eta_rfx']) > 0:
            break
    prior = Prior(rng, h)
    params = prior.sample(m=10)

    corr = params['corr_rfx']
    assert corr.shape == (q, q)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert np.all(np.linalg.eigvalsh(corr) >= -1e-10)


def test_corr_rfx_identity_when_uncorrelated():
    # force uncorrelated: q=1 always gives eta=0
    rng = np.random.default_rng(42)
    q = 1
    h = hypersample(d=4, q=q, rng=rng)
    assert float(h['eta_rfx']) == 0.0
    prior = Prior(rng, h)
    params = prior.sample(m=10)
    assert np.allclose(params['corr_rfx'], np.eye(q))


def test_reproducibility_same_seed_same_outputs():
    seed = 2026
    d, q, m = 4, 3, 8

    rng1 = np.random.default_rng(seed)
    h1 = hypersample(d=d, q=q, rng=rng1)
    prior1 = Prior(rng1, h1)
    p1 = prior1.sample(m=m)

    rng2 = np.random.default_rng(seed)
    h2 = hypersample(d=d, q=q, rng=rng2)
    prior2 = Prior(rng2, h2)
    p2 = prior2.sample(m=m)

    # Hyperparameters should match exactly given same RNG seed
    assert np.allclose(h1['nu_ffx'], h2['nu_ffx'])
    assert np.allclose(h1['tau_ffx'], h2['tau_ffx'])
    assert np.allclose(h1['tau_rfx'], h2['tau_rfx'])
    assert h1['tau_eps'] == h2['tau_eps']
    assert float(h1['eta_rfx']) == float(h2['eta_rfx'])

    # Sampled params should match exactly given same RNG seed and same hyperparams draw
    assert np.allclose(p1['ffx'], p2['ffx'])
    assert np.allclose(p1['sigma_rfx'], p2['sigma_rfx'])
    assert float(p1['sigma_eps']) == float(p2['sigma_eps'])
    assert np.allclose(p1['rfx'], p2['rfx'])


def test_different_seed_usually_changes_outputs():
    # Not a strict guarantee, but extremely likely.
    d, q, m = 4, 2, 5

    rng1 = np.random.default_rng(1)
    h1 = hypersample(d=d, q=q, rng=rng1)
    p1 = Prior(rng1, h1).sample(m=m)

    rng2 = np.random.default_rng(2)
    h2 = hypersample(d=d, q=q, rng=rng2)
    p2 = Prior(rng2, h2).sample(m=m)

    # At least one array should differ
    changed = (
        not np.allclose(p1['ffx'], p2['ffx'])
        or not np.allclose(p1['sigma_rfx'], p2['sigma_rfx'])
        or not np.allclose(p1['rfx'], p2['rfx'])
        or float(p1['sigma_eps']) != float(p2['sigma_eps'])
    )
    assert changed


def test_missing_required_hyperparams_raises_keyerror():
    rng = np.random.default_rng(0)
    h = hypersample(d=2, q=1, rng=rng)
    h.pop('tau_rfx')  # remove required key

    with pytest.raises(KeyError):
        _ = Prior(rng, h)


def test_hypersample_family_probabilities_match_families():
    assert np.isclose(sum(FFX_FAMILY_PROBS), 1.0)
    assert np.isclose(sum(SIGMA_RFX_FAMILY_PROBS), 1.0)
    assert np.isclose(sum(SIGMA_EPS_FAMILY_PROBS), 1.0)


def test_hypersample_weighted_family_sampling_hits_all_families():
    rng = np.random.default_rng(0)
    draws = 1000

    ffx_families = [int(hypersample(d=3, q=2, rng=rng)['family_ffx']) for _ in range(draws)]
    sigma_rfx_families = [
        int(hypersample(d=3, q=2, rng=rng)['family_sigma_rfx']) for _ in range(draws)
    ]
    sigma_eps_families = [
        int(hypersample(d=3, q=2, rng=rng)['family_sigma_eps']) for _ in range(draws)
    ]

    assert set(ffx_families) == {0, 1}
    assert set(sigma_rfx_families) == {0, 1, 2}
    assert set(sigma_eps_families) == {0, 1, 2}
