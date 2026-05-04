"""Tests for metabeta/simulation/emulator.py — lazy metadata loading."""

import numpy as np
import pytest

import metabeta.simulation.emulator as emu_mod
from metabeta.simulation.emulator import (
    PATHS,
    TEST_PATHS,
    _Y_TYPE_FAMILIES,
    _loadMeta,
    getDatabase,
    getTestDatabase,
    loadDataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_caches():
    """Clear all module-level caches so tests are independent."""
    emu_mod._META_DB = None
    emu_mod._TEST_META_DB = None
    emu_mod._DATA_CACHE.clear()


@pytest.fixture(autouse=True)
def isolated_caches():
    _reset_caches()
    yield
    _reset_caches()


# ---------------------------------------------------------------------------
# _loadMeta
# ---------------------------------------------------------------------------


def test_loadMeta_excludes_large_arrays():
    path = TEST_PATHS[0]
    meta = _loadMeta(path)
    assert 'X' not in meta
    assert 'y' not in meta
    assert 'groups' not in meta


def test_loadMeta_includes_scalar_fields():
    path = TEST_PATHS[0]
    meta = _loadMeta(path)
    assert 'd' in meta
    assert 'n' in meta
    assert meta['source'] == path


def test_loadMeta_grouped_has_m_and_ns():
    # test/ partition only contains grouped datasets
    path = TEST_PATHS[0]
    meta = _loadMeta(path)
    assert 'm' in meta
    assert 'ns' in meta


def test_loadMeta_y_families_from_y_type():
    path = TEST_PATHS[0]
    meta = _loadMeta(path)
    y_type = meta.get('y_type', '')
    expected = _Y_TYPE_FAMILIES.get(str(y_type), set())
    assert meta['y_families'] == expected


def test_loadMeta_validation_no_groups():
    from metabeta.simulation.emulator import VAL_PATHS

    # find a non-grouped validation file
    non_grouped = [p for p in VAL_PATHS if '__grp_' not in p.stem]
    if not non_grouped:
        pytest.skip('no non-grouped validation datasets present')
    meta = _loadMeta(non_grouped[0])
    assert 'm' not in meta
    assert 'ns' not in meta
    assert 'groups' not in meta


# ---------------------------------------------------------------------------
# getDatabase / getTestDatabase — metadata only until dataset selected
# ---------------------------------------------------------------------------


def test_getDatabase_returns_metadata_only():
    db = getDatabase()
    assert isinstance(db, list)
    assert len(db) == len(PATHS)
    for meta in db:
        assert 'X' not in meta
        assert 'y' not in meta


def test_getTestDatabase_returns_metadata_only():
    db = getTestDatabase()
    assert len(db) == len(TEST_PATHS)
    for meta in db:
        assert 'X' not in meta


def test_getDatabase_cache_is_populated_but_data_cache_empty():
    getDatabase()
    assert emu_mod._META_DB is not None
    assert len(emu_mod._DATA_CACHE) == 0, 'full datasets should not be loaded yet'


# ---------------------------------------------------------------------------
# loadDataset — on-demand, cached
# ---------------------------------------------------------------------------


def test_loadDataset_returns_full_arrays():
    path = TEST_PATHS[0]
    ds = loadDataset(path)
    assert 'X' in ds
    assert 'y' in ds
    assert 'groups' in ds


def test_loadDataset_populates_data_cache():
    assert len(emu_mod._DATA_CACHE) == 0
    path = TEST_PATHS[0]
    loadDataset(path)
    assert path in emu_mod._DATA_CACHE


def test_loadDataset_only_loads_selected():
    loadDataset(TEST_PATHS[0])
    assert len(emu_mod._DATA_CACHE) == 1, 'only one dataset should be in cache'


def test_loadDataset_cache_returns_same_object():
    path = TEST_PATHS[0]
    ds1 = loadDataset(path)
    ds2 = loadDataset(path)
    assert ds1 is ds2


# ---------------------------------------------------------------------------
# Emulator.sample — correct output, no full-DB load
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def ns_small(rng):
    from metabeta.utils.sampling import sampleCounts

    return sampleCounts(rng, 50, 4)


def test_emulator_sample_specific_source(rng, ns_small):
    from metabeta.simulation.emulator import Emulator

    source = TEST_PATHS[0].stem
    em = Emulator(rng, source=source, use_sgld=False)
    out = em.sample(d=3, ns=ns_small)
    assert 'X' in out and 'ns' in out and 'groups' in out
    assert out['X'].shape[1] == 3  # d predictors (incl. intercept)


def test_emulator_specific_source_loads_only_one_dataset(rng, ns_small):
    from metabeta.simulation.emulator import Emulator

    source = TEST_PATHS[0].stem
    Emulator(rng, source=source, use_sgld=False).sample(d=3, ns=ns_small)
    assert len(emu_mod._DATA_CACHE) == 1


def test_emulator_sample_all_source(rng, ns_small):
    from metabeta.simulation.emulator import Emulator

    em = Emulator(rng, source='all', use_sgld=False)
    out = em.sample(d=3, ns=ns_small)
    assert 'X' in out


def test_emulator_all_source_loads_at_most_one_dataset_per_call(rng, ns_small):
    from metabeta.simulation.emulator import Emulator

    Emulator(rng, source='all', use_sgld=False).sample(d=3, ns=ns_small)
    assert len(emu_mod._DATA_CACHE) <= 1


# ---------------------------------------------------------------------------
# Subsampler.sample
# ---------------------------------------------------------------------------


def test_subsampler_sample_normal(rng, ns_small):
    from metabeta.simulation.emulator import Subsampler

    source = TEST_PATHS[0].stem
    ss = Subsampler(rng, source=source, likelihood_family=0)
    out = ss.sample(d=3, ns=ns_small)
    assert 'X' in out and 'y' in out
    assert out['source'].item() == source


def test_subsampler_loads_only_one_dataset(rng, ns_small):
    from metabeta.simulation.emulator import Subsampler

    source = TEST_PATHS[0].stem
    Subsampler(rng, source=source, likelihood_family=0).sample(d=3, ns=ns_small)
    assert len(emu_mod._DATA_CACHE) == 1
