from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metabeta.utils.fits import FIT_TAGS, availableFits, fitPath, loadFits, saveFits


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    path = tmp_path / 'test.npz'
    np.savez(path, m=np.array([2, 3, 2]), y=np.zeros((3, 4)))
    return path


def _nuts(n: int = 3) -> dict[str, np.ndarray]:
    return {
        'nuts_ffx': np.zeros((n, 2, 5)),
        'nuts_duration': np.arange(n, dtype=np.float64),
        'nuts_failed': np.array([False] * n),
    }


def test_fit_path_sits_next_to_the_data(data_path: Path):
    assert fitPath(data_path, 'nuts2') == data_path.parent / 'test.nuts2.npz'
    assert fitPath(Path('train_ep00001.npz'), 'laplace') == Path('train_ep00001.laplace.npz')
    with pytest.raises(ValueError, match='unknown fit tag'):
        fitPath(data_path, 'fit')


def test_save_and_load_round_trip(data_path: Path):
    path = saveFits(data_path, 'nuts', _nuts())
    assert path == fitPath(data_path, 'nuts')
    assert availableFits(data_path) == ('nuts',)

    fits = loadFits(data_path, 'nuts')
    assert set(fits) == set(_nuts())
    np.testing.assert_array_equal(fits['nuts_duration'], np.arange(3.0))

    subset = loadFits(data_path, 'nuts', keys=('nuts_duration', 'nuts_missing'))
    assert set(subset) == {'nuts_duration'}


def test_save_validates_prefix_and_length(data_path: Path):
    with pytest.raises(ValueError, match="must start with 'nuts'_"):
        saveFits(data_path, 'nuts', {**_nuts(), 'advi_ffx': np.zeros((3, 1))})
    with pytest.raises(ValueError, match='leading axis of 3 datasets'):
        saveFits(data_path, 'nuts', _nuts(n=2))
    with pytest.raises(ValueError, match='leading axis of 3 datasets'):
        saveFits(data_path, 'nuts', {**_nuts(), 'nuts_draws': np.array(1000)})
    assert availableFits(data_path) == ()


def test_save_refuses_to_overwrite_without_force(data_path: Path):
    saveFits(data_path, 'nuts', _nuts())
    with pytest.raises(FileExistsError):
        saveFits(data_path, 'nuts', _nuts())
    saveFits(data_path, 'nuts', {'nuts_ffx': np.ones((3, 1, 1))}, force=True)
    assert set(loadFits(data_path, 'nuts')) == {'nuts_ffx'}


def test_load_rejects_fits_of_other_data(data_path: Path):
    saveFits(data_path, 'nuts', _nuts())
    np.savez(data_path, m=np.array([2, 3, 2]), y=np.ones((3, 4)))
    with pytest.raises(ValueError, match='checksum mismatch'):
        loadFits(data_path, 'nuts')


def test_load_rejects_files_without_checksum(data_path: Path):
    np.savez(fitPath(data_path, 'advi0'), advi0_ffx=np.zeros((3, 1, 1)))
    with pytest.raises(ValueError, match='no source_sha'):
        loadFits(data_path, 'advi0')


def test_every_tag_has_a_distinct_prefix():
    # key prefixes must not shadow each other ('nuts_' vs 'nuts0_' are distinct by design)
    prefixes = [f'{tag}_' for tag in FIT_TAGS]
    assert not any(a != b and b.startswith(a) for a in prefixes for b in prefixes)
