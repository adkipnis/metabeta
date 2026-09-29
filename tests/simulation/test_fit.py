from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from metabeta.simulation.fit import (
    Fitter,
    aggregateFits,
    markNonFinite,
    methodTags,
    tagMethodLevel,
)
from metabeta.utils.dataloader import Collection
from metabeta.utils.fits import FIT_TAGS, loadFits


def _writeBatch(root: Path, data_id: str = 'fit-n') -> Path:
    data_dir = root / data_id
    data_dir.mkdir(parents=True)
    B, n_max, d_max, q_max, m_max = 2, 5, 3, 2, 4
    X = np.zeros((B, n_max, d_max))
    X[..., 0] = 1.0
    d, q, m, n = np.array([2, 3]), np.array([1, 2]), np.array([2, 3]), np.array([4, 5])
    groups, ns = np.zeros((B, n_max), dtype=np.int64), np.zeros((B, m_max), dtype=np.int64)
    for i in range(B):
        groups[i, : n[i]] = np.repeat(np.arange(m[i]), [2, 2, 1][: m[i]])[: n[i]]
        ns[i, : m[i]] = np.bincount(groups[i, : n[i]], minlength=m[i])
    path = data_dir / 'test.npz'
    np.savez(
        path,
        y=np.zeros((B, n_max)),
        X=X,
        Z=X[..., :q_max].copy(),
        groups=groups,
        ns=ns,
        d=d,
        q=q,
        m=m,
        n=n,
        ffx=np.zeros((B, d_max)),
        sigma_rfx=np.ones((B, q_max)),
        sigma_eps=np.ones(B),
        rfx=np.zeros((B, m_max, q_max)),
        corr_rfx=np.tile(np.eye(q_max), (B, 1, 1)),
        nu_ffx=np.zeros((B, d_max)),
        tau_ffx=np.ones((B, d_max)),
        tau_rfx=np.ones((B, q_max)),
        tau_eps=np.ones(B),
        eta_rfx=np.zeros(B),
        sd_y=np.ones(B),
        likelihood_family=np.zeros(B, dtype=np.int64),
    )
    return path


def _cfg(**kwargs) -> argparse.Namespace:
    values = dict(
        data_id='fit-n', idx=0, method='nuts', level=0, partition='test', seed=0, diagonal=False
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_method_tags_round_trip():
    assert methodTags('nuts', 2) == ('nuts2',)
    assert methodTags('advi', 0) == ('advi0', 'advi1')
    assert methodTags('pathfinder', 1) == ('pathfinder1',)
    assert methodTags('laplace', 0) == ('laplace',)
    for tag in FIT_TAGS:
        if tag != 'nuts':
            assert tag in methodTags(*tagMethodLevel(tag))
    with pytest.raises(ValueError, match='levels 0..2'):
        methodTags('nuts', 3)
    with pytest.raises(ValueError, match='unknown method'):
        methodTags('vi', 0)


def test_aggregate_pads_and_fills_failed_fits():
    ok = {
        'advi1_ffx': np.ones((2, 5)),
        'advi1_names': np.array(['a', 'bb']),
        'advi1_failed': np.array(False),
        'advi1_error': np.array(''),
    }
    failed = {'advi1_failed': np.array(True), 'advi1_error': np.array('FloatingPointError: nan')}
    big = {**ok, 'advi1_ffx': np.full((3, 5), 2.0), 'advi1_names': np.array(['a', 'bb', 'ccc'])}

    out = aggregateFits([ok, failed, big])

    assert out['advi1_ffx'].shape == (3, 3, 5)
    assert np.isnan(out['advi1_ffx'][1]).all()
    assert out['advi1_ffx'][0, 2].tolist() == [0.0] * 5  # zero padding, not NaN
    np.testing.assert_array_equal(out['advi1_failed'], [False, True, False])
    assert out['advi1_error'][1] == 'FloatingPointError: nan'
    assert out['advi1_names'][2].tolist() == ['a', 'bb', 'ccc']
    assert out['advi1_names'][0].tolist() == ['a', 'bb', '']


def test_mark_non_finite_counts_nan_draws_as_failed():
    fits = {
        'laplace_ffx': np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]]),
        'laplace_rfx': np.array([[[0.5]], [[0.1]], [[np.inf]]]),
        'laplace_failed': np.array([False, False, False]),
        'laplace_error': np.array(['', '', '']),
        'laplace_duration': np.array([1.0, 2.0, 3.0]),
    }
    out = markNonFinite(fits, 'laplace')
    np.testing.assert_array_equal(out['laplace_failed'], [False, True, True])
    assert out['laplace_error'].tolist() == ['', 'non-finite draws', 'non-finite draws']
    assert np.isnan(out['laplace_ffx'][1:]).all() and np.isnan(out['laplace_rfx'][1:]).all()
    assert out['laplace_ffx'][0].tolist() == [1.0, 2.0]
    assert out['laplace_duration'].tolist() == [1.0, 2.0, 3.0]


def test_reintegrate_writes_checksummed_fit_files(tmp_path):
    path = _writeBatch(tmp_path)
    fitter = Fitter(_cfg(method='advi'), srcdir=tmp_path)
    assert fitter.tags == ('advi0', 'advi1')
    assert fitter.outPath('advi0') == tmp_path / 'fit-n' / 'fits' / 'test_advi0_000.npz'

    s = 6
    for idx, (d, q, m) in enumerate([(2, 1, 2), (3, 2, 3)]):
        for tag in fitter.tags:
            np.savez(
                fitter.outPath(tag, idx),
                **{
                    f'{tag}_ffx': np.full((d, s), idx + 1.0),
                    f'{tag}_sigma_rfx': np.ones((q, s)),
                    f'{tag}_sigma_eps': np.ones((1, s)),
                    f'{tag}_rfx': np.zeros((q, m, s)),
                    f'{tag}_corr_rfx': np.tile(np.eye(q)[None, None], (1, s, 1, 1)),
                    f'{tag}_duration': np.array(1.5),
                    f'{tag}_failed': np.array(False),
                    f'{tag}_error': np.array(''),
                },
            )
    fitter.reintegrate()

    fits = loadFits(path, 'advi1')
    assert fits['advi1_ffx'].shape == (2, 3, s)
    assert fits['advi1_ffx'][0, 2].tolist() == [0.0] * s
    np.testing.assert_array_equal(fits['advi1_duration'], [1.5, 1.5])
    col = Collection(path, permute=False, fits=('advi0', 'advi1'))
    assert col.fits == ('advi0', 'advi1')
    assert col[1]['advi0_ffx'].shape == (3, s)


def test_reintegrate_requires_every_dataset(tmp_path):
    _writeBatch(tmp_path)
    fitter = Fitter(_cfg(method='laplace'), srcdir=tmp_path)
    np.savez(fitter.outPath('laplace', 0), laplace_failed=np.array(True))
    with pytest.raises(FileNotFoundError, match='1/2 laplace fits missing'):
        fitter.reintegrate()


def _nutsFit(tag: str, level: int, s: int, rhat: float, duration: float) -> dict[str, np.ndarray]:
    d, q, m = 2, 1, 2
    return {
        f'{tag}_ffx': np.arange(d * s, dtype=np.float64).reshape(d, s),
        f'{tag}_sigma_rfx': np.ones((q, s)),
        f'{tag}_sigma_eps': np.ones((1, s)),
        f'{tag}_rfx': np.zeros((q, m, s)),
        f'{tag}_corr_rfx': np.tile(np.eye(q)[None, None], (1, s, 1, 1)),
        f'{tag}_rhat': np.array([1.0, rhat]),
        f'{tag}_ess': np.full(2, 900.0),
        f'{tag}_ess_tail': np.full(2, 900.0),
        f'{tag}_divergences': np.zeros(4, dtype=np.int64),
        f'{tag}_draws': np.array(s // 4),
        f'{tag}_level': np.array(level),
        f'{tag}_duration': np.array(duration),
        f'{tag}_failed': np.array(False),
        f'{tag}_error': np.array(''),
    }


def test_compose_nuts_picks_the_cheapest_converged_level(tmp_path):
    path = _writeBatch(tmp_path)
    fitter = Fitter(_cfg(method='nuts', level=0), srcdir=tmp_path)
    draws = {0: 4000, 1: 4000, 2: 8000}
    # dataset 0 converges at level 0; dataset 1 never converges and falls back to level 2
    for level in range(3):
        for idx in range(2):
            r = 1.0 if idx == 0 else 1.5
            np.savez(
                fitter.outPath(f'nuts{level}', idx),
                **_nutsFit(f'nuts{level}', level, draws[level], r, duration=10.0 * (level + 1)),
            )
    fitter.composeNuts()

    nuts = loadFits(path, 'nuts')
    np.testing.assert_array_equal(nuts['nuts_level'], [0, 2])
    np.testing.assert_array_equal(nuts['nuts_converged'], [True, False])
    np.testing.assert_array_equal(nuts['nuts_duration'], [10.0, 60.0])
    assert nuts['nuts_ffx'].shape == (2, 2, 4000)  # level 2 thinned from 8000 draws
    assert nuts['nuts_corr_rfx'].shape == (2, 1, 4000, 1, 1)
    np.testing.assert_array_equal(nuts['nuts_ffx'][1, 0, :3], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(nuts['nuts_draws'], [1000, 2000])
    col = Collection(path, permute=False, fits=('nuts',))
    assert 'nuts_level' in col.raw
