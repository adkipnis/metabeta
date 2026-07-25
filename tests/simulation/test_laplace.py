import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

from metabeta.simulation.laplace import LaplaceFitter, _stabilizePrecision, setup


def _write_batch(root: Path, data_id: str = 'test-n-laplace') -> None:
    data_dir = root / data_id
    data_dir.mkdir(parents=True)

    B, n_max, d_max, q_max, m_max = 2, 5, 3, 2, 4
    y = np.zeros((B, n_max), dtype=np.float64)
    X = np.zeros((B, n_max, d_max), dtype=np.float64)
    X[..., 0] = 1.0
    groups = np.zeros((B, n_max), dtype=np.int64)
    ns = np.zeros((B, m_max), dtype=np.int64)

    d = np.array([2, 3], dtype=np.int64)
    q = np.array([1, 2], dtype=np.int64)
    m = np.array([2, 3], dtype=np.int64)
    n = np.array([4, 5], dtype=np.int64)
    for i in range(B):
        groups[i, : n[i]] = np.repeat(np.arange(m[i]), [2, 2, 1][: m[i]])[: n[i]]
        ns[i, : m[i]] = np.bincount(groups[i, : n[i]], minlength=m[i])

    np.savez(
        data_dir / 'test.npz',
        y=y,
        X=X,
        groups=groups,
        ns=ns,
        d=d,
        q=q,
        m=m,
        n=n,
        nu_ffx=np.zeros((B, d_max), dtype=np.float64),
        tau_ffx=np.ones((B, d_max), dtype=np.float64),
        tau_rfx=np.ones((B, q_max), dtype=np.float64),
        ffx=np.zeros((B, d_max), dtype=np.float64),
        sigma_rfx=np.ones((B, q_max), dtype=np.float64),
        rfx=np.zeros((B, m_max, q_max), dtype=np.float64),
        corr_rfx=np.tile(np.eye(q_max), (B, 1, 1)),
        likelihood_family=np.zeros(B, dtype=np.int64),
    )


def _cfg(data_id: str = 'test-n-laplace', **kwargs) -> argparse.Namespace:
    values = {
        'data_id': data_id,
        'partition': 'test',
        'epoch': None,
        'draws': 3,
        'chains': 2,
        'seed': 7,
        'maxeval': 10,
        'optimizer': 'L-BFGS-B',
        'diagonal': False,
        'force': False,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_stabilize_precision_accepts_positive_definite():
    precision = np.array([[2.0, 0.2], [0.2, 1.5]])
    repaired, chol, jitter, used_repair, min_eig = _stabilizePrecision(precision)

    np.testing.assert_allclose(repaired, precision)
    np.testing.assert_allclose(chol @ chol.T, precision)
    assert jitter == 0.0
    assert used_repair is False
    assert min_eig > 0.0


def test_stabilize_precision_repairs_indefinite():
    precision = np.array([[1.0, 2.0], [2.0, 1.0]])
    repaired, chol, jitter, used_repair, min_eig = _stabilizePrecision(
        precision,
        max_tries=0,
        eig_floor=1e-4,
    )

    np.testing.assert_allclose(chol @ chol.T, repaired)
    assert np.linalg.eigvalsh(repaired).min() > 0.0
    assert jitter == pytest.approx(1e-4)
    assert used_repair is True
    assert min_eig < 0.0


def test_failure_result_shapes_gaussian(tmp_path):
    _write_batch(tmp_path)
    fitter = LaplaceFitter(_cfg(), srcdir=tmp_path)
    ds = fitter._getSingle(1)

    out = fitter._failureResult(ds, duration=0.5)

    assert out['laplace_ffx'].shape == (3, 6)
    assert out['laplace_sigma_rfx'].shape == (2, 6)
    assert out['laplace_sigma_eps'].shape == (1, 6)
    assert out['laplace_rfx'].shape == (2, 3, 6)
    assert out['laplace_corr_rfx'].shape == (1, 6, 2, 2)
    assert bool(out['laplace_failed'])
    assert float(out['laplace_duration']) == pytest.approx(0.5)


def test_go_writes_standalone_batch_file(tmp_path, monkeypatch):
    _write_batch(tmp_path)
    fitter = LaplaceFitter(_cfg(force=True), srcdir=tmp_path)

    def fake_fit_single(self, ds, rng):
        d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
        s = int(self.cfg.draws * self.cfg.chains)
        return {
            'laplace_ffx': np.full((d, s), 1.0),
            'laplace_sigma_rfx': np.full((q, s), 2.0),
            'laplace_sigma_eps': np.full((1, s), 3.0),
            'laplace_rfx': np.full((q, m, s), 4.0),
            'laplace_corr_rfx': np.tile(np.eye(q)[None, None], (1, s, 1, 1)),
            'laplace_duration': np.array(0.25),
            'laplace_failed': np.array(False),
            'laplace_hessian_jitter': np.array(0.0),
            'laplace_hessian_repaired': np.array(False),
            'laplace_hessian_min_eig': np.array(1.0),
        }

    monkeypatch.setattr(LaplaceFitter, '_fitSingle', fake_fit_single)
    fitter.go()

    outpath = tmp_path / 'test-n-laplace' / 'test.laplace.npz'
    assert outpath.exists()
    with np.load(outpath, allow_pickle=True) as raw:
        assert raw['laplace_ffx'].shape == (2, 3, 6)
        assert raw['laplace_sigma_rfx'].shape == (2, 2, 6)
        assert raw['laplace_sigma_eps'].shape == (2, 1, 6)
        assert raw['laplace_rfx'].shape == (2, 2, 4, 6)
        assert raw['laplace_corr_rfx'].shape == (2, 1, 6, 2, 2)
        assert np.all(raw['laplace_sigma_rfx'][0, 1] == 0.0)
        np.testing.assert_array_equal(raw['laplace_failed'], np.array([False, False]))


def test_go_refuses_existing_output_without_force(tmp_path):
    _write_batch(tmp_path)
    outpath = tmp_path / 'test-n-laplace' / 'test.laplace.npz'
    np.savez(outpath, laplace_failed=np.array([False]))

    fitter = LaplaceFitter(_cfg(force=False), srcdir=tmp_path)

    with pytest.raises(FileExistsError):
        fitter.go()


def test_train_partition_is_rejected(tmp_path):
    _write_batch(tmp_path)
    cfg = _cfg(partition='train')

    with pytest.raises(ValueError, match='test/valid'):
        LaplaceFitter(cfg, srcdir=tmp_path)


def test_setup_backfills_runtime_defaults_for_config(tmp_path, monkeypatch):
    config = tmp_path / 'config.yaml'
    config.write_text(
        '\n'.join(
            [
                'ds_type: sampled',
                'likelihood_family: 0',
                'max_d: 2',
                'max_q: 1',
                'min_m: 5',
                'max_m: 10',
                'min_n: 5',
                'max_n: 10',
                'max_n_total: 50',
                'data_id: configured-n-laplace',
            ]
        )
    )
    monkeypatch.setattr(sys, 'argv', ['laplace.py', '--config', str(config)])

    cfg = setup()

    assert cfg.data_id == 'configured-n-laplace'
    assert cfg.partition == 'test'
    assert cfg.draws == 1000
    assert cfg.chains == 4
    assert cfg.optimizer == 'L-BFGS-B'
    assert cfg.force is False
