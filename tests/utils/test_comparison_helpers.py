"""Tests for helper functions in experiments/analytical/glmm_inla_comparison.py.

Covers the diagnostic keys added for the Normal INLA diagnostic workflow:
- _loadInlaFits: backward-compatible loading of inla_sigma_rfx_mode
- _appendInlaRowRecord: sigma_rfx_mode_inla and sigma_rfx_map_{method} columns
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# import the experiment module without executing its __main__ block
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).parent.parent.parent / 'experiments' / 'analytical' / 'glmm_inla_comparison.py'
)


@pytest.fixture(scope='module')
def _cmp():
    spec = importlib.util.spec_from_file_location('glmm_inla_comparison', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _loadInlaFits: sigma_rfx_mode present vs. absent
# ---------------------------------------------------------------------------


def test_load_inla_fits_without_mode_key(tmp_path, _cmp):
    """Old .inla.npz files without inla_sigma_rfx_mode load cleanly; key absent in result."""
    n_ds, d, q, max_m = 8, 3, 2, 5
    np.savez(
        tmp_path / 'ds.inla.npz',
        inla_ffx=np.zeros((n_ds, d)),
        inla_sigma_rfx=np.ones((n_ds, q)),
        inla_rfx=np.zeros((n_ds, max_m, q)),
        inla_wall_s=np.ones(n_ds),
        inla_failed=np.zeros(n_ds, dtype=bool),
    )
    data = _cmp._loadInlaFits(tmp_path / 'ds.npz')
    assert 'sigma_rfx_mode' not in data


def test_load_inla_fits_with_mode_key(tmp_path, _cmp):
    """New .inla.npz files with inla_sigma_rfx_mode expose sigma_rfx_mode in result."""
    n_ds, d, q, max_m = 8, 3, 2, 5
    expected_mode = np.full((n_ds, q), 0.42)
    np.savez(
        tmp_path / 'ds.inla.npz',
        inla_ffx=np.zeros((n_ds, d)),
        inla_sigma_rfx=np.ones((n_ds, q)),
        inla_rfx=np.zeros((n_ds, max_m, q)),
        inla_wall_s=np.ones(n_ds),
        inla_failed=np.zeros(n_ds, dtype=bool),
        inla_sigma_rfx_mode=expected_mode,
    )
    data = _cmp._loadInlaFits(tmp_path / 'ds.npz')
    assert 'sigma_rfx_mode' in data
    np.testing.assert_array_equal(data['sigma_rfx_mode'], expected_mode)


# ---------------------------------------------------------------------------
# _appendInlaRowRecord: diagnostic columns
# ---------------------------------------------------------------------------


def _minimal_batch(B: int, m: int, max_n: int, d: int, q: int) -> dict[str, torch.Tensor]:
    return {
        'n': torch.full((B,), m * max_n, dtype=torch.long),
        'm': torch.full((B,), m, dtype=torch.long),
        'ffx': torch.zeros(B, d),
        'sigma_rfx': torch.ones(B, q),
        'rfx': torch.zeros(B, m, q),
        'mask_d': torch.ones(B, d, dtype=torch.bool),
        'mask_q': torch.ones(B, q, dtype=torch.bool),
        'mask_n': torch.ones(B, m, max_n),
        'ns': torch.full((B, m), float(max_n)),
        'y': torch.zeros(B, m, max_n),
    }


def test_append_row_includes_sigma_rfx_mode_inla_when_present(_cmp):
    """sigma_rfx_mode_inla is populated from est['sigma_rfx_mode'] when available."""
    B, m, max_n, d, q = 1, 4, 10, 2, 1
    batch = _minimal_batch(B, m, max_n, d, q)
    active_d = np.arange(d)
    active_q = np.arange(q)
    srfx_np = np.zeros((B, q))
    stats_np = {
        'current': {
            'beta': np.zeros((B, d)),
            'srfx': srfx_np,
            'blup': np.zeros((B, m, q)),
            'wall': 0.001,
        }
    }
    est = {
        'beta': np.zeros(d),
        'sigma_rfx': np.ones(q),
        'blups': np.zeros((m, q)),
        'sigma_rfx_mode': np.array([0.77]),
    }

    records: list[dict] = []
    _cmp._appendInlaRowRecord(
        records,
        data_id='small-n-sampled',
        partition='test',
        dataset_idx=0,
        batch=batch,
        b=0,
        active_d=active_d,
        active_q=active_q,
        stats_np=stats_np,
        stats_torch={'current': {}},
        analytical_methods=['current'],
        est=est,
        inla_wall_s=1.0,
        max_d=d,
        max_q=q,
    )

    assert len(records) == 1
    rec = records[0]
    assert 'sigma_rfx_mode_inla' in rec
    np.testing.assert_allclose(rec['sigma_rfx_mode_inla'], [0.77])


def test_append_row_sigma_rfx_mode_inla_nan_when_absent(_cmp):
    """sigma_rfx_mode_inla is NaN-padded when est has no sigma_rfx_mode."""
    B, m, max_n, d, q = 1, 4, 10, 2, 1
    batch = _minimal_batch(B, m, max_n, d, q)
    active_d = np.arange(d)
    active_q = np.arange(q)
    stats_np = {
        'current': {
            'beta': np.zeros((B, d)),
            'srfx': np.zeros((B, q)),
            'blup': np.zeros((B, m, q)),
            'wall': 0.001,
        }
    }
    est = {
        'beta': np.zeros(d),
        'sigma_rfx': np.ones(q),
        'blups': np.zeros((m, q)),
        # no sigma_rfx_mode key
    }

    records: list[dict] = []
    _cmp._appendInlaRowRecord(
        records,
        data_id='small-n-sampled',
        partition='test',
        dataset_idx=0,
        batch=batch,
        b=0,
        active_d=active_d,
        active_q=active_q,
        stats_np=stats_np,
        stats_torch={'current': {}},
        analytical_methods=['current'],
        est=est,
        inla_wall_s=1.0,
        max_d=d,
        max_q=q,
    )

    rec = records[0]
    assert 'sigma_rfx_mode_inla' in rec
    assert np.all(np.isnan(rec['sigma_rfx_mode_inla']))


def test_append_row_includes_sigma_rfx_map_when_present(_cmp):
    """sigma_rfx_map_{method} is saved when stats_np has map_srfx."""
    B, m, max_n, d, q = 1, 4, 10, 2, 1
    batch = _minimal_batch(B, m, max_n, d, q)
    active_d = np.arange(d)
    active_q = np.arange(q)
    stats_np = {
        'current': {
            'beta': np.zeros((B, d)),
            'srfx': np.zeros((B, q)),
            'blup': np.zeros((B, m, q)),
            'wall': 0.001,
            'map_srfx': np.full((B, q), 0.33),
        }
    }
    est = {
        'beta': np.zeros(d),
        'sigma_rfx': np.ones(q),
        'blups': np.zeros((m, q)),
    }

    records: list[dict] = []
    _cmp._appendInlaRowRecord(
        records,
        data_id='small-n-sampled',
        partition='test',
        dataset_idx=0,
        batch=batch,
        b=0,
        active_d=active_d,
        active_q=active_q,
        stats_np=stats_np,
        stats_torch={'current': {}},
        analytical_methods=['current'],
        est=est,
        inla_wall_s=1.0,
        max_d=d,
        max_q=q,
    )

    rec = records[0]
    assert 'sigma_rfx_map_current' in rec
    np.testing.assert_allclose(rec['sigma_rfx_map_current'], [0.33])


def test_append_row_no_sigma_rfx_map_when_absent(_cmp):
    """sigma_rfx_map_{method} must not appear when map_srfx is absent from stats_np."""
    B, m, max_n, d, q = 1, 4, 10, 2, 1
    batch = _minimal_batch(B, m, max_n, d, q)
    active_d = np.arange(d)
    active_q = np.arange(q)
    stats_np = {
        'current': {
            'beta': np.zeros((B, d)),
            'srfx': np.zeros((B, q)),
            'blup': np.zeros((B, m, q)),
            'wall': 0.001,
            # no map_srfx
        }
    }
    est = {
        'beta': np.zeros(d),
        'sigma_rfx': np.ones(q),
        'blups': np.zeros((m, q)),
    }

    records: list[dict] = []
    _cmp._appendInlaRowRecord(
        records,
        data_id='small-n-sampled',
        partition='test',
        dataset_idx=0,
        batch=batch,
        b=0,
        active_d=active_d,
        active_q=active_q,
        stats_np=stats_np,
        stats_torch={'current': {}},
        analytical_methods=['current'],
        est=est,
        inla_wall_s=1.0,
        max_d=d,
        max_q=q,
    )

    rec = records[0]
    assert 'sigma_rfx_map_current' not in rec
