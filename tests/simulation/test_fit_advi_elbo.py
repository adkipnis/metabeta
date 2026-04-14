"""Tests for ADVI ELBO curve recording in Fitter._fitAdvi."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / 'metabeta' / 'outputs' / 'data'
TEST_BATCH = DATA_DIR / 'tiny-n-toy' / 'test.npz'


def _requires_test_data():
    return pytest.mark.skipif(
        not TEST_BATCH.exists(),
        reason=f'test batch not found at {TEST_BATCH}',
    )


def make_advi_cfg(**overrides) -> argparse.Namespace:
    defaults = dict(
        data_id='tiny-n-toy',
        size='tiny',
        family=0,
        ds_type='toy',
        idx=0,
        method='advi',
        seed=42,
        tune=500,
        draws=100,
        chains=1,
        loop=False,
        viter=2000,        # small budget for tests
        lr=5e-3,
        reintegrate=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@_requires_test_data()
def test_elbo_array_saved():
    """_fitAdvi should include advi_elbo and advi_elbo_step in its output."""
    from metabeta.simulation.fit import Fitter

    cfg = make_advi_cfg(viter=2000)
    fitter = Fitter(cfg)
    ds = fitter._getSingle(fitter.batch, 0)
    out = fitter._fitAdvi(cfg, ds, elbo_every=200)

    assert 'advi_elbo' in out, 'advi_elbo missing from output'
    assert 'advi_elbo_step' in out, 'advi_elbo_step missing from output'


@_requires_test_data()
def test_elbo_array_shape():
    """ELBO array length should equal ceil(viter / elbo_every)."""
    from metabeta.simulation.fit import Fitter

    viter, every = 2000, 200
    cfg = make_advi_cfg(viter=viter)
    fitter = Fitter(cfg)
    ds = fitter._getSingle(fitter.batch, 0)
    out = fitter._fitAdvi(cfg, ds, elbo_every=every)

    elbo = out['advi_elbo']
    steps = out['advi_elbo_step']
    expected_len = len(range(0, viter, every))
    assert len(elbo) == expected_len, (
        f'expected {expected_len} ELBO points, got {len(elbo)}'
    )
    assert len(steps) == len(elbo), 'elbo and elbo_step must have the same length'


@_requires_test_data()
def test_elbo_steps_are_correct_multiples():
    """Step indices must be exact multiples of elbo_every."""
    from metabeta.simulation.fit import Fitter

    every = 200
    cfg = make_advi_cfg(viter=2000)
    fitter = Fitter(cfg)
    ds = fitter._getSingle(fitter.batch, 0)
    out = fitter._fitAdvi(cfg, ds, elbo_every=every)

    steps = out['advi_elbo_step']
    assert all(s % every == 0 for s in steps), 'some steps are not multiples of elbo_every'


@_requires_test_data()
def test_elbo_values_are_finite():
    """All recorded ELBO values should be finite (no NaN/Inf from diverged fits)."""
    from metabeta.simulation.fit import Fitter

    cfg = make_advi_cfg(viter=2000)
    fitter = Fitter(cfg)
    ds = fitter._getSingle(fitter.batch, 0)
    out = fitter._fitAdvi(cfg, ds, elbo_every=200)

    elbo = out['advi_elbo']
    assert np.all(np.isfinite(elbo)), f'non-finite ELBO values: {elbo[~np.isfinite(elbo)]}'


@_requires_test_data()
def test_elbo_dtype():
    """ELBO array should be float64, step array int64."""
    from metabeta.simulation.fit import Fitter

    cfg = make_advi_cfg(viter=2000)
    fitter = Fitter(cfg)
    ds = fitter._getSingle(fitter.batch, 0)
    out = fitter._fitAdvi(cfg, ds, elbo_every=200)

    assert out['advi_elbo'].dtype == np.float64
    assert out['advi_elbo_step'].dtype == np.int64


@_requires_test_data()
def test_existing_output_keys_still_present():
    """ELBO addition must not drop any pre-existing output keys."""
    from metabeta.simulation.fit import Fitter

    cfg = make_advi_cfg(viter=2000)
    fitter = Fitter(cfg)
    ds = fitter._getSingle(fitter.batch, 0)
    out = fitter._fitAdvi(cfg, ds, elbo_every=200)

    required = {'advi_ffx', 'advi_sigma_rfx', 'advi_rfx',
                'advi_ess', 'advi_duration', 'advi_elbo', 'advi_elbo_step'}
    missing = required - set(out.keys())
    assert not missing, f'missing keys: {missing}'
