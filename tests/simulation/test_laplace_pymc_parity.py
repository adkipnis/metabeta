import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from metabeta.simulation.laplace import LaplaceFitter as ScratchLaplaceFitter


if os.environ.get('METABETA_RUN_PYMC_LAPLACE') != '1':
    pytest.skip(
        'set METABETA_RUN_PYMC_LAPLACE=1 to run slow PyMC Laplace parity checks',
        allow_module_level=True,
    )

pytest.importorskip('pymc')


def _loadPymcReference():
    path = Path(__file__).with_name('laplacepymc_reference.py')
    spec = importlib.util.spec_from_file_location('laplacepymc_reference', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PymcLaplaceFitter = _loadPymcReference().LaplaceFitter


def _cfg(data_id: str, likelihood_family: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_id=data_id,
        partition='test',
        epoch=None,
        draws=200,
        chains=1,
        seed=123,
        maxeval=1000,
        optimizer='L-BFGS-B',
        diagonal=True,
        force=False,
    )


def _write_reference_batch(root: Path, data_id: str, likelihood_family: int) -> None:
    data_dir = root / data_id
    data_dir.mkdir(parents=True)

    B, n, d, q, m = 1, 18, 2, 1, 3
    groups = np.repeat(np.arange(m), n // m).astype(np.int64)
    X = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)]).astype(np.float64)
    Z = X[:, :q].copy()
    beta = np.array([0.25, -0.35], dtype=np.float64)
    rfx = np.array([[-0.4], [0.1], [0.3]], dtype=np.float64)
    eta = X @ beta + (Z * rfx[groups]).sum(axis=1)

    if likelihood_family == 0:
        sigma_eps = 0.45
        y = eta + np.linspace(-0.2, 0.2, n)
    elif likelihood_family == 1:
        sigma_eps = 1.0
        probs = 1.0 / (1.0 + np.exp(-eta))
        y = (probs > np.linspace(0.25, 0.75, n)).astype(np.float64)
    else:
        raise ValueError(f'unsupported test likelihood: {likelihood_family}')

    np.savez(
        data_dir / 'test.npz',
        y=y[None],
        X=X[None],
        Z=Z[None],
        groups=groups[None],
        ns=np.array([[n // m] * m], dtype=np.int64),
        d=np.array([d], dtype=np.int64),
        q=np.array([q], dtype=np.int64),
        m=np.array([m], dtype=np.int64),
        n=np.array([n], dtype=np.int64),
        nu_ffx=np.zeros((B, d), dtype=np.float64),
        tau_ffx=np.full((B, d), 2.0, dtype=np.float64),
        tau_rfx=np.full((B, q), 1.0, dtype=np.float64),
        tau_eps=np.ones(B, dtype=np.float64),
        family_ffx=np.zeros(B, dtype=np.int64),
        family_sigma_rfx=np.zeros(B, dtype=np.int64),
        family_sigma_eps=np.zeros(B, dtype=np.int64),
        ffx=beta[None],
        sigma_rfx=np.ones((B, q), dtype=np.float64),
        sigma_eps=np.array([sigma_eps], dtype=np.float64),
        rfx=rfx[None],
        corr_rfx=np.eye(q, dtype=np.float64)[None],
        eta_rfx=np.zeros(B, dtype=np.float64),
        likelihood_family=np.array([likelihood_family], dtype=np.int64),
        sd_y=np.ones(B, dtype=np.float64),
    )


@pytest.mark.parametrize('likelihood_family', [0, 1])
def test_scratch_laplace_matches_pymc_reference_on_simple_diagonal_models(
    tmp_path, likelihood_family
):
    data_id = f'pymc-parity-{likelihood_family}'
    _write_reference_batch(tmp_path, data_id, likelihood_family)
    cfg = _cfg(data_id, likelihood_family)
    rng = np.random.default_rng(123)

    scratch = ScratchLaplaceFitter(cfg, srcdir=tmp_path)
    pymc = PymcLaplaceFitter(cfg, srcdir=tmp_path)
    ds = scratch._getSingle(0)

    scratch_fit = scratch._fitSingle(ds, rng)
    pymc_fit = pymc._fitSingle(ds, np.random.default_rng(123))

    assert bool(scratch_fit['laplace_failed']) is False
    assert bool(pymc_fit['laplace_failed']) is False

    np.testing.assert_allclose(
        scratch_fit['laplace_ffx'].mean(axis=-1),
        pymc_fit['laplace_ffx'].mean(axis=-1),
        atol=0.35,
    )
    np.testing.assert_allclose(
        scratch_fit['laplace_sigma_rfx'].mean(axis=-1),
        pymc_fit['laplace_sigma_rfx'].mean(axis=-1),
        atol=0.5,
    )
    np.testing.assert_allclose(
        scratch_fit['laplace_rfx'].mean(axis=-1),
        pymc_fit['laplace_rfx'].mean(axis=-1),
        atol=0.6,
    )
    if likelihood_family == 0:
        np.testing.assert_allclose(
            scratch_fit['laplace_sigma_eps'].mean(axis=-1),
            pymc_fit['laplace_sigma_eps'].mean(axis=-1),
            atol=0.35,
        )
