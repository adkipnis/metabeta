"""Optional PyMC reference implementation for Laplace parity tests.

This module is intentionally kept under ``tests/`` rather than
``metabeta/simulation``.  The maintained rebuttal baseline is the fast scratch
implementation in ``metabeta.simulation.laplace``; this PyMC version is a slow,
partial reference used only by opt-in tests.

Batch output (<data_id>/<partition>.laplace.npz) contains only Laplace keys:
    laplace_ffx             (n_ds, d_max, S)
    laplace_sigma_rfx       (n_ds, q_max, S)
    laplace_sigma_eps       (n_ds, 1, S)              gaussian likelihood only
    laplace_rfx             (n_ds, q_max, m_max, S)
    laplace_corr_rfx        (n_ds, 1, S, q_max, q_max)
    laplace_duration        (n_ds,)
    laplace_failed          (n_ds,)
    laplace_hessian_jitter  (n_ds,)
    laplace_hessian_repaired (n_ds,)
    laplace_hessian_min_eig (n_ds,)

It is not intended as a command-line entry point.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from tqdm import tqdm

from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.names import datasetFilename
from metabeta.utils.padding import unpad
from metabeta.utils.pymc import buildPymc

_DEFAULT_SRCDIR = Path(__file__).resolve().parent.parent / 'outputs' / 'data'
_DIAGNOSTIC_KEYS = (
    'laplace_duration',
    'laplace_failed',
    'laplace_hessian_jitter',
    'laplace_hessian_repaired',
    'laplace_hessian_min_eig',
)


def _rfxLabel(j: int, suffix: str = '') -> str:
    return ('1|i' if j == 0 else f'x{j}|i') + suffix


def _stabilizePrecision(
    precision: np.ndarray,
    jitter_init: float = 1e-8,
    max_tries: int = 8,
    eig_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool, float]:
    """Return positive-definite precision and its Cholesky factor.

    The input is expected to be the negative Hessian of log posterior at the MAP.
    The returned Cholesky factor ``chol`` satisfies ``precision = chol @ chol.T``.
    """
    H = np.asarray(precision, dtype=np.float64)
    H = 0.5 * (H + H.T)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f'precision must be square, got shape {H.shape}')
    if not np.isfinite(H).all():
        raise ValueError('precision contains non-finite values')

    scale = max(float(np.mean(np.abs(np.diag(H)))) if H.size else 1.0, 1.0)
    min_eig = float(np.linalg.eigvalsh(H).min()) if H.size else 0.0
    eye = np.eye(H.shape[0], dtype=np.float64)

    for i in range(max_tries + 1):
        jitter = 0.0 if i == 0 else jitter_init * scale * (10.0 ** (i - 1))
        try:
            chol = np.linalg.cholesky(H + jitter * eye)
            return H + jitter * eye, chol, jitter, False, min_eig
        except np.linalg.LinAlgError:
            continue

    vals, vecs = np.linalg.eigh(H)
    floor = eig_floor if eig_floor is not None else jitter_init * scale
    clipped = np.maximum(vals, floor)
    repaired = (vecs * clipped) @ vecs.T
    repaired = 0.5 * (repaired + repaired.T)
    chol = np.linalg.cholesky(repaired)
    return repaired, chol, float(floor), True, min_eig


def _drawFromPrecision(
    rng: np.random.Generator,
    mean: np.ndarray,
    chol_precision: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Draw rows from ``N(mean, precision^-1)``."""
    z = rng.standard_normal((mean.shape[0], n_samples))
    noise = solve_triangular(chol_precision.T, z, lower=False, check_finite=False)
    return (mean[:, None] + noise).T


def _unravelSamples(flat_samples: np.ndarray, point_map_info: tuple) -> dict[str, np.ndarray]:
    """Map flat transformed samples to arrays keyed by PyMC value-variable name."""
    samples = {}
    last_idx = 0
    for name, shape, size, dtype in point_map_info:
        end = last_idx + size
        samples[name] = flat_samples[:, last_idx:end].reshape((flat_samples.shape[0], *shape))
        samples[name] = samples[name].astype(dtype, copy=False)
        last_idx = end
    return samples


class LaplaceFitter:
    """Fit all datasets in a partition with a transformed-space Laplace approximation."""

    def __init__(
        self,
        cfg: argparse.Namespace,
        srcdir: Path = _DEFAULT_SRCDIR,
    ) -> None:
        if cfg.partition == 'train':
            raise ValueError('Laplace fitting supports only test/valid partitions')
        self.cfg = cfg
        self.srcdir = Path(srcdir)

        epoch = getattr(cfg, 'epoch', None) or 1
        self.fname = datasetFilename(partition=cfg.partition, epoch=epoch)
        self.batch_path = self.srcdir / cfg.data_id / self.fname
        assert self.batch_path.exists(), f'{self.batch_path} does not exist'
        self.outpath = self.batch_path.with_suffix('.laplace.npz')

        self._batch_loaded = False

    def _loadBatch(self) -> None:
        if self._batch_loaded:
            return
        with np.load(self.batch_path, allow_pickle=True) as raw:
            self.batch = dict(raw)
        self.n_fit = len(self.batch['y'])
        self.d_max = int(self.batch['X'].shape[-1])
        self.q_max = int(self.batch['rfx'].shape[-1])
        self.m_max = int(self.batch['rfx'].shape[-2])
        self.likelihood_family = int(
            np.asarray(self.batch.get('likelihood_family', [0])).ravel()[0]
        )
        self.has_sigma_eps = hasSigmaEps(self.likelihood_family)
        self._batch_loaded = True

    def __len__(self) -> int:
        self._loadBatch()
        return self.n_fit

    def _getSingle(self, idx: int) -> dict[str, np.ndarray]:
        self._loadBatch()
        ds = {k: v[idx] for k, v in self.batch.items()}
        sizes = {'d': int(ds['d']), 'q': int(ds['q']), 'm': int(ds['m']), 'n': int(ds['n'])}
        return unpad(ds, sizes)

    def _emptyFit(self) -> dict[str, np.ndarray]:
        self._loadBatch()
        s = int(self.cfg.draws * self.cfg.chains)
        out: dict[str, np.ndarray] = {
            'laplace_ffx': np.zeros((self.n_fit, self.d_max, s), dtype=np.float64),
            'laplace_sigma_rfx': np.zeros((self.n_fit, self.q_max, s), dtype=np.float64),
            'laplace_rfx': np.zeros((self.n_fit, self.q_max, self.m_max, s), dtype=np.float64),
            'laplace_corr_rfx': np.zeros(
                (self.n_fit, 1, s, self.q_max, self.q_max), dtype=np.float64
            ),
            'laplace_duration': np.full(self.n_fit, np.nan, dtype=np.float64),
            'laplace_failed': np.ones(self.n_fit, dtype=bool),
            'laplace_hessian_jitter': np.full(self.n_fit, np.nan, dtype=np.float64),
            'laplace_hessian_repaired': np.zeros(self.n_fit, dtype=bool),
            'laplace_hessian_min_eig': np.full(self.n_fit, np.nan, dtype=np.float64),
        }
        if self.has_sigma_eps:
            out['laplace_sigma_eps'] = np.zeros((self.n_fit, 1, s), dtype=np.float64)
        return out

    def _failureResult(
        self,
        ds: dict[str, np.ndarray],
        duration: float = np.nan,
        jitter: float = np.nan,
        repaired: bool = False,
        min_eig: float = np.nan,
    ) -> dict[str, np.ndarray]:
        d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
        s = int(self.cfg.draws * self.cfg.chains)
        out: dict[str, np.ndarray] = {
            'laplace_ffx': np.full((d, s), np.nan, dtype=np.float64),
            'laplace_sigma_rfx': np.full((q, s), np.nan, dtype=np.float64),
            'laplace_rfx': np.full((q, m, s), np.nan, dtype=np.float64),
            'laplace_corr_rfx': np.full((1, s, q, q), np.nan, dtype=np.float64),
            'laplace_duration': np.array(duration, dtype=np.float64),
            'laplace_failed': np.array(True),
            'laplace_hessian_jitter': np.array(jitter, dtype=np.float64),
            'laplace_hessian_repaired': np.array(repaired, dtype=bool),
            'laplace_hessian_min_eig': np.array(min_eig, dtype=np.float64),
        }
        likelihood_family = (
            int(ds['likelihood_family']) if 'likelihood_family' in ds else self.likelihood_family
        )
        if hasSigmaEps(likelihood_family):
            out['laplace_sigma_eps'] = np.full((1, s), np.nan, dtype=np.float64)
        return out

    def _extractSamples(
        self,
        model: Any,
        flat_samples: np.ndarray,
        point_map_info: tuple,
        start_point: dict[str, np.ndarray],
        ds: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:

        d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
        s = flat_samples.shape[0]
        likelihood_family = (
            int(ds['likelihood_family']) if 'likelihood_family' in ds else self.likelihood_family
        )
        correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2 and not self.cfg.diagonal
        if not correlated:
            return self._extractDiagonalSamples(
                flat_samples, point_map_info, d, q, m, likelihood_family
            )

        from pymc.blocking import DictToArrayBijection, RaveledVars

        outputs = list(model.deterministics) + list(model.free_RVs)
        output_names = [v.name for v in outputs]
        eval_fn = model.compile_fn(
            outputs,
            inputs=model.value_vars,
            point_fn=False,
            on_unused_input='ignore',
        )

        ffx = np.empty((d, s), dtype=np.float64)
        sigma_rfx = np.empty((q, s), dtype=np.float64)
        rfx = np.empty((q, m, s), dtype=np.float64)
        corr_rfx = np.empty((1, s, q, q), dtype=np.float64)
        sigma_eps = np.empty((1, s), dtype=np.float64) if hasSigmaEps(likelihood_family) else None
        value_names = [v.name for v in model.value_vars]
        ffx_names = ['Intercept', *(f'x{j}' for j in range(1, d))]
        rfx_names = [_rfxLabel(j) for j in range(q)]
        sigma_rfx_names = [_rfxLabel(j, '_sigma') for j in range(q)]
        identity_corr = np.eye(q, dtype=np.float64)

        for sample_idx, flat_sample in enumerate(flat_samples):
            point = DictToArrayBijection.rmap(
                RaveledVars(flat_sample, point_map_info),
                start_point=start_point,
            )
            values = eval_fn(*[point[name] for name in value_names])
            sample = dict(zip(output_names, values, strict=True))

            for j, name in enumerate(ffx_names):
                ffx[j, sample_idx] = np.asarray(sample[name], dtype=np.float64)
            for j in range(q):
                sigma_rfx[j, sample_idx] = np.asarray(sample[sigma_rfx_names[j]], dtype=np.float64)
                rfx[j, :, sample_idx] = np.asarray(sample[rfx_names[j]], dtype=np.float64)[:m]
            if sigma_eps is not None:
                sigma_eps[0, sample_idx] = np.asarray(sample['sigma'], dtype=np.float64)
            if correlated:
                corr_rfx[0, sample_idx] = np.asarray(sample['_lkj_rfx_corr'], dtype=np.float64)
            else:
                corr_rfx[0, sample_idx] = identity_corr

        out = {
            'laplace_ffx': ffx,
            'laplace_sigma_rfx': sigma_rfx,
            'laplace_rfx': rfx,
            'laplace_corr_rfx': corr_rfx,
        }
        if sigma_eps is not None:
            out['laplace_sigma_eps'] = sigma_eps
        return out

    def _extractDiagonalSamples(
        self,
        flat_samples: np.ndarray,
        point_map_info: tuple,
        d: int,
        q: int,
        m: int,
        likelihood_family: int,
    ) -> dict[str, np.ndarray]:
        samples = _unravelSamples(flat_samples, point_map_info)
        s = flat_samples.shape[0]
        ffx_names = ['Intercept', *(f'x{j}' for j in range(1, d))]

        ffx = np.stack([samples[name] for name in ffx_names], axis=0).astype(np.float64)
        sigma_rfx = np.empty((q, s), dtype=np.float64)
        rfx = np.empty((q, m, s), dtype=np.float64)

        for j in range(q):
            sigma = np.exp(samples[_rfxLabel(j, '_sigma_log__')]).astype(np.float64)
            offset = samples[_rfxLabel(j, '_offset')][:, :m].astype(np.float64)
            sigma_rfx[j] = sigma
            rfx[j] = (offset * sigma[:, None]).T

        corr_rfx = np.tile(np.eye(q, dtype=np.float64), (1, s, 1, 1))
        out = {
            'laplace_ffx': ffx,
            'laplace_sigma_rfx': sigma_rfx,
            'laplace_rfx': rfx,
            'laplace_corr_rfx': corr_rfx,
        }
        if hasSigmaEps(likelihood_family):
            out['laplace_sigma_eps'] = np.exp(samples['sigma_log__'])[None].astype(np.float64)
        return out

    def _fitSingle(
        self,
        ds: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        import pymc as pm
        from pymc.blocking import DictToArrayBijection

        t0 = time.perf_counter()
        jitter = np.nan
        repaired = False
        min_eig = np.nan
        try:
            model = buildPymc(ds, force_diagonal=getattr(self.cfg, 'diagonal', False))
            with model:
                map_point = pm.find_MAP(
                    method=self.cfg.optimizer,
                    maxeval=self.cfg.maxeval,
                    include_transformed=True,
                    progressbar=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    precision = pm.find_hessian(map_point, model=model)

            _, chol, jitter, repaired, min_eig = _stabilizePrecision(precision)
            start_point = model.initial_point()
            map_value_point = {k: map_point[k] for k in start_point}
            raveled = DictToArrayBijection.map(map_value_point)
            flat_samples = _drawFromPrecision(
                rng,
                raveled.data.astype(np.float64),
                chol,
                int(self.cfg.draws * self.cfg.chains),
            )
            out = self._extractSamples(model, flat_samples, raveled.point_map_info, start_point, ds)
            out['laplace_duration'] = np.array(time.perf_counter() - t0, dtype=np.float64)
            out['laplace_failed'] = np.array(False)
            out['laplace_hessian_jitter'] = np.array(jitter, dtype=np.float64)
            out['laplace_hessian_repaired'] = np.array(repaired, dtype=bool)
            out['laplace_hessian_min_eig'] = np.array(min_eig, dtype=np.float64)
            return out
        except Exception as exc:
            print(f'Laplace failed: {exc}', file=sys.stderr, flush=True)
            return self._failureResult(
                ds,
                duration=time.perf_counter() - t0,
                jitter=jitter,
                repaired=repaired,
                min_eig=min_eig,
            )

    def _insertFit(
        self,
        out: dict[str, np.ndarray],
        idx: int,
        ds: dict[str, np.ndarray],
        fit: dict[str, np.ndarray],
    ) -> None:
        d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
        out['laplace_ffx'][idx, :d] = fit['laplace_ffx']
        out['laplace_sigma_rfx'][idx, :q] = fit['laplace_sigma_rfx']
        out['laplace_rfx'][idx, :q, :m] = fit['laplace_rfx']
        out['laplace_corr_rfx'][idx, :, :, :q, :q] = fit['laplace_corr_rfx']
        if 'laplace_sigma_eps' in out and 'laplace_sigma_eps' in fit:
            out['laplace_sigma_eps'][idx] = fit['laplace_sigma_eps']
        for key in _DIAGNOSTIC_KEYS:
            out[key][idx] = fit[key]

    def go(self) -> None:
        self._loadBatch()
        if self.outpath.exists() and not getattr(self.cfg, 'force', False):
            raise FileExistsError(f'{self.outpath} already exists; pass --force to overwrite')

        out = self._emptyFit()
        root_rng = np.random.default_rng(self.cfg.seed)
        seeds = root_rng.integers(0, np.iinfo(np.uint32).max, size=len(self), dtype=np.uint32)

        for idx in tqdm(range(len(self)), desc='laplace fits', unit='dataset'):
            ds = self._getSingle(idx)
            fit = self._fitSingle(ds, np.random.default_rng(int(seeds[idx])))
            self._insertFit(out, idx, ds, fit)

        np.savez_compressed(self.outpath, **out)
        n_ok = int(np.sum(~out['laplace_failed']))
        print(f'Saved Laplace fits to {self.outpath}  ({n_ok}/{len(self)} OK)')
