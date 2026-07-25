"""Fast scratch Laplace approximation for hierarchical datasets.

This module intentionally does not use PyMC.  It fits a pragmatic GLMM posterior
with a direct torch objective, computes a dense transformed-space Hessian at the
MAP, samples the Gaussian Laplace approximation, and writes method-only batch
outputs to ``<partition>.laplace.npz``.
The sidecar can be merged into ``<partition>.fit.npz`` for evaluation.

The random-effect covariance uses a simple unconstrained Cholesky
parameterization and prior.  This is not exact PyMC/LKJ prior parity; the PyMC
implementation is kept in ``laplacepymc.py`` as a reference backend.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from tqdm import tqdm
import torch

from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF, hasSigmaEps
from metabeta.utils.names import datasetFilename
from metabeta.utils.padding import unpad
from metabeta.utils.templates import setupConfigParser, generateSimulationConfig

_DEFAULT_SRCDIR = Path(__file__).resolve().parent.parent / 'outputs' / 'data'
_DIAGNOSTIC_KEYS = (
    'laplace_duration',
    'laplace_failed',
    'laplace_hessian_jitter',
    'laplace_hessian_repaired',
    'laplace_hessian_min_eig',
    'laplace_objective',
    'laplace_iterations',
)
_RUNTIME_DEFAULTS = {
    'partition': 'test',
    'epoch': None,
    'draws': 1000,
    'chains': 4,
    'seed': 42,
    'maxeval': 200,
    'optimizer': 'LBFGS',
    'diagonal': False,
    'force': False,
    'reintegrate': False,
}


def _numCovParams(q: int, diagonal: bool) -> int:
    return q if diagonal else q * (q + 1) // 2


def _lowerIndices(q: int, diagonal: bool) -> list[tuple[int, int]]:
    if diagonal:
        return [(j, j) for j in range(q)]
    return [(i, j) for i in range(q) for j in range(i + 1)]


def _stabilizePrecision(
    precision: np.ndarray,
    jitter_init: float = 1e-8,
    max_tries: int = 8,
    eig_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool, float]:
    """Return positive-definite precision and its Cholesky factor."""
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
    noise = solve_triangular(chol_precision, z, lower=True, check_finite=False)
    return (mean[:, None] + noise).T


def _unpack(
    theta: torch.Tensor,
    d: int,
    q: int,
    m: int,
    has_sigma_eps: bool,
    diagonal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    beta = theta[:d]
    pos = d
    cov_raw = theta[pos : pos + _numCovParams(q, diagonal)]
    pos += cov_raw.numel()

    L = theta.new_zeros((q, q))
    for k, (i, j) in enumerate(_lowerIndices(q, diagonal)):
        L[i, j] = torch.exp(cov_raw[k]) if i == j else cov_raw[k]

    rfx_offset = theta[pos : pos + m * q].reshape(m, q)
    pos += m * q
    log_sigma_eps = theta[pos] if has_sigma_eps else None
    return beta, L, rfx_offset, log_sigma_eps


def _packInitial(ds: dict[str, np.ndarray], diagonal: bool) -> np.ndarray:
    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    likelihood_family = int(ds.get('likelihood_family', 0))
    has_eps = hasSigmaEps(likelihood_family)
    y = ds['y'].astype(np.float64)
    X = ds['X'].astype(np.float64)
    nu_ffx = ds.get('nu_ffx', np.zeros(d, dtype=np.float64)).astype(np.float64)
    tau_rfx = np.maximum(ds.get('tau_rfx', np.ones(q, dtype=np.float64)).astype(np.float64), 1e-3)

    if likelihood_family == 0:
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = nu_ffx.copy()
    else:
        beta = nu_ffx.copy()

    cov = []
    for i, j in _lowerIndices(q, diagonal):
        cov.append(np.log(tau_rfx[i]) if i == j else 0.0)
    b = np.zeros(m * q, dtype=np.float64)

    parts = [beta, np.asarray(cov, dtype=np.float64), b]
    if has_eps:
        resid = y - X @ beta
        sigma0 = max(float(np.std(resid)), float(ds.get('tau_eps', 1.0)) * 0.25, 1e-3)
        parts.append(np.array([np.log(sigma0)], dtype=np.float64))
    return np.concatenate(parts)


def _asTorchData(ds: dict[str, np.ndarray], diagonal: bool) -> dict:
    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    likelihood_family = int(ds.get('likelihood_family', 0))
    return {
        'X': torch.as_tensor(ds['X'].astype(np.float64), dtype=torch.float64),
        'Z': torch.as_tensor(ds.get('Z', ds['X'][:, :q]).astype(np.float64), dtype=torch.float64),
        'y': torch.as_tensor(ds['y'].astype(np.float64), dtype=torch.float64),
        'groups': torch.as_tensor(ds['groups'].astype(np.int64), dtype=torch.long),
        'nu_ffx': torch.as_tensor(
            ds.get('nu_ffx', np.zeros(d)).astype(np.float64), dtype=torch.float64
        ),
        'tau_ffx': torch.as_tensor(
            np.maximum(ds.get('tau_ffx', np.ones(d)).astype(np.float64), 1e-6),
            dtype=torch.float64,
        ),
        'tau_rfx': torch.as_tensor(
            np.maximum(ds.get('tau_rfx', np.ones(q)).astype(np.float64), 1e-6),
            dtype=torch.float64,
        ),
        'tau_eps': torch.as_tensor(float(ds.get('tau_eps', 1.0)), dtype=torch.float64),
        'd': d,
        'q': q,
        'm': m,
        'likelihood_family': likelihood_family,
        'has_sigma_eps': hasSigmaEps(likelihood_family),
        'family_ffx': int(ds.get('family_ffx', 0)),
        'family_sigma_rfx': int(ds.get('family_sigma_rfx', 0)),
        'family_sigma_eps': int(ds.get('family_sigma_eps', 0)),
        'diagonal': diagonal,
    }


def _fixedPriorNlp(beta: torch.Tensor, data: dict) -> torch.Tensor:
    z = (beta - data['nu_ffx']) / data['tau_ffx']
    family = FFX_FAMILIES[data['family_ffx']]
    if family == 'normal':
        return 0.5 * torch.sum(z.square())
    if family == 'student':
        return 0.5 * (STUDENT_DF + 1.0) * torch.sum(torch.log1p(z.square() / STUDENT_DF))
    raise ValueError(f'unsupported fixed-effect prior family: {family}')


def _sigmaPriorNlp(log_sigma: torch.Tensor, scale: torch.Tensor, family_idx: int) -> torch.Tensor:
    sigma = torch.exp(log_sigma)
    family = SIGMA_FAMILIES[family_idx]
    if family == 'halfnormal':
        nlp = 0.5 * (sigma / scale).square()
    elif family == 'halfstudent':
        nlp = 0.5 * (STUDENT_DF + 1.0) * torch.log1p((sigma / scale).square() / STUDENT_DF)
    elif family == 'exponential':
        nlp = sigma / scale
    else:
        raise ValueError(f'unsupported sigma prior family: {family}')
    return nlp - log_sigma


def _objective(theta: torch.Tensor, data: dict) -> torch.Tensor:
    d, q, m = data['d'], data['q'], data['m']
    beta, L, rfx_offset, log_sigma_eps = _unpack(
        theta, d, q, m, data['has_sigma_eps'], data['diagonal']
    )
    b = rfx_offset.matmul(L.T)

    eta = data['X'].matmul(beta) + (data['Z'] * b[data['groups']]).sum(dim=-1)
    y = data['y']
    family = data['likelihood_family']
    if family == 0:
        sigma_eps = torch.exp(log_sigma_eps)
        ll_nlp = 0.5 * torch.sum(((y - eta) / sigma_eps).square() + 2.0 * log_sigma_eps)
    elif family == 1:
        ll_nlp = torch.sum(torch.nn.functional.softplus(eta) - y * eta)
    elif family == 2:
        eta_cap = eta.new_tensor(30.0)
        rate_cap = torch.exp(eta_cap)
        eta_capped = torch.minimum(eta, eta_cap)
        rate = torch.exp(eta_capped) + rate_cap * torch.clamp(eta - eta_cap, min=0.0)
        ll_nlp = torch.sum(rate - y * eta)
    else:
        raise ValueError(f'unsupported likelihood_family: {family}')

    prior = _fixedPriorNlp(beta, data)
    diag = torch.diagonal(L)
    log_diag = torch.log(diag)
    prior = prior + 0.5 * torch.sum(rfx_offset.square())

    prior = prior + torch.sum(_sigmaPriorNlp(log_diag, data['tau_rfx'], data['family_sigma_rfx']))
    if not data['diagonal'] and q > 1:
        offdiag = L[torch.tril_indices(q, q, offset=-1).unbind()]
        off_scale = torch.clamp(data['tau_rfx'].mean(), min=0.1)
        prior = prior + 0.5 * torch.sum((offdiag / off_scale).square())

    if data['has_sigma_eps']:
        prior = prior + _sigmaPriorNlp(log_sigma_eps, data['tau_eps'], data['family_sigma_eps'])
    return ll_nlp + prior


def _fitMap(
    init: np.ndarray,
    data: dict,
    max_iter: int,
) -> tuple[np.ndarray, float, int]:
    theta = torch.tensor(init, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max(1, max_iter * 2),
        tolerance_grad=1e-5,
        tolerance_change=1e-8,
        line_search_fn='strong_wolfe',
    )
    iterations = 0

    def closure():
        nonlocal iterations
        optimizer.zero_grad()
        loss = _objective(theta, data)
        loss.backward()
        iterations += 1
        return loss

    optimizer.step(closure)
    final_loss = _objective(theta, data)
    return theta.detach().numpy().copy(), float(final_loss.detach()), iterations


def _hessian(theta_map: np.ndarray, data: dict) -> np.ndarray:
    theta = torch.tensor(theta_map, dtype=torch.float64)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return _objective(x, data)

    H = torch.autograd.functional.hessian(fn, theta, vectorize=True)
    return H.detach().numpy()


def _naturalSamples(
    flat_samples: np.ndarray,
    d: int,
    q: int,
    m: int,
    has_sigma_eps: bool,
    diagonal: bool,
) -> dict[str, np.ndarray]:
    s = flat_samples.shape[0]
    pos = 0
    ffx = flat_samples[:, pos : pos + d].T.astype(np.float64)
    pos += d

    cov_raw = flat_samples[:, pos : pos + _numCovParams(q, diagonal)]
    pos += cov_raw.shape[1]
    L = np.zeros((s, q, q), dtype=np.float64)
    for k, (i, j) in enumerate(_lowerIndices(q, diagonal)):
        L[:, i, j] = np.exp(cov_raw[:, k]) if i == j else cov_raw[:, k]
    cov = L @ np.swapaxes(L, -1, -2)
    sigma = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-12))
    corr = cov / np.maximum(sigma[:, :, None] * sigma[:, None, :], 1e-12)

    rfx_offset = flat_samples[:, pos : pos + m * q].reshape(s, m, q)
    pos += m * q
    b = rfx_offset @ np.swapaxes(L, -1, -2)
    out = {
        'laplace_ffx': ffx,
        'laplace_sigma_rfx': sigma.T.astype(np.float64),
        'laplace_rfx': b.transpose(2, 1, 0).astype(np.float64),
        'laplace_corr_rfx': corr[None].astype(np.float64),
    }
    if has_sigma_eps:
        out['laplace_sigma_eps'] = np.exp(flat_samples[:, pos])[None].astype(np.float64)
    return out


class LaplaceFitter:
    """Fit all datasets in a partition with a scratch Laplace approximation."""

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
        ds = unpad(ds, sizes)
        if 'Z' in ds:
            ds['Z'] = ds['Z'][: sizes['n'], : sizes['q']]
        return ds

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
            'laplace_objective': np.full(self.n_fit, np.nan, dtype=np.float64),
            'laplace_iterations': np.zeros(self.n_fit, dtype=np.int64),
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
        objective: float = np.nan,
        iterations: int = 0,
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
            'laplace_objective': np.array(objective, dtype=np.float64),
            'laplace_iterations': np.array(iterations, dtype=np.int64),
        }
        likelihood_family = (
            int(ds['likelihood_family']) if 'likelihood_family' in ds else self.likelihood_family
        )
        if hasSigmaEps(likelihood_family):
            out['laplace_sigma_eps'] = np.full((1, s), np.nan, dtype=np.float64)
        return out

    def _fitSingle(
        self,
        ds: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        t0 = time.perf_counter()
        jitter = np.nan
        repaired = False
        min_eig = np.nan
        objective = np.nan
        iterations = 0
        try:
            diagonal = bool(getattr(self.cfg, 'diagonal', False))
            init = _packInitial(ds, diagonal)
            data = _asTorchData(ds, diagonal)
            theta_map, objective, iterations = _fitMap(init, data, int(self.cfg.maxeval))
            precision = _hessian(theta_map, data)
            _, chol, jitter, repaired, min_eig = _stabilizePrecision(precision)
            flat_samples = _drawFromPrecision(
                rng,
                theta_map,
                chol,
                int(self.cfg.draws * self.cfg.chains),
            )
            d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
            likelihood_family = (
                int(ds['likelihood_family'])
                if 'likelihood_family' in ds
                else self.likelihood_family
            )
            out = _naturalSamples(
                flat_samples,
                d,
                q,
                m,
                hasSigmaEps(likelihood_family),
                diagonal,
            )
            out['laplace_duration'] = np.array(time.perf_counter() - t0, dtype=np.float64)
            out['laplace_failed'] = np.array(False)
            out['laplace_hessian_jitter'] = np.array(jitter, dtype=np.float64)
            out['laplace_hessian_repaired'] = np.array(repaired, dtype=bool)
            out['laplace_hessian_min_eig'] = np.array(min_eig, dtype=np.float64)
            out['laplace_objective'] = np.array(objective, dtype=np.float64)
            out['laplace_iterations'] = np.array(iterations, dtype=np.int64)
            return out
        except Exception as exc:
            print(f'Laplace failed: {exc}', file=sys.stderr, flush=True)
            return self._failureResult(
                ds,
                duration=time.perf_counter() - t0,
                jitter=jitter,
                repaired=repaired,
                min_eig=min_eig,
                objective=objective,
                iterations=iterations,
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

    def reintegrate(self) -> None:
        """Merge ``<partition>.laplace.npz`` sidecar keys into ``<partition>.fit.npz``."""
        if not self.outpath.exists():
            raise FileNotFoundError(f'cannot merge: {self.outpath} does not exist')

        fit_path = self.batch_path.with_suffix('.fit.npz')
        base_path = fit_path if fit_path.exists() else self.batch_path
        with np.load(base_path, allow_pickle=True) as raw:
            merged = dict(raw)
        with np.load(self.outpath, allow_pickle=True) as raw:
            laplace = {key: raw[key] for key in raw.files if key.startswith('laplace_')}

        if not laplace:
            raise ValueError(f'{self.outpath} contains no laplace_* keys')
        n_laplace = len(next(iter(laplace.values())))
        n_data = len(merged['y'])
        if n_laplace != n_data:
            raise ValueError(
                f'{self.outpath} has {n_laplace} fits, but {base_path} has {n_data} datasets'
            )

        merged.update(laplace)
        np.savez_compressed(fit_path, **merged)
        n_ok = int(np.sum(~laplace['laplace_failed'].astype(bool)))
        print(f'Merged Laplace fits into {fit_path}  ({n_ok}/{n_data} OK)')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data (template-based, matching fit.py)
    parser.add_argument('--size', type=str, default='small', help='Size preset: tiny|small|medium|large|huge')
    parser.add_argument('--family', type=int, default=0, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type', type=str, default='sampled', help='Dataset type: toy|flat|scm|mixed|sampled|observed')
    parser.add_argument('--config', type=str, help='Path to a saved config.yaml; explicit CLI args override its values')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--epoch', type=int, default=None, help='Unused by Laplace; train partition is unsupported')

    # Laplace args
    parser.add_argument('--draws', type=int, default=1000, help='Posterior samples per chain (default=1000)')
    parser.add_argument('--chains', type=int, default=4, help='Number of chains to match sample count semantics (default=4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default=42)')
    parser.add_argument('--maxeval', type=int, default=200, help='Maximum LBFGS iterations (default=200)')
    parser.add_argument('--optimizer', type=str, default='LBFGS', help='Accepted for fit.py compatibility; scratch backend uses LBFGS')
    parser.add_argument('--diagonal', action='store_true', help='Force diagonal RFX covariance (default=False)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing <partition>.laplace.npz (default=False)')
    parser.add_argument('--reintegrate', action='store_true', help='Merge <partition>.laplace.npz into <partition>.fit.npz')
    cfg = setupConfigParser(parser, generateSimulationConfig, 'Fit hierarchical datasets with scratch Laplace approximation.')
    for key, value in _RUNTIME_DEFAULTS.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    if cfg.partition == 'all':
        cfg.partition = 'test'
    return cfg
# fmt: on


def main() -> int:
    cfg = setup()
    if cfg.partition == 'train':
        print('error: Laplace fitting supports only test/valid partitions', file=sys.stderr)
        return 1
    try:
        fitter = LaplaceFitter(cfg)
        if cfg.reintegrate:
            fitter.reintegrate()
        else:
            fitter.go()
    except (FileNotFoundError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
