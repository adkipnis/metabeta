"""R-INLA fitting for hierarchical datasets.

Saves point estimates in the same per-dataset / batch .fit.npz format as
fit.py (PyMC/NUTS/ADVI), enabling downstream comparison with NUTS reference fits.

Per-dataset output (<data_id>/fits/<partition>_inla_<idx:03d>.npz):
    inla_ffx        (d,)   — posterior means for fixed effects
    inla_sigma_rfx  (q,)   — E[1/sqrt(τ)] for each RE variance
    inla_rfx        (m, q) — group-level BLUPs
    inla_wall_s     scalar — wall time in seconds
    inla_failed     bool   — True if INLA failed or timed out

    When --samples S > 0 (joint posterior samples via inla.posterior.sample):
    inla_ffx_samples        (d, S)    — samples follow (dim, S) convention of nuts/advi
    inla_sigma_rfx_samples  (q, S)
    inla_rfx_samples        (q, m, S)

Batch fit file (<data_id>/<partition>.fit.npz) adds inla_* keys to the batch:
    inla_ffx        (n_ds, d_max)
    inla_sigma_rfx  (n_ds, q_max)
    inla_rfx        (n_ds, m_max, q_max)
    inla_wall_s     (n_ds,)
    inla_failed     (n_ds,)  dtype bool

    When --samples S > 0:
    inla_ffx_samples        (n_ds, d_max, S)
    inla_sigma_rfx_samples  (n_ds, q_max, S)
    inla_rfx_samples        (n_ds, q_max, m_max, S)

Usage (from repo root):
    uv run python -m metabeta.simulation.inla --size small --family 1 --ds_type sampled --idx 0
    uv run python -m metabeta.simulation.inla --size small --family 1 --ds_type sampled --reintegrate
    uv run python -m metabeta.simulation.inla --size small --family 1 --ds_type sampled --idx 0 --samples 1000
    uv run python -m metabeta.simulation.fit  --method inla --size small --family 1 --ds_type sampled --idx 0
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as _queue
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from metabeta.utils.io import datasetFilename
from metabeta.utils.padding import aggregate, unpad
from metabeta.utils.templates import setupConfigParser, generateSimulationConfig

INLA_DEFAULT_TIMEOUT_S = 120

_DEFAULT_SRCDIR = Path(__file__).resolve().parent.parent / 'outputs' / 'data'

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    _rinla = importr('INLA')
    _rbase = importr('base')
    _HAS_INLA = True
except Exception:
    _HAS_INLA = False


# ---------------------------------------------------------------------------
# Core INLA call
# ---------------------------------------------------------------------------


def _sigma_from_marginal(marg_hyper: object, name: str) -> float:
    """Compute E[1/sqrt(τ)] via numerical integration over an INLA precision marginal."""
    marg = marg_hyper.rx2(name)
    tau_v = np.array(marg.rx(True, 1)).ravel()
    den_v = np.array(marg.rx(True, 2)).ravel()
    return float(np.trapezoid(den_v / np.sqrt(np.maximum(tau_v, 1e-12)), tau_v))


def _draw_posterior_samples(
    result: object,
    n_samples: int,
    d: int,
    q: int,
    m: int,
    correlated: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Draw joint posterior samples via inla.posterior.sample.

    Returns (ffx_s, sigma_s, rfx_s) with shapes (S, d), (S, q), (S, q, m),
    or None if sampling fails.
    """
    try:
        r_ips = ro.r['inla.posterior.sample']
        samps = r_ips(n_samples, result)

        first = samps.rx2(1)
        # $latent is a FloatMatrix (n_latent, 1); rownames hold the parameter names
        latent_names = list(ro.r['rownames'](first.rx2('latent')))
        hyper_names = list(ro.r['names'](first.rx2('hyperpar')))
        n_latent = len(latent_names)
        n_hyper = len(hyper_names)

        latent_mat = np.empty((n_samples, n_latent))
        hyper_mat = np.empty((n_samples, n_hyper))
        for i in range(n_samples):
            samp = samps.rx2(i + 1)
            latent_mat[i] = np.asarray(samp.rx2('latent')).ravel()
            hyper_mat[i] = np.asarray(samp.rx2('hyperpar')).ravel()

        fe_names = ['(Intercept):1'] + [f'x{j}:1' for j in range(1, d)]
        fe_idx = [latent_names.index(nm) for nm in fe_names if nm in latent_names]
        ffx_s = np.zeros((n_samples, d))
        ffx_s[:, : len(fe_idx)] = latent_mat[:, fe_idx]

        sigma_s = np.zeros((n_samples, q))
        rfx_s = np.zeros((n_samples, q, m))

        if correlated:
            for j in range(q):
                for k in range(m):
                    nm = f'idx0:{j * m + k + 1}'
                    if nm in latent_names:
                        rfx_s[:, j, k] = latent_mat[:, latent_names.index(nm)]
                prec_nms = [
                    nm for nm in hyper_names
                    if 'idx0' in nm and f'component {j + 1}' in nm and 'Precision' in nm
                ]
                if prec_nms:
                    prec = hyper_mat[:, hyper_names.index(prec_nms[0])]
                    sigma_s[:, j] = 1.0 / np.sqrt(np.maximum(prec, 1e-12))
        else:
            for j in range(q):
                for k in range(m):
                    nm = f'group{j}:{k + 1}'
                    if nm in latent_names:
                        rfx_s[:, j, k] = latent_mat[:, latent_names.index(nm)]
                prec_nms = [
                    nm for nm in hyper_names
                    if f'group{j}' in nm and 'Precision' in nm
                ]
                if prec_nms:
                    prec = hyper_mat[:, hyper_names.index(prec_nms[0])]
                    sigma_s[:, j] = 1.0 / np.sqrt(np.maximum(prec, 1e-12))

        return ffx_s, sigma_s, rfx_s
    except Exception:
        return None


def estimate(
    ds: dict,
    likelihood_family: int,
    re_correlation: str = 'auto',
    n_samples: int = 0,
) -> dict | None:
    """Run R-INLA on a flat (unpadded) dataset using the simulation's true priors.

    Accepts flat datasets as returned by ``InlaFitter._getSingle`` (where
    ``X (n, d)``, ``y (n,)``, ``groups (n,)``) or by ``_flatten`` in
    ``glmm_inla_comparison.py`` (which provides ``Z`` separately).
    If ``ds['Z']`` is absent, ``Z = ds['X'][:, :q]`` is used.

    Uncorrelated (eta_rfx == 0 or q == 1): one iid term per RE dimension with
    PC prior P(sigma_j > tau_rfx[j]) = 0.317.

    Correlated (eta_rfx > 0 and q == 2): iid2d model with a Wishart prior
    parameterised to match HalfNormal(tau_rfx[j]) marginals; second dimension
    added via a copy term.  For q > 2 correlated: falls back to independent iid.

    FE prior: Normal(nu_ffx[j], tau_ffx[j]^2) via control.fixed.

    Returns dict with 'beta' (d,), 'sigma_rfx' (q,), 'blups' (m, q), or None.
    """
    if not _HAS_INLA:
        return None

    d, m, q = int(ds['d']), int(ds['m']), int(ds['q'])
    X = ds['X']
    Z = ds.get('Z', X[:, :q])
    y, groups = ds['y'], ds['groups']
    tau_rfx = ds.get('tau_rfx')
    tau_ffx = ds.get('tau_ffx')
    nu_ffx = ds.get('nu_ffx')
    tau_eps = ds.get('tau_eps')
    eta_rfx = ds.get('eta_rfx', 0.0)
    n = len(y)
    if n == 0 or m < 2:
        return None

    family_str = {0: 'gaussian', 1: 'binomial', 2: 'poisson'}.get(likelihood_family, 'gaussian')
    correlated = eta_rfx is not None and float(eta_rfx) > 0 and q == 2
    if re_correlation == 'diagonal':
        correlated = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # ---- Build R data.frame ----
            r_df = {'y': ro.FloatVector(y.astype(float))}
            for j in range(1, d):
                r_df[f'x{j}'] = ro.FloatVector(X[:, j].astype(float))

            fixed_part = ' + '.join(f'x{j}' for j in range(1, d)) if d > 1 else '1'

            if correlated:
                nu = q + 1
                t0 = float(tau_rfx[0]) if tau_rfx is not None and tau_rfx[0] > 0 else 1.0
                t1 = float(tau_rfx[1]) if tau_rfx is not None and tau_rfx[1] > 0 else 1.0
                V00 = 1.0 / (nu * t0**2)
                V11 = 1.0 / (nu * t1**2)
                wish = (
                    f"hyper=list(theta=list(prior='wishart2d',"
                    f' param=c({nu},{V00:.8f},0,{V11:.8f})))'
                )
                r_df['idx0'] = ro.IntVector((groups + 1).astype(int))
                r_df['idx1'] = ro.IntVector((groups + m + 1).astype(int))
                r_df['w0'] = ro.FloatVector(Z[:, 0].astype(float))
                r_df['w1'] = ro.FloatVector(Z[:, 1].astype(float))
                df = _rbase.as_data_frame(ro.ListVector(r_df))
                formula_str = (
                    f'y ~ {fixed_part} + '
                    f"f(idx0, w0, model='iid2d', n={2*m}, {wish}) + "
                    f"f(idx1, w1, copy='idx0', fixed=TRUE)"
                )
                formula = ro.Formula(formula_str)
                formula.environment['y'] = df.rx2('y')
                for j in range(1, d):
                    formula.environment[f'x{j}'] = df.rx2(f'x{j}')
                for k in ['idx0', 'idx1', 'w0', 'w1']:
                    formula.environment[k] = df.rx2(k)
            else:
                for j in range(q):
                    r_df[f'group{j}'] = ro.IntVector((groups + 1).astype(int))
                    if not np.allclose(Z[:, j], 1.0):
                        r_df[f'z{j}'] = ro.FloatVector(Z[:, j].astype(float))
                df = _rbase.as_data_frame(ro.ListVector(r_df))
                re_parts = []
                for j in range(q):
                    tau_j = float(tau_rfx[j]) if tau_rfx is not None and tau_rfx[j] > 0 else 1.0
                    pc = f"hyper=list(prec=list(prior='pc.prec', param=c({tau_j:.6f}, 0.317)))"
                    if np.allclose(Z[:, j], 1.0):
                        re_parts.append(f"f(group{j}, model='iid', {pc})")
                    else:
                        re_parts.append(f"f(group{j}, z{j}, model='iid', {pc})")
                formula_str = 'y ~ ' + fixed_part
                if re_parts:
                    formula_str += ' + ' + ' + '.join(re_parts)
                formula = ro.Formula(formula_str)
                formula.environment['y'] = df.rx2('y')
                for j in range(1, d):
                    formula.environment[f'x{j}'] = df.rx2(f'x{j}')
                for j in range(q):
                    formula.environment[f'group{j}'] = df.rx2(f'group{j}')
                    if not np.allclose(Z[:, j], 1.0):
                        formula.environment[f'z{j}'] = df.rx2(f'z{j}')

            # ---- FE prior ----
            fe_names = ['(Intercept)'] + [f'x{j}' for j in range(1, d)]
            if tau_ffx is not None and nu_ffx is not None:
                mean_list = {nm: float(nu_ffx[i]) for i, nm in enumerate(fe_names)}
                prec_list = {
                    nm: float(1.0 / max(tau_ffx[i] ** 2, 1e-8)) for i, nm in enumerate(fe_names)
                }
                ctrl_fixed = ro.ListVector(
                    {'mean': ro.ListVector(mean_list), 'prec': ro.ListVector(prec_list)}
                )
            else:
                ctrl_fixed = ro.ListVector({'mean': 0, 'prec': 0.001})

            inla_kwargs = {
                'formula': formula,
                'family': family_str,
                'data': df,
                'control.compute': ro.ListVector({'config': bool(n_samples > 0), 'return.marginals': True}),
                'control.predictor': ro.ListVector({'compute': False}),
                'control.fixed': ctrl_fixed,
                'verbose': False,
                'silent': True,
            }
            if likelihood_family == 1:
                inla_kwargs['Ntrials'] = ro.IntVector([1] * n)
            elif likelihood_family == 0 and tau_eps is not None and float(tau_eps) > 0:
                inla_kwargs['control.family'] = ro.ListVector(
                    {
                        'hyper': ro.ListVector(
                            {
                                'prec': ro.ListVector(
                                    {
                                        'prior': 'pc.prec',
                                        'param': ro.FloatVector([float(tau_eps), 0.317]),
                                    }
                                )
                            }
                        )
                    }
                )

            result = _rinla.inla(**inla_kwargs)

        # ---- Extract results ----
        sf = result.rx2('summary.fixed')
        beta = np.array(sf.rx(True, 'mean')).ravel()  # (d,)

        marg_hyper = result.rx2('marginals.hyperpar')
        hyper_names = list(marg_hyper.names)
        sr = result.rx2('summary.random')

        sigma_rfx = np.zeros(q)
        blups = np.zeros((m, q))

        if correlated:
            for j in range(q):
                comp_nm = [
                    nm
                    for nm in hyper_names
                    if 'idx0' in nm and f'component {j+1}' in nm and 'Precision' in nm
                ]
                if comp_nm:
                    sigma_rfx[j] = _sigma_from_marginal(marg_hyper, comp_nm[0])
                else:
                    sh = result.rx2('summary.hyperpar')
                    sigma_rfx[j] = 1.0 / np.sqrt(max(float(sh.rx(j + 1, 'mean')[0]), 1e-12))
            re_all = sr.rx2('idx0')
            means_all = np.array(re_all.rx(True, 'mean')).ravel()
            blups[:, 0] = means_all[:m]
            blups[:, 1] = means_all[m : 2 * m]
        else:
            for j in range(q):
                matched = [nm for nm in hyper_names if f'group{j}' in nm and 'Precision' in nm]
                if matched:
                    sigma_rfx[j] = _sigma_from_marginal(marg_hyper, matched[0])
                else:
                    sh = result.rx2('summary.hyperpar')
                    sigma_rfx[j] = 1.0 / np.sqrt(max(float(sh.rx(j + 1, 'mean')[0]), 1e-12))
                re_j = sr.rx2(f'group{j}')
                means_j = np.array(re_j.rx(True, 'mean')).ravel()
                blups[: len(means_j), j] = means_j[:m]

        out = {'beta': beta, 'sigma_rfx': sigma_rfx, 'blups': blups}
        if n_samples > 0:
            drawn = _draw_posterior_samples(result, n_samples, d, q, m, correlated)
            if drawn is not None:
                ffx_s, sigma_s, rfx_s = drawn
                out['ffx_samples'] = ffx_s      # (S, d)
                out['sigma_rfx_samples'] = sigma_s  # (S, q)
                out['rfx_samples'] = rfx_s      # (S, q, m)
        return out

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Persistent worker subprocess (process-level timeout)
# ---------------------------------------------------------------------------


def _worker_loop(task_q: mp.Queue, result_q: mp.Queue) -> None:
    """Run in a child process: receive tasks, call estimate, put results back."""
    while True:
        task = task_q.get()
        if task is None:
            break
        ds, likelihood_family, re_correlation, n_samples = task
        result_q.put(estimate(ds, likelihood_family, re_correlation, n_samples))


class Worker:
    """Persistent subprocess for R-INLA with process-level timeout.

    SIGALRM cannot interrupt blocking rpy2/R C calls, so we fork a worker
    process and use result_q.get(timeout=...) to enforce the deadline.
    On timeout, we terminate/kill the worker and spawn a fresh one.
    """

    def __init__(self, timeout_s: int = INLA_DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s
        self._ctx = mp.get_context('fork')
        self._task_q: mp.Queue | None = None
        self._result_q: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._spawn()

    def _spawn(self) -> None:
        self._task_q = self._ctx.Queue()
        self._result_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_loop,
            args=(self._task_q, self._result_q),
            daemon=True,
        )
        self._proc.start()

    def _kill(self) -> None:
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(1)

    def estimate(
        self,
        ds: dict,
        likelihood_family: int,
        re_correlation: str = 'auto',
        n_samples: int = 0,
    ) -> dict | None:
        self._task_q.put((ds, likelihood_family, re_correlation, n_samples))
        try:
            return self._result_q.get(timeout=self.timeout_s)
        except _queue.Empty:
            d = int(ds.get('d', '?'))
            q = int(ds.get('q', '?'))
            m = int(ds.get('m', '?'))
            n = len(ds.get('y', []))
            print(
                f'\nINLA timeout after {self.timeout_s}s'
                f' (d={d}, q={q}, m={m}, n={n}) — skipping, restarting worker',
                file=sys.stderr,
                flush=True,
            )
            self._kill()
            self._spawn()
            return None

    def close(self) -> None:
        try:
            self._task_q.put(None)
            self._proc.join(5)
        except Exception:
            pass
        self._kill()


# ---------------------------------------------------------------------------
# InlaFitter — mirrors fit.py's Fitter API
# ---------------------------------------------------------------------------


class InlaFitter:
    """Fit a batch of datasets with R-INLA, mirroring the Fitter API in fit.py.

    Individual per-dataset fits are saved to <data_id>/fits/ and then aggregated
    back into <data_id>/<partition>.fit.npz via reintegrate().
    """

    def __init__(
        self,
        cfg: argparse.Namespace,
        srcdir: Path = _DEFAULT_SRCDIR,
    ) -> None:
        self.cfg = cfg
        self.srcdir = Path(srcdir)
        self.outdir = self.srcdir / cfg.data_id / 'fits'
        self.outdir.mkdir(parents=True, exist_ok=True)

        epoch = getattr(cfg, 'epoch', 1)
        self.fname = datasetFilename(partition=cfg.partition, epoch=epoch)
        self.batch_path = self.srcdir / cfg.data_id / self.fname
        assert self.batch_path.exists(), f'{self.batch_path} does not exist'

        with np.load(self.batch_path, allow_pickle=True) as raw:
            self.batch = dict(raw)

        self.likelihood_family = int(
            np.asarray(self.batch.get('likelihood_family', [0])).ravel()[0]
        )
        assert 0 <= cfg.idx < len(self), 'idx out of bounds'
        self.outpath = self.outdir / self._outname(cfg.idx)

    def __len__(self) -> int:
        return len(self.batch['y'])

    def _outname(self, idx: int) -> str:
        return f'{self.batch_path.stem}_inla_{idx:03d}.npz'

    def _getSingle(self, idx: int) -> dict:
        ds = {k: v[idx] for k, v in self.batch.items()}
        sizes = {'d': int(ds['d']), 'q': int(ds['q']), 'm': int(ds['m']), 'n': int(ds['n'])}
        return unpad(ds, sizes)

    def go(self) -> None:
        ds = self._getSingle(self.cfg.idx)
        re_correlation = getattr(self.cfg, 're_correlation', 'auto')
        timeout_s = getattr(self.cfg, 'timeout_s', INLA_DEFAULT_TIMEOUT_S)
        n_samples = getattr(self.cfg, 'n_samples', 0)

        worker = Worker(timeout_s)
        t0 = time.perf_counter()
        result = worker.estimate(ds, self.likelihood_family, re_correlation, n_samples)
        wall_s = time.perf_counter() - t0
        worker.close()

        d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
        if result is not None:
            out = {
                'inla_ffx': result['beta'].astype(np.float64),
                'inla_sigma_rfx': result['sigma_rfx'].astype(np.float64),
                'inla_rfx': result['blups'].astype(np.float64),
                'inla_wall_s': np.array(wall_s, dtype=np.float64),
                'inla_failed': np.array(False),
            }
            if n_samples > 0:
                if 'ffx_samples' in result:
                    # transpose to (dim, S) convention matching nuts/advi storage
                    out['inla_ffx_samples'] = result['ffx_samples'].T.astype(np.float64)
                    out['inla_sigma_rfx_samples'] = result['sigma_rfx_samples'].T.astype(np.float64)
                    out['inla_rfx_samples'] = result['rfx_samples'].transpose(1, 2, 0).astype(np.float64)
                else:
                    out['inla_ffx_samples'] = np.full((d, n_samples), np.nan, dtype=np.float64)
                    out['inla_sigma_rfx_samples'] = np.full((q, n_samples), np.nan, dtype=np.float64)
                    out['inla_rfx_samples'] = np.full((q, m, n_samples), np.nan, dtype=np.float64)
        else:
            out = {
                'inla_ffx': np.full(d, np.nan, dtype=np.float64),
                'inla_sigma_rfx': np.full(q, np.nan, dtype=np.float64),
                'inla_rfx': np.full((m, q), np.nan, dtype=np.float64),
                'inla_wall_s': np.array(wall_s, dtype=np.float64),
                'inla_failed': np.array(True),
            }
            if n_samples > 0:
                out['inla_ffx_samples'] = np.full((d, n_samples), np.nan, dtype=np.float64)
                out['inla_sigma_rfx_samples'] = np.full((q, n_samples), np.nan, dtype=np.float64)
                out['inla_rfx_samples'] = np.full((q, m, n_samples), np.nan, dtype=np.float64)

        np.savez_compressed(self.outpath, **out)
        status = 'OK' if result is not None else 'FAILED'
        print(f'[{status}] idx={self.cfg.idx}  wall={wall_s:.1f}s  → {self.outpath}')

    def _aggregate(self) -> dict:
        paths = [self.outdir / self._outname(i) for i in range(len(self))]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f'Cannot aggregate: {len(missing)} fit file(s) missing, e.g. {missing[0]}'
            )
        fits = []
        for p in paths:
            with np.load(p, allow_pickle=True) as f:
                fits.append(dict(f))
        return aggregate(fits)

    def reintegrate(self) -> None:
        inla_data = self._aggregate()
        fit_path = self.batch_path.with_suffix('.fit.npz')
        if fit_path.exists():
            with np.load(fit_path, allow_pickle=True) as f:
                merged = dict(f)
        else:
            merged = dict(self.batch)
        merged.update(inla_data)
        np.savez_compressed(fit_path, **merged, allow_pickle=True)
        n_ok = int(np.sum(~inla_data['inla_failed'].astype(bool)))
        print(f'Reintegrated INLA fits into {fit_path}  ({n_ok}/{len(self)} OK)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data (template-based, matching fit.py)
    parser.add_argument('--size', type=str, default='small', help='Size preset: tiny|small|medium|large|huge')
    parser.add_argument('--family', type=int, default=0, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type', type=str, default='sampled', help='Dataset type: toy|flat|scm|mixed|sampled|observed')
    parser.add_argument('--config', type=str, help='Path to a saved config.yaml; explicit CLI args override its values')

    parser.add_argument('--idx', type=int, default=0, help='Dataset index to fit (default=0)')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch number (required when --partition train)')
    parser.add_argument('--samples', dest='n_samples', type=int, default=0,
                        help='Draw S joint posterior samples via inla.posterior.sample (default: 0 = point estimates only)')
    parser.add_argument('--reintegrate', action='store_true',
                        help='Aggregate individual fit files back into the batch .fit.npz')
    parser.add_argument('--re-correlation', dest='re_correlation', default='diagonal',
                        choices=['auto', 'diagonal'],
                        help='RE correlation: diagonal forces iid per dim (default: diagonal)')
    parser.add_argument('--timeout', dest='timeout_s', type=int, default=INLA_DEFAULT_TIMEOUT_S,
                        help=f'Per-dataset timeout in seconds (default: {INLA_DEFAULT_TIMEOUT_S})')
    return setupConfigParser(parser, generateSimulationConfig, 'Fit hierarchical datasets with R-INLA.')
# fmt: on


if __name__ == '__main__':
    cfg = setup()
    for _k, _v in [
        ('idx', 0), ('reintegrate', False), ('partition', 'test'),
        ('epoch', None), ('n_samples', 0), ('re_correlation', 'diagonal'),
        ('timeout_s', INLA_DEFAULT_TIMEOUT_S),
    ]:
        if not hasattr(cfg, _k):
            setattr(cfg, _k, _v)

    if cfg.partition == 'train' and cfg.epoch is None:
        print('error: --epoch is required when --partition train', file=sys.stderr)
        sys.exit(1)

    if not _HAS_INLA:
        print('R-INLA not available (rpy2 or INLA package missing).', file=sys.stderr)
        sys.exit(1)

    fitter = InlaFitter(cfg)
    if cfg.reintegrate:
        fitter.reintegrate()
    else:
        fitter.go()
