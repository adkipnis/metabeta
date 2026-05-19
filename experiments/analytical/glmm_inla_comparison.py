"""Experiment: analytical GLMM vs R-INLA on Bernoulli/Normal GLMM datasets.

Compares analytical GLMM summaries against R-INLA on Bernoulli or Normal datasets.
The analytical side can run the raw estimator and the current EB-refined estimator.

RE prior: PC prior P(σ_j > τ_rfx_j) = 0.317 per dimension for uncorrelated
datasets (eta_rfx=0, or q=1).  For correlated datasets (eta_rfx>0, q=2):
iid2d model with Wishart prior W(q+1, V) where V_{jj}=1/((q+1)*τ_rfx_j²),
plus a copy term for the second dimension.
FE prior: N(ν_ffx_j, τ_ffx_j²) via control.fixed.
Normal family residual prior: PC prior P(σ_eps > τ_eps) = 0.317.

Usage (from repo root):
    uv run python -u experiments/analytical/glmm_inla_comparison.py
    uv run python -u experiments/analytical/glmm_inla_comparison.py \\
        --data-ids small-b-sampled,small-n-sampled --n-inla 100
    uv run python -u experiments/analytical/glmm_inla_comparison.py \\
        --data-ids small-n-sampled --n-inla 200 --partition test \\
        --analytical-methods raw,current
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as _queue
import sys
import time
import warnings

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.analytical.fit import glmm
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


ANALYTICAL_METHODS = ('raw', 'current', 'normal_eb', 'normal_beta_grid')

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    _rinla = importr('INLA')
    _rbase = importr('base')
    _HAS_INLA = True
except Exception:
    _HAS_INLA = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INLA_DEFAULT_TIMEOUT_S = 120


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.std(truth))
    return float(np.sqrt(np.mean(err**2))) / max(denom, 1e-8)


def _bias(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


def _parseAnalyticalMethods(value: str) -> list[str]:
    methods = [method.strip().lower() for method in value.split(',') if method.strip()]
    invalid = sorted(set(methods) - set(ANALYTICAL_METHODS))
    if invalid:
        raise ValueError(f'unsupported analytical method(s): {", ".join(invalid)}')
    return methods or ['raw', 'current']


def _methodLabel(method: str, likelihood_family: int) -> str:
    if method == 'raw':
        return 'RAW'
    if method in {'current', 'normal_eb'} and likelihood_family == 1:
        return 'BERNOULLI-EB'
    if method == 'current' and likelihood_family == 0:
        return 'NORMAL-EB-TAIL-BETA'
    if method == 'normal_eb' and likelihood_family == 0:
        return 'NORMAL-EB'
    if method == 'normal_beta_grid' and likelihood_family == 0:
        return 'NORMAL-EB-BETA-GRID'
    if method == 'normal_eb':
        return 'NORMAL-EB'
    if method == 'current' and likelihood_family == 2:
        return 'POISSON-EB-SIGMA-GRID'
    return method.upper()


def _metricLists() -> dict[str, list[np.ndarray] | list[float]]:
    return {
        'be': [],
        'bt': [],
        'se': [],
        'st': [],
        're': [],
        'rt': [],
        'wall': [],
    }


def _flat(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values) if values else np.array([np.nan])


def _sliceBatch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    out = {}
    B = batch['X'].shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (B,):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _flatten(batch: dict, b: int, active_d: np.ndarray, active_q: np.ndarray) -> dict:
    """Reconstruct flat (n, d) + (n, q) dataset from item b in a grouped batch."""
    m = int(batch['m'][b].item())
    d = len(active_d)
    q = len(active_q)
    X_parts, y_parts, g_parts, Z_parts = [], [], [], []
    for g in range(m):
        ng = int(batch['ns'][b, g].item())
        if ng == 0:
            continue
        X_g = batch['X'][b, g, :ng].cpu().numpy()[:, active_d]
        y_g = batch['y'][b, g, :ng].cpu().numpy()
        Z_g = batch['Z'][b, g, :ng].cpu().numpy()[:, active_q]
        X_parts.append(X_g)
        y_parts.append(y_g)
        g_parts.append(np.full(ng, g, dtype=int))
        Z_parts.append(Z_g)

    # Prior parameters for this dataset.
    tau_rfx_np = batch['tau_rfx'][b, active_q].cpu().numpy() if 'tau_rfx' in batch else None
    tau_ffx_np = batch['tau_ffx'][b, active_d].cpu().numpy() if 'tau_ffx' in batch else None
    nu_ffx_np = batch['nu_ffx'][b, active_d].cpu().numpy() if 'nu_ffx' in batch else None
    tau_eps_np = (
        float(batch['tau_eps'][b].item())
        if 'tau_eps' in batch and batch['tau_eps'] is not None
        else None
    )
    # eta_rfx > 0 means Ψ is full (correlated); == 0 means diagonal.
    eta_rfx_b = (
        float(batch['eta_rfx'][b].item())
        if 'eta_rfx' in batch and batch['eta_rfx'] is not None
        else 0.0
    )

    base = {
        'm': m,
        'd': d,
        'q': q,
        'tau_rfx': tau_rfx_np,
        'tau_ffx': tau_ffx_np,
        'nu_ffx': nu_ffx_np,
        'tau_eps': tau_eps_np,
        'eta_rfx': eta_rfx_b,
    }
    if not X_parts:
        return {
            **base,
            'X': np.zeros((0, d)),
            'Z': np.zeros((0, q)),
            'y': np.zeros(0),
            'groups': np.zeros(0, dtype=int),
        }
    return {
        **base,
        'X': np.vstack(X_parts),
        'Z': np.vstack(Z_parts),
        'y': np.concatenate(y_parts),
        'groups': np.concatenate(g_parts),
    }


def _sigma_from_marginal(marg_hyper: object, name: str) -> float:
    """Compute E[1/sqrt(τ)] via numerical integration over an INLA precision marginal."""
    marg = marg_hyper.rx2(name)
    tau_v = np.array(marg.rx(True, 1)).ravel()
    den_v = np.array(marg.rx(True, 2)).ravel()
    return float(np.trapezoid(den_v / np.sqrt(np.maximum(tau_v, 1e-12)), tau_v))


def _inla_estimate(
    ds_flat: dict,
    likelihood_family: int,
    re_correlation: str = 'auto',
) -> dict | None:
    """Run R-INLA on a flat dataset using the simulation's true priors.

    Uncorrelated (eta_rfx == 0 or q == 1): one iid term per RE dimension with
    PC prior P(sigma_j > tau_rfx[j]) = 0.317.

    Correlated (eta_rfx > 0 and q == 2): iid2d model with a Wishart prior
    parameterised to match HalfNormal(tau_rfx[j]) marginals on each component;
    the second dimension is added via a copy term at index offset m.
    For q > 2 correlated: falls back to independent iid per dimension.

    FE prior: Normal(nu_ffx[j], tau_ffx[j]^2) via control.fixed.

    Returns dict with 'beta' (d,), 'sigma_rfx' (q,), 'blups' (m, q) or None.
    """
    if not _HAS_INLA:
        return None

    d, m, q = ds_flat['d'], ds_flat['m'], ds_flat['q']
    X, Z, y, groups = ds_flat['X'], ds_flat['Z'], ds_flat['y'], ds_flat['groups']
    tau_rfx = ds_flat.get('tau_rfx')
    tau_ffx = ds_flat.get('tau_ffx')
    nu_ffx = ds_flat.get('nu_ffx')
    tau_eps = ds_flat.get('tau_eps')
    eta_rfx = ds_flat.get('eta_rfx', 0.0)
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
                # iid2d: component j uses index g+j*m+1 (1-based).
                # Wishart prior W(nu, V) with nu=q+1 and V_{jj}=1/(nu*tau_rfx[j]^2)
                # gives E[Q_{jj}] = nu*V_{jj} = 1/tau_rfx[j]^2, matching HalfNormal.
                nu = q + 1  # minimum df = 3 for q=2
                t0 = float(tau_rfx[0]) if tau_rfx is not None and tau_rfx[0] > 0 else 1.0
                t1 = float(tau_rfx[1]) if tau_rfx is not None and tau_rfx[1] > 0 else 1.0
                V00 = 1.0 / (nu * t0**2)
                V11 = 1.0 / (nu * t1**2)
                wish = (
                    f"hyper=list(theta=list(prior='wishart2d',"
                    f' param=c({nu},{V00:.8f},0,{V11:.8f})))'
                )
                # idx0 = group+1, idx1 = group+m+1; weights = Z[:,j]
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
                # Independent iid per dimension with PC prior.
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
                'control.compute': ro.ListVector({'config': False, 'return.marginals': True}),
                'control.predictor': ro.ListVector({'compute': False}),
                'control.fixed': ctrl_fixed,
                'verbose': False,
                'silent': True,
            }
            if likelihood_family == 1:
                inla_kwargs['Ntrials'] = ro.IntVector([1] * n)
            elif likelihood_family == 0 and tau_eps is not None and tau_eps > 0:
                inla_kwargs['control.family'] = ro.ListVector(
                    {
                        'hyper': ro.ListVector(
                            {
                                'prec': ro.ListVector(
                                    {
                                        'prior': 'pc.prec',
                                        'param': ro.FloatVector([tau_eps, 0.317]),
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
            # iid2d: hyperpar names "Precision for idx0 (component j+1)".
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
            # BLUPs: summary.random['idx0'] has 2*m rows.
            # Rows 1..m → component 0, rows m+1..2m → component 1.
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

        return {'beta': beta, 'sigma_rfx': sigma_rfx, 'blups': blups}

    except Exception:
        return None


def _inla_worker_loop(task_q: mp.Queue, result_q: mp.Queue) -> None:
    """Run in a child process: receive tasks, run INLA, put results back."""
    while True:
        task = task_q.get()
        if task is None:
            break
        ds_flat, likelihood_family, re_correlation = task
        result = _inla_estimate(ds_flat, likelihood_family, re_correlation)
        result_q.put(result)


class _InlaWorker:
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
            target=_inla_worker_loop,
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
        ds_flat: dict,
        likelihood_family: int,
        re_correlation: str = 'auto',
    ) -> dict | None:
        self._task_q.put((ds_flat, likelihood_family, re_correlation))
        try:
            return self._result_q.get(timeout=self.timeout_s)
        except _queue.Empty:
            print(
                f'\nINLA timeout after {self.timeout_s}s'
                f' (d={ds_flat["d"]}, q={ds_flat["q"]},'
                f' m={ds_flat["m"]}, n={len(ds_flat["y"])}) — skipping, restarting worker',
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


def _breakdown_srfx(err: np.ndarray, truth: np.ndarray, label: str) -> str:
    """Bias/RMSE breakdown by true σ_rfx in four quantile bins."""
    finite = np.isfinite(truth) & np.isfinite(err)
    edges = np.nanpercentile(truth[finite], [0, 25, 50, 75, 100])
    edges = np.unique(np.round(edges, 3))
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = finite & (truth >= lo) & (truth <= (hi if i == len(edges) - 2 else hi - 1e-9))
        if sel.sum() < 2:
            continue
        e = err[sel]
        rows.append(
            [
                f'{lo:.3f}–{hi:.3f}',
                int(sel.sum()),
                f'{float(np.nanmean(e)):+.4f}',
                f'{float(np.sqrt(np.mean(e**2))):.4f}',
            ]
        )
    return tabulate(rows, headers=[f'σ_rfx_true ({label})', 'N', 'Bias', 'RMSE'], tablefmt='simple')


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def run_one_dataset(
    data_id: str,
    partition: str = 'test',
    n_inla: int = 100,
    n_total: int = 0,
    n_epochs: int = 1,
    analytical_methods: list[str] | None = None,
    re_correlation: str = 'auto',
    inla_timeout_s: int = INLA_DEFAULT_TIMEOUT_S,
    device: torch.device | None = None,
) -> dict:
    """Run analytical GLMM methods vs R-INLA on one dataset, return metrics dict."""
    if device is None:
        device = torch.device('cpu')
    if analytical_methods is None:
        analytical_methods = ['raw', 'current']
    data_cfg = loadDataConfig(data_id)
    max_d = data_cfg['max_d']
    max_q = data_cfg['max_q']
    likelihood_family = data_cfg.get('likelihood_family', 0)

    if partition == 'train':
        paths = [dataFilePath(data_cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
        paths = [p for p in paths if p.exists()]
    elif partition == 'test':
        paths = [dataFilePath(data_cfg['data_id'], 'test')]
    else:
        paths = [dataFilePath(data_cfg['data_id'], 'valid')]

    assert paths and paths[0].exists(), f'No data at {paths[0] if paths else data_id}/{partition}'

    all_metrics = {method: _metricLists() for method in analytical_methods}
    matched_metrics = {method: _metricLists() for method in analytical_methods}
    inla_metrics = _metricLists()
    inla_n_tried = inla_n_ok = 0

    inla_worker = _InlaWorker(inla_timeout_s) if _HAS_INLA else None

    ds_bar = tqdm(
        total=n_total if n_total > 0 else None,
        desc=f'{data_id}/{partition} datasets',
        unit='ds',
        file=sys.stderr,
        leave=False,
    )
    inla_bar = tqdm(
        total=n_inla if _HAS_INLA else 0,
        desc='INLA',
        unit='ds',
        file=sys.stderr,
        leave=True,
    )

    with torch.no_grad():
        n_seen = 0
        done = False
        for path in paths:
            if done:
                break
            dl = Dataloader(path, batch_size=32, shuffle=False)
            for batch in dl:
                if n_total > 0 and n_seen >= n_total:
                    done = True
                    break
                if n_total > 0:
                    remaining = n_total - n_seen
                    if batch['X'].shape[0] > remaining:
                        batch = _sliceBatch(batch, remaining)
                batch = toDevice(batch, device)
                B = batch['X'].shape[0]
                Zm = batch['Z'][..., :max_q]

                stats_np = {}
                for method in analytical_methods:
                    if method == 'raw':
                        method_kwargs = {
                            'map_refine': False,
                            'bernoulli_laplace_eb': False,
                            'normal_laplace_eb': False,
                            'poisson_laplace_eb': False,
                        }
                    elif method == 'normal_eb' and likelihood_family == 0:
                        method_kwargs = {
                            'map_refine': True,
                            'bernoulli_laplace_eb': False,
                            'normal_laplace_eb': True,
                            'normal_laplace_eb_sigma_grid_refine': False,
                            'normal_beta_sigma_grid': False,
                            'normal_beta_tail_grid': False,
                        }
                    elif method == 'normal_beta_grid' and likelihood_family == 0:
                        method_kwargs = {
                            'map_refine': True,
                            'bernoulli_laplace_eb': False,
                            'normal_laplace_eb': True,
                            'normal_laplace_eb_sigma_grid_refine': False,
                            'normal_beta_sigma_grid': True,
                            'normal_beta_sigma_grid_scales': (0.75, 1.0, 1.3333333),
                            'normal_beta_tail_grid': False,
                        }
                    else:
                        method_kwargs = {'map_refine': True}
                    t0 = time.perf_counter()
                    stats = glmm(
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        batch['ns'].clamp(min=1).float(),
                        batch['n'].float(),
                        likelihood_family=likelihood_family,
                        eta_rfx=batch.get('eta_rfx'),
                        mask_q=batch.get('mask_q'),
                        nu_ffx=batch.get('nu_ffx'),
                        tau_ffx=batch.get('tau_ffx'),
                        family_ffx=batch.get('family_ffx'),
                        tau_rfx=batch.get('tau_rfx'),
                        family_sigma_rfx=batch.get('family_sigma_rfx'),
                        tau_eps=batch.get('tau_eps'),
                        family_sigma_eps=batch.get('family_sigma_eps'),
                        mask_d=batch.get('mask_d'),
                        **method_kwargs,
                    )
                    elapsed = time.perf_counter() - t0
                    stats_np[method] = {
                        'beta': stats['beta_est'].cpu().numpy(),
                        'srfx': stats['sigma_rfx_est'].cpu().numpy(),
                        'blup': stats['blup_est'].cpu().numpy(),
                        'wall': elapsed / B,
                    }

                ffx_true = batch['ffx'].cpu().numpy()
                srfx_true = batch['sigma_rfx'].cpu().numpy()
                rfx_true = batch['rfx'].cpu().numpy()
                mask_d_np = (
                    batch['mask_d'].cpu().numpy().astype(bool)
                    if 'mask_d' in batch
                    else np.ones((B, max_d), dtype=bool)
                )
                mask_q_np = (
                    batch['mask_q'].cpu().numpy().astype(bool)
                    if 'mask_q' in batch
                    else np.ones((B, max_q), dtype=bool)
                )
                m_np = batch['m'].cpu().numpy()

                for b in range(B):
                    active_d = np.flatnonzero(mask_d_np[b])
                    active_q = np.flatnonzero(mask_q_np[b])
                    m_b = int(m_np[b])
                    re_t = rfx_true[b, :m_b][:, active_q].reshape(-1)

                    per_method_errors = {}
                    for method in analytical_methods:
                        be = stats_np[method]['beta'][b, active_d] - ffx_true[b, active_d]
                        se = stats_np[method]['srfx'][b, active_q] - srfx_true[b, active_q]
                        re_e = (
                            stats_np[method]['blup'][b, :m_b][:, active_q]
                            - rfx_true[b, :m_b][:, active_q]
                        ).reshape(-1)
                        per_method_errors[method] = {'be': be, 'se': se, 're': re_e}
                        store = all_metrics[method]
                        store['be'].append(be)
                        store['bt'].append(ffx_true[b, active_d])
                        store['se'].append(se)
                        store['st'].append(srfx_true[b, active_q])
                        store['re'].append(re_e)
                        store['rt'].append(re_t)
                        store['wall'].append(stats_np[method]['wall'])

                    n_seen += 1
                    ds_bar.update(1)

                    if not _HAS_INLA or inla_n_tried >= n_inla:
                        continue

                    inla_n_tried += 1
                    ds_flat = _flatten(batch, b, active_d, active_q)
                    t_i = time.perf_counter()
                    est = inla_worker.estimate(ds_flat, likelihood_family, re_correlation)
                    inla_elapsed = time.perf_counter() - t_i
                    inla_metrics['wall'].append(inla_elapsed)
                    inla_bar.set_postfix(ok=inla_n_ok, s=f'{inla_elapsed:.1f}')
                    inla_bar.update(1)

                    if est is None:
                        continue
                    inla_n_ok += 1
                    for method in analytical_methods:
                        store = matched_metrics[method]
                        store['be'].append(per_method_errors[method]['be'])
                        store['bt'].append(ffx_true[b, active_d])
                        store['se'].append(per_method_errors[method]['se'])
                        store['st'].append(srfx_true[b, active_q])
                        store['re'].append(per_method_errors[method]['re'])
                        store['rt'].append(re_t)
                    inla_metrics['be'].append(est['beta'] - ffx_true[b, active_d])
                    inla_metrics['bt'].append(ffx_true[b, active_d])
                    inla_metrics['se'].append(est['sigma_rfx'] - srfx_true[b, active_q])
                    inla_metrics['st'].append(srfx_true[b, active_q])
                    inla_metrics['re'].append(
                        (est['blups'][:m_b] - rfx_true[b, :m_b][:, active_q]).reshape(-1)
                    )
                    inla_metrics['rt'].append(re_t)

    ds_bar.close()
    inla_bar.close()
    if inla_worker:
        inla_worker.close()
    all_flat = {
        method: {key: _flat(values) for key, values in store.items() if key != 'wall'}
        for method, store in all_metrics.items()
    }
    matched_flat = {
        method: {key: _flat(values) for key, values in store.items() if key != 'wall'}
        for method, store in matched_metrics.items()
    }
    inla = {key: _flat(values) for key, values in inla_metrics.items() if key != 'wall'}

    n_all = len(all_metrics[analytical_methods[0]]['be']) if analytical_methods else 0
    metrics = {
        'data_id': data_id,
        'n_analytical': n_all,
        'n_inla': inla_n_ok,
        'analytical_methods': analytical_methods,
        'inla_ffx': _nrmse(inla['be'], inla['bt']) if inla_n_ok > 0 else float('nan'),
        'inla_srfx': _nrmse(inla['se'], inla['st']) if inla_n_ok > 0 else float('nan'),
        'inla_blup': _nrmse(inla['re'], inla['rt']) if inla_n_ok > 0 else float('nan'),
    }
    for method in analytical_methods:
        vals = all_flat[method]
        metrics[f'{method}_ffx'] = _nrmse(vals['be'], vals['bt'])
        metrics[f'{method}_srfx'] = _nrmse(vals['se'], vals['st'])
        metrics[f'{method}_blup'] = _nrmse(vals['re'], vals['rt'])
        method_wall = np.array(all_metrics[method]['wall'])
        metrics[f'{method}_wall_ms'] = float(method_wall.mean() * 1000.0)
    inla_wall = np.array(inla_metrics['wall'])
    metrics['inla_wall_s'] = float(inla_wall.mean()) if inla_wall.size else float('nan')

    sep = '=' * 70
    print(sep)
    print(f'  {data_id}  |  partition={partition}  N={n_all}  family={likelihood_family}')
    print(sep)

    family_label = {0: 'Normal', 1: 'Bernoulli', 2: 'Poisson'}.get(
        likelihood_family, str(likelihood_family)
    )
    print(f'Family: {family_label}')

    for method in analytical_methods:
        label = _methodLabel(method, likelihood_family)
        vals = all_flat[method]
        print(f'\n{label} — all q (N={n_all})')
        print(
            tabulate(
                [
                    ['FFX (β)', f'{metrics[f"{method}_ffx"]:.4f}', f'{_bias(vals["be"]):+.4f}'],
                    ['σ_rfx', f'{metrics[f"{method}_srfx"]:.4f}', f'{_bias(vals["se"]):+.4f}'],
                    ['BLUP', f'{metrics[f"{method}_blup"]:.4f}', f'{_bias(vals["re"]):+.4f}'],
                ],
                headers=['Parameter', 'NRMSE', 'Bias'],
                tablefmt='simple',
            )
        )

    if inla_n_ok > 0:
        print(
            f'\nAnalytical vs R-INLA — matched'
            f' (N={len(matched_metrics[analytical_methods[0]]["be"])},'
            f' INLA {inla_n_ok}/{inla_n_tried})'
        )
        rows = []
        for method in analytical_methods:
            label = _methodLabel(method, likelihood_family)
            vals = matched_flat[method]
            rows.extend(
                [
                    [
                        label,
                        'FFX (β)',
                        f'{_nrmse(vals["be"], vals["bt"]):.4f}',
                        f'{_bias(vals["be"]):+.4f}',
                    ],
                    [
                        label,
                        'σ_rfx',
                        f'{_nrmse(vals["se"], vals["st"]):.4f}',
                        f'{_bias(vals["se"]):+.4f}',
                    ],
                    [
                        label,
                        'BLUP',
                        f'{_nrmse(vals["re"], vals["rt"]):.4f}',
                        f'{_bias(vals["re"]):+.4f}',
                    ],
                ]
            )
        rows.extend(
            [
                ['R-INLA', 'FFX (β)', f'{metrics["inla_ffx"]:.4f}', f'{_bias(inla["be"]):+.4f}'],
                ['R-INLA', 'σ_rfx', f'{metrics["inla_srfx"]:.4f}', f'{_bias(inla["se"]):+.4f}'],
                ['R-INLA', 'BLUP', f'{metrics["inla_blup"]:.4f}', f'{_bias(inla["re"]):+.4f}'],
            ]
        )
        print(tabulate(rows, headers=['Method', 'Parameter', 'NRMSE', 'Bias'], tablefmt='simple'))

    for method in analytical_methods:
        label = _methodLabel(method, likelihood_family)
        vals = all_flat[method]
        print(f'\nσ_rfx bias by true σ_rfx bin — {label}')
        print(_breakdown_srfx(vals['se'], vals['st'], label))
    if inla_n_ok > 0:
        print('\nσ_rfx bias by true σ_rfx bin — R-INLA')
        print(_breakdown_srfx(inla['se'], inla['st'], 'R-INLA'))

    print()
    for method in analytical_methods:
        label = _methodLabel(method, likelihood_family)
        w = np.array(all_metrics[method]['wall'])
        print(
            f'Wall time — {label}: mean={w.mean()*1000:.2f} ms  '
            f'median={np.median(w)*1000:.2f} ms/ds'
        )
    if inla_metrics['wall']:
        w = np.array(inla_metrics['wall'])
        print(f'Wall time — INLA: mean={w.mean():.3f} s  median={np.median(w):.3f} s/ds')
    print()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_ids: list[str],
    partition: str = 'test',
    n_epochs: int = 1,
    n_inla: int = 100,
    n_total: int = 0,
    analytical_methods: list[str] | None = None,
    re_correlation: str = 'auto',
    inla_timeout_s: int = INLA_DEFAULT_TIMEOUT_S,
) -> None:
    if analytical_methods is None:
        analytical_methods = ['raw', 'current']
    print(
        f'R-INLA: {"enabled" if _HAS_INLA else "DISABLED"}   limit={n_inla}'
        + (f'   n_total={n_total}' if n_total > 0 else '')
    )
    print(f'Analytical methods: {", ".join(analytical_methods)}')
    print(f'R-INLA random-effects correlation: {re_correlation}')
    print(f'R-INLA per-dataset timeout: {inla_timeout_s}s')
    print()

    all_metrics = []
    for data_id in data_ids:
        all_metrics.append(
            run_one_dataset(
                data_id=data_id,
                partition=partition,
                n_inla=n_inla,
                n_total=n_total,
                n_epochs=n_epochs,
                analytical_methods=analytical_methods,
                re_correlation=re_correlation,
                inla_timeout_s=inla_timeout_s,
            )
        )

    if len(all_metrics) > 1:
        print('=' * 70)
        print('  SUMMARY — NRMSE across datasets  (F=FFX, S=σ_rfx, B=BLUP)')
        print('=' * 70)
        rows = []
        for m in all_metrics:

            def fmt(v: float) -> str:
                return f'{v:.4f}' if not np.isnan(v) else '—'

            rows.append(
                [m['data_id'], m['n_analytical']]
                + [
                    fmt(m[f'{method}_{suffix}'])
                    for method in analytical_methods
                    for suffix in ['ffx', 'srfx', 'blup']
                ]
                + [fmt(m['inla_ffx']), fmt(m['inla_srfx']), fmt(m['inla_blup'])]
            )
        method_headers = [
            f'{method.upper()}-{short}'
            for method in analytical_methods
            for short in ['F', 'S', 'B']
        ]
        print(
            tabulate(
                rows,
                headers=['Dataset', 'N'] + method_headers + ['INLA-F', 'INLA-S', 'INLA-B'],
                tablefmt='simple',
            )
        )
        time_rows = []
        for m in all_metrics:
            time_rows.append(
                [m['data_id']]
                + [f'{m[f"{method}_wall_ms"]:.2f}' for method in analytical_methods]
                + [fmt(m['inla_wall_s'])]
            )
        print()
        print('  SUMMARY — wall time per dataset')
        print(
            tabulate(
                time_rows,
                headers=['Dataset']
                + [f'{method.upper()} ms' for method in analytical_methods]
                + ['INLA s'],
                tablefmt='simple',
            )
        )


if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description='Analytical GLMM vs R-INLA comparison')
    parser.add_argument('--data-ids',  default='small-b-sampled',
                        help='comma-separated data config ids (default: small-b-sampled)')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-epochs',  default=1,   type=int)
    parser.add_argument('--n-inla',    default=100, type=int, help='max datasets for INLA per data_id')
    parser.add_argument('--n-total',   default=0,   type=int, help='cap total datasets per data_id (0=all)')
    parser.add_argument('--analytical-methods', default='raw,current',
                        help='comma-separated analytical methods: raw,current,normal_eb,normal_beta_grid')
    parser.add_argument('--re-correlation', default='diagonal',
                        choices=['auto', 'diagonal'],
                        help='R-INLA RE correlation: diagonal forces iid per dim for all families')
    parser.add_argument('--inla-timeout', default=INLA_DEFAULT_TIMEOUT_S, type=int,
                        help=f'per-dataset INLA timeout in seconds (default: {INLA_DEFAULT_TIMEOUT_S})')
    # fmt: on
    a = parser.parse_args()
    main(
        data_ids=[x.strip() for x in a.data_ids.split(',')],
        partition=a.partition,
        n_epochs=a.n_epochs,
        n_inla=a.n_inla,
        n_total=a.n_total,
        analytical_methods=_parseAnalyticalMethods(a.analytical_methods),
        re_correlation=a.re_correlation,
        inla_timeout_s=a.inla_timeout,
    )
