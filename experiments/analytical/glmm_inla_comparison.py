"""Experiment: PQL vs R-INLA on Bernoulli/Normal GLMM datasets.

Compares our PQL estimator (+ P5/P6 refinements) against R-INLA on test
datasets.  INLA supports any q via separate iid random-effect terms per
RE dimension (diagonal Ψ assumed).

For Bernoulli: inla(..., family="binomial", Ntrials=1).
For Normal:    inla(..., family="gaussian").

Prior matching is approximate: INLA uses a log-gamma(1, 5e-5) prior on
precision by default; we use HalfNormal(τ_rfx) on σ.  The comparison is
frequentist — accuracy relative to simulated ground truth.

Usage (from repo root):
    uv run python experiments/analytical/glmm_inla_comparison.py
    uv run python experiments/analytical/glmm_inla_comparison.py \\
        --data-ids small-b-sampled,small-n-sampled --n-inla 100
    uv run python experiments/analytical/glmm_inla_comparison.py \\
        --data-ids small-b-sampled --n-inla 200 --partition test
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import torch
from tabulate import tabulate

from metabeta.analytical.glmm import glmm
from metabeta.analytical.map import refineBernoulliMapBeta, refineBernoulliNagqSrfx
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath

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


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.std(truth))
    return float(np.sqrt(np.mean(err**2))) / max(denom, 1e-8)


def _bias(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


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

    # Extract prior parameters for this dataset (per active dimension).
    # tau_rfx: HalfNormal scale per RE dim. tau_ffx/nu_ffx: Normal scale/mean per FE dim.
    tau_rfx_np = batch['tau_rfx'][b, active_q].cpu().numpy() if 'tau_rfx' in batch else None
    tau_ffx_np = batch['tau_ffx'][b, active_d].cpu().numpy() if 'tau_ffx' in batch else None
    nu_ffx_np = batch['nu_ffx'][b, active_d].cpu().numpy() if 'nu_ffx' in batch else None
    tau_eps_np = (
        float(batch['tau_eps'][b].item())
        if 'tau_eps' in batch and batch['tau_eps'] is not None
        else None
    )

    if not X_parts:
        return {
            'X': np.zeros((0, d)),
            'Z': np.zeros((0, q)),
            'y': np.zeros(0),
            'groups': np.zeros(0, dtype=int),
            'm': m,
            'd': d,
            'q': q,
            'tau_rfx': tau_rfx_np,
            'tau_ffx': tau_ffx_np,
            'nu_ffx': nu_ffx_np,
            'tau_eps': tau_eps_np,
        }
    return {
        'X': np.vstack(X_parts),
        'Z': np.vstack(Z_parts),
        'y': np.concatenate(y_parts),
        'groups': np.concatenate(g_parts),
        'm': m,
        'd': d,
        'q': q,
        'tau_rfx': tau_rfx_np,
        'tau_ffx': tau_ffx_np,
        'nu_ffx': nu_ffx_np,
        'tau_eps': tau_eps_np,
    }


def _inla_estimate(ds_flat: dict, likelihood_family: int) -> dict | None:
    """Run R-INLA on a flat dataset using the simulation's true priors.

    RE prior: HalfNormal(tau_rfx[j]) on sigma_j, expressed as INLA PC prior
    P(sigma_j > tau_rfx[j]) = 0.317 (matches the HalfNormal(scale) mass above scale).

    FE prior: Normal(nu_ffx[j], tau_ffx[j]^2) on beta_j, passed via control.fixed.

    For q>1: each RE dimension gets an independent iid term (diagonal Ψ).

    Returns dict with:
      'beta'      : (d,)   fixed-effect posterior means
      'sigma_rfx' : (q,)   per-dim posterior std devs  E[σ_j] = E[1/sqrt(τ_j)]
      'blups'     : (m, q) per-group RE means
    or None on failure.
    """
    if not _HAS_INLA:
        return None

    d, m, q = ds_flat['d'], ds_flat['m'], ds_flat['q']
    X, Z, y, groups = ds_flat['X'], ds_flat['Z'], ds_flat['y'], ds_flat['groups']
    tau_rfx = ds_flat.get('tau_rfx')   # (q,) HalfNormal scales, or None
    tau_ffx = ds_flat.get('tau_ffx')   # (d,) Normal scales, or None
    nu_ffx = ds_flat.get('nu_ffx')     # (d,) Normal means, or None
    tau_eps = ds_flat.get('tau_eps')   # float HalfNormal scale for eps, or None
    n = len(y)
    if n == 0 or m < 2:
        return None

    family_str = 'binomial' if likelihood_family == 1 else 'gaussian'

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Build R data.frame
            r_df = {'y': ro.FloatVector(y.astype(float))}
            for j in range(1, d):
                r_df[f'x{j}'] = ro.FloatVector(X[:, j].astype(float))
            # Group indices (1-based for R)
            for j in range(q):
                r_df[f'group{j}'] = ro.IntVector((groups + 1).astype(int))
                if not np.allclose(Z[:, j], 1.0):
                    r_df[f'z{j}'] = ro.FloatVector(Z[:, j].astype(float))

            df = _rbase.as_data_frame(ro.ListVector(r_df))

            # Build formula string.
            # RE prior: PC prior P(sigma_j > U_j) = 0.317 where U_j = tau_rfx[j].
            # 0.317 matches P(HalfNormal(scale) > scale) = P(|N(0,1)| > 1).
            fixed_part = ' + '.join(f'x{j}' for j in range(1, d)) if d > 1 else '1'
            re_parts = []
            for j in range(q):
                if tau_rfx is not None and tau_rfx[j] > 0:
                    pc_hyper = (
                        f"hyper=list(prec=list(prior='pc.prec',"
                        f' param=c({tau_rfx[j]:.6f}, 0.317)))'
                    )
                else:
                    pc_hyper = "hyper=list(prec=list(prior='pc.prec', param=c(1, 0.317)))"
                if np.allclose(Z[:, j], 1.0):
                    re_parts.append(f"f(group{j}, model='iid', {pc_hyper})")
                else:
                    re_parts.append(f"f(group{j}, z{j}, model='iid', {pc_hyper})")
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

            # FE prior: N(nu_ffx[j], tau_ffx[j]^2).  INLA control.fixed uses
            # precision = 1/tau^2 and a per-covariate list form.
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
                # Normal: also set a PC prior on residual std dev.
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

        # --- Fixed effects ---
        sf = result.rx2('summary.fixed')
        beta = np.array(sf.rx(True, 'mean')).ravel()  # (d,)

        # --- Hyperparameters → sigma_rfx ---
        # Each iid term contributes one precision hyperparameter.
        # Use marginals for proper E[1/sqrt(τ)].
        marg_hyper = result.rx2('marginals.hyperpar')
        hyper_names = list(marg_hyper.names)

        # Identify the q RE precisions: INLA names them "Precision for group{j}".
        # E[σ_j] = E[1/sqrt(τ_j)] via numerical integration over the posterior marginal.
        sigma_rfx = np.zeros(q)
        for j in range(q):
            matched = [nm for nm in hyper_names if f'group{j}' in nm and 'Precision' in nm]
            if matched:
                marg = marg_hyper.rx2(matched[0])
                tau_vals = np.array(marg.rx(True, 1)).ravel()
                dens_vals = np.array(marg.rx(True, 2)).ravel()
                integrand = dens_vals / np.sqrt(np.maximum(tau_vals, 1e-12))
                sigma_rfx[j] = float(np.trapezoid(integrand, tau_vals))
            else:
                sh = result.rx2('summary.hyperpar')
                sigma_rfx[j] = float(1.0 / np.sqrt(max(float(sh.rx(j + 1, 'mean')[0]), 1e-12)))

        # --- BLUPs ---
        sr = result.rx2('summary.random')
        blups = np.zeros((m, q))
        for j in range(q):
            key = f'group{j}'
            re_j = sr.rx2(key)
            means_j = np.array(re_j.rx(True, 'mean')).ravel()  # length m (one per group)
            blups[: len(means_j), j] = means_j[:m]

        return {'beta': beta, 'sigma_rfx': sigma_rfx, 'blups': blups}

    except Exception:
        return None


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
    device: torch.device | None = None,
) -> dict:
    """Run PQL+P5+P6 vs R-INLA on one dataset, return metrics dict."""
    if device is None:
        device = torch.device('cpu')
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

    pql_a_be, pql_a_bt = [], []
    pql_a_se, pql_a_st = [], []
    pql_a_re, pql_a_rt = [], []

    p6_a_be, p6_a_bt = [], []
    p6_a_se, p6_a_st = [], []
    p6_a_re, p6_a_rt = [], []

    # matched subset (same datasets INLA ran on)
    pql_cv_be, pql_cv_bt = [], []
    pql_cv_se, pql_cv_st = [], []
    pql_cv_re, pql_cv_rt = [], []
    p6_cv_be, p6_cv_bt = [], []
    p6_cv_se, p6_cv_st = [], []
    p6_cv_re, p6_cv_rt = [], []

    inla_be, inla_bt = [], []
    inla_se, inla_st = [], []
    inla_re, inla_rt = [], []
    inla_wall: list[float] = []
    pql_wall: list[float] = []
    p6_wall: list[float] = []
    inla_n_tried = inla_n_ok = 0

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
                batch = toDevice(batch, device)
                B = batch['X'].shape[0]
                Zm = batch['Z'][..., :max_q]

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
                )
                pql_batch_wall = time.perf_counter() - t0

                if likelihood_family == 1:
                    stats_p5 = refineBernoulliNagqSrfx(
                        stats,
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        mask_q=batch.get('mask_q'),
                    )
                    t0_p6 = time.perf_counter()
                    stats_p6 = refineBernoulliMapBeta(
                        stats_p5,
                        batch['X'],
                        batch['y'],
                        Zm,
                        batch['mask_n'].float(),
                        batch['mask_m'].float(),
                        nu_ffx=batch.get('nu_ffx'),
                        tau_ffx=batch.get('tau_ffx'),
                        family_ffx=batch.get('family_ffx'),
                    )
                    p6_batch_wall = time.perf_counter() - t0_p6
                else:
                    stats_p6 = stats
                    p6_batch_wall = 0.0

                beta_est = stats['beta_est'].cpu().numpy()
                srfx_est = stats['sigma_rfx_est'].cpu().numpy()
                blup_est = stats['blup_est'].cpu().numpy()
                beta_est_p6 = stats_p6['beta_est'].cpu().numpy()
                srfx_est_p6 = stats_p6['sigma_rfx_est'].cpu().numpy()
                blup_est_p6 = stats_p6['blup_est'].cpu().numpy()
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

                    be = beta_est[b, active_d] - ffx_true[b, active_d]
                    se = srfx_est[b, active_q] - srfx_true[b, active_q]
                    re_e = (
                        blup_est[b, :m_b][:, active_q] - rfx_true[b, :m_b][:, active_q]
                    ).reshape(-1)
                    re_t = rfx_true[b, :m_b][:, active_q].reshape(-1)

                    be_p6 = beta_est_p6[b, active_d] - ffx_true[b, active_d]
                    se_p6 = srfx_est_p6[b, active_q] - srfx_true[b, active_q]
                    re_e_p6 = (
                        blup_est_p6[b, :m_b][:, active_q] - rfx_true[b, :m_b][:, active_q]
                    ).reshape(-1)

                    pql_a_be.append(be)
                    pql_a_bt.append(ffx_true[b, active_d])
                    pql_a_se.append(se)
                    pql_a_st.append(srfx_true[b, active_q])
                    pql_a_re.append(re_e)
                    pql_a_rt.append(re_t)
                    pql_wall.append(pql_batch_wall / B)

                    p6_a_be.append(be_p6)
                    p6_a_bt.append(ffx_true[b, active_d])
                    p6_a_se.append(se_p6)
                    p6_a_st.append(srfx_true[b, active_q])
                    p6_a_re.append(re_e_p6)
                    p6_a_rt.append(re_t)
                    p6_wall.append(p6_batch_wall / B)

                    n_seen += 1

                    if not _HAS_INLA or inla_n_tried >= n_inla:
                        continue

                    inla_n_tried += 1
                    pql_cv_be.append(be)
                    pql_cv_bt.append(ffx_true[b, active_d])
                    pql_cv_se.append(se)
                    pql_cv_st.append(srfx_true[b, active_q])
                    pql_cv_re.append(re_e)
                    pql_cv_rt.append(re_t)
                    p6_cv_be.append(be_p6)
                    p6_cv_bt.append(ffx_true[b, active_d])
                    p6_cv_se.append(se_p6)
                    p6_cv_st.append(srfx_true[b, active_q])
                    p6_cv_re.append(re_e_p6)
                    p6_cv_rt.append(re_t)

                    ds_flat = _flatten(batch, b, active_d, active_q)
                    t_i = time.perf_counter()
                    est = _inla_estimate(ds_flat, likelihood_family)
                    inla_wall.append(time.perf_counter() - t_i)

                    if est is None:
                        continue
                    inla_n_ok += 1
                    inla_be.append(est['beta'] - ffx_true[b, active_d])
                    inla_bt.append(ffx_true[b, active_d])
                    inla_se.append(est['sigma_rfx'] - srfx_true[b, active_q])
                    inla_st.append(srfx_true[b, active_q])
                    inla_re.append(
                        (est['blups'][:m_b] - rfx_true[b, :m_b][:, active_q]).reshape(-1)
                    )
                    inla_rt.append(re_t)

    def flat(lst: list) -> np.ndarray:
        return np.concatenate(lst) if lst else np.array([np.nan])

    pql_a = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(),
            [pql_a_be, pql_a_bt, pql_a_se, pql_a_st, pql_a_re, pql_a_rt],
        )
    }
    p6_a = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(), [p6_a_be, p6_a_bt, p6_a_se, p6_a_st, p6_a_re, p6_a_rt]
        )
    }
    pql_cv = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(),
            [pql_cv_be, pql_cv_bt, pql_cv_se, pql_cv_st, pql_cv_re, pql_cv_rt],
        )
    }
    p6_cv = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(),
            [p6_cv_be, p6_cv_bt, p6_cv_se, p6_cv_st, p6_cv_re, p6_cv_rt],
        )
    }
    inla = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(), [inla_be, inla_bt, inla_se, inla_st, inla_re, inla_rt]
        )
    }

    n_pql_all = len(pql_a_be)
    metrics = {
        'data_id': data_id,
        'n_pql': n_pql_all,
        'n_inla': inla_n_ok,
        'pql_ffx': _nrmse(pql_a['be'], pql_a['bt']),
        'pql_srfx': _nrmse(pql_a['se'], pql_a['st']),
        'pql_blup': _nrmse(pql_a['re'], pql_a['rt']),
        'p6_ffx': _nrmse(p6_a['be'], p6_a['bt']),
        'p6_srfx': _nrmse(p6_a['se'], p6_a['st']),
        'p6_blup': _nrmse(p6_a['re'], p6_a['rt']),
        'inla_ffx': _nrmse(inla['be'], inla['bt']) if inla_n_ok > 0 else float('nan'),
        'inla_srfx': _nrmse(inla['se'], inla['st']) if inla_n_ok > 0 else float('nan'),
        'inla_blup': _nrmse(inla['re'], inla['rt']) if inla_n_ok > 0 else float('nan'),
    }

    sep = '=' * 70
    print(sep)
    print(f'  {data_id}  |  partition={partition}  N={n_pql_all}  family={likelihood_family}')
    print(sep)

    family_label = {0: 'Normal', 1: 'Bernoulli', 2: 'Poisson'}.get(
        likelihood_family, str(likelihood_family)
    )
    p6_label = 'P6' if likelihood_family == 1 else 'MAP'

    print(f'\nPQL — all q (N={n_pql_all})')
    print(
        tabulate(
            [
                ['FFX (β)', f'{metrics["pql_ffx"]:.4f}', f'{_bias(pql_a["be"]):+.4f}'],
                ['σ_rfx', f'{metrics["pql_srfx"]:.4f}', f'{_bias(pql_a["se"]):+.4f}'],
                ['BLUP', f'{metrics["pql_blup"]:.4f}', f'{_bias(pql_a["re"]):+.4f}'],
            ],
            headers=['Parameter', 'NRMSE', 'Bias'],
            tablefmt='simple',
        )
    )

    print(f'\n{p6_label} — all q (N={n_pql_all})')
    print(
        tabulate(
            [
                ['FFX (β)', f'{metrics["p6_ffx"]:.4f}', f'{_bias(p6_a["be"]):+.4f}'],
                ['σ_rfx', f'{metrics["p6_srfx"]:.4f}', f'{_bias(p6_a["se"]):+.4f}'],
                ['BLUP', f'{metrics["p6_blup"]:.4f}', f'{_bias(p6_a["re"]):+.4f}'],
            ],
            headers=['Parameter', 'NRMSE', 'Bias'],
            tablefmt='simple',
        )
    )

    if inla_n_ok > 0:
        print(
            f'\nPQL vs {p6_label} vs R-INLA — matched'
            f' (N={len(pql_cv_be)}, INLA {inla_n_ok}/{inla_n_tried})'
        )
        print(
            tabulate(
                [
                    [
                        'PQL',
                        'FFX (β)',
                        f'{_nrmse(pql_cv["be"], pql_cv["bt"]):.4f}',
                        f'{_bias(pql_cv["be"]):+.4f}',
                    ],
                    [
                        'PQL',
                        'σ_rfx',
                        f'{_nrmse(pql_cv["se"], pql_cv["st"]):.4f}',
                        f'{_bias(pql_cv["se"]):+.4f}',
                    ],
                    [
                        'PQL',
                        'BLUP',
                        f'{_nrmse(pql_cv["re"], pql_cv["rt"]):.4f}',
                        f'{_bias(pql_cv["re"]):+.4f}',
                    ],
                    [
                        p6_label,
                        'FFX (β)',
                        f'{_nrmse(p6_cv["be"], p6_cv["bt"]):.4f}',
                        f'{_bias(p6_cv["be"]):+.4f}',
                    ],
                    [
                        p6_label,
                        'σ_rfx',
                        f'{_nrmse(p6_cv["se"], p6_cv["st"]):.4f}',
                        f'{_bias(p6_cv["se"]):+.4f}',
                    ],
                    [
                        p6_label,
                        'BLUP',
                        f'{_nrmse(p6_cv["re"], p6_cv["rt"]):.4f}',
                        f'{_bias(p6_cv["re"]):+.4f}',
                    ],
                    [
                        'R-INLA',
                        'FFX (β)',
                        f'{metrics["inla_ffx"]:.4f}',
                        f'{_bias(inla["be"]):+.4f}',
                    ],
                    ['R-INLA', 'σ_rfx', f'{metrics["inla_srfx"]:.4f}', f'{_bias(inla["se"]):+.4f}'],
                    ['R-INLA', 'BLUP', f'{metrics["inla_blup"]:.4f}', f'{_bias(inla["re"]):+.4f}'],
                ],
                headers=['Method', 'Parameter', 'NRMSE', 'Bias'],
                tablefmt='simple',
            )
        )

    print('\nσ_rfx bias by true σ_rfx bin — PQL')
    print(_breakdown_srfx(pql_a['se'], pql_a['st'], 'PQL'))
    print(f'\nσ_rfx bias by true σ_rfx bin — {p6_label}')
    print(_breakdown_srfx(p6_a['se'], p6_a['st'], p6_label))
    if inla_n_ok > 0:
        print('\nσ_rfx bias by true σ_rfx bin — R-INLA')
        print(_breakdown_srfx(inla['se'], inla['st'], 'R-INLA'))

    w_pql = np.array(pql_wall)
    w_p6 = np.array(p6_wall)
    print(
        f'\nWall time — PQL:  mean={w_pql.mean()*1000:.2f} ms  median={np.median(w_pql)*1000:.2f} ms/ds'
    )
    if likelihood_family == 1:
        print(
            f'Wall time — P6:   mean={w_p6.mean()*1000:.2f} ms  median={np.median(w_p6)*1000:.2f} ms/ds'
        )
    if inla_wall:
        w = np.array(inla_wall)
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
) -> None:
    print(
        f'R-INLA: {"enabled" if _HAS_INLA else "DISABLED"}   limit={n_inla}'
        + (f'   n_total={n_total}' if n_total > 0 else '')
    )
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
                [
                    m['data_id'],
                    m['n_pql'],
                    fmt(m['pql_ffx']),
                    fmt(m['pql_srfx']),
                    fmt(m['pql_blup']),
                    fmt(m['p6_ffx']),
                    fmt(m['p6_srfx']),
                    fmt(m['p6_blup']),
                    fmt(m['inla_ffx']),
                    fmt(m['inla_srfx']),
                    fmt(m['inla_blup']),
                ]
            )
        print(
            tabulate(
                rows,
                headers=[
                    'Dataset',
                    'N',
                    'PQL-F',
                    'PQL-S',
                    'PQL-B',
                    'P6-F',
                    'P6-S',
                    'P6-B',
                    'INLA-F',
                    'INLA-S',
                    'INLA-B',
                ],
                tablefmt='simple',
            )
        )


if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description='PQL vs R-INLA comparison on GLMM datasets')
    parser.add_argument('--data-ids',  default='small-b-sampled',
                        help='comma-separated data config ids (default: small-b-sampled)')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-epochs',  default=1,   type=int)
    parser.add_argument('--n-inla',    default=100, type=int, help='max datasets for INLA per data_id')
    parser.add_argument('--n-total',   default=0,   type=int, help='cap total datasets per data_id (0=all)')
    # fmt: on
    a = parser.parse_args()
    main(
        data_ids=[x.strip() for x in a.data_ids.split(',')],
        partition=a.partition,
        n_epochs=a.n_epochs,
        n_inla=a.n_inla,
        n_total=a.n_total,
    )
