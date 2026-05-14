"""Experiment: PQL vs statsmodels BinomialBayesMixedGLM (CAVI) on Bernoulli data.

Compares our PQL estimator against CAVI (coordinate-ascent variational inference)
on Bernoulli test datasets.  CAVI supports any q via independent variance components
per RE dimension (diagonal Ψ).

lme4::glmer was evaluated and removed: ML without regularization diverges
catastrophically for small groups with q>1 (σ_rfx NRMSE 33–64 on medium datasets).

Goal: establish headroom above PQL for the nAGQ (P5) and True Laplace LML (P6)
improvement directions.

Prior matching is approximate: CAVI uses a Gaussian prior on log(σ²) via vcp_p;
we use HalfNormal(τ_rfx) on σ.  The comparison is frequentist — accuracy relative
to the simulated ground truth — not a matched-prior Bayesian comparison.

Usage (from repo root):
    uv run python experiments/analytical/glmm_reference_comparison.py
    uv run python experiments/analytical/glmm_reference_comparison.py \\
        --data-ids small-b-sampled,small-b-mixed --n-cavi 200
    uv run python experiments/analytical/glmm_reference_comparison.py \\
        --data-ids small-b-sampled,small-b-mixed,medium-b-sampled,medium-b-mixed \\
        --n-cavi 200
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from metabeta.analytical.glmm import glmm
from metabeta.analytical.map import refineBernoulliMapBeta, refineBernoulliNagqSrfx
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath

try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    _HAS_CAVI = True
except ImportError:
    _HAS_CAVI = False

# Prior variance on log(σ²) for the CAVI model — weakly informative.
_VCP_PRIOR = 4.0


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
    if not X_parts:
        return {
            'X': np.zeros((0, d)),
            'Z': np.zeros((0, q)),
            'y': np.zeros(0),
            'groups': np.zeros(0, dtype=int),
            'm': m,
            'd': d,
            'q': q,
        }
    return {
        'X': np.vstack(X_parts),
        'Z': np.vstack(Z_parts),
        'y': np.concatenate(y_parts),
        'groups': np.concatenate(g_parts),
        'm': m,
        'd': d,
        'q': q,
    }


def _cavi_estimate(ds_flat: dict) -> dict | None:
    """Run statsmodels CAVI on a flat Bernoulli dataset (any q).

    For q>1: each RE dimension gets an independent variance component (diagonal Ψ).
    Intercept Z-columns (all 1s) use '0 + C(group)'; slope columns use
    '0 + z{j}:C(group)'.

    Returns dict with:
      'beta'      : (d,)   fixed-effect posterior means
      'sigma_rfx' : (q,)   per-dim posterior std devs
      'blups'     : (m, q) per-group RE means
    or None on failure.

    API notes (statsmodels 0.14+):
      result.fe_mean (k_fep,), result.vcp_mean (q,) log σ² per VC.
      RE params after [k_fep + q:] ordered VC-major, group-minor.
    """
    if not _HAS_CAVI:
        return None

    d, m, q = ds_flat['d'], ds_flat['m'], ds_flat['q']
    X, Z, y, groups = ds_flat['X'], ds_flat['Z'], ds_flat['y'], ds_flat['groups']

    df_data: dict = {'y': y.astype(float), 'group': groups.astype(int)}
    for j in range(1, d):
        df_data[f'x{j}'] = X[:, j].astype(float)

    vc_formula: dict[str, str] = {}
    for j in range(q):
        if np.allclose(Z[:, j], 1.0):
            vc_formula[f'rfx_{j}'] = '0 + C(group)'
        else:
            df_data[f'z{j}'] = Z[:, j].astype(float)
            vc_formula[f'rfx_{j}'] = f'0 + z{j}:C(group)'

    formula = 'y ~ ' + (' + '.join(f'x{j}' for j in range(1, d)) if d > 1 else '1')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = pd.DataFrame(df_data)
            model = BinomialBayesMixedGLM.from_formula(formula, vc_formula, df, vcp_p=_VCP_PRIOR)
            result = model.fit_vb()

        fe = np.asarray(result.fe_mean, dtype=float)
        vcp = np.asarray(result.vcp_mean, dtype=float)
        sigma_rfx = np.exp(vcp / 2.0)

        k_fe, k_vcp = len(fe), len(vcp)
        re_all = np.asarray(result.params[k_fe + k_vcp :], dtype=float)
        blups = re_all.reshape(q, m).T  # (m, q) — VC-major → transpose

        return {'beta': fe, 'sigma_rfx': sigma_rfx, 'blups': blups}
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
    n_cavi: int = 200,
    n_total: int = 0,
    n_epochs: int = 1,
    skip_cavi: bool = False,
    device: torch.device | None = None,
) -> dict:
    """Run PQL vs CAVI on one dataset, return metrics dict."""
    if device is None:
        device = torch.device('cpu')
    data_cfg = loadDataConfig(data_id)
    max_d = data_cfg['max_d']
    max_q = data_cfg['max_q']
    likelihood_family = data_cfg.get('likelihood_family', 0)

    if likelihood_family != 1:
        print(f'  WARNING: likelihood_family={likelihood_family}, expected 1 (Bernoulli)')

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

    p5_a_be, p5_a_bt = [], []
    p5_a_se, p5_a_st = [], []
    p5_a_re, p5_a_rt = [], []
    p5_wall: list[float] = []

    p6_a_be, p6_a_bt = [], []
    p6_a_se, p6_a_st = [], []
    p6_a_re, p6_a_rt = [], []

    pql_cv_be, pql_cv_bt = [], []
    pql_cv_se, pql_cv_st = [], []
    pql_cv_re, pql_cv_rt = [], []

    p6_cv_be, p6_cv_bt = [], []
    p6_cv_se, p6_cv_st = [], []
    p6_cv_re, p6_cv_rt = [], []

    cavi_be, cavi_bt = [], []
    cavi_se, cavi_st = [], []
    cavi_re, cavi_rt = [], []
    cavi_wall: list[float] = []
    pql_wall: list[float] = []
    p6_wall: list[float] = []
    # p5_wall declared above
    cavi_n_tried = cavi_n_ok = 0

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

                t0_p5 = time.perf_counter()
                stats_p5 = refineBernoulliNagqSrfx(
                    stats,
                    batch['X'],
                    batch['y'],
                    Zm,
                    batch['mask_n'].float(),
                    batch['mask_m'].float(),
                    mask_q=batch.get('mask_q'),
                )
                p5_batch_wall = time.perf_counter() - t0_p5

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

                beta_est = stats['beta_est'].cpu().numpy()
                srfx_est = stats['sigma_rfx_est'].cpu().numpy()
                blup_est = stats['blup_est'].cpu().numpy()
                beta_est_p5 = stats_p5['beta_est'].cpu().numpy()
                srfx_est_p5 = stats_p5['sigma_rfx_est'].cpu().numpy()
                blup_est_p5 = stats_p5['blup_est'].cpu().numpy()
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

                    be_p5 = beta_est_p5[b, active_d] - ffx_true[b, active_d]
                    se_p5 = srfx_est_p5[b, active_q] - srfx_true[b, active_q]
                    re_e_p5 = (
                        blup_est_p5[b, :m_b][:, active_q] - rfx_true[b, :m_b][:, active_q]
                    ).reshape(-1)
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

                    p5_a_be.append(be_p5)
                    p5_a_bt.append(ffx_true[b, active_d])
                    p5_a_se.append(se_p5)
                    p5_a_st.append(srfx_true[b, active_q])
                    p5_a_re.append(re_e_p5)
                    p5_a_rt.append(re_t)
                    p5_wall.append(p5_batch_wall / B)

                    p6_a_be.append(be_p6)
                    p6_a_bt.append(ffx_true[b, active_d])
                    p6_a_se.append(se_p6)
                    p6_a_st.append(srfx_true[b, active_q])
                    p6_a_re.append(re_e_p6)
                    p6_a_rt.append(re_t)
                    p6_wall.append(p6_batch_wall / B)

                    n_seen += 1

                    if cavi_n_tried < n_cavi:
                        cavi_n_tried += 1
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

                        if _HAS_CAVI and not skip_cavi:
                            ds_flat = _flatten(batch, b, active_d, active_q)
                            t_c = time.perf_counter()
                            est = _cavi_estimate(ds_flat)
                            cavi_wall.append(time.perf_counter() - t_c)

                            if est is not None:
                                cavi_n_ok += 1
                                cavi_be.append(est['beta'] - ffx_true[b, active_d])
                                cavi_bt.append(ffx_true[b, active_d])
                                cavi_se.append(est['sigma_rfx'] - srfx_true[b, active_q])
                                cavi_st.append(srfx_true[b, active_q])
                                cavi_re.append(
                                    (est['blups'][:m_b] - rfx_true[b, :m_b][:, active_q]).reshape(
                                        -1
                                    )
                                )
                                cavi_rt.append(re_t)

    def flat(lst: list) -> np.ndarray:
        return np.concatenate(lst) if lst else np.array([np.nan])

    pql_a = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(),
            [pql_a_be, pql_a_bt, pql_a_se, pql_a_st, pql_a_re, pql_a_rt],
        )
    }
    p5_a = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(),
            [p5_a_be, p5_a_bt, p5_a_se, p5_a_st, p5_a_re, p5_a_rt],
        )
    }
    p6_a = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(),
            [p6_a_be, p6_a_bt, p6_a_se, p6_a_st, p6_a_re, p6_a_rt],
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
    cavi = {
        k: flat(v)
        for k, v in zip(
            'be bt se st re rt'.split(), [cavi_be, cavi_bt, cavi_se, cavi_st, cavi_re, cavi_rt]
        )
    }

    n_pql_all = len(pql_a_be)
    metrics = {
        'data_id': data_id,
        'n_pql': n_pql_all,
        'n_cavi': cavi_n_ok,
        'pql_ffx': _nrmse(pql_a['be'], pql_a['bt']),
        'pql_srfx': _nrmse(pql_a['se'], pql_a['st']),
        'pql_blup': _nrmse(pql_a['re'], pql_a['rt']),
        'p5_ffx': _nrmse(p5_a['be'], p5_a['bt']),
        'p5_srfx': _nrmse(p5_a['se'], p5_a['st']),
        'p5_blup': _nrmse(p5_a['re'], p5_a['rt']),
        'p6_ffx': _nrmse(p6_a['be'], p6_a['bt']),
        'p6_srfx': _nrmse(p6_a['se'], p6_a['st']),
        'p6_blup': _nrmse(p6_a['re'], p6_a['rt']),
        'cavi_ffx': _nrmse(cavi['be'], cavi['bt']) if cavi_n_ok > 0 else float('nan'),
        'cavi_srfx': _nrmse(cavi['se'], cavi['st']) if cavi_n_ok > 0 else float('nan'),
        'cavi_blup': _nrmse(cavi['re'], cavi['rt']) if cavi_n_ok > 0 else float('nan'),
    }

    sep = '=' * 70
    print(sep)
    print(f'  {data_id}  |  partition={partition}  N={n_pql_all}')
    print(sep)

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

    print(f'\nP5 — q=1 only (N={n_pql_all})')
    print(
        tabulate(
            [
                ['FFX (β)', f'{metrics["p5_ffx"]:.4f}', f'{_bias(p5_a["be"]):+.4f}'],
                ['σ_rfx', f'{metrics["p5_srfx"]:.4f}', f'{_bias(p5_a["se"]):+.4f}'],
                ['BLUP', f'{metrics["p5_blup"]:.4f}', f'{_bias(p5_a["re"]):+.4f}'],
            ],
            headers=['Parameter', 'NRMSE', 'Bias'],
            tablefmt='simple',
        )
    )

    print(f'\nP6 — all q (N={n_pql_all})')
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

    if pql_cv_be:
        cavi_info = f', CAVI {cavi_n_ok}/{cavi_n_tried}' if cavi_n_ok > 0 else ''
        vs_label = 'PQL vs P6' + (' vs CAVI' if cavi_n_ok > 0 else '')
        print(f'\n{vs_label} — matched (N={len(pql_cv_be)}{cavi_info})')
        rows_matched = [
            ['PQL', 'FFX (β)', f'{_nrmse(pql_cv["be"],pql_cv["bt"]):.4f}', f'{_bias(pql_cv["be"]):+.4f}'],
            ['PQL', 'σ_rfx',   f'{_nrmse(pql_cv["se"],pql_cv["st"]):.4f}', f'{_bias(pql_cv["se"]):+.4f}'],
            ['PQL', 'BLUP',    f'{_nrmse(pql_cv["re"],pql_cv["rt"]):.4f}', f'{_bias(pql_cv["re"]):+.4f}'],
            ['P6',  'FFX (β)', f'{_nrmse(p6_cv["be"],p6_cv["bt"]):.4f}',  f'{_bias(p6_cv["be"]):+.4f}'],
            ['P6',  'σ_rfx',   f'{_nrmse(p6_cv["se"],p6_cv["st"]):.4f}',  f'{_bias(p6_cv["se"]):+.4f}'],
            ['P6',  'BLUP',    f'{_nrmse(p6_cv["re"],p6_cv["rt"]):.4f}',  f'{_bias(p6_cv["re"]):+.4f}'],
        ]
        if cavi_n_ok > 0:
            rows_matched += [
                ['CAVI', 'FFX (β)', f'{metrics["cavi_ffx"]:.4f}', f'{_bias(cavi["be"]):+.4f}'],
                ['CAVI', 'σ_rfx',   f'{metrics["cavi_srfx"]:.4f}', f'{_bias(cavi["se"]):+.4f}'],
                ['CAVI', 'BLUP',    f'{metrics["cavi_blup"]:.4f}', f'{_bias(cavi["re"]):+.4f}'],
            ]
        print(tabulate(rows_matched, headers=['Method', 'Parameter', 'NRMSE', 'Bias'], tablefmt='simple'))

    print('\nσ_rfx bias by true σ_rfx bin — PQL')
    print(_breakdown_srfx(pql_a['se'], pql_a['st'], 'PQL'))
    print('\nσ_rfx bias by true σ_rfx bin — P5')
    print(_breakdown_srfx(p5_a['se'], p5_a['st'], 'P5'))
    print('\nσ_rfx bias by true σ_rfx bin — P6')
    print(_breakdown_srfx(p6_a['se'], p6_a['st'], 'P6'))
    if cavi_n_ok > 0:
        print('\nσ_rfx bias by true σ_rfx bin — CAVI')
        print(_breakdown_srfx(cavi['se'], cavi['st'], 'CAVI'))

    w_pql = np.array(pql_wall)
    w_p5 = np.array(p5_wall)
    w_p6 = np.array(p6_wall)
    print(
        f'\nWall time — PQL: mean={w_pql.mean()*1000:.2f} ms  median={np.median(w_pql)*1000:.2f} ms/ds'
    )
    print(
        f'Wall time — P5:  mean={w_p5.mean()*1000:.2f} ms  median={np.median(w_p5)*1000:.2f} ms/ds'
    )
    print(
        f'Wall time — P6:  mean={w_p6.mean()*1000:.2f} ms  median={np.median(w_p6)*1000:.2f} ms/ds'
    )
    if cavi_wall:
        w = np.array(cavi_wall)
        print(f'Wall time — CAVI: mean={w.mean():.3f} s  median={np.median(w):.3f} s/ds')
    print()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_ids: list[str],
    partition: str = 'test',
    n_epochs: int = 1,
    n_cavi: int = 200,
    n_total: int = 0,
    skip_cavi: bool = False,
) -> None:
    print(
        f'CAVI: {"enabled" if _HAS_CAVI else "DISABLED"}   vcp_prior={_VCP_PRIOR}   limit={n_cavi}'
        + (f'   n_total={n_total}' if n_total > 0 else '')
    )
    print()

    all_metrics = []
    for data_id in data_ids:
        all_metrics.append(
            run_one_dataset(
                data_id=data_id,
                partition=partition,
                n_cavi=n_cavi,
                n_total=n_total,
                n_epochs=n_epochs,
                skip_cavi=skip_cavi,
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
                    fmt(m['p5_ffx']),
                    fmt(m['p5_srfx']),
                    fmt(m['p5_blup']),
                    fmt(m['p6_ffx']),
                    fmt(m['p6_srfx']),
                    fmt(m['p6_blup']),
                    fmt(m['cavi_ffx']),
                    fmt(m['cavi_srfx']),
                    fmt(m['cavi_blup']),
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
                    'P5-F',
                    'P5-S',
                    'P5-B',
                    'P6-F',
                    'P6-S',
                    'P6-B',
                    'CAVI-F',
                    'CAVI-S',
                    'CAVI-B',
                ],
                tablefmt='simple',
            )
        )


if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description='PQL vs CAVI comparison on Bernoulli datasets')
    parser.add_argument('--data-ids',  default='small-b-sampled',
                        help='comma-separated data config ids (default: small-b-sampled)')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-epochs',  default=1, type=int)
    parser.add_argument('--n-cavi',    default=200, type=int, help='max datasets for CAVI per data_id')
    parser.add_argument('--n-total',   default=0,   type=int, help='cap total datasets per data_id (0=all)')
    parser.add_argument('--no-cavi',   action='store_true', help='skip CAVI; still track matched-N PQL/P6 subset')
    # fmt: on
    a = parser.parse_args()
    main(
        data_ids=[x.strip() for x in a.data_ids.split(',')],
        partition=a.partition,
        n_epochs=a.n_epochs,
        n_cavi=a.n_cavi,
        n_total=a.n_total,
        skip_cavi=a.no_cavi,
    )
