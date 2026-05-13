"""Experiment: PQL vs CAVI (statsmodels) vs lme4::glmer (rpy2) on Bernoulli data.

Compares our PQL estimator against two reference methods on Bernoulli test datasets:
  - CAVI (BinomialBayesMixedGLM): variational Bayes, diagonal Ψ for q>1.
  - lme4::glmer: marginal ML via Laplace, full correlated Ψ — same model as PQL.

Goal: establish how much headroom exists above PQL for the nAGQ (P5) and
True Laplace LML (P6) improvement directions.

CAVI uses a Gaussian prior on log(σ²); lme4 uses no prior (ML).
The comparison is frequentist — accuracy relative to the simulated ground truth.

Usage (from repo root):
    uv run python experiments/analytical/glmm_reference_comparison.py
    uv run python experiments/analytical/glmm_reference_comparison.py \\
        --data-ids small-b-sampled,small-b-mixed --n-cavi 200 --n-lme4 50
    uv run python experiments/analytical/glmm_reference_comparison.py \\
        --data-ids small-b-sampled,small-b-mixed,medium-b-sampled,medium-b-mixed \\
        --n-cavi 100 --n-lme4 30
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
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    _HAS_CAVI = True
except ImportError:
    _HAS_CAVI = False

try:
    import rpy2.robjects as _ro
    from rpy2.robjects import pandas2ri as _pandas2ri
    from rpy2.robjects.packages import importr as _importr

    _r_lme4 = _importr('lme4')
    _r_base = _importr('base')
    _HAS_LME4 = True
except Exception:
    _HAS_LME4 = False

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
        Z_g = batch['Z'][b, g, :ng].cpu().numpy()[:, active_q]  # (ng, q)
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


def _lme4_estimate(ds_flat: dict) -> dict | None:
    """Run lme4::glmer via rpy2 on a flat Bernoulli dataset (any q).

    Uses correlated random effects (1 + z1 + ... | group) — full Ψ, same model
    as PQL.  Optimizer: bobyqa for robustness.

    Returns dict with:
      'beta'      : (d,)   ML fixed-effect estimates
      'sigma_rfx' : (q,)   sqrt(diag(Ψ)) — marginal RE std devs
      'blups'     : (m, q) conditional modes sorted by group index
    or None on failure.
    """
    if not _HAS_LME4:
        return None

    d, m, q = ds_flat['d'], ds_flat['m'], ds_flat['q']
    X, Z, y, groups = ds_flat['X'], ds_flat['Z'], ds_flat['y'], ds_flat['groups']

    df_data: dict = {'y': y.astype(float), 'group': [str(int(g)) for g in groups]}
    for j in range(1, d):
        df_data[f'x{j}'] = X[:, j].astype(float)

    has_intercept = False
    slope_names: list[str] = []
    for j in range(q):
        if np.allclose(Z[:, j], 1.0):
            has_intercept = True
        else:
            sname = f'z{j}'
            df_data[sname] = Z[:, j].astype(float)
            slope_names.append(sname)

    re_int = '1' if has_intercept else '0'
    re_str = ' + '.join([re_int] + slope_names)
    fe_str = ' + '.join(f'x{j}' for j in range(1, d)) if d > 1 else '1'
    formula_str = f'y ~ {fe_str} + ({re_str} | group)'

    # S4 generics must be called via ro.r('lme4::...'), not through importr binding
    _fixef = _ro.r('lme4::fixef')
    _VarCorr = _ro.r('lme4::VarCorr')
    _ranef = _ro.r('lme4::ranef')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = pd.DataFrame(df_data)
            with (_ro.default_converter + _pandas2ri.converter).context():
                r_df = _ro.conversion.get_conversion().py2rpy(df)

            ctrl = _r_lme4.glmerControl(optimizer='bobyqa', optCtrl=_ro.r('list(maxfun=2e4)'))
            model = _r_lme4.glmer(
                _ro.Formula(formula_str), data=r_df, family='binomial', control=ctrl
            )

        # Fixed effects
        beta = np.array(_fixef(model))  # (d,)

        # σ_rfx from as.data.frame(VarCorr): diagonal rows (var2 is R NA_character_)
        vc_df_r = _r_base.as_data_frame(_VarCorr(model))
        with (_ro.default_converter + _pandas2ri.converter).context():
            vc_df = _ro.conversion.get_conversion().rpy2py(vc_df_r)
        # rpy2 converts R NA_character_ to the string 'NA_character_', not NaN
        diag_mask = vc_df['var2'].apply(
            lambda v: pd.isna(v) or str(v) in ('NA_character_', 'NA', '<NA>')
        )
        sigma_rfx = vc_df.loc[diag_mask, 'sdcor'].values.astype(float)  # (q,)

        # BLUPs from ranef(model)[['group']] — sorted by group index
        ranef_r = _ranef(model)
        ranef_group_r = ranef_r.rx2('group')
        with (_ro.default_converter + _pandas2ri.converter).context():
            ranef_df = _ro.conversion.get_conversion().rpy2py(ranef_group_r)
        group_int = [int(g) for g in list(ranef_group_r.rownames)]
        blups = ranef_df.values[np.argsort(group_int)]  # (m, q) sorted

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
    n_cavi: int = 200,
    n_lme4: int = 50,
    n_epochs: int = 1,
    device: torch.device | None = None,
) -> dict:
    """Run PQL vs CAVI vs lme4 on one dataset, return metrics dict."""
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

    # Accumulators: _all = all PQL, _cv = PQL on CAVI-matched subset,
    # _lv = PQL on lme4-matched subset
    def _acc() -> tuple[list, list]:
        return [], []

    pql_a_be, pql_a_bt = _acc()
    pql_a_se, pql_a_st = _acc()
    pql_a_re, pql_a_rt = _acc()

    pql_cv_be, pql_cv_bt = _acc()
    pql_cv_se, pql_cv_st = _acc()
    pql_cv_re, pql_cv_rt = _acc()

    pql_lv_be, pql_lv_bt = _acc()
    pql_lv_se, pql_lv_st = _acc()
    pql_lv_re, pql_lv_rt = _acc()

    cavi_be, cavi_bt = _acc()
    cavi_se, cavi_st = _acc()
    cavi_re, cavi_rt = _acc()
    cavi_wall: list[float] = []
    cavi_n_tried = cavi_n_ok = 0

    lme4_be, lme4_bt = _acc()
    lme4_se, lme4_st = _acc()
    lme4_re, lme4_rt = _acc()
    lme4_wall: list[float] = []
    lme4_n_tried = lme4_n_ok = 0

    pql_wall: list[float] = []

    all_batches = []
    for path in paths:
        dl = Dataloader(path, batch_size=32, shuffle=False)
        all_batches.extend(list(dl))

    with torch.no_grad():
        for batch in all_batches:
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

            beta_est = stats['beta_est'].cpu().numpy()
            srfx_est = stats['sigma_rfx_est'].cpu().numpy()
            blup_est = stats['blup_est'].cpu().numpy()
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
                re_e = (blup_est[b, :m_b][:, active_q] - rfx_true[b, :m_b][:, active_q]).reshape(-1)
                re_t = rfx_true[b, :m_b][:, active_q].reshape(-1)

                pql_a_be.append(be)
                pql_a_bt.append(ffx_true[b, active_d])
                pql_a_se.append(se)
                pql_a_st.append(srfx_true[b, active_q])
                pql_a_re.append(re_e)
                pql_a_rt.append(re_t)
                pql_wall.append(pql_batch_wall / B)

                run_cavi = _HAS_CAVI and cavi_n_tried < n_cavi
                run_lme4 = _HAS_LME4 and lme4_n_tried < n_lme4
                ds_flat = _flatten(batch, b, active_d, active_q) if (run_cavi or run_lme4) else None

                if run_cavi:
                    cavi_n_tried += 1
                    pql_cv_be.append(be)
                    pql_cv_bt.append(ffx_true[b, active_d])
                    pql_cv_se.append(se)
                    pql_cv_st.append(srfx_true[b, active_q])
                    pql_cv_re.append(re_e)
                    pql_cv_rt.append(re_t)
                    t_c = time.perf_counter()
                    est_c = _cavi_estimate(ds_flat)
                    cavi_wall.append(time.perf_counter() - t_c)
                    if est_c is not None:
                        cavi_n_ok += 1
                        cavi_be.append(est_c['beta'] - ffx_true[b, active_d])
                        cavi_bt.append(ffx_true[b, active_d])
                        cavi_se.append(est_c['sigma_rfx'] - srfx_true[b, active_q])
                        cavi_st.append(srfx_true[b, active_q])
                        cavi_re.append(
                            (est_c['blups'][:m_b] - rfx_true[b, :m_b][:, active_q]).reshape(-1)
                        )
                        cavi_rt.append(re_t)

                if run_lme4:
                    lme4_n_tried += 1
                    pql_lv_be.append(be)
                    pql_lv_bt.append(ffx_true[b, active_d])
                    pql_lv_se.append(se)
                    pql_lv_st.append(srfx_true[b, active_q])
                    pql_lv_re.append(re_e)
                    pql_lv_rt.append(re_t)
                    t_l = time.perf_counter()
                    est_l = _lme4_estimate(ds_flat)
                    lme4_wall.append(time.perf_counter() - t_l)
                    if est_l is not None:
                        lme4_n_ok += 1
                        lme4_be.append(est_l['beta'] - ffx_true[b, active_d])
                        lme4_bt.append(ffx_true[b, active_d])
                        lme4_se.append(est_l['sigma_rfx'] - srfx_true[b, active_q])
                        lme4_st.append(srfx_true[b, active_q])
                        lme4_re.append(
                            (est_l['blups'][:m_b] - rfx_true[b, :m_b][:, active_q]).reshape(-1)
                        )
                        lme4_rt.append(re_t)

    def flat(lst: list) -> np.ndarray:
        return np.concatenate(lst) if lst else np.array([np.nan])

    pql_a = {
        'be': flat(pql_a_be),
        'bt': flat(pql_a_bt),
        'se': flat(pql_a_se),
        'st': flat(pql_a_st),
        're': flat(pql_a_re),
        'rt': flat(pql_a_rt),
    }
    pql_cv = {
        'be': flat(pql_cv_be),
        'bt': flat(pql_cv_bt),
        'se': flat(pql_cv_se),
        'st': flat(pql_cv_st),
        're': flat(pql_cv_re),
        'rt': flat(pql_cv_rt),
    }
    pql_lv = {
        'be': flat(pql_lv_be),
        'bt': flat(pql_lv_bt),
        'se': flat(pql_lv_se),
        'st': flat(pql_lv_st),
        're': flat(pql_lv_re),
        'rt': flat(pql_lv_rt),
    }
    cavi = {
        'be': flat(cavi_be),
        'bt': flat(cavi_bt),
        'se': flat(cavi_se),
        'st': flat(cavi_st),
        're': flat(cavi_re),
        'rt': flat(cavi_rt),
    }
    lme4d = {
        'be': flat(lme4_be),
        'bt': flat(lme4_bt),
        'se': flat(lme4_se),
        'st': flat(lme4_st),
        're': flat(lme4_re),
        'rt': flat(lme4_rt),
    }

    n_pql_all = len(pql_a_be)

    # Summary metrics for cross-dataset table
    metrics = {
        'data_id': data_id,
        'n_pql': n_pql_all,
        'n_cavi': cavi_n_ok,
        'n_lme4': lme4_n_ok,
        'pql_ffx': _nrmse(pql_a['be'], pql_a['bt']),
        'pql_srfx': _nrmse(pql_a['se'], pql_a['st']),
        'pql_blup': _nrmse(pql_a['re'], pql_a['rt']),
        'cavi_ffx': _nrmse(cavi['be'], cavi['bt']) if cavi_n_ok > 0 else float('nan'),
        'cavi_srfx': _nrmse(cavi['se'], cavi['st']) if cavi_n_ok > 0 else float('nan'),
        'cavi_blup': _nrmse(cavi['re'], cavi['rt']) if cavi_n_ok > 0 else float('nan'),
        'lme4_ffx': _nrmse(lme4d['be'], lme4d['bt']) if lme4_n_ok > 0 else float('nan'),
        'lme4_srfx': _nrmse(lme4d['se'], lme4d['st']) if lme4_n_ok > 0 else float('nan'),
        'lme4_blup': _nrmse(lme4d['re'], lme4d['rt']) if lme4_n_ok > 0 else float('nan'),
    }

    # -----------------------------------------------------------------------
    # Print per-dataset results
    # -----------------------------------------------------------------------
    sep = '=' * 72
    print(sep)
    print(f'  {data_id}  |  partition={partition}  N={n_pql_all}')
    print(sep)

    # 1. PQL overall
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

    # 2. PQL vs CAVI (matched)
    if cavi_n_ok > 0:
        print(f'\nPQL vs CAVI — matched (PQL N={len(pql_cv_be)}, CAVI {cavi_n_ok}/{cavi_n_tried})')
        print(
            tabulate(
                [
                    [
                        'PQL',
                        'FFX (β)',
                        f'{_nrmse(pql_cv["be"],pql_cv["bt"]):.4f}',
                        f'{_bias(pql_cv["be"]):+.4f}',
                    ],
                    [
                        'PQL',
                        'σ_rfx',
                        f'{_nrmse(pql_cv["se"],pql_cv["st"]):.4f}',
                        f'{_bias(pql_cv["se"]):+.4f}',
                    ],
                    [
                        'PQL',
                        'BLUP',
                        f'{_nrmse(pql_cv["re"],pql_cv["rt"]):.4f}',
                        f'{_bias(pql_cv["re"]):+.4f}',
                    ],
                    ['CAVI', 'FFX (β)', f'{metrics["cavi_ffx"]:.4f}', f'{_bias(cavi["be"]):+.4f}'],
                    ['CAVI', 'σ_rfx', f'{metrics["cavi_srfx"]:.4f}', f'{_bias(cavi["se"]):+.4f}'],
                    ['CAVI', 'BLUP', f'{metrics["cavi_blup"]:.4f}', f'{_bias(cavi["re"]):+.4f}'],
                ],
                headers=['Method', 'Parameter', 'NRMSE', 'Bias'],
                tablefmt='simple',
            )
        )

    # 3. PQL vs lme4 (matched)
    if lme4_n_ok > 0:
        print(f'\nPQL vs lme4 — matched (PQL N={len(pql_lv_be)}, lme4 {lme4_n_ok}/{lme4_n_tried})')
        print(
            tabulate(
                [
                    [
                        'PQL',
                        'FFX (β)',
                        f'{_nrmse(pql_lv["be"],pql_lv["bt"]):.4f}',
                        f'{_bias(pql_lv["be"]):+.4f}',
                    ],
                    [
                        'PQL',
                        'σ_rfx',
                        f'{_nrmse(pql_lv["se"],pql_lv["st"]):.4f}',
                        f'{_bias(pql_lv["se"]):+.4f}',
                    ],
                    [
                        'PQL',
                        'BLUP',
                        f'{_nrmse(pql_lv["re"],pql_lv["rt"]):.4f}',
                        f'{_bias(pql_lv["re"]):+.4f}',
                    ],
                    ['lme4', 'FFX (β)', f'{metrics["lme4_ffx"]:.4f}', f'{_bias(lme4d["be"]):+.4f}'],
                    ['lme4', 'σ_rfx', f'{metrics["lme4_srfx"]:.4f}', f'{_bias(lme4d["se"]):+.4f}'],
                    ['lme4', 'BLUP', f'{metrics["lme4_blup"]:.4f}', f'{_bias(lme4d["re"]):+.4f}'],
                ],
                headers=['Method', 'Parameter', 'NRMSE', 'Bias'],
                tablefmt='simple',
            )
        )

    # 4. σ_rfx bias breakdown
    print('\nσ_rfx bias by true σ_rfx bin — PQL')
    print(_breakdown_srfx(pql_a['se'], pql_a['st'], 'PQL'))
    if cavi_n_ok > 0:
        print('\nσ_rfx bias by true σ_rfx bin — CAVI')
        print(_breakdown_srfx(cavi['se'], cavi['st'], 'CAVI'))
    if lme4_n_ok > 0:
        print('\nσ_rfx bias by true σ_rfx bin — lme4')
        print(_breakdown_srfx(lme4d['se'], lme4d['st'], 'lme4'))

    # 5. Wall time
    w_pql = np.array(pql_wall)
    lines = [
        f'PQL:  mean={w_pql.mean()*1000:.2f} ms  median={np.median(w_pql)*1000:.2f} ms / dataset'
    ]
    if cavi_wall:
        w = np.array(cavi_wall)
        lines.append(f'CAVI: mean={w.mean():.3f} s  median={np.median(w):.3f} s / dataset')
    if lme4_wall:
        w = np.array(lme4_wall)
        lines.append(f'lme4: mean={w.mean():.3f} s  median={np.median(w):.3f} s / dataset')
    print('\nWall time')
    print('\n'.join(lines))
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
    n_lme4: int = 50,
) -> None:
    print(
        f'CAVI: {"enabled" if _HAS_CAVI else "DISABLED"}   vcp_prior={_VCP_PRIOR}   limit={n_cavi}'
    )
    print(f'lme4: {"enabled" if _HAS_LME4 else "DISABLED"}   optimizer=bobyqa   limit={n_lme4}')
    print()

    all_metrics = []
    for data_id in data_ids:
        m = run_one_dataset(
            data_id=data_id,
            partition=partition,
            n_cavi=n_cavi,
            n_lme4=n_lme4,
            n_epochs=n_epochs,
        )
        all_metrics.append(m)

    if len(all_metrics) > 1:
        print('=' * 72)
        print('  SUMMARY — NRMSE across datasets  (B=BLUP, F=FFX, S=σ_rfx)')
        print('=' * 72)
        headers = [
            'Dataset',
            'N',
            'PQL-F',
            'PQL-S',
            'PQL-B',
            'CAVI-F',
            'CAVI-S',
            'CAVI-B',
            'lme4-F',
            'lme4-S',
            'lme4-B',
        ]
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
                    fmt(m['cavi_ffx']),
                    fmt(m['cavi_srfx']),
                    fmt(m['cavi_blup']),
                    fmt(m['lme4_ffx']),
                    fmt(m['lme4_srfx']),
                    fmt(m['lme4_blup']),
                ]
            )
        print(tabulate(rows, headers=headers, tablefmt='simple'))


if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description='PQL vs CAVI vs lme4 comparison on Bernoulli datasets')
    parser.add_argument('--data-ids',  default='small-b-sampled',
                        help='comma-separated data config ids (default: small-b-sampled)')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-epochs',  default=1, type=int)
    parser.add_argument('--n-cavi',    default=200, type=int, help='max datasets for CAVI per data_id')
    parser.add_argument('--n-lme4',    default=50,  type=int, help='max datasets for lme4 per data_id')
    # fmt: on
    a = parser.parse_args()
    main(
        data_ids=[x.strip() for x in a.data_ids.split(',')],
        partition=a.partition,
        n_epochs=a.n_epochs,
        n_cavi=a.n_cavi,
        n_lme4=a.n_lme4,
    )
