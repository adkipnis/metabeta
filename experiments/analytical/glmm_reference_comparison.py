"""Experiment: PQL (glmm) vs statsmodels BinomialBayesMixedGLM (CAVI).

Compares our PQL estimator against a CAVI reference on Bernoulli test datasets.
CAVI is only run on q=1 datasets (scalar random intercept); PQL results are
reported both globally and on the same q=1 subset so the comparison is apples-
to-apples.

Goal: establish how much headroom exists above PQL for the nAGQ (P5) and
True Laplace LML (P6) improvement directions.

Note: prior matching is approximate.  CAVI uses a Gaussian prior on log(σ²) via
vcp_p; we use HalfNormal(τ_rfx) on σ.  The comparison is frequentist — accuracy
relative to the simulated ground truth — not a matched-prior Bayesian comparison.

Usage (from repo root):
    uv run python experiments/analytical/glmm_reference_comparison.py
    uv run python experiments/analytical/glmm_reference_comparison.py \\
        --data-id small-b-sampled --n-cavi 200
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


def _flatten(batch: dict, b: int, active_d: np.ndarray) -> dict:
    """Reconstruct flat (n, len(active_d)) dataset from item b in a grouped batch."""
    m = int(batch['m'][b].item())
    d = len(active_d)
    X_parts, y_parts, g_parts = [], [], []
    for g in range(m):
        ng = int(batch['ns'][b, g].item())
        if ng == 0:
            continue
        X_g = batch['X'][b, g, :ng].cpu().numpy()[:, active_d]
        y_g = batch['y'][b, g, :ng].cpu().numpy()
        X_parts.append(X_g)
        y_parts.append(y_g)
        g_parts.append(np.full(ng, g, dtype=int))
    if not X_parts:
        return {
            'X': np.zeros((0, d)),
            'y': np.zeros(0),
            'groups': np.zeros(0, dtype=int),
            'm': m,
            'd': d,
        }
    return {
        'X': np.vstack(X_parts),
        'y': np.concatenate(y_parts),
        'groups': np.concatenate(g_parts),
        'm': m,
        'd': d,
    }


def _cavi_estimate(ds_flat: dict) -> dict | None:
    """Run statsmodels CAVI on a flat q=1 Bernoulli dataset.

    Returns dict with 'beta' (d,), 'sigma_rfx' (scalar), 'blups' (m,),
    or None if the fit fails.

    API notes (statsmodels 0.14+):
      - result.fe_mean : fixed-effect posterior means, shape (d,)
      - result.vcp_mean : log-variance component means, shape (1,); vcp = log σ²
      - sigma_rfx = exp(vcp_mean[0] / 2)
      - RE estimates: result.params[len(fe_mean) + len(vcp_mean):]  — one per group
      - model.k_fep / model.k_vcp exist but model.k_fe / model.k_vc do not
    """
    if not _HAS_CAVI:
        return None

    d = int(ds_flat['d'])
    m = int(ds_flat['m'])
    X = ds_flat['X']  # (n, d)
    y = ds_flat['y']  # (n,)
    groups = ds_flat['groups']  # (n,) ints 0..m-1

    df_data: dict = {'y': y.astype(float), 'group': groups.astype(int)}
    for j in range(1, d):
        df_data[f'x{j}'] = X[:, j].astype(float)
    df = pd.DataFrame(df_data)

    # '0 + C(group)' gives one coefficient per group with a shared variance component
    formula = 'y ~ ' + (' + '.join(f'x{j}' for j in range(1, d)) if d > 1 else '1')
    vc_formula = {'rfx': '0 + C(group)'}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = BinomialBayesMixedGLM.from_formula(formula, vc_formula, df, vcp_p=_VCP_PRIOR)
            result = model.fit_vb()

        fe = np.asarray(result.fe_mean, dtype=float)    # (d,)
        vcp = np.asarray(result.vcp_mean, dtype=float)  # (1,); log σ²
        sigma_rfx = float(np.exp(vcp[0] / 2.0))

        k_fe = len(fe)
        k_vcp = len(vcp)
        re = np.asarray(result.params[k_fe + k_vcp :], dtype=float)  # (m,) sorted by group

        return {'beta': fe, 'sigma_rfx': sigma_rfx, 'blups': re[:m]}
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
# Main
# ---------------------------------------------------------------------------


def run_comparison(
    data_id: str = 'small-b-sampled',
    partition: str = 'test',
    n_epochs: int = 1,
    n_cavi: int = 200,
) -> None:
    device = torch.device('cpu')
    data_cfg = loadDataConfig(data_id)
    max_d = data_cfg['max_d']
    max_q = data_cfg['max_q']
    likelihood_family = data_cfg.get('likelihood_family', 0)

    if likelihood_family != 1:
        print(f'WARNING: likelihood_family={likelihood_family}, expected 1 (Bernoulli)')

    if partition == 'train':
        paths = [dataFilePath(data_cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
        paths = [p for p in paths if p.exists()]
    elif partition == 'test':
        paths = [dataFilePath(data_cfg['data_id'], 'test')]
    else:
        paths = [dataFilePath(data_cfg['data_id'], 'valid')]

    assert paths and paths[0].exists(), f'No data at {paths[0] if paths else data_id}/{partition}'

    print(f'data_id={data_id}  partition={partition}  likelihood_family={likelihood_family}')
    print(f'CAVI enabled: {_HAS_CAVI}   vcp_prior={_VCP_PRIOR}   n_cavi_limit={n_cavi}')
    print()

    # Accumulators: _all = all-q PQL, _q1 = q=1 only (for fair side-by-side)
    pql_all_be, pql_all_bt = [], []
    pql_all_se, pql_all_st = [], []
    pql_all_re, pql_all_rt = [], []

    pql_q1_be, pql_q1_bt = [], []
    pql_q1_se, pql_q1_st = [], []
    pql_q1_re, pql_q1_rt = [], []

    cavi_be, cavi_bt = [], []
    cavi_se, cavi_st = [], []
    cavi_re, cavi_rt = [], []
    cavi_wall_list: list[float] = []
    pql_q1_wall_list: list[float] = []
    cavi_n_tried = cavi_n_ok = 0

    all_batches = []
    for path in paths:
        dl = Dataloader(path, batch_size=32, shuffle=False)
        all_batches.extend(list(dl))

    print(f'Total batches: {len(all_batches)}')

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
            mask_m_np = batch['mask_m'].cpu().numpy().astype(bool)
            m_np = batch['m'].cpu().numpy()

            for b in range(B):
                active_d = np.flatnonzero(mask_d_np[b])
                active_q = np.flatnonzero(mask_q_np[b])
                m_b = int(m_np[b])
                q_b = len(active_q)

                be = beta_est[b, active_d] - ffx_true[b, active_d]
                se = srfx_est[b, active_q] - srfx_true[b, active_q]
                re_e = (blup_est[b, :m_b][:, active_q] - rfx_true[b, :m_b][:, active_q]).reshape(-1)

                pql_all_be.append(be)
                pql_all_bt.append(ffx_true[b, active_d])
                pql_all_se.append(se)
                pql_all_st.append(srfx_true[b, active_q])
                pql_all_re.append(re_e)
                pql_all_rt.append(rfx_true[b, :m_b][:, active_q].reshape(-1))

                if q_b != 1:
                    continue

                # q=1 PQL accumulator (wall time prorated per dataset)
                pql_q1_be.append(be)
                pql_q1_bt.append(ffx_true[b, active_d])
                pql_q1_se.append(se)
                pql_q1_st.append(srfx_true[b, active_q])
                pql_q1_re.append(re_e)
                pql_q1_rt.append(rfx_true[b, :m_b][:, active_q].reshape(-1))
                pql_q1_wall_list.append(pql_batch_wall / B)

                if cavi_n_tried >= n_cavi:
                    continue

                # CAVI
                cavi_n_tried += 1
                ds_flat = _flatten(batch, b, active_d)
                t_cavi = time.perf_counter()
                est = _cavi_estimate(ds_flat)
                cavi_wall_list.append(time.perf_counter() - t_cavi)

                if est is None:
                    continue
                cavi_n_ok += 1

                fe_cavi = est['beta']  # (d,) aligned with ffx_true[b, active_d]
                cavi_be.append(fe_cavi - ffx_true[b, active_d])
                cavi_bt.append(ffx_true[b, active_d])
                cavi_se.append(np.array([est['sigma_rfx']]) - srfx_true[b, active_q])
                cavi_st.append(srfx_true[b, active_q])

                blups_cavi = est['blups'][:m_b]  # (m,)
                re_true_b = rfx_true[b, :m_b, active_q[0]]  # (m,)
                cavi_re.append(blups_cavi - re_true_b)
                cavi_rt.append(re_true_b)

    # -----------------------------------------------------------------------
    # Flatten accumulators
    # -----------------------------------------------------------------------
    def flat(lst: list) -> np.ndarray:
        return np.concatenate(lst) if lst else np.array([np.nan])

    pql_a_be, pql_a_bt = flat(pql_all_be), flat(pql_all_bt)
    pql_a_se, pql_a_st = flat(pql_all_se), flat(pql_all_st)
    pql_a_re, pql_a_rt = flat(pql_all_re), flat(pql_all_rt)

    pql_1_be, pql_1_bt = flat(pql_q1_be), flat(pql_q1_bt)
    pql_1_se, pql_1_st = flat(pql_q1_se), flat(pql_q1_st)
    pql_1_re, pql_1_rt = flat(pql_q1_re), flat(pql_q1_rt)

    c_be, c_bt = flat(cavi_be), flat(cavi_bt)
    c_se, c_st = flat(cavi_se), flat(cavi_st)
    c_re, c_rt = flat(cavi_re), flat(cavi_rt)

    n_pql_all = len(pql_all_be)
    n_pql_q1 = len(pql_q1_be)

    # -----------------------------------------------------------------------
    # 1. PQL overall
    # -----------------------------------------------------------------------
    print('=' * 65)
    print(f'  PQL — all q  (N={n_pql_all} datasets)')
    print('=' * 65)
    rows = [
        ['FFX (β)', f'{_nrmse(pql_a_be, pql_a_bt):.4f}', f'{_bias(pql_a_be):+.4f}'],
        ['σ_rfx', f'{_nrmse(pql_a_se, pql_a_st):.4f}', f'{_bias(pql_a_se):+.4f}'],
        ['BLUP', f'{_nrmse(pql_a_re, pql_a_rt):.4f}', f'{_bias(pql_a_re):+.4f}'],
    ]
    print(tabulate(rows, headers=['Parameter', 'NRMSE', 'Bias'], tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 2. PQL vs CAVI on q=1 subset
    # -----------------------------------------------------------------------
    print()
    print('=' * 65)
    print(f'  PQL vs CAVI — q=1 only  (PQL N={n_pql_q1}, CAVI {cavi_n_ok}/{cavi_n_tried})')
    print('=' * 65)
    rows2 = [
        ['PQL', 'FFX (β)', f'{_nrmse(pql_1_be, pql_1_bt):.4f}', f'{_bias(pql_1_be):+.4f}'],
        ['PQL', 'σ_rfx', f'{_nrmse(pql_1_se, pql_1_st):.4f}', f'{_bias(pql_1_se):+.4f}'],
        ['PQL', 'BLUP', f'{_nrmse(pql_1_re, pql_1_rt):.4f}', f'{_bias(pql_1_re):+.4f}'],
    ]
    if cavi_n_ok > 0:
        rows2 += [
            ['CAVI', 'FFX (β)', f'{_nrmse(c_be, c_bt):.4f}', f'{_bias(c_be):+.4f}'],
            ['CAVI', 'σ_rfx', f'{_nrmse(c_se, c_st):.4f}', f'{_bias(c_se):+.4f}'],
            ['CAVI', 'BLUP', f'{_nrmse(c_re, c_rt):.4f}', f'{_bias(c_re):+.4f}'],
        ]
    print(tabulate(rows2, headers=['Method', 'Parameter', 'NRMSE', 'Bias'], tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 3. σ_rfx bias by true σ_rfx — key diagnostic for Breslow-Lin bias
    # -----------------------------------------------------------------------
    print()
    print('=== σ_rfx bias by true σ_rfx bin — PQL (q=1 subset) ===')
    print(_breakdown_srfx(pql_1_se, pql_1_st, 'PQL q=1'))

    if cavi_n_ok > 0:
        print()
        print('=== σ_rfx bias by true σ_rfx bin — CAVI ===')
        print(_breakdown_srfx(c_se, c_st, 'CAVI'))

    # -----------------------------------------------------------------------
    # 4. Wall time
    # -----------------------------------------------------------------------
    print()
    print('=== Wall time per dataset (seconds) ===')
    wt_rows = []
    if pql_q1_wall_list:
        w = np.array(pql_q1_wall_list)
        wt_rows.append(
            ['PQL (q=1 prorated)', f'{w.mean()*1000:.2f} ms', f'{np.median(w)*1000:.2f} ms']
        )
    if cavi_wall_list:
        w = np.array(cavi_wall_list)
        wt_rows.append(['CAVI', f'{w.mean():.3f} s', f'{np.median(w):.3f} s'])
    if wt_rows:
        print(tabulate(wt_rows, headers=['Method', 'Mean', 'Median'], tablefmt='simple'))

    if not _HAS_CAVI:
        print('\nWARNING: statsmodels not available; CAVI comparison skipped.')


if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description='PQL vs CAVI comparison on Bernoulli datasets')
    parser.add_argument('--data-id',   default='small-b-sampled', help='data config id (default: small-b-sampled)')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--n-epochs',  default=1, type=int, help='train epochs to load (train partition only)')
    parser.add_argument('--n-cavi',    default=200, type=int, help='max q=1 datasets to run CAVI on (default: 200)')
    # fmt: on
    a = parser.parse_args()
    run_comparison(
        data_id=a.data_id,
        partition=a.partition,
        n_epochs=a.n_epochs,
        n_cavi=a.n_cavi,
    )
