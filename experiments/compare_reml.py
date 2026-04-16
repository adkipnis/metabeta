"""Compare glmm.py estimates against statsmodels REML on a representative sample.

Runs statsmodels MixedLM (REML) on individual datasets and compares against:
  - glmm.py estimates (our batched implementation)
  - ground truth

Usage (from metabeta/simulation/):
    uv run python compare_reml.py [--n_datasets 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent / 'metabeta'
sys.path.insert(0, str(ROOT.parent))

from metabeta.utils.glmm import glmm
from metabeta.utils.dataloader import Collection, Dataloader, collateGrouped, toDevice
from metabeta.utils.config import loadDataConfig


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    """NaN-safe NRMSE; NaN errors (REML failures) are excluded."""
    finite = np.isfinite(err)
    if finite.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.nanmean(err[finite]**2))) / max(float(np.std(truth[finite])), 1e-8)


def _bias(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


def _reml_estimates(
    X: np.ndarray,      # (N, d) — col 0 is intercept (all 1s)
    y: np.ndarray,      # (N,)
    Z: np.ndarray,      # (N, q) — col 0 is intercept; col 1 (if q=2) is random slope
    group: np.ndarray,  # (N,) int group ids
    q: int,
    m: int,
) -> dict | None:
    """Run statsmodels MixedLM (REML).

    Returns dict with sigma_eps, sigma_rfx, beta (d,), blups (m, q), or None on failure.
    sigma_rfx = sqrt(mean of Psi diagonal) to match glmm.py convention.
    """
    import statsmodels.regression.mixed_linear_model as mlm

    N, d = X.shape
    exog_fe = np.hstack([np.ones((N, 1)), X[:, 1:]])  # intercept + FE covariates

    # q=1: None → statsmodels default random intercept
    # q=2: full Z = [1, z1] → random intercept + slope; statsmodels builds 2×2 Ψ
    exog_re = None if q == 1 else Z

    if np.linalg.matrix_rank(exog_fe) < exog_fe.shape[1]:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model = mlm.MixedLM(endog=y, exog=exog_fe, groups=group, exog_re=exog_re)
            result = model.fit(reml=True, method='lbfgs', disp=False)
            if not result.converged:
                result = model.fit(reml=True, method='bfgs', disp=False)
            if not result.converged:
                result = model.fit(reml=True, method='nm', disp=False)

            sigma_eps = float(result.scale**0.5)

            cov_re_raw = result.cov_re
            try:
                cov_diag = np.diag(np.atleast_2d(np.array(cov_re_raw, dtype=float)))
            except Exception:
                cov_diag = np.array([float(cov_re_raw)])
            sigma_rfx = float(np.sqrt(cov_diag.mean()))

            # Fixed effects: fe_params ordered as [intercept, x1, ..., x_{d-1}]
            beta = np.asarray(result.fe_params, dtype=float).ravel()  # (d,)

            # BLUPs: random_effects is a dict {group_id: Series/scalar}
            # Keys are np.int64 but compare equal to Python int, so .get(g) works.
            blups = np.zeros((m, q))
            for g in range(m):
                re_g = result.random_effects.get(g)
                if re_g is None:
                    continue
                re_arr = np.atleast_1d(np.array(re_g, dtype=float)).flatten()
                blups[g, :min(q, len(re_arr))] = re_arr[:min(q, len(re_arr))]

            return {
                'sigma_eps': sigma_eps,
                'sigma_rfx': sigma_rfx,
                'beta': beta,
                'blups': blups,
            }
        except Exception:
            return None


def run_comparison(n_datasets: int = 1000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    data_cfg = loadDataConfig('small-n-mixed')
    max_q = data_cfg['max_q']
    data_dir = ROOT / 'outputs' / 'data' / data_cfg['data_id']

    from metabeta.utils.io import datasetFilename
    paths = [data_dir / datasetFilename('train', ep) for ep in range(1, 21)]
    paths = [p for p in paths if p.exists()]

    # Collect all batches then sample (permute=False so X col 0 = intercept)
    print(f'Loading {len(paths)} training files ...')
    all_batches: list = []
    for path in paths:
        col = Collection(path, permute=False)
        dl = torch.utils.data.DataLoader(
            col, batch_size=32, shuffle=False, collate_fn=collateGrouped
        )
        all_batches.extend(list(dl))

    # Flatten to per-dataset records
    records: list[dict] = []
    for batch in all_batches:
        B = batch['X'].shape[0]
        for b in range(B):
            records.append({'batch': batch, 'b': b})
    print(f'Total datasets: {len(records)}')

    # Stratified sample: spread across bg_df range
    idx = rng.choice(len(records), size=min(n_datasets, len(records)), replace=False)
    sample = [records[i] for i in idx]
    print(f'Sampled {len(sample)} datasets for REML comparison\n')

    device = torch.device('cpu')

    # Results
    seps_our, seps_reml, seps_true_list = [], [], []
    srfx_our, srfx_reml, srfx_true_list = [], [], []
    # Per-coefficient FFX errors (flat across datasets × d)
    beta_err_our_all, beta_err_reml_all, beta_true_all = [], [], []
    # Per-BLUP RFX errors (flat across datasets × m × q)
    rfx_err_our_all, rfx_err_reml_all, rfx_true_all, rfx_ng_our_all = [], [], [], []
    failed = 0
    meta: list[dict] = []

    for rec in sample:
        batch = toDevice(rec['batch'], device)
        b = rec['b']

        B = batch['X'].shape[0]
        Zm = batch['Z'][..., :max_q]

        # Our estimate (batched glmm on the whole batch, take element b)
        with torch.no_grad():
            stats = glmm(
                batch['X'], batch['y'], Zm,
                batch['mask_n'].float(), batch['mask_m'].float(),
                batch['ns'].clamp(min=1).float(), batch['n'].float(),
                likelihood_family=0,
                eta_rfx=batch.get('eta_rfx'),
            )

        # Boolean masks for active dims — robust to column permutation.
        # permute=False is required above so that X[:, 0] remains the intercept,
        # which _reml_estimates passes to statsmodels as the implicit constant term.
        mask_d    = batch['mask_d'][b].cpu().numpy().astype(bool)   # (d_max,)
        mask_q    = batch['mask_q'][b].cpu().numpy().astype(bool)   # (q_max,)
        mask_m    = batch['mask_m'][b].cpu().numpy().astype(bool)   # (m_max,)
        mask_n_np = batch['mask_n'][b].cpu().numpy().astype(bool)   # (m_max, n_max)
        ns_b      = batch['ns'][b].cpu().numpy()                    # (m_max,)

        d = int(mask_d.sum())
        q = int(mask_q.sum())
        m = int(mask_m.sum())
        n_total = int(batch['n'][b].item())

        srfx_true    = float(batch['sigma_rfx'][b].cpu().numpy()[mask_q].mean())
        seps_true    = float(batch['sigma_eps'][b].item())
        srfx_est_our = float(stats['sigma_rfx_est'][b].cpu().numpy()[mask_q].mean())
        seps_est_our = float(stats['sigma_eps_est'][b, 0].item())

        # Ground-truth FFX and RFX (active dims only)
        beta_true = batch['ffx'][b].cpu().numpy()[mask_d]           # (d,)
        rfx_true  = batch['rfx'][b].cpu().numpy()[mask_m, :][:, mask_q]  # (m, q)

        # Our FFX and RFX estimates (active dims only)
        beta_our = stats['beta_est'][b].cpu().numpy()[mask_d]        # (d,)
        rfx_our  = stats['blup_est'][b].cpu().numpy()[mask_m, :][:, mask_q]  # (m, q)

        # Build flat arrays for statsmodels, using boolean masks to select active
        # dims and observations — correct regardless of column ordering.
        Xnp = batch['X'][b].cpu().numpy()       # (m_max, n_max, d_max)
        ynp = batch['y'][b].cpu().numpy()        # (m_max, n_max)
        Znp = batch['Z'][b].cpu().numpy()        # (m_max, n_max, q_max)

        X_flat, y_flat, Z_flat, grp_flat = [], [], [], []
        for g in range(m):
            for i in range(int(ns_b[g])):
                if mask_n_np[g, i]:
                    X_flat.append(Xnp[g, i][mask_d])
                    y_flat.append(ynp[g, i])
                    Z_flat.append(Znp[g, i][mask_q])
                    grp_flat.append(g)

        X_flat = np.array(X_flat)
        y_flat = np.array(y_flat)
        Z_flat = np.array(Z_flat)
        grp_flat = np.array(grp_flat)

        if len(y_flat) < d + q + 2:
            failed += 1
            continue

        res = _reml_estimates(X_flat, y_flat, Z_flat, grp_flat, q, m)
        if res is None:
            failed += 1
            continue

        seps_r   = res['sigma_eps']
        srfx_r   = res['sigma_rfx']
        beta_reml = res['beta']        # (d,)
        rfx_reml  = res['blups'][:m, :q]  # (m, q)

        seps_our.append(seps_est_our - seps_true)
        seps_reml.append(seps_r - seps_true)
        seps_true_list.append(seps_true)

        srfx_our.append(srfx_est_our - srfx_true)
        srfx_reml.append(srfx_r - srfx_true)
        srfx_true_list.append(srfx_true)

        # Accumulate per-coefficient FFX errors (flat).
        # Flag REML beta as unreliable when any element exceeds 50× true scale
        # (statsmodels q=2 random-slopes model can diverge for small samples).
        beta_scale = float(np.abs(beta_true).max()) + 1.0
        beta_reml_ok = float(np.abs(beta_reml).max()) < 50.0 * beta_scale
        beta_err_our_all.append(beta_our - beta_true)
        beta_err_reml_all.append(
            beta_reml - beta_true if beta_reml_ok else np.full_like(beta_true, np.nan)
        )
        beta_true_all.append(beta_true)

        # Accumulate per-BLUP RFX errors (flat m×q) and group sizes per BLUP
        rfx_err_our_all.append((rfx_our - rfx_true).ravel())
        rfx_err_reml_all.append((rfx_reml - rfx_true).ravel())
        rfx_true_all.append(rfx_true.ravel())
        # repeat each group's n_g for each rfx dimension
        ns_active = batch['ns'][b].cpu().numpy()[mask_m]   # (m,)
        rfx_ng_our_all.append(np.repeat(ns_active, q))

        meta.append({
            'n': n_total, 'm': m, 'd': d, 'q': q,
            'bg_df': max(m - d, 1),
            'srfx_true': srfx_true,
            'seps_true': seps_true,
            'srfx_err_our': srfx_est_our - srfx_true,
            'srfx_err_reml': srfx_r - srfx_true,
            'seps_err_our': seps_est_our - seps_true,
            'seps_err_reml': seps_r - seps_true,
            'beta_rmse_our':  float(np.sqrt(np.mean((beta_our - beta_true)**2))),
            'beta_rmse_reml': float(np.sqrt(np.mean((beta_reml - beta_true)**2))),
            'rfx_rmse_our':   float(np.sqrt(np.mean((rfx_our - rfx_true)**2))),
            'rfx_rmse_reml':  float(np.sqrt(np.mean((rfx_reml - rfx_true)**2))),
        })

    print(f'Converged: {len(meta)}/{len(sample)}  failed/skipped: {failed}\n')

    seps_our = np.array(seps_our)
    seps_reml = np.array(seps_reml)
    seps_true_arr = np.array(seps_true_list)
    srfx_our = np.array(srfx_our)
    srfx_reml = np.array(srfx_reml)
    srfx_true_arr = np.array(srfx_true_list)
    beta_err_our  = np.concatenate(beta_err_our_all)   # all β errors pooled
    beta_err_reml = np.concatenate(beta_err_reml_all)
    beta_true_flat = np.concatenate(beta_true_all)
    rfx_err_our   = np.concatenate(rfx_err_our_all)    # all BLUP errors pooled
    rfx_err_reml  = np.concatenate(rfx_err_reml_all)
    rfx_true_flat  = np.concatenate(rfx_true_all)

    # -----------------------------------------------------------------------
    # 1. Overall NRMSE / bias table
    # -----------------------------------------------------------------------
    print('=' * 70)
    print('  glmm.py  vs  statsmodels REML  vs  ground truth')
    print('=' * 70)
    rows = [
        ['FFX (β)',
         f'{_nrmse(beta_err_our, beta_true_flat):.4f}', f'{_bias(beta_err_our):+.4f}',
         f'{_nrmse(beta_err_reml, beta_true_flat):.4f}', f'{_bias(beta_err_reml):+.4f}'],
        ['RFX (BLUPs)',
         f'{_nrmse(rfx_err_our, rfx_true_flat):.4f}', f'{_bias(rfx_err_our):+.4f}',
         f'{_nrmse(rfx_err_reml, rfx_true_flat):.4f}', f'{_bias(rfx_err_reml):+.4f}'],
        ['Sigma(RFX)',
         f'{_nrmse(srfx_our, srfx_true_arr):.4f}', f'{_bias(srfx_our):+.4f}',
         f'{_nrmse(srfx_reml, srfx_true_arr):.4f}', f'{_bias(srfx_reml):+.4f}'],
        ['Sigma(Eps)',
         f'{_nrmse(seps_our, seps_true_arr):.4f}', f'{_bias(seps_our):+.4f}',
         f'{_nrmse(seps_reml, seps_true_arr):.4f}', f'{_bias(seps_reml):+.4f}'],
    ]
    print(tabulate(rows,
        headers=['', 'NRMSE (ours)', 'Bias (ours)', 'NRMSE (REML)', 'Bias (REML)'],
        tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 2. Sigma(RFX) breakdown by true value
    # -----------------------------------------------------------------------
    def col(k): return np.array([r[k] for r in meta])
    srfx_true_d = col('srfx_true')
    bg_df_arr   = col('bg_df')
    q_arr       = col('q')

    print('\n=== Sigma(RFX) error breakdown by true sigma_rfx ===')
    edges = np.nanpercentile(srfx_true_d, [0, 25, 50, 75, 100])
    edges = np.unique(edges)
    rows = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mask = (srfx_true_d >= lo) & (srfx_true_d <= hi)
        eo = srfx_our[mask]; er = srfx_reml[mask]; t = srfx_true_d[mask]
        if len(eo) < 2: continue
        rows.append([
            f'{lo:.3f}–{hi:.3f}', int(mask.sum()),
            f'{_bias(eo):+.4f}', f'{np.sqrt(np.mean(eo**2)):.4f}',
            f'{_bias(er):+.4f}', f'{np.sqrt(np.mean(er**2)):.4f}',
        ])
    print(tabulate(rows,
        headers=['sigma_rfx range', 'N',
                 'Bias (ours)', 'RMSE (ours)',
                 'Bias (REML)', 'RMSE (REML)'],
        tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 3. Breakdown by bg_df
    # -----------------------------------------------------------------------
    print('\n=== Sigma(RFX) error breakdown by between-group df (m−d) ===')
    bg_vals = sorted(np.unique(bg_df_arr[np.isfinite(bg_df_arr)]).astype(int))
    # bin into groups of 3 for readability
    rows = []
    bins = [(1,3),(4,7),(8,12),(13,23)]
    for lo, hi in bins:
        mask = (bg_df_arr >= lo) & (bg_df_arr <= hi)
        if mask.sum() < 2: continue
        eo = srfx_our[mask]; er = srfx_reml[mask]
        rows.append([
            f'{lo}–{hi}', int(mask.sum()),
            f'{_bias(eo):+.4f}', f'{np.sqrt(np.mean(eo**2)):.4f}',
            f'{_bias(er):+.4f}', f'{np.sqrt(np.mean(er**2)):.4f}',
        ])
    print(tabulate(rows,
        headers=['bg_df', 'N',
                 'Bias (ours)', 'RMSE (ours)',
                 'Bias (REML)', 'RMSE (REML)'],
        tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 4. Breakdown by q (random intercept only vs intercept+slope)
    # -----------------------------------------------------------------------
    print('\n=== Sigma(RFX) error breakdown by q (random effects dimension) ===')
    rows = []
    for qv in sorted(np.unique(q_arr).astype(int)):
        mask = q_arr == qv
        if mask.sum() < 2: continue
        eo = srfx_our[mask]; er = srfx_reml[mask]; t = srfx_true_d[mask]
        rows.append([
            f'q={qv}', int(mask.sum()),
            f'{_nrmse(eo,t):.4f}', f'{_bias(eo):+.4f}',
            f'{_nrmse(er,t):.4f}', f'{_bias(er):+.4f}',
        ])
    print(tabulate(rows,
        headers=['q', 'N',
                 'NRMSE (ours)', 'Bias (ours)',
                 'NRMSE (REML)', 'Bias (REML)'],
        tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 5. Sigma(RFX) clip rate (ours) vs REML boundary (=0)
    # -----------------------------------------------------------------------
    n_clip_our  = int((srfx_our + srfx_true_arr == 0.0).sum())
    n_clip_reml = int((srfx_reml + srfx_true_arr <= 1e-6).sum())
    print(f'\nSigma(RFX) = 0 rate:  ours={n_clip_our/len(srfx_our)*100:.1f}%  '
          f'REML={n_clip_reml/len(srfx_reml)*100:.1f}%')

    # -----------------------------------------------------------------------
    # 6. Where does REML help most? — cases where REML error << ours
    # -----------------------------------------------------------------------
    delta = np.abs(srfx_our) - np.abs(srfx_reml)  # positive = REML better
    top_idx = np.argsort(delta)[-10:][::-1]
    print('\n=== Top 10 cases where REML error < ours (biggest improvement) ===')
    rows = [[
        meta[i]['n'], meta[i]['m'], meta[i]['d'], meta[i]['q'], meta[i]['bg_df'],
        f'{meta[i]["srfx_true"]:.3f}',
        f'{meta[i]["srfx_err_our"]:+.3f}',
        f'{meta[i]["srfx_err_reml"]:+.3f}',
    ] for i in top_idx]
    print(tabulate(rows,
        headers=['n', 'm', 'd', 'q', 'bg_df', 'σ_rfx_true', 'err(ours)', 'err(REML)'],
        tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 7. Where does REML hurt? — cases where REML error >> ours
    # -----------------------------------------------------------------------
    bot_idx = np.argsort(delta)[:10]
    print('\n=== Top 10 cases where REML error > ours (REML is worse) ===')
    rows = [[
        meta[i]['n'], meta[i]['m'], meta[i]['d'], meta[i]['q'], meta[i]['bg_df'],
        f'{meta[i]["srfx_true"]:.3f}',
        f'{meta[i]["srfx_err_our"]:+.3f}',
        f'{meta[i]["srfx_err_reml"]:+.3f}',
    ] for i in bot_idx]
    print(tabulate(rows,
        headers=['n', 'm', 'd', 'q', 'bg_df', 'σ_rfx_true', 'err(ours)', 'err(REML)'],
        tablefmt='simple'))

    def col(k): return np.array([r[k] for r in meta])  # noqa: redefined for convenience

    # -----------------------------------------------------------------------
    # 8. FFX (β) breakdown by n/d and by d
    # -----------------------------------------------------------------------
    print('\n=== FFX (β) breakdown by n_total/d ===')
    n_over_d = col('n') / col('d')
    beta_rmse_our_ds  = col('beta_rmse_our')
    beta_rmse_reml_ds = col('beta_rmse_reml')
    edges_nd = np.nanpercentile(n_over_d, [0, 25, 50, 75, 100])
    edges_nd = np.unique(edges_nd)
    rows = []
    for i in range(len(edges_nd) - 1):
        lo, hi = edges_nd[i], edges_nd[i + 1]
        mask = (n_over_d >= lo) & (n_over_d <= hi)
        if mask.sum() < 2:
            continue
        # flat β errors for datasets in this bin
        idx_list = np.where(mask)[0]
        eo_flat = np.concatenate([beta_err_our_all[i] for i in idx_list])
        er_flat = np.concatenate([beta_err_reml_all[i] for i in idx_list])
        tr_flat = np.concatenate([beta_true_all[i] for i in idx_list])
        rows.append([
            f'{lo:.1f}–{hi:.1f}', int(mask.sum()),
            f'{_nrmse(eo_flat, tr_flat):.4f}', f'{_bias(eo_flat):+.4f}',
            f'{_nrmse(er_flat, tr_flat):.4f}', f'{_bias(er_flat):+.4f}',
        ])
    print(tabulate(rows,
        headers=['n/d', 'N_ds', 'NRMSE (ours)', 'Bias (ours)', 'NRMSE (REML)', 'Bias (REML)'],
        tablefmt='simple'))

    print('\n=== FFX (β) breakdown by q ===')
    rows = []
    for qv in sorted(np.unique(q_arr).astype(int)):
        mask = q_arr == qv
        if mask.sum() < 2:
            continue
        idx_list = np.where(mask)[0]
        eo_flat = np.concatenate([beta_err_our_all[i] for i in idx_list])
        er_flat = np.concatenate([beta_err_reml_all[i] for i in idx_list])
        tr_flat = np.concatenate([beta_true_all[i] for i in idx_list])
        rows.append([
            f'q={qv}', int(mask.sum()),
            f'{_nrmse(eo_flat, tr_flat):.4f}', f'{_bias(eo_flat):+.4f}',
            f'{_nrmse(er_flat, tr_flat):.4f}', f'{_bias(er_flat):+.4f}',
        ])
    print(tabulate(rows,
        headers=['q', 'N_ds', 'NRMSE (ours)', 'Bias (ours)', 'NRMSE (REML)', 'Bias (REML)'],
        tablefmt='simple'))

    # -----------------------------------------------------------------------
    # 9. RFX (BLUPs) breakdown by sigma_rfx and by q
    # -----------------------------------------------------------------------
    print('\n=== RFX (BLUPs) breakdown by true sigma_rfx ===')
    edges_rfx = np.nanpercentile(srfx_true_arr, [0, 25, 50, 75, 100])
    edges_rfx = np.unique(edges_rfx)
    rows = []
    for i in range(len(edges_rfx) - 1):
        lo, hi = edges_rfx[i], edges_rfx[i + 1]
        mask = (srfx_true_arr >= lo) & (srfx_true_arr <= hi)
        if mask.sum() < 2:
            continue
        idx_list = np.where(mask)[0]
        eo_flat = np.concatenate([rfx_err_our_all[i] for i in idx_list])
        er_flat = np.concatenate([rfx_err_reml_all[i] for i in idx_list])
        tr_flat = np.concatenate([rfx_true_all[i] for i in idx_list])
        rows.append([
            f'{lo:.3f}–{hi:.3f}', int(mask.sum()),
            f'{_nrmse(eo_flat, tr_flat):.4f}', f'{_bias(eo_flat):+.4f}',
            f'{_nrmse(er_flat, tr_flat):.4f}', f'{_bias(er_flat):+.4f}',
        ])
    print(tabulate(rows,
        headers=['sigma_rfx range', 'N_ds', 'NRMSE (ours)', 'Bias (ours)', 'NRMSE (REML)', 'Bias (REML)'],
        tablefmt='simple'))

    print('\n=== RFX (BLUPs) breakdown by q ===')
    rows = []
    for qv in sorted(np.unique(q_arr).astype(int)):
        mask = q_arr == qv
        if mask.sum() < 2:
            continue
        idx_list = np.where(mask)[0]
        eo_flat = np.concatenate([rfx_err_our_all[i] for i in idx_list])
        er_flat = np.concatenate([rfx_err_reml_all[i] for i in idx_list])
        tr_flat = np.concatenate([rfx_true_all[i] for i in idx_list])
        rows.append([
            f'q={qv}', int(mask.sum()),
            f'{_nrmse(eo_flat, tr_flat):.4f}', f'{_bias(eo_flat):+.4f}',
            f'{_nrmse(er_flat, tr_flat):.4f}', f'{_bias(er_flat):+.4f}',
        ])
    print(tabulate(rows,
        headers=['q', 'N_ds', 'NRMSE (ours)', 'Bias (ours)', 'NRMSE (REML)', 'Bias (REML)'],
        tablefmt='simple'))

    print('\n=== RFX (BLUPs) breakdown by n_g (group size) ===')
    ng_all   = np.concatenate(rfx_ng_our_all)
    eo_all   = np.concatenate(rfx_err_our_all)
    er_all   = np.concatenate(rfx_err_reml_all)
    tr_all_r = np.concatenate(rfx_true_all)
    ng_edges = [5, 8, 12, 17, 25]
    rows = []
    for lo, hi in zip(ng_edges[:-1], ng_edges[1:]):
        mask = (ng_all >= lo) & (ng_all < hi)
        if mask.sum() < 2:
            continue
        rows.append([
            f'{lo}–{hi-1}', int(mask.sum()),
            f'{_nrmse(eo_all[mask], tr_all_r[mask]):.4f}', f'{_bias(eo_all[mask]):+.4f}',
            f'{_nrmse(er_all[mask], tr_all_r[mask]):.4f}', f'{_bias(er_all[mask]):+.4f}',
        ])
    print(tabulate(rows,
        headers=['n_g', 'N_blups', 'NRMSE (ours)', 'Bias (ours)', 'NRMSE (REML)', 'Bias (REML)'],
        tablefmt='simple'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_datasets', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run_comparison(n_datasets=args.n_datasets, seed=args.seed)
