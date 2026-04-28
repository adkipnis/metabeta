"""Diagnostic: compare glmm.py analytical estimates to ground truth.

Runs over the validation split, collects per-dataset and per-component errors,
and reports:

  1. NRMSE / Bias per parameter  (same NRMSE metric as NN evaluation)
  2. Error quantile profiles
  3. sigma_rfx clip rate (MoM clamped to 0) broken down by between-group df
  4. sigma_eps error broken down by n, sigma_eps_true, and SNR  (normal only)
  5. BLUP error broken down by group size (n_g)
  6. Worst-case examples

Usage (from metabeta/benchmarks/):
    uv run python diagnose_glmm.py                                 # normal (small-n-mixed)
    uv run python diagnose_glmm.py --data-id small-b-mixed         # bernoulli
    uv run python diagnose_glmm.py --data-id small-b-mixed --partition valid
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from metabeta.utils.glmm import glmm
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.config import loadDataConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    """sqrt(MSE(err)) / std(truth) — matches evaluation/point.py getRMSE."""
    denom = float(np.std(truth))
    return float(np.sqrt(np.mean(err ** 2))) / max(denom, 1e-8)


def _bias(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


def _quantile_row(arr: np.ndarray) -> list[str]:
    qs = np.nanpercentile(np.abs(arr), [25, 50, 75, 95, 99])
    return [f'{q:.4f}' for q in qs]


def _breakdown(
    err: np.ndarray,
    factor: np.ndarray,
    factor_label: str,
    n_bins: int = 4,
    factor_is_int: bool = False,
) -> str:
    """Bias / RMSE / p95 of err broken down by bins of factor."""
    if factor_is_int:
        vals = sorted(np.unique(factor[np.isfinite(factor)]).astype(int))
        bins = [(str(v), factor == v) for v in vals]
    else:
        finite = np.isfinite(factor)
        edges = np.nanpercentile(factor[finite], np.linspace(0, 100, n_bins + 1))
        edges = np.unique(np.round(edges, 4))
        bins = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = finite & (factor >= lo) & (factor <= (hi if i == len(edges) - 2 else hi - 1e-9))
            bins.append((f'{lo:.3f}–{hi:.3f}', mask))

    rows = []
    for label, mask in bins:
        e = err[mask]
        if len(e) < 2:
            continue
        rows.append([
            label, int(mask.sum()),
            f'{_bias(e):+.4f}',
            f'{np.sqrt(np.mean(e**2)):.4f}',
            f'{np.nanpercentile(np.abs(e), 95):.4f}',
        ])
    return tabulate(rows, headers=[factor_label, 'N', 'Bias', 'RMSE', '|err| p95'], tablefmt='simple')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostic(
    data_id: str = 'small-b-mixed',
    partition: str = 'train',
    n_epochs: int = 20,
) -> None:
    """
    Args:
        data_id:   data config id (default: small-n-mixed, i.e. training data)
        partition: 'train' loops over n_epochs files; 'valid' loads valid.npz
        n_epochs:  number of training epochs to load (only used when partition='train')
    """
    from metabeta.utils.io import datasetFilename

    device = torch.device('cpu')

    data_cfg = loadDataConfig(data_id)
    max_d = data_cfg['max_d']
    max_q = data_cfg['max_q']
    likelihood_family = data_cfg.get('likelihood_family', 0)

    data_dir = ROOT / 'metabeta' / 'outputs' / 'data' / data_cfg['data_id']

    if partition == 'train':
        paths = [data_dir / datasetFilename('train', ep) for ep in range(1, n_epochs + 1)]
        paths = [p for p in paths if p.exists()]
        assert paths, f'No training files found in {data_dir}'
    elif partition == 'test':
        paths = [data_dir / 'test.npz']
        assert paths[0].exists(), f'missing: {paths[0]}'
    else:
        paths = [data_dir / 'valid.npz']
        assert paths[0].exists(), f'missing: {paths[0]}'

    print(f'data_id={data_id}  partition={partition}  files={len(paths)}  likelihood_family={likelihood_family}\n')

    all_batches: list = []
    for path in paths:
        dl = Dataloader(path, batch_size=32, shuffle=False)
        all_batches.extend(list(dl))
    print(f'Total batches: {len(all_batches)}')

    # Per-component flat arrays (all active dims across all datasets)
    beta_err_list:  list[np.ndarray] = []
    beta_true_list: list[np.ndarray] = []
    srfx_err_list:  list[np.ndarray] = []
    srfx_true_list: list[np.ndarray] = []
    seps_err_list:  list[float] = []
    seps_true_list: list[float] = []
    rfx_err_list:   list[np.ndarray] = []
    rfx_true_list:  list[np.ndarray] = []
    # blup_var calibration: predicted posterior variance vs actual squared error
    blup_var_list:  list[np.ndarray] = []  # predicted Var(b_g)
    blup_ns_list:   list[np.ndarray] = []  # group sizes, aligned with blup arrays

    # Per-dataset scalars for breakdown
    ds: list[dict] = []

    with torch.no_grad():
        for batch in all_batches:
            batch = toDevice(batch, device)
            B = batch['X'].shape[0]

            Zm = batch['Z'][..., :max_q]
            stats = glmm(
                batch['X'], batch['y'], Zm,
                batch['mask_n'].float(), batch['mask_m'].float(),
                batch['ns'].clamp(min=1).float(), batch['n'].float(),
                likelihood_family=likelihood_family,
                eta_rfx=batch.get('eta_rfx'),
            )

            # Numpy copies
            beta_est  = stats['beta_est'].cpu().numpy()          # (B, d)
            srfx_est  = stats['sigma_rfx_est'].cpu().numpy()     # (B, q)
            blup_est  = stats['blup_est'].cpu().numpy()          # (B, m, q)
            blup_var  = stats['blup_var'].cpu().numpy()          # (B, m, q)
            ffx_true  = batch['ffx'].cpu().numpy()               # (B, d)
            srfx_true = batch['sigma_rfx'].cpu().numpy()         # (B, q)
            rfx_true  = batch['rfx'].cpu().numpy()               # (B, m, q)
            mask_d    = batch['mask_d'].cpu().numpy().astype(bool)  # (B, d)
            mask_q    = batch['mask_q'].cpu().numpy().astype(bool)  # (B, q)
            mask_m    = batch['mask_m'].cpu().numpy().astype(bool)  # (B, m)
            ns_np     = batch['ns'].cpu().numpy()                # (B, m)
            n_np      = batch['n'].cpu().numpy()                 # (B,)
            m_np      = batch['m'].cpu().numpy()                 # (B,)

            if likelihood_family == 0:
                seps_est_np  = stats['sigma_eps_est'].squeeze(-1).cpu().numpy()  # (B,)
                seps_true_np = batch['sigma_eps'].cpu().numpy()                  # (B,)
            else:
                seps_est_np = seps_true_np = np.full(B, float('nan'))

            for b in range(B):
                d = int(mask_d[b].sum())
                q = int(mask_q[b].sum())
                m = int(m_np[b])

                # Active-component errors
                be = beta_est[b, :d] - ffx_true[b, :d]           # (d,)
                se = srfx_est[b, :q] - srfx_true[b, :q]          # (q,)
                eps_e = float(seps_est_np[b] - seps_true_np[b])

                # Active group × rfx errors
                rfx_e = blup_est[b, :m, :q] - rfx_true[b, :m, :q]  # (m, q)
                bv    = blup_var[b, :m, :q]                          # (m, q)
                grp_ns = ns_np[b, :m]                               # (m,)

                beta_err_list.append(be)
                beta_true_list.append(ffx_true[b, :d])
                srfx_err_list.append(se)
                srfx_true_list.append(srfx_true[b, :q])
                seps_err_list.append(eps_e)
                seps_true_list.append(float(seps_true_np[b]))
                rfx_err_list.append(rfx_e.reshape(-1))
                rfx_true_list.append(rfx_true[b, :m, :q].reshape(-1))
                blup_var_list.append(bv.reshape(-1))
                blup_ns_list.append(np.repeat(grp_ns, q))

                ds.append({
                    'n': int(n_np[b]),
                    'm': m,
                    'd': d,
                    'q': q,
                    'n_over_d': int(n_np[b]) / max(d, 1),
                    'bg_df': max(m - d, 1),
                    'snr': float(srfx_true[b, :q].mean() / seps_true_np[b]) if likelihood_family == 0 else float('nan'),
                    'seps_true': float(seps_true_np[b]),
                    'srfx_true_mean': float(srfx_true[b, :q].mean()),
                    'seps_err': eps_e,
                    'srfx_err_mean': float(se.mean()),
                    'srfx_clipped': bool((srfx_est[b, :q] == 0.0).any()),
                    'rfx_rmse': float(np.sqrt(np.mean(rfx_e ** 2))),
                    '_rfx_err': rfx_e.reshape(-1).tolist(),
                    '_rfx_ns':  np.repeat(grp_ns, q).tolist(),
                    '_beta_err': be.tolist(),
                    '_beta_true': ffx_true[b, :d].tolist(),
                })

    print(f'Collected {len(ds)} datasets\n')

    # Flat arrays for global stats
    beta_err_flat  = np.concatenate(beta_err_list)
    beta_true_flat = np.concatenate(beta_true_list)
    srfx_err_flat  = np.concatenate(srfx_err_list)
    srfx_true_flat = np.concatenate(srfx_true_list)
    seps_err_arr   = np.array(seps_err_list)
    seps_true_arr  = np.array(seps_true_list)
    rfx_err_flat   = np.concatenate(rfx_err_list)
    rfx_true_flat  = np.concatenate(rfx_true_list)
    rfx_ns_flat    = np.array([x for r in ds for x in r['_rfx_ns']])
    blup_var_flat  = np.concatenate(blup_var_list)
    blup_ns_flat   = np.concatenate(blup_ns_list)

    # Per-dataset scalars
    def col(k): return np.array([r[k] for r in ds], dtype=float)
    n_arr        = col('n')
    m_arr        = col('m')
    d_arr        = col('d')
    n_over_d_arr = col('n_over_d')
    bg_df_arr    = col('bg_df')
    snr_arr      = col('snr')
    seps_true_d  = col('seps_true')
    srfx_true_d  = col('srfx_true_mean')
    seps_err_d   = col('seps_err')
    srfx_err_d   = col('srfx_err_mean')
    srfx_clip    = np.array([r['srfx_clipped'] for r in ds])
    beta_err_ds  = np.array([np.sqrt(np.mean(np.array(r['_beta_err'])**2)) for r in ds])
    # Use the global std(beta_true) as scale — consistent with _nrmse and robust to
    # near-zero per-dataset beta values (which occur when the Emulator draws real datasets
    # with small coefficients on the training partition).
    beta_scale_global = max(float(np.std(beta_true_flat)), 1e-8)

    # ===========================================================================
    # 1. NRMSE / Bias table
    # ===========================================================================
    print('=' * 65)
    print('  GLMM Estimates vs Ground Truth  (NN eval uses same NRMSE)')
    print('=' * 65)
    rows_nrmse = [
        ['FFX (β)',       f'{_nrmse(beta_err_flat,  beta_true_flat):.4f}',  f'{_bias(beta_err_flat):+.4f}'],
        ['Sigma(RFX)',    f'{_nrmse(srfx_err_flat,  srfx_true_flat):.4f}',  f'{_bias(srfx_err_flat):+.4f}'],
    ]
    if likelihood_family == 0:
        rows_nrmse.append(
            ['Sigma(Eps)', f'{_nrmse(seps_err_arr, seps_true_arr):.4f}', f'{_bias(seps_err_arr):+.4f}']
        )
    rows_nrmse.append(
        ['RFX (BLUPs)', f'{_nrmse(rfx_err_flat, rfx_true_flat):.4f}', f'{_bias(rfx_err_flat):+.4f}']
    )
    print(tabulate(rows_nrmse, headers=['Parameter', 'NRMSE', 'Bias (est−truth)'], tablefmt='simple'))

    # ===========================================================================
    # 2. Absolute error quantiles
    # ===========================================================================
    print('\n=== Absolute error quantiles ===')
    rows_q = [
        ['FFX (β)']     + _quantile_row(beta_err_flat),
        ['Sigma(RFX)']  + _quantile_row(srfx_err_flat),
    ]
    if likelihood_family == 0:
        rows_q.append(['Sigma(Eps)'] + _quantile_row(seps_err_arr))
    rows_q.append(['RFX (BLUPs)'] + _quantile_row(rfx_err_flat))
    print(tabulate(rows_q, headers=['', '|err| p25', '|err| p50', '|err| p75', '|err| p95', '|err| p99'],
    tablefmt='simple'))

    # ===========================================================================
    # 3. sigma_rfx: clip rate and breakdown by between-group df
    # ===========================================================================
    clip_rate = srfx_clip.mean() * 100
    print(f'\n=== sigma_rfx: {clip_rate:.1f}% of datasets clipped to 0 (negative MoM) ===')
    print('\nBreakdown by between-group df (m − d):')
    print(_breakdown(srfx_err_d, bg_df_arr, 'bg_df (m−d)', factor_is_int=True))
    print('\nBreakdown by true sigma_rfx:')
    print(_breakdown(srfx_err_d, srfx_true_d, 'sigma_rfx_true'))

    # ===========================================================================
    # 4. sigma_eps: breakdown
    # ===========================================================================
    if likelihood_family == 0:
        print(f'\n=== sigma_eps errors  (global bias={_bias(seps_err_arr):+.4f}) ===')
        print('\nBy true sigma_eps:')
        print(_breakdown(seps_err_d, seps_true_d, 'sigma_eps_true'))
        print('\nBy total n:')
        print(_breakdown(seps_err_d, n_arr, 'n (total obs)'))
        print('\nBy SNR = mean(sigma_rfx) / sigma_eps:')
        print(_breakdown(seps_err_d, snr_arr, 'SNR'))
        print('\nBy between-group df (m − d):')
        print(_breakdown(seps_err_d, bg_df_arr, 'bg_df', factor_is_int=True))

    # ===========================================================================
    # 5. RFX (BLUP) by group size
    # ===========================================================================
    print('\n=== BLUP errors by group size (n_g) ===')
    print(_breakdown(rfx_err_flat, rfx_ns_flat, 'n_g'))

    # ===========================================================================
    # 5b. blup_var calibration: predicted posterior variance vs actual MSE
    #     ratio > 1  → underconfident (predicted var too large)
    #     ratio < 1  → overconfident  (predicted var too small, BLUPs overshrunk)
    # ===========================================================================
    print('\n=== blup_var calibration: mean(err²) / mean(blup_var) by group size ===')
    print('  Ratio > 1 → overconfident (blup_var too small; actual errors exceed predicted variance → BLUPs overshrunk)')
    print('  Ratio < 1 → underconfident (blup_var too large)')
    rfx_err2_flat = rfx_err_flat ** 2
    finite_mask = np.isfinite(rfx_err2_flat) & np.isfinite(blup_var_flat) & (blup_var_flat > 0)
    edges_ng = np.nanpercentile(blup_ns_flat[finite_mask], np.linspace(0, 100, 6))
    edges_ng = np.unique(np.round(edges_ng))
    calib_rows = []
    for i in range(len(edges_ng) - 1):
        lo, hi = edges_ng[i], edges_ng[i + 1]
        sel = finite_mask & (blup_ns_flat >= lo) & (blup_ns_flat <= (hi if i == len(edges_ng) - 2 else hi - 1e-9))
        if sel.sum() < 2:
            continue
        mse  = float(np.mean(rfx_err2_flat[sel]))
        mvar = float(np.mean(blup_var_flat[sel]))
        ratio = mse / max(mvar, 1e-30)
        calib_rows.append([f'{lo:.0f}–{hi:.0f}', int(sel.sum()),
                           f'{mvar:.4f}', f'{mse:.4f}', f'{ratio:.3f}'])
    print(tabulate(calib_rows,
        headers=['n_g', 'N', 'mean(blup_var)', 'mean(err²)', 'ratio'],
        tablefmt='simple'))

    # ===========================================================================
    # 5c. FFX (beta) breakdown by n/d — checks if poor NRMSE is unavoidable
    # ===========================================================================
    print('\n=== FFX (β) RMSE breakdown by n_total/d (effective obs per param) ===')
    print(_breakdown(beta_err_ds / beta_scale_global, n_over_d_arr, 'n/d'))

    # ===========================================================================
    # 6. Worst-case examples
    # ===========================================================================
    worst_targets = [('sigma_rfx', srfx_err_d, 'srfx_err_mean')]
    if likelihood_family == 0:
        worst_targets.append(('sigma_eps', seps_err_d, 'seps_err'))

    for label, err_col, thresh_key in worst_targets:
        thresh = np.nanpercentile(np.abs(err_col), 99)
        worst  = [r for r in ds if abs(r[thresh_key]) >= thresh][:10]
        print(f'\n=== Worst 1% by |{label} error| (threshold={thresh:.3f}) ===')
        if likelihood_family == 0:
            rows = [[r['n'], r['m'], r['d'], r['bg_df'],
                     f'{r["srfx_true_mean"]:.3f}', f'{r["seps_true"]:.3f}',
                     f'{r["srfx_err_mean"]:+.3f}', f'{r["seps_err"]:+.3f}',
                     f'{r["snr"]:.2f}', '✓' if r['srfx_clipped'] else '']
                    for r in worst]
            print(tabulate(rows,
                headers=['n', 'm', 'd', 'bg_df', 'σ_rfx_true', 'σ_eps_true',
                         'σ_rfx_err', 'σ_eps_err', 'SNR', 'clip'],
                tablefmt='simple'))
        else:
            rows = [[r['n'], r['m'], r['d'], r['bg_df'],
                     f'{r["srfx_true_mean"]:.3f}',
                     f'{r["srfx_err_mean"]:+.3f}',
                     '✓' if r['srfx_clipped'] else '']
                    for r in worst]
            print(tabulate(rows,
                headers=['n', 'm', 'd', 'bg_df', 'σ_rfx_true', 'σ_rfx_err', 'clip'],
                tablefmt='simple'))


if __name__ == '__main__':
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(description='Diagnose GLMM analytical estimates vs ground truth')
    parser.add_argument('--data-id',   default='small-b-mixed', help='data config id (default: small-b-mixed)')
    parser.add_argument('--partition', default='train',          choices=['train', 'valid', 'test'], help='which split to evaluate')
    parser.add_argument('--n-epochs',  default=20, type=int,    help='number of train epochs to load (train partition only)')
    # fmt: on
    args = parser.parse_args()
    run_diagnostic(data_id=args.data_id, partition=args.partition, n_epochs=args.n_epochs)
