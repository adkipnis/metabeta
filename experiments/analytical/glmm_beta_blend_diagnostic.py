"""Beta blend diagnostic for final BLUP residuals.

Sweeps beta_alpha_low / beta_alpha_high gating strategies
(beta_for_blup = (1-alpha)*gls_beta + alpha*pooled_ols, gated by active_d_count<=8)
to identify whether changing the OLS blend for low-d or high-d rows improves BLUP
without regressions.

Tests are run on both the raw MoM/EM path and the production MAP path.

Baselines:
  raw         — raw MoM/EM, current blend (d<=8 → 0.65, d>8 → 0.75)
  current_map — production MAP, same blend (reference)

Ceiling:
  oracle_beta — raw path, true beta substituted for BLUP residuals
  (implemented via an internal recompute; separate from glmm alpha params)

Each method passes (beta_alpha_low, beta_alpha_high) to glmm().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from metabeta.analytical.glmm import glmm
from metabeta.analytical.linalg import _adaptiveRidge, _safeSolve
from metabeta.analytical.normal import _normalGlsAndBlups
from metabeta.utils.config import loadDataConfig
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.experiments import dataFilePath


SIZES = ['small', 'medium', 'large', 'huge']

# (path, beta_alpha_low, beta_alpha_high)
# 'raw' and 'current_map' use the production glmm() path.
# 'oracle_beta_raw' requires a manual recompute (see below).
_CONFIGS: dict[str, tuple[str, float, float]] = {
    'raw': ('raw', 0.65, 0.75),  # raw, current blend
    'raw_060_075': ('raw', 0.60, 0.75),
    'raw_065_075': ('raw', 0.65, 0.75),  # same as raw (explicit reference)
    'raw_070_075': ('raw', 0.70, 0.75),
    'raw_075_075': ('raw', 0.75, 0.75),  # flat 0.75
    'raw_080_075': ('raw', 0.80, 0.75),
    'raw_065_065': ('raw', 0.65, 0.65),  # flat 0.65
    'current_map': ('map', 0.65, 0.75),  # MAP, current blend
    'map_060_075': ('map', 0.60, 0.75),
    'map_065_075': ('map', 0.65, 0.75),  # same as current_map (explicit reference)
    'map_070_075': ('map', 0.70, 0.75),
    'map_075_075': ('map', 0.75, 0.75),
    'map_080_075': ('map', 0.80, 0.75),
    'map_065_065': ('map', 0.65, 0.65),
}

DEFAULT_METHODS = [
    'raw',
    'raw_065_075',
    'raw_070_075',
    'raw_075_075',
    'raw_080_075',
    'current_map',
    'map_065_075',
    'map_070_075',
    'map_075_075',
    'map_080_075',
]

ALL_METHODS = list(_CONFIGS.keys()) + ['oracle_beta_raw']


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _oracleBetaBlups(
    raw_stats: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_q: int,
) -> torch.Tensor:
    """Recompute BLUPs using the MAP Psi but substituting true beta for residuals.

    Uses the same diagonal MAP Psi as raw_stats['Psi'] is unavailable internally.
    This is an approximation of the oracle ceiling: true beta + raw Psi.
    """
    Xm = batch['X']
    ym = batch['y']
    Zm = batch['Z'][..., :max_q]
    if 'mask_q' in batch:
        Zm = Zm * batch['mask_q'][..., :max_q].to(Zm.dtype)[:, None, None, :]
    mask_n = batch['mask_n'].float()
    mask_m = batch['mask_m'].float()
    B, m, _, _ = Xm.shape
    q = Zm.shape[-1]
    device = Xm.device
    dtype = Xm.dtype
    active = mask_m.bool()
    mask4 = mask_m[:, :, None, None]
    eye_q = torch.eye(q, device=device, dtype=dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)
    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)
    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)
    se2 = raw_stats['sigma_eps_est'].squeeze(-1).clamp(min=1e-6).square()
    Psi = raw_stats['Psi']
    beta_fallback = raw_stats.get('beta_wg', raw_stats['beta_est'])
    gls = _normalGlsAndBlups(
        Xm,
        ym,
        Zm,
        mask_n,
        ZtZ_safe,
        Zty,
        ZtX,
        XtX,
        Xty,
        XtZ,
        Psi,
        se2,
        eye_q,
        eye_q_bm,
        mask4,
        beta_fallback,
    )
    resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, batch['ffx'])) * mask_n
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
    blups = torch.einsum('bmqp,bmp->bmq', gls.W_g, Ztr)
    return blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)


class _BlupStore:
    def __init__(self) -> None:
        self.errs: list[np.ndarray] = []
        self.truths: list[np.ndarray] = []
        self.n_total = 0

    def add(self, blup_est: torch.Tensor, batch: dict[str, torch.Tensor], max_q: int) -> None:
        mask_q = batch['mask_q'][..., :max_q].bool() if 'mask_q' in batch else None
        mask_m = batch['mask_m'].bool()
        for b in range(blup_est.shape[0]):
            mq = mask_q[b] if mask_q is not None else torch.ones(max_q, dtype=torch.bool)
            mm = mask_m[b]
            self.errs.append(
                (blup_est[b][mm][:, mq] - batch['rfx'][b][mm][:, mq]).reshape(-1).cpu().numpy()
            )
            self.truths.append(batch['rfx'][b][mm][:, mq].reshape(-1).cpu().numpy())
        self.n_total += blup_est.shape[0]

    def nrmse(self) -> float:
        if not self.errs:
            return float('nan')
        return _nrmse(np.concatenate(self.errs), np.concatenate(self.truths))


def run_diagnostic(args: argparse.Namespace) -> None:
    size_stores: dict[str, dict[str, _BlupStore]] = {}
    print('method,dataset,partition,N,BLUP', flush=True)

    for size in args.sizes:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])

        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            stores = {m: _BlupStore() for m in args.methods}

            with torch.no_grad():
                for path in _paths(data_id, partition, n_epochs):
                    for batch_idx, batch in enumerate(
                        Dataloader(path, batch_size=args.batch_size, shuffle=False)
                    ):
                        if args.max_batches is not None and batch_idx >= args.max_batches:
                            break
                        batch = toDevice(batch, torch.device('cpu'))
                        Xm = batch['X']
                        ym = batch['y']
                        Zm = batch['Z'][..., :max_q]
                        mask_n = batch['mask_n'].float()
                        mask_m = batch['mask_m'].float()
                        ns = batch['ns'].clamp(min=1).float()
                        n = batch['n'].float()
                        common = dict(
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

                        # Cache raw/map stats to avoid re-running glmm for same alpha combo
                        _raw_cache: dict[tuple[float, float], dict] = {}
                        _map_cache: dict[tuple[float, float], dict] = {}

                        for method in args.methods:
                            if method == 'oracle_beta_raw':
                                key = (0.65, 0.75)
                                if key not in _raw_cache:
                                    _raw_cache[key] = glmm(
                                        Xm,
                                        ym,
                                        Zm,
                                        mask_n,
                                        mask_m,
                                        ns,
                                        n,
                                        map_refine=False,
                                        beta_alpha_low=key[0],
                                        beta_alpha_high=key[1],
                                        **common,
                                    )
                                blup = _oracleBetaBlups(_raw_cache[key], batch, max_q)
                            else:
                                path_type, al, ah = _CONFIGS[method]
                                key = (al, ah)
                                if path_type == 'raw':
                                    if key not in _raw_cache:
                                        _raw_cache[key] = glmm(
                                            Xm,
                                            ym,
                                            Zm,
                                            mask_n,
                                            mask_m,
                                            ns,
                                            n,
                                            map_refine=False,
                                            beta_alpha_low=al,
                                            beta_alpha_high=ah,
                                            **common,
                                        )
                                    blup = _raw_cache[key]['blup_est']
                                else:
                                    if key not in _map_cache:
                                        _map_cache[key] = glmm(
                                            Xm,
                                            ym,
                                            Zm,
                                            mask_n,
                                            mask_m,
                                            ns,
                                            n,
                                            beta_alpha_low=al,
                                            beta_alpha_high=ah,
                                            **common,
                                        )
                                    blup = _map_cache[key]['blup_est']
                            stores[method].add(blup, batch, max_q)

            key = f'{data_id}/{partition}'
            if key not in size_stores:
                size_stores[key] = {}
            for m in args.methods:
                size_stores[key][m] = stores[m]
                print(
                    f'{m},{data_id},{partition},{stores[m].n_total},{stores[m].nrmse():.4f}',
                    flush=True,
                )

    _print_totals(size_stores, args.methods, args.sizes)


def _print_totals(
    size_stores: dict[str, dict[str, _BlupStore]],
    methods: list[str],
    sizes: list[str],
) -> None:
    print('\n--- BLUP NRMSE by size ---')
    header = f'{"method":<18}' + ''.join(f'  {s[:5]:>8}' for s in sizes) + f'  {"global":>8}'
    print(header)
    for method in methods:
        vals = []
        for size in sizes:
            size_keys = [
                f'{size}-n-mixed/train',
                f'{size}-n-sampled/valid',
                f'{size}-n-sampled/test',
            ]
            nrmses = [
                size_stores[k][method].nrmse()
                for k in size_keys
                if k in size_stores and method in size_stores[k]
            ]
            vals.append(float(np.mean(nrmses)) if nrmses else float('nan'))
        global_mean = float(np.nanmean(vals))
        print(f'{method:<18}' + ''.join(f'  {v:8.4f}' for v in vals) + f'  {global_mean:8.4f}')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=SIZES, choices=SIZES)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS, choices=ALL_METHODS)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-batches', type=int, default=None)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run_diagnostic(setup())
