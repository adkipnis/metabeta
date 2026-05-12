"""Beta blend diagnostic for final BLUP residuals.

Sweeps alpha gating strategies (beta_for_blup = (1-alpha)*gls_beta + alpha*pooled_ols)
to identify whether raising alpha for low-d rows recovers oracle-beta BLUP gains without
regressing large/huge rows.

Baselines:
  raw         — raw MoM/EM with current blend (d<=8 → 0.65, d>8 → 0.75)
  current_map — production MAP with current blend

Ceiling:
  oracle_beta — raw path, true beta used for BLUP residuals

Sweep: raise alpha for d<=8 rows while leaving d>8 at 0.75.

Alpha only affects blup_est; beta_est and sigma estimates are identical across
sweep methods (shared from the same raw glmm() call).
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

# Each entry: (alpha for d<=8, alpha for d<=16, alpha for d>16)
# None = use default current blend
_ALPHA_CONFIGS: dict[str, tuple[float, float, float] | None] = {
    'raw': None,          # current blend (0.65 / 0.75) — computed by glmm() itself
    'current_map': None,  # production MAP — computed by glmm() itself
    'oracle_beta': None,  # raw + true beta — special case
    'd8_065_075': (0.65, 0.75, 0.75),  # current (reference)
    'd8_070_075': (0.70, 0.75, 0.75),
    'd8_075_075': (0.75, 0.75, 0.75),  # flat 0.75
    'd8_080_075': (0.80, 0.75, 0.75),
    'd8_085_075': (0.85, 0.75, 0.75),
    'd8_065_065': (0.65, 0.65, 0.65),  # flat 0.65
    'd8_075_080': (0.75, 0.80, 0.80),  # uniform raise
}

DEFAULT_METHODS = [
    'raw',
    'current_map',
    'oracle_beta',
    'd8_065_075',
    'd8_070_075',
    'd8_075_075',
    'd8_080_075',
    'd8_085_075',
]

ALL_METHODS = list(_ALPHA_CONFIGS.keys())


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _recomputeBlup(
    stats: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    alpha_low: float,
    alpha_mid: float,
    alpha_high: float,
    beta_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute BLUPs from raw stats with a custom alpha schedule.

    alpha schedule by active_d_count:
      <= 8   → alpha_low
      <= 16  → alpha_mid
      > 16   → alpha_high

    Returns blup_est (B, m, q).
    """
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

    se2 = stats['sigma_eps_est'].squeeze(-1).clamp(min=1e-6).square()
    beta_fallback = stats.get('beta_wg', stats['beta_est'])
    Psi = stats['Psi']

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

    if beta_override is not None:
        beta_for_blup = beta_override
    else:
        beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
        active_d_count = (XtX.diagonal(dim1=-2, dim2=-1).abs() > 1e-8).sum(dim=-1)
        alpha = torch.where(
            active_d_count <= 8,
            se2.new_full(se2.shape, alpha_low),
            torch.where(
                active_d_count <= 16,
                se2.new_full(se2.shape, alpha_mid),
                se2.new_full(se2.shape, alpha_high),
            ),
        )
        beta_for_blup = (
            (1.0 - alpha[:, None]) * gls.beta + alpha[:, None] * beta_ols
        ).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_for_blup)) * mask_n
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
    blups = torch.einsum('bmqp,bmp->bmq', gls.W_g, Ztr)
    return blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)


class _BlupStore:
    def __init__(self) -> None:
        self.errs: list[np.ndarray] = []
        self.truths: list[np.ndarray] = []
        self.n_total = 0

    def add(
        self,
        blup_est: torch.Tensor,
        batch: dict[str, torch.Tensor],
        max_q: int,
    ) -> None:
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
                        if 'mask_q' in batch:
                            Zm = Zm * batch['mask_q'][..., :max_q].to(Zm.dtype)[:, None, None, :]
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

                        needs_raw = any(
                            m not in ('raw', 'current_map') for m in args.methods
                        )
                        needs_raw = needs_raw or 'raw' in args.methods
                        raw_stats = None
                        if needs_raw:
                            raw_stats = glmm(
                                Xm,
                                ym,
                                Zm,
                                mask_n,
                                mask_m,
                                batch['ns'].clamp(min=1).float(),
                                batch['n'].float(),
                                map_refine=False,
                                **common,
                            )

                        current_stats = None
                        if 'current_map' in args.methods:
                            current_stats = glmm(
                                Xm,
                                ym,
                                Zm,
                                mask_n,
                                mask_m,
                                batch['ns'].clamp(min=1).float(),
                                batch['n'].float(),
                                **common,
                            )

                        for method in args.methods:
                            if method == 'raw':
                                blup = raw_stats['blup_est']
                            elif method == 'current_map':
                                blup = current_stats['blup_est']
                            elif method == 'oracle_beta':
                                blup = _recomputeBlup(
                                    raw_stats, Xm, ym, Zm, mask_n, mask_m,
                                    0.0, 0.0, 0.0,
                                    beta_override=batch['ffx'],
                                )
                            else:
                                al, am, ah = _ALPHA_CONFIGS[method]
                                blup = _recomputeBlup(
                                    raw_stats, Xm, ym, Zm, mask_n, mask_m, al, am, ah
                                )
                            stores[method].add(blup, batch, max_q)

            key = f'{data_id}/{partition}'
            if key not in size_stores:
                size_stores[key] = {}
            for m in args.methods:
                size_stores[key][m] = stores[m]
                n = stores[m].n_total
                blup = stores[m].nrmse()
                print(f'{m},{data_id},{partition},{n},{blup:.4f}', flush=True)

    _print_totals(size_stores, args.methods, args.sizes)


def _print_totals(
    size_stores: dict[str, dict[str, _BlupStore]],
    methods: list[str],
    sizes: list[str],
) -> None:
    keys = []
    for size in sizes:
        keys.append(f'{size}-n-mixed/train')
        keys.extend(f'{size}-n-sampled/{p}' for p in ('valid', 'test'))

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
            nrmses = []
            for k in size_keys:
                if k in size_stores and method in size_stores[k]:
                    nrmses.append(size_stores[k][method].nrmse())
            vals.append(float(np.mean(nrmses)) if nrmses else float('nan'))
        global_mean = float(np.nanmean(vals))
        row = f'{method:<18}' + ''.join(f'  {v:8.4f}' for v in vals) + f'  {global_mean:8.4f}'
        print(row)


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
