"""Raw MoM/EM attribution diagnostic for the Gaussian analytical GLMM.

This script is experiment-only. It compares production MAP and raw MoM/EM to
oracle final-stage substitutions that identify which raw stage limits FFX,
sigma(Eps), sigma(RFX), and BLUP accuracy.
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
METHODS = [
    'current',
    'raw',
    'output_psi_recompute',
    'map_psi_diag',
    'oracle_sigma_eps',
    'oracle_beta_blup',
    'oracle_psi_diag',
    'oracle_psi_diag_est_corr',
    'oracle_psi_corr_est_diag',
    'oracle_psi',
]
DEFAULT_METHODS = [
    'current',
    'raw',
    'oracle_sigma_eps',
    'oracle_beta_blup',
    'oracle_psi_diag',
    'oracle_psi',
]


def _nrmse(err: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)) / max(float(np.std(truth)), 1e-8))


def _paths(data_id: str, partition: str, n_epochs: int) -> list[Path]:
    cfg = loadDataConfig(data_id)
    if partition == 'train':
        return [dataFilePath(cfg['data_id'], 'train', ep) for ep in range(1, n_epochs + 1)]
    return [dataFilePath(cfg['data_id'], partition)]


def _truePsi(batch: dict[str, torch.Tensor], max_q: int) -> torch.Tensor:
    sigma = batch['sigma_rfx'][..., :max_q]
    if 'corr_rfx' in batch:
        corr = batch['corr_rfx'][..., :max_q, :max_q]
    else:
        corr = torch.eye(max_q, device=sigma.device, dtype=sigma.dtype).expand(
            sigma.shape[0], max_q, max_q
        )
    if 'eta_rfx' in batch:
        eye = torch.eye(max_q, device=sigma.device, dtype=sigma.dtype).expand_as(corr)
        corr = torch.where(batch['eta_rfx'][:, None, None] > 0, corr, eye)
    return corr * sigma[:, :, None] * sigma[:, None, :]


def _corrFromPsi(Psi: torch.Tensor) -> torch.Tensor:
    std = Psi.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    corr = (Psi / (std[:, :, None] * std[:, None, :])).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(Psi.shape[-1], device=Psi.device, dtype=Psi.dtype)
    return corr.clamp(-0.95, 0.95) - torch.diag_embed(corr.diagonal(dim1=-2, dim2=-1)) + eye


def _psiFromSigmaCorr(sigma: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    return corr * sigma[:, :, None] * sigma[:, None, :]


def _replacePsiDiag(
    stats: dict[str, torch.Tensor],
    sigma_rfx: torch.Tensor,
    Psi: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out = dict(stats)
    out['sigma_rfx_est'] = sigma_rfx
    out['Psi'] = Psi
    return out


def _recomputeFinal(
    stats: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_q: int,
    *,
    sigma_eps: torch.Tensor | None = None,
    Psi: torch.Tensor | None = None,
    beta_for_blup: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    Xm = batch['X']
    ym = batch['y']
    Zm = batch['Z'][..., :max_q]
    if 'mask_q' in batch:
        Zm = Zm * batch['mask_q'][..., :max_q].to(Zm.dtype)[:, None, None, :]

    mask_n = batch['mask_n'].float()
    mask_m = batch['mask_m'].float()
    B, m, _, d = Xm.shape
    q = Zm.shape[-1]
    device = Xm.device
    dtype = Xm.dtype

    eye_q = torch.eye(q, device=device, dtype=dtype)
    eye_q_bm = eye_q.expand(B, m, q, q)
    active = mask_m.bool()
    mask4 = mask_m[:, :, None, None]

    ZtZ = torch.einsum('bmnq,bmnr->bmqr', Zm, Zm)
    ZtZ_safe = torch.where(active[:, :, None, None], ZtZ, eye_q)
    Zty = torch.einsum('bmnq,bmn->bmq', Zm, ym)
    ZtX = torch.einsum('bmnq,bmnd->bmqd', Zm, Xm)
    XtX = torch.einsum('bmnd,bmnk->bdk', Xm, Xm)
    Xty = torch.einsum('bmnd,bmn->bd', Xm, ym)
    XtZ = torch.einsum('bmnd,bmnq->bmdq', Xm, Zm)

    se = stats['sigma_eps_est'].squeeze(-1) if sigma_eps is None else sigma_eps
    se2 = se.clamp(min=1e-6).square()
    Psi_use = stats['Psi'] if Psi is None else Psi
    beta_fallback = stats.get('beta_wg', stats['beta_est'])

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
        Psi_use,
        se2,
        eye_q,
        eye_q_bm,
        mask4,
        beta_fallback,
    )

    if beta_for_blup is None:
        beta_ols = _safeSolve(XtX + _adaptiveRidge(XtX), Xty)
        active_d_count = (XtX.diagonal(dim1=-2, dim2=-1).abs() > 1e-8).sum(dim=-1)
        alpha = torch.where(
            active_d_count <= 8,
            se2.new_full(se2.shape, 0.65),
            se2.new_full(se2.shape, 0.75),
        )
        beta_for_blup = ((1.0 - alpha[:, None]) * gls.beta + alpha[:, None] * beta_ols).nan_to_num(
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    resid = (ym - torch.einsum('bmnd,bd->bmn', Xm, beta_for_blup)) * mask_n
    Ztr = torch.einsum('bmnq,bmn->bmq', Zm, resid)
    blups = torch.einsum('bmqp,bmp->bmq', gls.W_g, Ztr)

    out = dict(stats)
    out['beta_est'] = gls.beta
    out['sigma_eps_est'] = se.unsqueeze(-1)
    out['blup_est'] = blups.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
    out['blup_var'] = gls.blup_var
    if Psi is not None:
        out['Psi'] = Psi
    return out


def _oracleStats(
    method: str,
    raw_stats: dict[str, torch.Tensor],
    current_stats: dict[str, torch.Tensor] | None,
    batch: dict[str, torch.Tensor],
    max_q: int,
) -> dict[str, torch.Tensor]:
    if method == 'output_psi_recompute':
        return _recomputeFinal(raw_stats, batch, max_q, Psi=raw_stats['Psi'])
    if method == 'map_psi_diag':
        if current_stats is None:
            raise ValueError('map_psi_diag requires current MAP stats')
        sigma = current_stats['sigma_rfx_est'][..., :max_q]
        Psi = torch.diag_embed(sigma.square())
        out = _recomputeFinal(raw_stats, batch, max_q, Psi=Psi)
        return _replacePsiDiag(out, sigma, Psi)
    if method == 'oracle_sigma_eps':
        return _recomputeFinal(
            raw_stats,
            batch,
            max_q,
            sigma_eps=batch['sigma_eps'],
        )
    if method == 'oracle_beta_blup':
        out = _recomputeFinal(
            raw_stats,
            batch,
            max_q,
            beta_for_blup=batch['ffx'],
        )
        out['beta_est'] = raw_stats['beta_est']
        return out
    if method == 'oracle_psi_diag':
        sigma = batch['sigma_rfx'][..., :max_q]
        Psi = torch.diag_embed(sigma.square())
        out = _recomputeFinal(raw_stats, batch, max_q, Psi=Psi)
        return _replacePsiDiag(out, sigma, Psi)
    if method == 'oracle_psi_diag_est_corr':
        sigma = batch['sigma_rfx'][..., :max_q]
        Psi = _psiFromSigmaCorr(sigma, _corrFromPsi(raw_stats['Psi']))
        out = _recomputeFinal(raw_stats, batch, max_q, Psi=Psi)
        return _replacePsiDiag(out, sigma, Psi)
    if method == 'oracle_psi_corr_est_diag':
        sigma = raw_stats['sigma_rfx_est'][..., :max_q]
        true_corr = _corrFromPsi(_truePsi(batch, max_q))
        Psi = _psiFromSigmaCorr(sigma, true_corr)
        out = _recomputeFinal(raw_stats, batch, max_q, Psi=Psi)
        return _replacePsiDiag(out, sigma, Psi)
    if method == 'oracle_psi':
        Psi = _truePsi(batch, max_q)
        out = _recomputeFinal(raw_stats, batch, max_q, Psi=Psi)
        return _replacePsiDiag(out, batch['sigma_rfx'][..., :max_q], Psi)
    raise ValueError(f'unsupported oracle method: {method}')


def run_raw_diagnostic(args: argparse.Namespace) -> None:
    stores_all = {method: _MetricStore() for method in args.methods}
    print('method,dataset,partition,N,FFX,sRFX,sEps,BLUP', flush=True)
    for size in args.sizes:
        combos = [(f'{size}-n-mixed', 'train', 2)]
        combos.extend((f'{size}-n-sampled', part, 0) for part in ['valid', 'test'])
        for data_id, partition, n_epochs in combos:
            cfg = loadDataConfig(data_id)
            max_q = cfg['max_q']
            likelihood_family = cfg.get('likelihood_family', 0)
            stores = {method: _MetricStore() for method in args.methods}

            with torch.no_grad():
                for path in _paths(data_id, partition, n_epochs):
                    for batch_idx, batch in enumerate(
                        Dataloader(path, batch_size=args.batch_size, shuffle=False)
                    ):
                        if args.max_batches is not None and batch_idx >= args.max_batches:
                            break
                        batch = toDevice(batch, torch.device('cpu'))
                        Zm = batch['Z'][..., :max_q]
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
                        raw_stats = glmm(
                            batch['X'],
                            batch['y'],
                            Zm,
                            batch['mask_n'].float(),
                            batch['mask_m'].float(),
                            batch['ns'].clamp(min=1).float(),
                            batch['n'].float(),
                            likelihood_family=likelihood_family,
                            map_refine=False,
                            **common,
                        )
                        stats_by_method = {'raw': raw_stats}
                        needs_current = 'current' in args.methods or any(
                            method.startswith('map_') for method in args.methods
                        )
                        if needs_current:
                            stats_by_method['current'] = glmm(
                                batch['X'],
                                batch['y'],
                                Zm,
                                batch['mask_n'].float(),
                                batch['mask_m'].float(),
                                batch['ns'].clamp(min=1).float(),
                                batch['n'].float(),
                                likelihood_family=likelihood_family,
                                map_refine=True,
                                **common,
                            )
                        for method in args.methods:
                            if method not in stats_by_method:
                                stats_by_method[method] = _oracleStats(
                                    method,
                                    raw_stats,
                                    stats_by_method.get('current'),
                                    batch,
                                    max_q,
                                )
                            stores[method].add(stats_by_method[method], batch, max_q)
                            stores_all[method].add(stats_by_method[method], batch, max_q)

            for method in args.methods:
                print(stores[method].row(method, data_id, partition), flush=True)

    for method in args.methods:
        print(stores_all[method].row(method, 'all', 'all'), flush=True)


class _MetricStore:
    def __init__(self) -> None:
        self.beta_errs: list[np.ndarray] = []
        self.beta_truths: list[np.ndarray] = []
        self.srfx_errs: list[np.ndarray] = []
        self.srfx_truths: list[np.ndarray] = []
        self.seps_errs: list[np.ndarray] = []
        self.seps_truths: list[np.ndarray] = []
        self.blup_errs: list[np.ndarray] = []
        self.blup_truths: list[np.ndarray] = []
        self.n_total = 0

    def add(
        self, stats: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], max_q: int
    ) -> None:
        mask_d = batch['mask_d'].bool()
        mask_q = batch['mask_q'][..., :max_q].bool()
        mask_m = batch['mask_m'].bool()
        self.n_total += batch['X'].shape[0]
        for b in range(batch['X'].shape[0]):
            self.beta_errs.append(
                (stats['beta_est'][b][mask_d[b]] - batch['ffx'][b][mask_d[b]]).cpu().numpy()
            )
            self.beta_truths.append(batch['ffx'][b][mask_d[b]].cpu().numpy())
            self.srfx_errs.append(
                (stats['sigma_rfx_est'][b][mask_q[b]] - batch['sigma_rfx'][b][mask_q[b]])
                .cpu()
                .numpy()
            )
            self.srfx_truths.append(batch['sigma_rfx'][b][mask_q[b]].cpu().numpy())
            self.seps_errs.append(
                (stats['sigma_eps_est'][b, 0] - batch['sigma_eps'][b]).reshape(1).cpu().numpy()
            )
            self.seps_truths.append(batch['sigma_eps'][b].reshape(1).cpu().numpy())
            blup_est = stats['blup_est'][b][mask_m[b]][:, mask_q[b]]
            blup_true = batch['rfx'][b][mask_m[b]][:, mask_q[b]]
            self.blup_errs.append((blup_est - blup_true).reshape(-1).cpu().numpy())
            self.blup_truths.append(blup_true.reshape(-1).cpu().numpy())

    def row(self, method: str, data_id: str, partition: str) -> str:
        return ','.join(
            [
                method,
                data_id,
                partition,
                str(self.n_total),
                f'{_nrmse(np.concatenate(self.beta_errs), np.concatenate(self.beta_truths)):.4f}',
                f'{_nrmse(np.concatenate(self.srfx_errs), np.concatenate(self.srfx_truths)):.4f}',
                f'{_nrmse(np.concatenate(self.seps_errs), np.concatenate(self.seps_truths)):.4f}',
                f'{_nrmse(np.concatenate(self.blup_errs), np.concatenate(self.blup_truths)):.4f}',
            ]
        )


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=SIZES, choices=SIZES)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS, choices=METHODS)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-batches', type=int, default=None)
    return parser.parse_args()
# fmt: on


if __name__ == '__main__':
    run_raw_diagnostic(setup())
