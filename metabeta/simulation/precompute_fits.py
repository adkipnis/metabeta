"""Pre-compute analytical GLMM statistics and save into dataset npz files.

Adds beta_est, sigma_rfx_est, blup_est, blup_var, Psi, and sigma_eps_est /
phi_pearson to each npz so that during training data['stats'] is populated
directly from the batch instead of running glmm() live.

Files that already contain beta_est are skipped unless --overwrite is given.

Usage (from any directory):
    uv run python metabeta/simulation/precompute_fits.py \\
        --paths outputs/data/toy/train_epoch0.npz outputs/data/toy/valid.npz

    uv run python metabeta/simulation/precompute_fits.py \\
        --paths outputs/data/toy/*.npz --refinement full --device cuda
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.analytical.fit import glmm


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pre-compute analytical fits for dataset npz files.')
    parser.add_argument('--paths', nargs='+', required=True,
                        help='npz file paths to process')
    parser.add_argument('--refinement', default='light', choices=['light', 'full'],
                        help='Analytical refinement: light (EB only) or full (MAP+EB) (default: light)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Datasets per batch (default: 32)')
    parser.add_argument('--device', default='cpu',
                        help='Compute device: cpu|cuda (default: cpu)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if stats are already present')
    return parser.parse_args()
# fmt: on


def _compute_file(path: Path, refinement: str, batch_size: int, device: torch.device) -> int:
    """Compute and save stats for one npz file. Returns number of datasets processed."""
    with np.load(path, allow_pickle=True) as f:
        raw = dict(f)

    N = len(raw['y'])
    d_file = int(raw['d'].max())
    q_file = int(raw['q'].max())
    m_max = int(raw['ns'].shape[1])

    lf = int(np.asarray(raw['likelihood_family']).flat[0])
    has_eps = lf == 0
    map_refine = refinement == 'full'

    out: dict[str, np.ndarray] = {
        'beta_est':      np.zeros((N, d_file),        dtype=np.float32),
        'sigma_rfx_est': np.zeros((N, q_file),        dtype=np.float32),
        'blup_est':      np.zeros((N, m_max, q_file), dtype=np.float32),
        'blup_var':      np.zeros((N, m_max, q_file), dtype=np.float32),
    }
    if has_eps:
        out['sigma_eps_est'] = np.zeros((N, 1), dtype=np.float32)
    else:
        out['phi_pearson'] = np.zeros(N, dtype=np.float32)
    psi_buf = np.zeros((N, q_file, q_file), dtype=np.float32) if q_file >= 2 else None
    has_psi = False

    dl = Dataloader(path, batch_size=batch_size, sortish=False, shuffle=False)
    idx = 0
    for batch in dl:
        batch = toDevice(batch, device)
        B = int(batch['X'].shape[0])
        ms = batch['m'].cpu().numpy().astype(int)

        map_kwargs: dict = {
            'nu_ffx':           batch['nu_ffx'],
            'tau_ffx':          batch['tau_ffx'],
            'family_ffx':       batch['family_ffx'],
            'tau_rfx':          batch['tau_rfx'],
            'family_sigma_rfx': batch['family_sigma_rfx'],
            'mask_d':           batch.get('mask_d'),
        }
        if has_eps:
            map_kwargs['tau_eps']          = batch['tau_eps']
            map_kwargs['family_sigma_eps'] = batch['family_sigma_eps']

        stats = glmm(
            batch['X'],
            batch['y'],
            batch['Z'][..., :q_file],
            batch['mask_n'].float(),
            batch['mask_m'].float(),
            batch['ns'].clamp(min=1).float(),
            batch['n'].float(),
            likelihood_family=lf,
            eta_rfx=batch.get('eta_rfx'),
            mask_q=batch.get('mask_q'),
            map_refine=map_refine,
            **map_kwargs,
        )

        if 'Psi_lap' in stats and 'Psi' not in stats:
            stats['Psi'] = stats['Psi_lap']

        out['beta_est'][idx:idx+B]      = stats['beta_est'].cpu().numpy()[:, :d_file]
        out['sigma_rfx_est'][idx:idx+B] = stats['sigma_rfx_est'].cpu().numpy()[:, :q_file]
        if has_eps:
            out['sigma_eps_est'][idx:idx+B] = stats['sigma_eps_est'].cpu().numpy()[:B, :1]
        else:
            out['phi_pearson'][idx:idx+B] = stats['phi_pearson'].cpu().numpy()[:B]

        blup  = stats['blup_est'].cpu().numpy()
        blupv = stats['blup_var'].cpu().numpy()
        for b in range(B):
            m_b = ms[b]
            out['blup_est'][idx+b, :m_b] = blup[b,  :m_b, :q_file]
            out['blup_var'][idx+b, :m_b] = blupv[b, :m_b, :q_file]

        if psi_buf is not None and 'Psi' in stats:
            has_psi = True
            psi_buf[idx:idx+B] = stats['Psi'].cpu().numpy()[:B, :q_file, :q_file]

        idx += B

    if has_psi:
        out['Psi'] = psi_buf

    raw.update(out)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **raw)
    os.replace(tmp, path)
    return N


def main() -> None:
    cfg = setup()
    device = torch.device(cfg.device)

    for path_str in cfg.paths:
        path = Path(path_str)
        if not path.exists():
            print(f'SKIP (not found):    {path}')
            continue

        with np.load(path, allow_pickle=True) as f:
            already_done = 'beta_est' in f
        if already_done and not cfg.overwrite:
            print(f'SKIP (already done): {path}')
            continue

        print(f'Processing {path} ...', end=' ', flush=True)
        t0 = time.perf_counter()
        n = _compute_file(path, cfg.refinement, cfg.batch_size, device)
        dt = time.perf_counter() - t0
        print(f'{n} datasets  {dt:.1f}s  ({dt / n * 1000:.1f} ms/ds)')


if __name__ == '__main__':
    main()
