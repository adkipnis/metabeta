"""Pre-compute analytical GLMM statistics and save into a dataset npz file.

Adds beta_est, sigma_rfx_est, blup_est, blup_var, Psi, and sigma_eps_est /
phi_pearson to the npz so that during training data['stats'] is populated
directly from the batch instead of running glmm() live.

The file is skipped if it already contains beta_est unless --overwrite is given.

Usage (from metabeta/analytical/):
    python precompute.py --size small --family 0 --ds_type mixed --partition valid
    python precompute.py --size small --family 0 --ds_type mixed --partition train --epoch 1
    python precompute.py --size small --family 1 --ds_type mixed --partition train --epoch 42 --overwrite
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.io import datasetFilename
from metabeta.utils.templates import FAMILY_NAMES
from metabeta.analytical.fit import glmm

OUTDIR = Path(__file__).resolve().parent / '..' / 'outputs' / 'data'


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pre-compute analytical fits for a dataset npz file.')
    parser.add_argument('--size',      required=True,           help='Dataset size: tiny|small|medium|large|huge')
    parser.add_argument('--family',    type=int, required=True, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type',   type=str, default='mixed', help='Dataset type: toy|flat|scm|mixed|sampled|real (default: mixed)')
    parser.add_argument('--partition', type=str, default='valid', help='Partition: train|valid|test|eval (default: valid)')
    parser.add_argument('--epoch',     type=int, default=1,     help='Training epoch, only used for partition=train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,   help='Datasets per batch (default: 32)')
    parser.add_argument('--device',    default='cpu',           help='Compute device: cpu|cuda (default: cpu)')
    parser.add_argument('--overwrite', action='store_true',     help='Recompute even if stats are already present')
    return parser.parse_args()
# fmt: on


def buildPath(size: str, family: int, ds_type: str, partition: str, epoch: int) -> Path:
    data_id = f'{size}-{FAMILY_NAMES[family]}-{ds_type}'
    return OUTDIR / data_id / datasetFilename(partition, epoch)


def run(path: Path, batch_size: int, device: torch.device) -> int:
    """Compute and save stats for one npz file. Returns number of datasets processed."""
    with np.load(path, allow_pickle=True) as f:
        raw = dict(f)

    N = len(raw['y'])
    d_file = int(raw['d'].max())
    q_file = int(raw['q'].max())
    m_max = int(raw['ns'].shape[1])

    lf = int(np.asarray(raw['likelihood_family']).flat[0])
    has_eps = lf == 0

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
    for batch in tqdm(dl, desc='batches', unit='batch'):
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
            map_refine=True,
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
    path = buildPath(cfg.size, cfg.family, cfg.ds_type, cfg.partition, cfg.epoch)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as f:
        already_done = 'beta_est' in f
    if already_done and not cfg.overwrite:
        print(f'SKIP (already done): {path}')
        return

    device = torch.device(cfg.device)
    print(f'Processing {path} ...', end=' ', flush=True)
    t0 = time.perf_counter()
    n = run(path, cfg.batch_size, device)
    dt = time.perf_counter() - t0
    print(f'{n} datasets  {dt:.1f}s  ({dt / n * 1000:.1f} ms/ds)')


if __name__ == '__main__':
    main()
