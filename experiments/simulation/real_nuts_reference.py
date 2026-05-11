"""Run NUTS reference fits for preprocessed real grouped datasets.

The script packages real datasets into the same batch schema used by
``metabeta.simulation.fit.Fitter`` and then runs PyMC NUTS with that fitter's
defaults.  It currently uses a random-intercept model (q=1), which is the only
real-data random-effect structure available without extra formula metadata.

Usage from repo root:
    uv run python experiments/simulation/real_nuts_reference.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from metabeta.simulation.fit import Fitter
from metabeta.simulation.prior import bambiDefaultPriors
from metabeta.utils.evaluation import nutsConvergeMask
from metabeta.utils.experiments import DATA_DIR, PREPROCESSED_DATA_DIR
from metabeta.utils.padding import aggregate


_Y_TYPE_TO_FAMILY = {
    'continuous': 0,
    'binary': 1,
    'count': 2,
}


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=Path, default=PREPROCESSED_DATA_DIR / 'test')
    parser.add_argument('--data-id', default='real-normal-reference')
    parser.add_argument('--y-type', default='continuous', choices=['continuous', 'binary', 'count'])
    parser.add_argument('--q', type=int, default=1, help='Number of leading columns used as random effects.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tune', type=int, default=2000)
    parser.add_argument('--target_accept', type=float, default=0.8)
    parser.add_argument('--max_treedepth', type=int, default=10)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--mp_ctx', type=str, default='forkserver')
    parser.add_argument('--diagonal', action='store_true')
    parser.add_argument('--force', action='store_true', help='Refit datasets even when an individual fit exists.')
    return parser.parse_args()
# fmt: on


def _scalar(data, key: str) -> int:
    return int(np.asarray(data[key]).item())


def _yType(data) -> str:
    value = data['y_type']
    return str(value.item() if np.asarray(value).shape == () else value)


def _datasetFromRealFile(path: Path, y_type: str, q: int) -> dict[str, np.ndarray] | None:
    with np.load(path, allow_pickle=True) as data:
        if _yType(data) != y_type:
            return None

        X_raw = np.asarray(data['X'], dtype=np.float64)
        y = np.asarray(data['y'], dtype=np.float64)
        n = _scalar(data, 'n')
        d = _scalar(data, 'd')
        m = _scalar(data, 'm')
        q_i = min(q, d)
        family = _Y_TYPE_TO_FAMILY[y_type]
        hyper = bambiDefaultPriors(d, q_i, likelihood_family=family)
        X = np.column_stack([np.ones(n, dtype=np.float64), X_raw])
        ns = np.asarray(data['ns'], dtype=np.int64)
        groups = np.asarray(data['groups'], dtype=np.int64)

    out: dict[str, np.ndarray] = {
        'ffx': np.full(d, np.nan, dtype=np.float64),
        'sigma_rfx': np.full(q_i, np.nan, dtype=np.float64),
        'sigma_eps': np.array(np.nan, dtype=np.float64),
        'corr_rfx': np.eye(q_i, dtype=np.float64),
        'rfx': np.full((m, q_i), np.nan, dtype=np.float64),
        'y': y,
        'X': X,
        'groups': groups,
        'm': np.array(m, dtype=np.int64),
        'n': np.array(n, dtype=np.int64),
        'ns': ns,
        'd': np.array(d, dtype=np.int64),
        'q': np.array(q_i, dtype=np.int64),
        'sd_y': np.array(np.nanstd(y), dtype=np.float64),
        'r_squared': np.array(np.nan, dtype=np.float64),
        'source': np.array(path.stem, dtype=object),
    }
    out.update(hyper)
    return out


def makeBatch(test_dir: Path, y_type: str, q: int) -> dict[str, np.ndarray]:
    datasets = []
    for path in sorted(test_dir.glob('*.npz')):
        ds = _datasetFromRealFile(path, y_type=y_type, q=q)
        if ds is not None:
            datasets.append(ds)
    if not datasets:
        raise ValueError(f'no real datasets found for y_type={y_type} in {test_dir}')
    return aggregate(datasets)


def _fitCfg(args: argparse.Namespace, idx: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_id=args.data_id,
        idx=idx,
        method='nuts',
        seed=args.seed,
        tune=args.tune,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        draws=args.draws,
        chains=args.chains,
        loop=args.loop,
        mp_ctx=args.mp_ctx,
        diagonal=args.diagonal,
        partition='test',
    )


def _toTorchBatch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if key.startswith('nuts_') and np.issubdtype(value.dtype, np.number):
            if key == 'nuts_draws':
                value = np.asarray(value).reshape(-1)
                value = value[:1]
            out[key] = torch.as_tensor(value)
    return out


def _summarizeConvergence(batch: dict[str, np.ndarray]) -> None:
    torch_batch = _toTorchBatch(batch)
    names = batch.get('source', np.array([str(i) for i in range(len(batch['n']))]))
    for mode in ('strict', 'liberal'):
        mask = nutsConvergeMask(torch_batch, mode=mode)
        if mask is None:
            continue
        print(f'NUTS convergence ({mode}): {int(mask.sum())} / {len(mask)}')
        failed = np.asarray(names)[~mask]
        if len(failed):
            print('  failed:', ', '.join(str(x) for x in failed))


def main() -> None:
    args = setup()
    outdir = DATA_DIR / args.data_id
    outdir.mkdir(parents=True, exist_ok=True)
    batch_path = outdir / 'test.npz'

    if args.force or not batch_path.exists():
        batch = makeBatch(args.test_dir, y_type=args.y_type, q=args.q)
        np.savez_compressed(batch_path, **batch, allow_pickle=True)
        print(f'Wrote real-data batch to {batch_path} ({len(batch["n"])} datasets)')
    else:
        with np.load(batch_path, allow_pickle=True) as data:
            batch = dict(data)
        print(f'Using existing real-data batch {batch_path} ({len(batch["n"])} datasets)')

    for idx in range(len(batch['n'])):
        cfg = _fitCfg(args, idx=idx)
        fitter = Fitter(cfg, srcdir=DATA_DIR)
        if fitter.outpath.exists() and not args.force:
            print(f'Skipping existing fit {fitter.outpath}')
            continue
        fitter.go()

    cfg0 = _fitCfg(args, idx=0)
    fitter0 = Fitter(cfg0, srcdir=DATA_DIR)
    batch.update(fitter0._aggregate('nuts'))
    fit_path = batch_path.with_suffix('.fit.npz')
    np.savez_compressed(fit_path, **batch, allow_pickle=True)
    print(f'Wrote NUTS reference batch to {fit_path}')
    _summarizeConvergence(batch)


if __name__ == '__main__':
    main()
