import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from metabeta.simulation.fit import Fitter
from metabeta.utils.io import datasetFilename


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check generated training datasets and fit files for a d_tag.'
    )
    parser.add_argument(
        '--data_id',
        type=str,
        nargs='+',
        required=True,
        help='one or more data config tags',
    )
    parser.add_argument(
        '--n_train_epochs',
        type=int,
        default=500,
        help='number of expected train partitions (default = 500)',
    )
    parser.add_argument(
        '--n_fits',
        type=int,
        default=512,
        help='number of expected fit files per method (default = 512)',
    )
    parser.add_argument(
        '--srcdir',
        type=str,
        default=str(Path(__file__).resolve().parent / '..' / 'outputs' / 'data'),
        help='directory containing generated data and fits',
    )
    parser.add_argument(
        '--no_reintegrate',
        action='store_true',
        help='only check files; do not reintegrate fits even when complete',
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='load each npz file to inspect readability (default = False)',
    )
    return parser.parse_args()

def _checkReadable(
    paths: list[Path], inspect: bool = False
) -> tuple[list[Path], list[tuple[Path, str]]]:
    missing = []
    broken = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        if not inspect:
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                _ = data.files
        except Exception as exc:
            broken.append((path, str(exc)))
    return missing, broken


def _printPathPreview(paths: list[Path], label: str) -> None:
    if not paths:
        return
    print(f'{label} ({len(paths)}):')
    for p in paths:
        print(f'  - {p}')


def _printBrokenPreview(items: list[tuple[Path, str]], label: str) -> None:
    if not items:
        return
    print(f'{label} ({len(items)}):')
    for p, err in items:
        print(f'  - {p}: {err}')


def _fitCfg(data_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        d_tag=data_id,
        idx=0,
        reintegrate=True,
        respecify_ffx=False,
        method='nuts',
        seed=42,
        tune=2000,
        draws=1000,
        chains=4,
        loop=False,
        viter=50_000,
        lr=5e-3,
    )


def _runForTag(data_id: str, cfg: argparse.Namespace, srcdir: Path) -> int:
    fits_dir = srcdir / 'fits'

    print('Running dataset and fit checks')
    print(f'  data_id: {data_id}')
    print(f'  expected train files: {cfg.n_train_epochs}')
    print(f'  expected pymc (nuts) fits: {cfg.n_fits}')
    print(f'  expected advi fits: {cfg.n_fits}')
    print(f'  expected total fit files: {2 * cfg.n_fits}')
    print(f'  data directory: {srcdir}')
    print(f'  fits directory: {fits_dir}')
    print(f'  npz inspection: {cfg.inspect}')

    # check expected training partitions
    train_paths = [
        srcdir / data_id / datasetFilename(partition='train', epoch=epoch)
        for epoch in range(1, cfg.n_train_epochs + 1)
    ]
    train_missing, train_broken = _checkReadable(train_paths, inspect=cfg.inspect)
    train_ok = len(train_missing) == 0 and len(train_broken) == 0
    print(
        f'train partitions: {cfg.n_train_epochs - len(train_missing) - len(train_broken)}/{cfg.n_train_epochs} ok'
    )
    _printPathPreview(train_missing, 'missing train partitions')
    _printBrokenPreview(train_broken, 'broken train partitions')

    # check expected fit files
    test_fname = datasetFilename(partition='test')
    test_path = srcdir / data_id / test_fname
    if not test_path.exists():
        print(f'missing test batch used for fit naming: {test_path}')

    stem = Path(test_fname).stem
    pymc_paths = [fits_dir / f'{stem}_nuts_{idx:03d}.npz' for idx in range(cfg.n_fits)]
    advi_paths = [fits_dir / f'{stem}_advi_{idx:03d}.npz' for idx in range(cfg.n_fits)]

    pymc_missing, pymc_broken = _checkReadable(pymc_paths, inspect=cfg.inspect)
    advi_missing, advi_broken = _checkReadable(advi_paths, inspect=cfg.inspect)

    pymc_ok = len(pymc_missing) == 0 and len(pymc_broken) == 0
    advi_ok = len(advi_missing) == 0 and len(advi_broken) == 0
    fits_ok = pymc_ok and advi_ok

    print(f'pymc (nuts) fits: {cfg.n_fits - len(pymc_missing) - len(pymc_broken)}/{cfg.n_fits} ok')
    _printPathPreview(pymc_missing, 'missing pymc (nuts) fits')
    _printBrokenPreview(pymc_broken, 'broken pymc (nuts) fits')

    print(f'advi fits: {cfg.n_fits - len(advi_missing) - len(advi_broken)}/{cfg.n_fits} ok')
    _printPathPreview(advi_missing, 'missing advi fits')
    _printBrokenPreview(advi_broken, 'broken advi fits')

    # reintegrate only when all fits exist and are readable
    reintegrated = False
    if fits_ok and not cfg.no_reintegrate:
        try:
            fitter = Fitter(_fitCfg(data_id), srcdir=srcdir)
            fitter.reintegrate()
            reintegrated = True
        except Exception as exc:
            print(f'reintegration failed: {exc}')
            return 1

    print(f"reintegrated: {'yes' if reintegrated else 'no'}")

    if train_ok and fits_ok:
        return 0
    return 1


def main() -> int:
    cfg = setup()
    srcdir = Path(cfg.srcdir).resolve()

    status_by_tag = {}
    for i, data_id in enumerate(cfg.data_id):
        if i > 0:
            print('-' * 60)
        status_by_tag[data_id] = _runForTag(data_id, cfg, srcdir) == 0

    print('-' * 60)
    print('Final status by data_id:')
    for data_id, ok in status_by_tag.items():
        print(f"  {data_id}: {'ok' if ok else 'failed'}")

    return 0 if all(status_by_tag.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
