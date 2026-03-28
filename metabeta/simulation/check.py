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
    parser.add_argument('--d_tag', type=str, required=True, help='name of data config file')
    parser.add_argument(
        '--n_train_epochs',
        type=int,
        default=250,
        help='number of expected train partitions (default = 250)',
    )
    parser.add_argument(
        '--n_fits',
        type=int,
        default=128,
        help='number of expected fit files per method (default = 128)',
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


def _loadDataCfg(d_tag: str) -> dict:
    cfg_path = Path(__file__).resolve().parent / 'configs' / f'{d_tag}.yaml'
    assert cfg_path.exists(), f'config file {cfg_path} does not exist'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


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
    for p in paths[:10]:
        print(f'  - {p}')
    if len(paths) > 10:
        print(f'  ... and {len(paths) - 10} more')


def _printBrokenPreview(items: list[tuple[Path, str]], label: str) -> None:
    if not items:
        return
    print(f'{label} ({len(items)}):')
    for p, err in items[:10]:
        print(f'  - {p}: {err}')
    if len(items) > 10:
        print(f'  ... and {len(items) - 10} more')


def _fitCfg(d_tag: str) -> argparse.Namespace:
    return argparse.Namespace(
        d_tag=d_tag,
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


def main() -> int:
    cfg = setup()
    srcdir = Path(cfg.srcdir).resolve()
    fits_dir = srcdir / 'fits'

    data_cfg = _loadDataCfg(cfg.d_tag)

    # check expected training partitions
    train_paths = [
        srcdir / datasetFilename(data_cfg, partition='train', epoch=epoch)
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
    test_fname = datasetFilename(data_cfg, partition='test')
    test_path = srcdir / test_fname
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
            fitter = Fitter(_fitCfg(cfg.d_tag), srcdir=srcdir)
            fitter.reintegrate()
            reintegrated = True
        except Exception as exc:
            print(f'reintegration failed: {exc}')
            return 1

    print(f"reintegrated: {'yes' if reintegrated else 'no'}")

    if train_ok and fits_ok:
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
