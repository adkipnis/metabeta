import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from metabeta.simulation.fit import Fitter
from metabeta.utils.io import datasetFilename

# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check generated data files for one or more data config tags.'
    )
    parser.add_argument('--data_id', type=str, nargs='+', required=True,
                        help='one or more data config tags')
    parser.add_argument('--mode', choices=['train', 'test'], required=True,
                        help='check training partitions (train) or test fits (test)')
    parser.add_argument('-b', type=int, default=1000,
                        help='batch size: expected number of train partitions (default: 1000)')
    parser.add_argument('--n_fits', type=int, default=512,
                        help='expected fit files per method (default: 512)')
    parser.add_argument('--srcdir', type=str,
                        default=str(Path(__file__).resolve().parent / '..' / 'outputs' / 'data'),
                        help='root data directory')
    parser.add_argument('--no_reintegrate', action='store_true',
                        help='skip reintegration even when all fits are present (test mode only)')
    parser.add_argument('--inspect', action='store_true',
                        help='load each npz file to verify readability')
    return parser.parse_args()
# fmt: on


def _check(paths: list[Path], label: str, inspect: bool = False) -> tuple[list[Path], list[tuple[Path, str]]]:
    missing, broken = [], []
    for p in tqdm(paths, desc=label, unit='file'):
        if not p.exists():
            missing.append(p)
        elif inspect:
            try:
                with np.load(p, allow_pickle=True) as f:
                    _ = f.files
            except Exception as exc:
                broken.append((p, str(exc)))
    return missing, broken


def _report(label: str, n_ok: int, total: int, missing: list[Path], broken: list[tuple[Path, str]]) -> None:
    print(f'{label}: {n_ok}/{total} ok')
    for p in missing:
        print(f'  missing  {p}')
    for p, err in broken:
        print(f'  broken   {p}: {err}')


def _checkTrain(data_id: str, cfg: argparse.Namespace, srcdir: Path) -> bool:
    paths = [
        srcdir / data_id / datasetFilename(partition='train', epoch=e)
        for e in range(1, cfg.b + 1)
    ]
    missing, broken = _check(paths, 'train partitions', inspect=cfg.inspect)
    _report('train partitions', len(paths) - len(missing) - len(broken), len(paths), missing, broken)
    return not missing and not broken


def _checkTest(data_id: str, cfg: argparse.Namespace, srcdir: Path) -> bool:
    fits_dir = srcdir / data_id / 'fits'
    stem = Path(datasetFilename(partition='test')).stem

    pymc_paths = [fits_dir / f'{stem}_nuts_{i:03d}.npz' for i in range(cfg.n_fits)]
    advi_paths = [fits_dir / f'{stem}_advi_{i:03d}.npz' for i in range(cfg.n_fits)]

    pymc_missing, pymc_broken = _check(pymc_paths, 'nuts fits', inspect=cfg.inspect)
    advi_missing, advi_broken = _check(advi_paths, 'advi fits', inspect=cfg.inspect)
    _report('nuts fits', len(pymc_paths) - len(pymc_missing) - len(pymc_broken), len(pymc_paths), pymc_missing, pymc_broken)
    _report('advi fits', len(advi_paths) - len(advi_missing) - len(advi_broken), len(advi_paths), advi_missing, advi_broken)

    fits_ok = not any([pymc_missing, pymc_broken, advi_missing, advi_broken])
    if fits_ok and not cfg.no_reintegrate:
        _reintegrate(data_id, srcdir)
    return fits_ok


def _reintegrate(data_id: str, srcdir: Path) -> None:
    fit_cfg = argparse.Namespace(data_id=data_id, idx=0, method='nuts')
    try:
        Fitter(fit_cfg, srcdir=srcdir).reintegrate()
    except Exception as exc:
        print(f'reintegration failed: {exc}')


def main() -> int:
    cfg = setup()
    srcdir = Path(cfg.srcdir).resolve()
    check_fn = _checkTrain if cfg.mode == 'train' else _checkTest

    results = {}
    for i, data_id in enumerate(cfg.data_id):
        if i > 0:
            print()
        print(f'--- {data_id} ---')
        results[data_id] = check_fn(data_id, cfg, srcdir)

    if len(cfg.data_id) > 1:
        print('\nSummary:')
        for data_id, ok in results.items():
            print(f"  {data_id}: {'ok' if ok else 'failed'}")

    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
