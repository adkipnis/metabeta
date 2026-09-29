"""Check generated data files and per-dataset reference fits; reintegrate complete fit tags.

For the test and valid partitions every fit tag with at least one per-dataset file under
``fits/`` is checked for completeness; complete tags are aggregated into
``{partition}.{tag}.npz`` unless ``--no_reintegrate``. Missing or broken fits print the
``scripts/fit-ref.sh`` command that refits exactly those datasets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from metabeta.simulation.fit import Fitter, tagMethodLevel
from metabeta.utils.fits import FIT_TAGS, availableFits
from metabeta.utils.names import datasetFilename

# tags a fit run writes; the composite `nuts` is composed from the level files, not fitted
FITTED_TAGS = tuple(tag for tag in FIT_TAGS if tag != 'nuts')


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check generated data files for one or more data config tags.'
    )
    parser.add_argument('--data_id', type=str, nargs='+', required=True,
                        help='one or more data config tags')
    parser.add_argument('--partition', choices=['train', 'test', 'valid'], required=True,
                        help='check training epochs (train) or reference fits (test, valid)')
    parser.add_argument('-b', type=int, default=12000,
                        help='batch size: expected number of train epochs (default: 12000)')
    parser.add_argument('--srcdir', type=str,
                        default=str(Path(__file__).resolve().parent / '..' / 'outputs' / 'data'),
                        help='root data directory')
    parser.add_argument('--no_reintegrate', action='store_true',
                        help='skip reintegration of complete fit tags')
    parser.add_argument('--inspect', action='store_true',
                        help='load each npz file to verify readability')
    return parser.parse_args()
# fmt: on


def _check(
    paths: list[Path], label: str, inspect: bool = False
) -> tuple[list[Path], list[tuple[Path, str]]]:
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


def _report(
    label: str, n_ok: int, total: int, missing: list[Path], broken: list[tuple[Path, str]]
) -> None:
    print(f'{label}: {n_ok}/{total} ok')
    for p in missing:
        print(f'  missing  {p}')
    for p, err in broken:
        print(f'  broken   {p}: {err}')


def _failedIndices(missing: list[Path], broken: list[tuple[Path, str]]) -> list[int]:
    paths = [*missing, *(p for p, _ in broken)]
    return sorted({int(p.stem.rsplit('_', maxsplit=1)[-1]) for p in paths})


def _printRefitCommand(data_id: str, tag: str, failed_idx: list[int], partition: str) -> None:
    method, level = tagMethodLevel(tag)
    idx_args = ' '.join(str(idx) for idx in failed_idx)
    print('  rerun with:')
    print(
        f'    scripts/fit-ref.sh --method {method} --level {level} --data_id {data_id}'
        f' --partition {partition} --idx {idx_args}'
    )


def _checkTrain(data_id: str, cfg: argparse.Namespace, srcdir: Path) -> bool:
    paths = [
        srcdir / data_id / datasetFilename(partition='train', epoch=e) for e in range(1, cfg.b + 1)
    ]
    missing, broken = _check(paths, 'train partitions', inspect=cfg.inspect)
    _report(
        'train partitions', len(paths) - len(missing) - len(broken), len(paths), missing, broken
    )
    return not missing and not broken


def _checkFits(data_id: str, cfg: argparse.Namespace, srcdir: Path) -> bool:
    """Check every started fit tag of the partition; reintegrate the complete ones."""
    data_path = srcdir / data_id / datasetFilename(partition=cfg.partition)
    fits_dir = data_path.parent / 'fits'
    with np.load(data_path, allow_pickle=True) as raw:
        n = int(raw['m'].shape[0])
    stem = data_path.stem

    ok = True
    for tag in FITTED_TAGS:
        paths = [fits_dir / f'{stem}_{tag}_{i:03d}.npz' for i in range(n)]
        if not any(p.exists() for p in paths):
            continue
        missing, broken = _check(paths, f'{tag} fits', inspect=cfg.inspect)
        _report(f'{tag} fits', n - len(missing) - len(broken), n, missing, broken)
        if missing or broken:
            _printRefitCommand(data_id, tag, _failedIndices(missing, broken), cfg.partition)
            ok = False
        elif not cfg.no_reintegrate:
            method, level = tagMethodLevel(tag)
            fit_cfg = argparse.Namespace(
                data_id=data_id, idx=0, method=method, level=level, partition=cfg.partition
            )
            Fitter(fit_cfg, srcdir=srcdir).reintegrate(tags=(tag,))
    if (
        ok
        and not cfg.no_reintegrate
        and any(t.startswith('nuts') for t in availableFits(data_path))
    ):
        Fitter(
            argparse.Namespace(
                data_id=data_id, idx=0, method='nuts', level=0, partition=cfg.partition
            ),
            srcdir=srcdir,
        ).composeNuts()
    done = availableFits(data_path)
    if done:
        print(f'reintegrated: {" ".join(done)}')
    return ok


def main() -> int:
    cfg = setup()
    srcdir = Path(cfg.srcdir).resolve()
    check_fn = _checkTrain if cfg.partition == 'train' else _checkFits

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
