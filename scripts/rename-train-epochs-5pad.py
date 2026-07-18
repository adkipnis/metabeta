#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


TRAIN_EPOCH_RE = re.compile(r'^train_ep(\d{4})\.npz$')


def find_renames(root: Path) -> list[tuple[Path, Path]]:
    renames = []
    for path in sorted(root.rglob('train_ep????.npz')):
        match = TRAIN_EPOCH_RE.fullmatch(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        target = path.with_name(f'train_ep{epoch:05d}.npz')
        if target != path:
            renames.append((path, target))
    return renames


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rename 4-padded train epoch files to 5-padded names.'
    )
    parser.add_argument(
        'root',
        nargs='?',
        type=Path,
        default=Path('metabeta/outputs/data'),
        help='Root directory to scan recursively (default: metabeta/outputs/data).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Perform the rename. Without this flag, only print the planned changes.',
    )
    cfg = parser.parse_args()

    root = cfg.root
    if not root.exists():
        raise FileNotFoundError(f'root directory does not exist: {root}')

    renames = find_renames(root)
    if not renames:
        print('No 4-padded train epoch files found.')
        return

    collisions = [(src, dst) for src, dst in renames if dst.exists()]
    if collisions:
        for src, dst in collisions:
            print(f'collision: {src} -> {dst} already exists')
        raise FileExistsError('refusing to rename because one or more targets already exist')

    action = 'Renaming' if cfg.apply else 'Would rename'
    for src, dst in renames:
        print(f'{action}: {src} -> {dst}')
        if cfg.apply:
            src.rename(dst)

    if cfg.apply:
        print(f'Renamed {len(renames)} files.')
    else:
        print(f'Dry run only. Pass --apply to rename {len(renames)} files.')


if __name__ == '__main__':
    main()
