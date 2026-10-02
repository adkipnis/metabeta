"""Per-method reference fits stored as ``{partition}.{tag}.npz`` next to ``{partition}.npz``.

One file per method and budget level (the *tag*), arrays only, every key prefixed
``{tag}_`` and one dataset per leading axis entry. Each file carries ``source_sha``, the
SHA-256 of the dataset file it was fitted on, so a fit can never be paired with data it was
not computed on; ``Collection(path, fits=(...))`` merges the requested tags into its key
space and fails loudly on a mismatch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np

# NUTS budget ladder (nuts0 < nuts1 < nuts2) plus the composite `nuts` written by
# reintegration, ADVI at PyMC's default budget (advi0) and at 100k iterations (advi1),
# Pathfinder with 4 (pathfinder0) and 20 (pathfinder1) paths, and the pymc-extras Laplace fit.
FIT_TAGS = (
    'nuts0',
    'nuts1',
    'nuts2',
    'nuts',
    'advi0',
    'advi1',
    'pathfinder0',
    'pathfinder1',
    'laplace',
)
# the top of the NUTS ladder: the reference posterior, and whose convergence decides which
# datasets enter the comparisons
REFERENCE_TAG = 'nuts2'
_SHA_KEY = 'source_sha'


def fitPath(data_path: Path, tag: str) -> Path:
    """``.../test.npz`` → ``.../test.{tag}.npz``."""
    if tag not in FIT_TAGS:
        raise ValueError(f'unknown fit tag {tag!r}; expected one of {FIT_TAGS}')
    return Path(data_path).with_suffix(f'.{tag}.npz')


def availableFits(data_path: Path) -> tuple[str, ...]:
    """Tags whose fit file exists next to ``data_path``, in ``FIT_TAGS`` order."""
    return tuple(tag for tag in FIT_TAGS if fitPath(data_path, tag).exists())


def sourceSha(data_path: Path) -> str:
    sha = hashlib.sha256()
    with open(data_path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b''):
            sha.update(chunk)
    return sha.hexdigest()


def _checkKeys(keys: Iterable[str], tag: str, path: Path) -> None:
    bad = [k for k in keys if not k.startswith(f'{tag}_')]
    if bad:
        raise ValueError(f'{path}: keys must start with {tag!r}_, got {bad}')


def saveFits(data_path: Path, tag: str, arrays: dict[str, np.ndarray], force: bool = False) -> Path:
    """Write ``arrays`` as the ``tag`` fit of ``data_path``; returns the fit file path.

    Every key must start with ``{tag}_`` and every array must have one leading entry per
    dataset in ``data_path``. Refuses to overwrite an existing file unless ``force``.
    """
    path = fitPath(data_path, tag)
    if path.exists() and not force:
        raise FileExistsError(f'{path} already exists; pass force=True to overwrite')
    _checkKeys(arrays, tag, path)
    with np.load(data_path, allow_pickle=True) as raw:
        n = int(raw['m'].shape[0])
    short = {k: v.shape for k, v in arrays.items() if np.ndim(v) == 0 or v.shape[0] != n}
    if short:
        raise ValueError(f'{path}: every array needs a leading axis of {n} datasets, got {short}')
    np.savez_compressed(path, **arrays, **{_SHA_KEY: np.array(sourceSha(data_path))})
    return path


def loadFits(data_path: Path, tag: str, keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Arrays of the ``tag`` fit of ``data_path`` after verifying the source checksum.

    ``keys`` restricts the load to those members (absent ones are left out, so callers test
    membership as they would on the npz); the default is every array except the checksum.
    """
    path = fitPath(data_path, tag)
    with np.load(path, allow_pickle=True) as raw:
        if _SHA_KEY not in raw.files:
            raise ValueError(f'{path} has no {_SHA_KEY}; not a per-method fit file')
        if str(raw[_SHA_KEY]) != sourceSha(data_path):
            raise ValueError(
                f'{path} was fitted on a different {data_path.name} (checksum mismatch)'
            )
        names = [k for k in raw.files if k != _SHA_KEY]
        _checkKeys(names, tag, path)
        if keys is not None:
            names = [k for k in keys if k in names]
        return {k: raw[k] for k in names}
