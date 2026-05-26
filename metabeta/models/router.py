"""Checkpoint packaging utilities for routed model inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from metabeta.utils.constants import LIKELIHOOD_FAMILIES
from metabeta.utils.experiments import CHECKPOINT_DIR


JOINT_CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_PREFIXES = ('best', 'latest')

ROUTING_KEYS = (
    'likelihood_family',
    'min_d',
    'max_d',
    'min_q',
    'max_q',
    'min_m',
    'max_m',
    'min_n',
    'max_n',
    'max_n_total',
    'min_bg_df',
    'min_within_df',
    'shape_profile',
    'model_id',
    'data_id',
)


def joinCheckpoints(
    checkpoints: Mapping[str, str | Path] | Sequence[str | Path],
    output_path: str | Path | None = None,
    *,
    prefixes: Mapping[str, str] | Sequence[str] | str | None = None,
    ids: Sequence[str] | None = None,
    map_location: str | torch.device = 'cpu',
) -> Path:
    """Join model weights from multiple checkpoints into one routed checkpoint.

    The source checkpoints are expected to be trusted local training checkpoints.
    They may contain optimizer and RNG state, but this function writes only
    model weights plus the config metadata needed by the router.

    If ``output_path`` is omitted, the checkpoint is written to
    ``metabeta/outputs/checkpoints/joint_{family}_v{version}.pt``. If
    ``output_path`` is a directory, the same default filename is used inside it.
    """

    entries = _normalizeCheckpointEntries(checkpoints, prefixes=prefixes, ids=ids)

    submodels: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        submodel_id = entry['id']
        if submodel_id in seen_ids:
            raise ValueError(f'duplicate checkpoint id: {submodel_id}')
        seen_ids.add(submodel_id)

        checkpoint_path = _resolveCheckpointPath(entry['path'], entry['prefix'])
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if 'model_state' not in payload:
            raise KeyError(f'checkpoint is missing model_state: {checkpoint_path}')

        trainer_cfg = _sanitizeConfig(payload.get('trainer_cfg', {}))
        data_cfg = _sanitizeConfig(payload.get('data_cfg', {}))
        model_cfg = _sanitizeConfig(payload.get('model_cfg', {}))

        submodels.append(
            {
                'id': submodel_id,
                'source': str(checkpoint_path),
                'prefix': entry['prefix'],
                'trainer_cfg': trainer_cfg,
                'data_cfg': data_cfg,
                'model_cfg': model_cfg,
                'routing': _routingMetadata(trainer_cfg, data_cfg, model_cfg),
                'model_state': _cpuModelState(payload['model_state']),
            }
        )

    output_path = _resolveOutputPath(output_path, submodels)
    joint_payload = {
        '_version': JOINT_CHECKPOINT_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'submodels': submodels,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    torch.save(joint_payload, tmp_path)
    tmp_path.replace(output_path)
    return output_path


def _resolveOutputPath(
    output_path: str | Path | None, submodels: Sequence[Mapping[str, Any]]
) -> Path:
    filename = _jointCheckpointFilename(submodels)
    if output_path is None:
        return CHECKPOINT_DIR / filename

    path = Path(output_path)
    if path.exists() and path.is_dir():
        return path / filename
    if path.suffix == '':
        return path / filename
    return path


def _jointCheckpointFilename(submodels: Sequence[Mapping[str, Any]]) -> str:
    family_ids = {
        submodel.get('routing', {}).get('likelihood_family')
        for submodel in submodels
        if submodel.get('routing', {}).get('likelihood_family') is not None
    }
    if len(family_ids) != 1:
        raise ValueError('default joint checkpoint naming requires one shared likelihood_family')

    family_id = int(family_ids.pop())
    family = (
        LIKELIHOOD_FAMILIES[family_id]
        if 0 <= family_id < len(LIKELIHOOD_FAMILIES)
        else f'family{family_id}'
    )
    return f'joint_{family}_v{JOINT_CHECKPOINT_VERSION}.pt'


def _normalizeCheckpointEntries(
    checkpoints: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    prefixes: Mapping[str, str] | Sequence[str] | str | None,
    ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if isinstance(checkpoints, Mapping):
        if ids is not None:
            raise ValueError('ids cannot be provided when checkpoints is a mapping')
        return [
            {
                'id': str(checkpoint_id),
                'path': Path(path),
                'prefix': _prefixFor(checkpoint_id, prefixes),
            }
            for checkpoint_id, path in checkpoints.items()
        ]

    if isinstance(checkpoints, (str, Path)):
        raise TypeError('checkpoints must be a mapping or a sequence of checkpoint paths')

    if ids is not None and len(ids) != len(checkpoints):
        raise ValueError('ids must have the same length as checkpoints')
    if (
        isinstance(prefixes, Sequence)
        and not isinstance(prefixes, str)
        and len(prefixes) != len(checkpoints)
    ):
        raise ValueError('prefixes must have the same length as checkpoints')

    entries = []
    for i, path in enumerate(checkpoints):
        checkpoint_path = Path(path)
        checkpoint_id = str(ids[i]) if ids is not None else checkpoint_path.stem
        prefix_key = checkpoint_id if isinstance(prefixes, Mapping) else i
        entries.append(
            {
                'id': checkpoint_id,
                'path': checkpoint_path,
                'prefix': _prefixFor(prefix_key, prefixes),
            }
        )
    return entries


def _prefixFor(
    key: str | int, prefixes: Mapping[str, str] | Sequence[str] | str | None
) -> str | None:
    if prefixes is None:
        return None
    if isinstance(prefixes, str):
        return prefixes
    if isinstance(prefixes, Mapping):
        return prefixes.get(str(key))
    return prefixes[int(key)]


def _resolveCheckpointPath(path: Path, prefix: str | None) -> Path:
    if path.is_dir():
        if prefix is not None:
            candidate = path / f'{prefix}.pt'
            if not candidate.exists():
                raise FileNotFoundError(f'checkpoint not found: {candidate}')
            return candidate

        for default_prefix in DEFAULT_CHECKPOINT_PREFIXES:
            candidate = path / f'{default_prefix}.pt'
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f'checkpoint directory contains none of: {", ".join(DEFAULT_CHECKPOINT_PREFIXES)}'
        )

    if prefix is not None:
        raise ValueError(f'checkpoint prefix was provided for a file path: {path}')
    if not path.exists():
        raise FileNotFoundError(f'checkpoint not found: {path}')
    return path


def _cpuModelState(model_state: Mapping[str, Any]) -> dict[str, Any]:
    cpu_state = {}
    for key, value in model_state.items():
        if isinstance(value, torch.Tensor):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def _routingMetadata(
    trainer_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    routing: dict[str, Any] = {}
    for key in ROUTING_KEYS:
        value = _firstConfigValue(key, data_cfg, trainer_cfg, model_cfg)
        if value is not None:
            routing[key] = value

    if 'max_d' not in routing and 'd_ffx' in model_cfg:
        routing['max_d'] = model_cfg['d_ffx']
    if 'max_q' not in routing and 'd_rfx' in model_cfg:
        routing['max_q'] = model_cfg['d_rfx']

    return routing


def _firstConfigValue(key: str, *configs: Mapping[str, Any]) -> Any:
    for cfg in configs:
        if key in cfg:
            return cfg[key]
    return None


def _sanitizeConfig(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitizeConfig(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitizeConfig(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitizeConfig(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
