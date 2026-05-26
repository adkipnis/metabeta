"""Checkpoint packaging utilities for routed model inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import warnings

import torch

from metabeta.models.approximator import Approximator
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.constants import LIKELIHOOD_FAMILIES
from metabeta.utils.dataloader import toDevice
from metabeta.utils.evaluation import Proposal
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

REQUIRED_BATCH_KEYS = (
    'X',
    'Z',
    'y',
    'ns',
    'm',
    'n',
    'mask_d',
    'mask_q',
    'mask_n',
    'mask_m',
    'mask_mq',
    'mask_corr',
    'nu_ffx',
    'tau_ffx',
    'tau_rfx',
    'eta_rfx',
    'family_ffx',
    'family_sigma_rfx',
)


@dataclass
class RouterResult:
    """Inference output and routing metadata."""

    proposal: Proposal | None
    routes: list[str]
    validation: list[dict[str, Any]]
    log_probs: dict[str, torch.Tensor] | None = None


class Router:
    """Route dataloader-formatted datasets through a joint checkpoint.

    Models are instantiated lazily when their first compatible batch is run.

    TODO: accept raw pandas DataFrames, optional single/multiple prior
    specifications using default Bambi-style priors when absent, and an lme4- or
    Bambi-like formula string identifying y, fixed predictors, random-effect
    terms, and grouping variables. The current implementation expects data that
    is already preprocessed or path-backed by the existing metabeta dataloader.
    """

    def __init__(
        self,
        joint_checkpoint: str | Path,
        *,
        device: str | torch.device = 'cpu',
        batch_size: int | None = None,
    ) -> None:
        self.joint_checkpoint = Path(joint_checkpoint)
        self.device = torch.device(device)
        self.batch_size = batch_size

        payload = torch.load(self.joint_checkpoint, map_location='cpu', weights_only=True)
        if payload.get('_version') != JOINT_CHECKPOINT_VERSION:
            raise ValueError(f'unsupported joint checkpoint version: {payload.get("_version")!r}')
        if 'submodels' not in payload:
            raise KeyError('joint checkpoint is missing submodels')
        self.submodels = list(payload['submodels'])
        if not self.submodels:
            raise ValueError('joint checkpoint contains no submodels')
        self.submodels.sort(key=self._routingSortKey)
        self._submodel_by_id = {str(entry['id']): entry for entry in self.submodels}
        if len(self._submodel_by_id) != len(self.submodels):
            raise ValueError('joint checkpoint contains duplicate submodel ids')
        self._models: dict[str, Approximator] = {}

    def model(self, submodel_id: str) -> Approximator:
        """Return the lazily instantiated model for ``submodel_id``."""

        if submodel_id in self._models:
            return self._models[submodel_id]

        try:
            entry = self._submodel_by_id[submodel_id]
        except KeyError as exc:
            raise KeyError(f'unknown submodel id: {submodel_id}') from exc

        model_cfg = ApproximatorConfig(**entry['model_cfg'])
        model = Approximator(model_cfg).to(self.device)
        model.load_state_dict(entry['model_state'])
        model.eval()
        self._models[submodel_id] = model
        return model

    def prepareData(self, data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate and return a collated dataloader-style batch."""

        if not isinstance(data, Mapping):
            raise TypeError('router input must be a collated dataloader batch dict')
        if not self._isCollatedBatch(data):
            raise TypeError('router input must match the dataloader collated batch format')
        return dict(data)

    def route(self, data: Mapping[str, torch.Tensor]) -> list[str]:
        """Return the selected submodel id for each dataset in ``data``."""

        batch = self.prepareData(data)
        routes, _ = self._routeBatch(batch)
        return routes

    @torch.no_grad()
    def sample(
        self,
        data: Mapping[str, torch.Tensor],
        *,
        n_samples: int = 1,
    ) -> RouterResult:
        """Sample from the posterior through the routed submodel."""

        batch = self.prepareData(data)
        self._validateBatchFormat(batch)
        routes, validation = self._routeBatch(batch)
        submodel_ids = set(routes)
        if len(submodel_ids) != 1:
            raise NotImplementedError(
                'mixed-submodel batches are not reassembled yet; route one compatible '
                'dataset family at a time'
            )

        submodel_id = routes[0]
        model = self.model(submodel_id)
        self._validateBatchMatchesModel(batch, model)
        batch = toDevice(batch, self.device)
        proposal = model.estimate(batch, n_samples=n_samples)
        return RouterResult(proposal=proposal, routes=routes, validation=validation)

    @torch.no_grad()
    def forward(self, data: Mapping[str, torch.Tensor]) -> RouterResult:
        """Evaluate the forward log-probability path for batches with parameters."""

        batch = self.prepareData(data)
        self._validateBatchFormat(batch)
        routes, validation = self._routeBatch(batch)
        submodel_ids = set(routes)
        if len(submodel_ids) != 1:
            raise NotImplementedError(
                'mixed-submodel batches are not reassembled yet; route one compatible '
                'dataset family at a time'
            )

        submodel_id = routes[0]
        model = self.model(submodel_id)
        self._validateBatchMatchesModel(batch, model)
        batch = toDevice(batch, self.device)
        log_probs = model(batch)
        return RouterResult(
            proposal=None,
            routes=routes,
            validation=validation,
            log_probs=log_probs,
        )

    def _routeBatch(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        self._validateRoutingInputs(batch)
        routes = []
        validation = []
        batch_size = int(batch['X'].shape[0])
        for i in range(batch_size):
            selected, failures = self._selectSubmodel(batch, i)
            submodel_id = str(selected['id'])
            routes.append(submodel_id)
            validation.append(
                {
                    'index': i,
                    'submodel_id': submodel_id,
                    'dimensions': self._datasetDimensions(batch, i),
                    'online_stats': 'stats' not in batch,
                    'failures_by_submodel': failures,
                }
            )
        return routes, validation

    def _selectSubmodel(
        self, batch: Mapping[str, torch.Tensor], i: int
    ) -> tuple[Mapping[str, Any], dict[str, list[str]]]:
        failures_by_submodel = {}
        for entry in self.submodels:
            failures = self._compatibilityFailures(batch, i, entry)
            if not failures:
                return entry, failures_by_submodel
            failures_by_submodel[str(entry['id'])] = failures

        warnings.warn(
            f'dataset {i} is incompatible with every routed submodel',
            RuntimeWarning,
            stacklevel=2,
        )
        raise ValueError(f'dataset {i} is outside every routed submodel: {failures_by_submodel}')

    def _compatibilityFailures(
        self, batch: Mapping[str, torch.Tensor], i: int, entry: Mapping[str, Any]
    ) -> list[str]:
        routing = entry.get('routing', {})
        dims = self._datasetDimensions(batch, i)
        failures = []

        for dim_key, route_key in (
            ('d', 'max_d'),
            ('q', 'max_q'),
            ('m', 'max_m'),
            ('n', 'max_n_total'),
        ):
            value = dims[dim_key]
            bound = routing.get(route_key)
            if bound is not None and value > int(bound):
                failures.append(f'{dim_key}={value} > {route_key}={bound}')

        for dim_key, route_key in (
            ('d', 'min_d'),
            ('q', 'min_q'),
            ('m', 'min_m'),
        ):
            value = dims[dim_key]
            bound = routing.get(route_key)
            if bound is not None and value < int(bound):
                failures.append(f'{dim_key}={value} < {route_key}={bound}')

        min_n = routing.get('min_n')
        if min_n is not None and dims['min_n_i'] < int(min_n):
            failures.append(f'min_n_i={dims["min_n_i"]} < min_n={min_n}')

        max_n = routing.get('max_n')
        if max_n is not None and dims['max_n_i'] > int(max_n):
            failures.append(f'max_n_i={dims["max_n_i"]} > max_n={max_n}')

        min_bg_df = routing.get('min_bg_df')
        if min_bg_df is not None:
            min_groups = max(dims['d'], dims['q'] * (dims['q'] + 1) // 2) + int(min_bg_df)
            if dims['m'] < min_groups:
                failures.append(f'm={dims["m"]} < min groups required by min_bg_df={min_groups}')

        min_within_df = routing.get('min_within_df')
        if min_within_df is not None:
            min_group_n = dims['q'] + int(min_within_df)
            if dims['min_n_i'] < min_group_n:
                failures.append(f'min_n_i={dims["min_n_i"]} < q + min_within_df={min_group_n}')

        likelihood_family = routing.get('likelihood_family')
        if likelihood_family is not None and dims['likelihood_family'] is not None:
            if dims['likelihood_family'] != int(likelihood_family):
                failures.append(
                    f'likelihood_family={dims["likelihood_family"]} != {likelihood_family}'
                )

        if 'max_d' not in routing and entry.get('model_cfg', {}).get('d_ffx') is None:
            failures.append('missing hard max_d routing metadata')
        if 'max_q' not in routing and entry.get('model_cfg', {}).get('d_rfx') is None:
            failures.append('missing hard max_q routing metadata')

        return failures

    def _datasetDimensions(
        self, batch: Mapping[str, torch.Tensor], i: int
    ) -> dict[str, int | None]:
        mask_m = batch['mask_m'][i].bool()
        ns = batch['ns'][i]
        active_ns = ns[mask_m]
        likelihood_family = None
        if 'likelihood_family' in batch:
            family = batch['likelihood_family'][i]
            likelihood_family = int(family.item() if torch.is_tensor(family) else family)
        return {
            'd': int(batch['mask_d'][i].sum().item()),
            'q': int(batch['mask_q'][i].sum().item()),
            'm': int(batch['m'][i].item()),
            'n': int(batch['n'][i].item()),
            'min_n_i': int(active_ns.min().item()) if active_ns.numel() else 0,
            'max_n_i': int(active_ns.max().item()) if active_ns.numel() else 0,
            'likelihood_family': likelihood_family,
        }

    def _validateBatchFormat(self, batch: Mapping[str, Any]) -> None:
        missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f'batch is missing dataloader keys: {missing}')

        if batch['X'].dim() != 4:
            raise ValueError('X must have shape (B, m, n_i, d)')
        if batch['Z'].dim() != 4:
            raise ValueError('Z must have shape (B, m, n_i, q)')
        if batch['y'].dim() != 3:
            raise ValueError('y must have shape (B, m, n_i)')
        if batch['mask_n'].shape != batch['y'].shape:
            raise ValueError('mask_n must match y shape')

        finite_keys = ('X', 'Z', 'y', 'nu_ffx', 'tau_ffx', 'tau_rfx', 'eta_rfx')
        for key in finite_keys:
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f'batch contains non-finite values in {key}')

        ns_sum = (batch['ns'] * batch['mask_m'].to(batch['ns'].dtype)).sum(dim=-1)
        if not torch.equal(ns_sum, batch['n']):
            raise ValueError('n must equal ns.sum() over active groups')

    def _validateRoutingInputs(self, batch: Mapping[str, Any]) -> None:
        missing = [
            key
            for key in ('X', 'Z', 'ns', 'm', 'n', 'mask_d', 'mask_q', 'mask_m')
            if key not in batch
        ]
        if missing:
            raise KeyError(f'batch is missing routing keys: {missing}')

    def _validateBatchMatchesModel(
        self, batch: Mapping[str, torch.Tensor], model: Approximator
    ) -> None:
        d_file = int(batch['X'].shape[-1])
        q_file = int(batch['Z'].shape[-1])
        if d_file != model.d_ffx or q_file != model.d_rfx:
            raise ValueError(
                'batch padding does not match selected submodel: '
                f'batch has d={d_file}, q={q_file}; '
                f'{model.cfg.d_ffx=}, {model.cfg.d_rfx=}'
            )

    @staticmethod
    def _routeValue(entry: Mapping[str, Any], key: str) -> Any:
        routing = entry.get('routing', {})
        if key in routing:
            return routing[key]
        model_cfg = entry.get('model_cfg', {})
        if key == 'max_d':
            return model_cfg.get('d_ffx')
        if key == 'max_q':
            return model_cfg.get('d_rfx')
        return None

    @classmethod
    def _routingSortKey(cls, entry: Mapping[str, Any]) -> tuple[float, str]:
        # For current checkpoint families, max_q and the n/m bounds are tied to
        # max_d. If that changes, sort by those routing markers here too.
        value = cls._routeValue(entry, 'max_d')
        return (float('inf') if value is None else float(value), str(entry['id']))

    @staticmethod
    def _isCollatedBatch(data: Mapping[str, Any]) -> bool:
        return (
            'X' in data and torch.is_tensor(data['X']) and data['X'].dim() == 4 and 'mask_n' in data
        )


CheckpointRouter = Router


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
