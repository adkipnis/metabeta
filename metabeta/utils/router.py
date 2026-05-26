"""Shared helpers for routed model inference and checkpoint packaging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from metabeta.simulation.prior import bambiDefaultPriors
from metabeta.utils.constants import FFX_FAMILIES, LIKELIHOOD_FAMILIES, SIGMA_FAMILIES, hasSigmaEps
from metabeta.utils.experiments import CHECKPOINT_DIR


JOINT_CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_PREFIXES = ('best', 'latest')
MAX_ROUTER_Q = 5

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

Y_TYPE_TO_LIKELIHOOD = {
    'continuous': 0,
    'binary': 1,
    'count': 2,
}

LIKELIHOOD_NAME_TO_ID = {name: i for i, name in enumerate(LIKELIHOOD_FAMILIES)}
FFX_FAMILY_NAME_TO_ID = {name: i for i, name in enumerate(FFX_FAMILIES)}
SIGMA_FAMILY_NAME_TO_ID = {name: i for i, name in enumerate(SIGMA_FAMILIES)}

CANONICAL_PRIOR_KEYS = {
    'nu_ffx',
    'tau_ffx',
    'family_ffx',
    'tau_rfx',
    'family_sigma_rfx',
    'tau_eps',
    'family_sigma_eps',
    'eta_rfx',
}
TERM_PRIOR_KEYS = {'fixed', 'random_sd', 'sigma_eps', 'corr_rfx', 'name'}


@dataclass(frozen=True)
class FormulaSpec:
    """Minimal formula representation for router-side design construction."""

    target: str | None
    fixed_terms: tuple[str, ...]
    random_terms: tuple[str, ...]
    group_name: str | None
    intercept: bool = True


def routeValue(entry: Mapping[str, Any], key: str) -> Any:
    routing = entry.get('routing', {})
    if key in routing:
        return routing[key]
    model_cfg = entry.get('model_cfg', {})
    if key == 'max_d':
        return model_cfg.get('d_ffx')
    if key == 'max_q':
        return model_cfg.get('d_rfx')
    return None


def routingSortKey(entry: Mapping[str, Any]) -> tuple[float, str]:
    # For current checkpoint families, max_q and the n/m bounds are tied to
    # max_d. If that changes, sort by those routing markers here too.
    value = routeValue(entry, 'max_d')
    return (float('inf') if value is None else float(value), str(entry['id']))


def parseFormula(formula: str | None) -> FormulaSpec:
    if formula is None:
        return FormulaSpec(
            target=None,
            fixed_terms=(),
            random_terms=(),
            group_name=None,
        )

    if '~' not in formula:
        raise ValueError('formula must contain "~"')
    lhs, rhs = formula.split('~', 1)
    target = lhs.strip().lower()
    if not target:
        raise ValueError('formula target is empty')

    random_matches = re.findall(r'\(([^|]+)\|([^)]+)\)', rhs)
    if len(random_matches) > 1:
        raise NotImplementedError('router formula support currently accepts one random term')

    random_terms: tuple[str, ...] = ()
    group_name = None
    if random_matches:
        random_rhs, group_name = random_matches[0]
        random_terms = tuple(
            term.strip().lower()
            for term in random_rhs.split('+')
            if term.strip() and term.strip() != '0'
        )
        if len(random_terms) > MAX_ROUTER_Q:
            raise ValueError(f'random-effect dimension q must be <= {MAX_ROUTER_Q}')
        group_name = group_name.strip().lower()

    fixed_rhs = re.sub(r'\([^|]+\|[^)]+\)', '', rhs)
    fixed_terms = []
    intercept = True
    for term in fixed_rhs.split('+'):
        term = term.strip().lower()
        if not term:
            continue
        if term in {'1'}:
            intercept = True
            continue
        if term in {'0', '-1'}:
            intercept = False
            continue
        fixed_terms.append(term)

    return FormulaSpec(
        target=target,
        fixed_terms=tuple(fixed_terms),
        random_terms=random_terms,
        group_name=group_name,
        intercept=intercept,
    )


def defaultRandomTerms(q: int, columns: Sequence[str]) -> tuple[str, ...]:
    if q < 1:
        raise ValueError('q must be positive')
    if q > MAX_ROUTER_Q:
        raise ValueError(f'random-effect dimension q must be <= {MAX_ROUTER_Q}')
    if q - 1 > len(columns):
        raise ValueError(f'q={q} requires {q - 1} non-intercept predictor columns')
    return tuple(['1', *[columns[i] for i in range(q - 1)]])


def resolveFixedIndices(terms: Sequence[str], columns: Sequence[str]) -> list[int]:
    if not terms:
        return list(range(len(columns)))

    out: list[int] = []
    seen: set[int] = set()
    for term in terms:
        for idx in resolveColumnTerm(term, columns):
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
    return out


def resolveColumnTerm(term: str, columns: Sequence[str]) -> list[int]:
    term = term.strip().lower()
    lower_columns = [column.lower() for column in columns]
    exact = [i for i, column in enumerate(lower_columns) if column == term]
    if exact:
        return exact

    prefix = f'{term}_'
    prefixed = [i for i, column in enumerate(lower_columns) if column.startswith(prefix)]
    if prefixed:
        return prefixed

    raise KeyError(f'formula term not found in preprocessed columns: {term}')


def buildRandomDesign(
    X_pre: np.ndarray,
    columns: Sequence[str],
    random_terms: Sequence[str],
) -> np.ndarray:
    if len(random_terms) > MAX_ROUTER_Q:
        raise ValueError(f'random-effect dimension q must be <= {MAX_ROUTER_Q}')

    parts = []
    for term in random_terms:
        term = term.strip().lower()
        if term == '1':
            parts.append(np.ones((X_pre.shape[0], 1), dtype=float))
            continue
        for idx in resolveColumnTerm(term, columns):
            parts.append(X_pre[:, idx : idx + 1])
    if not parts:
        return np.empty((X_pre.shape[0], 0), dtype=float)
    return np.concatenate(parts, axis=1)


def resolveLikelihoodFamily(likelihood_family: int | str | None, y_type: str) -> int:
    if likelihood_family is None:
        try:
            return Y_TYPE_TO_LIKELIHOOD[y_type]
        except KeyError as exc:
            raise ValueError(f'unsupported y_type for router inference: {y_type}') from exc

    if isinstance(likelihood_family, str):
        key = likelihood_family.lower()
        try:
            return LIKELIHOOD_NAME_TO_ID[key]
        except KeyError as exc:
            raise ValueError(f'unknown likelihood family: {likelihood_family}') from exc
    return int(likelihood_family)


def coercePriors(
    priors: Mapping[str, Any] | None,
    *,
    d: int,
    q: int,
    likelihood_family: int,
) -> dict[str, np.ndarray]:
    if q > MAX_ROUTER_Q:
        raise ValueError(f'random-effect dimension q must be <= {MAX_ROUTER_Q}')

    values = bambiDefaultPriors(d, q, likelihood_family=likelihood_family)
    if priors is not None:
        values.update({str(key): np.asarray(value) for key, value in priors.items()})

    required = ['nu_ffx', 'tau_ffx', 'tau_rfx', 'eta_rfx', 'family_ffx', 'family_sigma_rfx']
    if hasSigmaEps(likelihood_family):
        required += ['tau_eps', 'family_sigma_eps']
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f'priors are missing required keys: {missing}')

    values['likelihood_family'] = np.array(likelihood_family)
    values['nu_ffx'] = np.asarray(values['nu_ffx'], dtype=float)
    values['tau_ffx'] = np.asarray(values['tau_ffx'], dtype=float)
    values['tau_rfx'] = np.asarray(values['tau_rfx'], dtype=float)
    values['eta_rfx'] = np.asarray(values['eta_rfx'], dtype=float)
    values['family_ffx'] = np.asarray(values['family_ffx'], dtype=np.int64)
    values['family_sigma_rfx'] = np.asarray(values['family_sigma_rfx'], dtype=np.int64)
    if hasSigmaEps(likelihood_family):
        values['tau_eps'] = np.asarray(values['tau_eps'], dtype=float)
        values['family_sigma_eps'] = np.asarray(values['family_sigma_eps'], dtype=np.int64)

    if values['nu_ffx'].shape != (d,):
        raise ValueError(f'nu_ffx must have shape ({d},), got {values["nu_ffx"].shape}')
    if values['tau_ffx'].shape != (d,):
        raise ValueError(f'tau_ffx must have shape ({d},), got {values["tau_ffx"].shape}')
    if values['tau_rfx'].shape != (q,):
        raise ValueError(f'tau_rfx must have shape ({q},), got {values["tau_rfx"].shape}')
    return values


def resolvePriors(
    priors: Any,
    *,
    fixed_names: Sequence[str],
    random_names: Sequence[str],
    likelihood_family: int,
) -> list[tuple[str | None, dict[str, np.ndarray]]]:
    """Resolve one or more prior specifications to canonical model prior arrays."""

    d = len(fixed_names)
    q = len(random_names)
    variants = _priorVariants(priors)
    return [
        (
            name,
            _resolveSinglePrior(
                prior,
                fixed_names=fixed_names,
                random_names=random_names,
                d=d,
                q=q,
                likelihood_family=likelihood_family,
            ),
        )
        for name, prior in variants
    ]


def _priorVariants(priors: Any) -> list[tuple[str | None, Mapping[str, Any] | None]]:
    if priors is None:
        return [(None, None)]

    if isinstance(priors, Sequence) and not isinstance(priors, (str, bytes, bytearray, Mapping)):
        if len(priors) == 0:
            raise ValueError('priors sequence cannot be empty')
        return [(_priorName(prior, i), _priorMapping(prior)) for i, prior in enumerate(priors)]

    if isinstance(priors, Mapping):
        if _isNamedPriorCollection(priors):
            return [(str(name), _priorMapping(prior)) for name, prior in priors.items()]
        return [(_priorName(priors, 0), priors)]

    raise TypeError('priors must be None, a mapping, a sequence of mappings, or a named mapping')


def _isNamedPriorCollection(priors: Mapping[str, Any]) -> bool:
    keys = {str(key) for key in priors}
    if keys & (CANONICAL_PRIOR_KEYS | TERM_PRIOR_KEYS):
        return False
    return all(isinstance(value, Mapping) for value in priors.values())


def _priorMapping(prior: Any) -> Mapping[str, Any]:
    if not isinstance(prior, Mapping):
        raise TypeError('each prior specification must be a mapping')
    return prior


def _priorName(prior: Any, i: int) -> str | None:
    if isinstance(prior, Mapping) and 'name' in prior:
        return str(prior['name'])
    return None


def _resolveSinglePrior(
    prior: Mapping[str, Any] | None,
    *,
    fixed_names: Sequence[str],
    random_names: Sequence[str],
    d: int,
    q: int,
    likelihood_family: int,
) -> dict[str, np.ndarray]:
    if prior is None:
        return coercePriors(None, d=d, q=q, likelihood_family=likelihood_family)

    canonical = {key: value for key, value in prior.items() if key in CANONICAL_PRIOR_KEYS}
    values = coercePriors(canonical, d=d, q=q, likelihood_family=likelihood_family)

    if 'fixed' in prior:
        _applyFixedTermPriors(values, prior['fixed'], fixed_names)
    if 'random_sd' in prior:
        _applyRandomSdTermPriors(values, prior['random_sd'], random_names)
    if 'sigma_eps' in prior:
        if not hasSigmaEps(likelihood_family):
            raise ValueError('sigma_eps prior is only valid for likelihood_family=normal')
        _applySigmaEpsPrior(values, prior['sigma_eps'])
    if 'corr_rfx' in prior:
        _applyCorrelationPrior(values, prior['corr_rfx'])

    return coercePriors(values, d=d, q=q, likelihood_family=likelihood_family)


def _applyFixedTermPriors(
    values: dict[str, np.ndarray],
    spec: Any,
    fixed_names: Sequence[str],
) -> None:
    if not isinstance(spec, Mapping):
        raise TypeError('fixed priors must be a mapping from term name to prior spec')

    family_id: int | None = None
    for term, term_spec in spec.items():
        idxs = _termIndices(str(term), fixed_names)
        term_spec = _requireMapping(term_spec, f'fixed prior for {term}')
        if 'family' in term_spec:
            term_family = _ffxFamilyId(term_spec['family'])
            family_id = term_family if family_id is None else family_id
            if term_family != family_id:
                raise ValueError('per-term fixed-effect prior families are not supported')
        if 'mu' in term_spec:
            values['nu_ffx'][idxs] = float(term_spec['mu'])
        if 'nu' in term_spec:
            values['nu_ffx'][idxs] = float(term_spec['nu'])
        if 'sigma' in term_spec:
            values['tau_ffx'][idxs] = float(term_spec['sigma'])
        if 'tau' in term_spec:
            values['tau_ffx'][idxs] = float(term_spec['tau'])

    if family_id is not None:
        values['family_ffx'] = np.array(family_id)


def _applyRandomSdTermPriors(
    values: dict[str, np.ndarray],
    spec: Any,
    random_names: Sequence[str],
) -> None:
    if not isinstance(spec, Mapping):
        raise TypeError('random_sd priors must be a mapping from term name to prior spec')

    family_id: int | None = None
    for term, term_spec in spec.items():
        idxs = _termIndices(str(term), random_names)
        term_spec = _requireMapping(term_spec, f'random_sd prior for {term}')
        if 'family' in term_spec:
            term_family = _sigmaFamilyId(term_spec['family'])
            family_id = term_family if family_id is None else family_id
            if term_family != family_id:
                raise ValueError('per-term random-effect SD prior families are not supported')
        if 'sigma' in term_spec:
            values['tau_rfx'][idxs] = float(term_spec['sigma'])
        if 'tau' in term_spec:
            values['tau_rfx'][idxs] = float(term_spec['tau'])

    if family_id is not None:
        values['family_sigma_rfx'] = np.array(family_id)


def _applySigmaEpsPrior(values: dict[str, np.ndarray], spec: Any) -> None:
    spec = _requireMapping(spec, 'sigma_eps prior')
    if 'family' in spec:
        values['family_sigma_eps'] = np.array(_sigmaFamilyId(spec['family']))
    if 'sigma' in spec:
        values['tau_eps'] = np.asarray(float(spec['sigma']))
    if 'tau' in spec:
        values['tau_eps'] = np.asarray(float(spec['tau']))


def _applyCorrelationPrior(values: dict[str, np.ndarray], spec: Any) -> None:
    spec = _requireMapping(spec, 'corr_rfx prior')
    if 'eta' in spec:
        values['eta_rfx'] = np.asarray(float(spec['eta']))


def _requireMapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping')
    return value


def _termIndices(term: str, names: Sequence[str]) -> list[int]:
    term_lower = term.lower()
    names_lower = [name.lower() for name in names]
    exact = [i for i, name in enumerate(names_lower) if name == term_lower]
    if exact:
        return exact

    prefix = f'{term_lower}_'
    prefixed = [i for i, name in enumerate(names_lower) if name.startswith(prefix)]
    if prefixed:
        return prefixed

    raise KeyError(f'prior term not found in model design: {term}')


def _ffxFamilyId(value: Any) -> int:
    if isinstance(value, str):
        try:
            return FFX_FAMILY_NAME_TO_ID[value.lower()]
        except KeyError as exc:
            raise ValueError(f'unknown fixed-effect prior family: {value}') from exc
    return int(value)


def _sigmaFamilyId(value: Any) -> int:
    if isinstance(value, str):
        try:
            return SIGMA_FAMILY_NAME_TO_ID[value.lower()]
        except KeyError as exc:
            raise ValueError(f'unknown sigma prior family: {value}') from exc
    return int(value)


def preprocessedSdY(preprocessed: Mapping[str, Any]) -> float:
    if 'sd_y' in preprocessed:
        return float(np.asarray(preprocessed['sd_y']).item())
    return 1.0


def padVector(values: np.ndarray, size: int, fill: float = 0.0) -> np.ndarray:
    values = np.asarray(values)
    if values.shape == ():
        return values
    if values.shape[0] > size:
        raise ValueError(f'cannot pad vector with leading shape {values.shape[0]} to {size}')
    out = np.full((size, *values.shape[1:]), fill, dtype=values.dtype)
    out[: values.shape[0]] = values
    return out


def padModelDataset(
    dataset: Mapping[str, np.ndarray],
    *,
    max_d: int,
    max_q: int,
) -> dict[str, np.ndarray]:
    d = int(np.asarray(dataset['d']).item())
    q = int(np.asarray(dataset['q']).item())
    if d > max_d:
        raise ValueError(f'd={d} exceeds selected model max_d={max_d}')
    if q > max_q:
        raise ValueError(f'q={q} exceeds selected model max_q={max_q}')

    out = {key: np.array(value, copy=True) for key, value in dataset.items()}
    n = out['X'].shape[0]
    if out['X'].shape[-1] < max_d:
        X = np.zeros((n, max_d), dtype=out['X'].dtype)
        X[:, :d] = out['X'][:, :d]
        out['X'] = X
        for key in ('ffx', 'nu_ffx', 'tau_ffx'):
            out[key] = padVector(out[key], max_d)

    if out['Z'].shape[-1] < max_q:
        Z = np.zeros((n, max_q), dtype=out['Z'].dtype)
        Z[:, :q] = out['Z'][:, :q]
        out['Z'] = Z
        for key in ('sigma_rfx', 'tau_rfx'):
            out[key] = padVector(out[key], max_q)

        m = int(np.asarray(out['m']).item())
        rfx = np.zeros((m, max_q), dtype=out['rfx'].dtype)
        rfx[:, :q] = out['rfx'][:, :q]
        out['rfx'] = rfx

        corr = np.eye(max_q, dtype=out['corr_rfx'].dtype)
        corr[:q, :q] = out['corr_rfx'][:q, :q]
        out['corr_rfx'] = corr

    return out


def isPreprocessedDict(data: Mapping[str, Any]) -> bool:
    return (
        'X' in data
        and isinstance(data['X'], np.ndarray)
        and data['X'].ndim == 2
        and 'groups' in data
        and 'columns' in data
        and 'mask_n' not in data
        and 'Z' not in data
    )


def isModelDataset(data: Mapping[str, Any]) -> bool:
    return (
        'X' in data
        and 'Z' in data
        and isinstance(data['X'], np.ndarray)
        and isinstance(data['Z'], np.ndarray)
        and data['X'].ndim == 2
        and data['Z'].ndim == 2
        and 'nu_ffx' in data
        and 'tau_rfx' in data
    )


def isCollatedBatch(data: Mapping[str, Any]) -> bool:
    return 'X' in data and torch.is_tensor(data['X']) and data['X'].dim() == 4 and 'mask_n' in data


def returnPreparedStage(
    batch: dict[str, torch.Tensor],
    stage: str,
) -> dict[str, torch.Tensor]:
    if stage != 'batch':
        raise ValueError(f'cannot return stage={stage!r} from already-collated input')
    return batch


def attachPreprocessorMetadata(
    data: dict[str, np.ndarray],
    preprocessor: Any,
) -> dict[str, np.ndarray]:
    if 'sd_y' not in data and hasattr(preprocessor, '_y_std'):
        data = dict(data)
        data['sd_y'] = np.array(float(preprocessor._y_std))
    return data


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
