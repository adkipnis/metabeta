"""Checkpoint packaging utilities for routed model inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from metabeta.datasets.preprocessor import DataPreprocessor, PreprocessReport
from metabeta.models.approximator import Approximator
from metabeta.simulation.prior import bambiDefaultPriors
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.constants import LIKELIHOOD_FAMILIES, hasSigmaEps
from metabeta.utils.dataloader import Dataloader, collateGrouped, toDevice
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


Y_TYPE_TO_LIKELIHOOD = {
    'continuous': 0,
    'binary': 1,
    'count': 2,
}

LIKELIHOOD_NAME_TO_ID = {name: i for i, name in enumerate(LIKELIHOOD_FAMILIES)}


@dataclass
class RouterResult:
    """Inference output and routing metadata."""

    proposal: Proposal | None
    routes: list[str]
    validation: list[dict[str, Any]]
    log_probs: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class FormulaSpec:
    """Minimal formula representation for router-side design construction."""

    target: str | None
    fixed_terms: tuple[str, ...]
    random_terms: tuple[str, ...]
    group_name: str | None
    intercept: bool = True


class Router:
    """Route dataloader-formatted datasets through a joint checkpoint.

    Models are instantiated lazily when their first compatible batch is run.

    Tabular inputs are normalized through explicit stages: dataframe/parquet,
    preprocessed numpy dict, model dataset dict, then collated dataloader batch.
    Formula support is intentionally narrow and currently covers additive fixed
    terms plus one lme4-style random-effect term.
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

    def prepareData(
        self,
        data: Any,
        *,
        formula: str | None = None,
        priors: Mapping[str, Any] | None = None,
        preprocessor: DataPreprocessor | str | Path | None = None,
        fit_preprocessor: bool = False,
        group_name: str | None = None,
        likelihood_family: int | str | None = None,
        q: int | None = None,
        stage: str = 'batch',
        dry_run: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, np.ndarray] | PreprocessReport:
        """Normalize supported input stages to a collated dataloader-style batch.

        Stages are intentionally layered:

        - DataFrames/parquet paths are tabular inputs.
        - ``DataPreprocessor`` turns tabular inputs into numpy dictionaries.
        - Formula/prior handling turns numpy dictionaries into model datasets.
        - ``collateGrouped`` turns model datasets into the tensor batch consumed
          by the router and approximator.
        """

        if stage not in {'preprocessed', 'dataset', 'batch'}:
            raise ValueError(f'unknown prepareData stage: {stage}')

        if isinstance(data, Dataloader):
            batch = data.fullBatch()
            return self._returnPreparedStage(batch, stage)

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == '.npz':
                loader = Dataloader(path, batch_size=self.batch_size)
                batch = loader.fullBatch()
                return self._returnPreparedStage(batch, stage)
            if path.suffix == '.parquet':
                data = pd.read_parquet(path)
            else:
                raise ValueError(f'unsupported router input path suffix: {path.suffix}')

        if isinstance(data, pd.DataFrame):
            preprocessed = self._preprocessTabular(
                data,
                formula=formula,
                preprocessor=preprocessor,
                fit_preprocessor=fit_preprocessor,
                group_name=group_name,
                dry_run=dry_run,
            )
            if dry_run or stage == 'preprocessed':
                return preprocessed
            data = preprocessed

        if isinstance(data, Mapping) and self._isCollatedBatch(data):
            return self._returnPreparedStage(dict(data), stage)

        if isinstance(data, Mapping) and self._isModelDataset(data):
            dataset = {str(k): v for k, v in data.items()}
            if stage == 'dataset':
                return dataset
            batch = collateGrouped([dataset])
            return self._returnPreparedStage(batch, stage)

        if isinstance(data, Mapping) and self._isPreprocessedDict(data):
            if stage == 'preprocessed':
                return {str(k): v for k, v in data.items()}
            dataset = self._buildModelDataset(
                data,
                formula=formula,
                priors=priors,
                likelihood_family=likelihood_family,
                q=q,
            )
            if stage == 'dataset':
                return dataset
            batch = self._collateForSelectedSubmodel(dataset)
            return self._returnPreparedStage(batch, stage)

        if not isinstance(data, Mapping):
            raise TypeError('router input must be tabular, path-backed, preprocessed, or collated')
        raise TypeError('router input mapping does not match a supported data stage')

    def route(self, data: Any, **prepare_kwargs: Any) -> list[str]:
        """Return the selected submodel id for each dataset in ``data``."""

        batch = self.prepareData(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not self._isCollatedBatch(batch):
            raise TypeError('route requires data prepared to batch stage')
        routes, _ = self._routeBatch(batch)
        return routes

    @torch.no_grad()
    def sample(
        self,
        data: Any,
        *,
        n_samples: int = 1,
        **prepare_kwargs: Any,
    ) -> RouterResult:
        """Sample from the posterior through the routed submodel."""

        batch = self.prepareData(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not self._isCollatedBatch(batch):
            raise TypeError('sample requires data prepared to batch stage')
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
    def forward(self, data: Any, **prepare_kwargs: Any) -> RouterResult:
        """Evaluate the forward log-probability path for batches with parameters."""

        batch = self.prepareData(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not self._isCollatedBatch(batch):
            raise TypeError('forward requires data prepared to batch stage')
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

    def _preprocessTabular(
        self,
        df: pd.DataFrame,
        *,
        formula: str | None,
        preprocessor: DataPreprocessor | str | Path | None,
        fit_preprocessor: bool,
        group_name: str | None,
        dry_run: bool,
    ) -> dict[str, np.ndarray] | PreprocessReport:
        spec = self._parseFormula(formula)
        df = df.copy()
        df.columns = df.columns.str.lower()

        target = spec.target.lower() if spec.target is not None else 'y'
        if target != 'y':
            if target not in df.columns:
                raise KeyError(f'formula target column not found: {target}')
            df = df.rename(columns={target: 'y'})

        group_name = (group_name or spec.group_name or '').lower()
        if preprocessor is not None:
            prep = (
                DataPreprocessor.load(preprocessor)
                if isinstance(preprocessor, (str, Path))
                else preprocessor
            )
            if dry_run:
                return prep.audit(df)
            return self._attachPreprocessorMetadata(prep.transform(df), prep)

        prep = DataPreprocessor(group_name=group_name)
        if dry_run:
            return prep.audit(df)
        if not fit_preprocessor:
            raise ValueError(
                'tabular router input requires a fitted preprocessor or fit_preprocessor=True'
            )
        return self._attachPreprocessorMetadata(prep.fit_transform(df), prep)

    @staticmethod
    def _attachPreprocessorMetadata(
        data: dict[str, np.ndarray],
        preprocessor: DataPreprocessor,
    ) -> dict[str, np.ndarray]:
        if 'sd_y' not in data and hasattr(preprocessor, '_y_std'):
            data = dict(data)
            data['sd_y'] = np.array(float(preprocessor._y_std))
        return data

    def _buildModelDataset(
        self,
        preprocessed: Mapping[str, Any],
        *,
        formula: str | None,
        priors: Mapping[str, Any] | None,
        likelihood_family: int | str | None,
        q: int | None,
    ) -> dict[str, np.ndarray]:
        required = ('X', 'y', 'groups', 'columns', 'd', 'n', 'ns', 'm', 'y_type')
        missing = [key for key in required if key not in preprocessed]
        if missing:
            raise KeyError(f'preprocessed data is missing keys: {missing}')

        y_type = str(np.asarray(preprocessed['y_type']).item())
        if y_type == 'multiclass':
            raise ValueError('multiclass targets are not supported by the router')
        likelihood = self._resolveLikelihoodFamily(likelihood_family, y_type)

        X_pre = np.asarray(preprocessed['X'], dtype=float)
        y = np.asarray(preprocessed['y'])
        groups = np.asarray(preprocessed['groups'])
        columns = tuple(str(c) for c in np.asarray(preprocessed['columns']).tolist())
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError('preprocessed groups must be a 1D array aligned with y')
        if X_pre.ndim != 2 or X_pre.shape[0] != len(y):
            raise ValueError('preprocessed X must have shape (n, d) aligned with y')

        spec = self._parseFormula(formula)
        fixed_indices = self._resolveFixedIndices(spec.fixed_terms, columns)
        if spec.intercept:
            X_parts = [np.ones((X_pre.shape[0], 1), dtype=float)]
        else:
            X_parts = []
        X_parts += [X_pre[:, idx : idx + 1] for idx in fixed_indices]
        X = np.concatenate(X_parts, axis=1) if X_parts else np.empty((X_pre.shape[0], 0))

        random_terms = spec.random_terms
        if not random_terms:
            if q is None:
                random_terms = ('1',)
            else:
                if q < 1:
                    raise ValueError('q must be positive')
                random_terms = tuple(['1', *[columns[i] for i in range(min(q - 1, len(columns)))]])
                if len(random_terms) != q:
                    raise ValueError(f'q={q} requires {q - 1} non-intercept predictor columns')
        Z = self._buildRandomDesign(X_pre, columns, random_terms)

        actual_d = int(X.shape[-1])
        actual_q = int(Z.shape[-1])
        if actual_d < 1:
            raise ValueError('fixed-effect design must include at least one column')
        if actual_q < 1:
            raise ValueError('random-effect design must include at least one column')

        prior_values = self._coercePriors(
            priors,
            d=actual_d,
            q=actual_q,
            likelihood_family=likelihood,
        )

        groups = groups.astype(np.int64, copy=False)
        if np.any(groups < 0):
            raise ValueError('group indices must be non-negative')
        unique_groups = np.unique(groups)
        if not np.array_equal(unique_groups, np.arange(len(unique_groups))):
            raise ValueError('group indices must be contiguous integers starting at 0')
        ns = np.asarray(preprocessed['ns'], dtype=np.int64)
        m = int(np.asarray(preprocessed['m']).item())
        n = int(np.asarray(preprocessed['n']).item())

        dataset = {
            'X': X.astype(np.float64),
            'Z': Z.astype(np.float64),
            'y': y.astype(np.float64),
            'groups': groups,
            'd': np.array(actual_d),
            'q': np.array(actual_q),
            'n': np.array(n),
            'm': np.array(m),
            'ns': ns,
            'sd_y': np.array(self._preprocessedSdY(preprocessed)),
            'ffx': np.zeros(actual_d, dtype=np.float64),
            'sigma_rfx': np.ones(actual_q, dtype=np.float64),
            'corr_rfx': np.eye(actual_q, dtype=np.float64),
            'rfx': np.zeros((m, actual_q), dtype=np.float64),
            **prior_values,
        }
        if hasSigmaEps(likelihood):
            dataset['sigma_eps'] = np.array(1.0)
        return dataset

    def _collateForSelectedSubmodel(
        self, dataset: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        tentative = collateGrouped([dataset])
        routes, _ = self._routeBatch(tentative)
        selected = self._submodel_by_id[routes[0]]
        max_d = self._routeValue(selected, 'max_d')
        max_q = self._routeValue(selected, 'max_q')
        if max_d is None or max_q is None:
            raise ValueError(f'selected submodel {routes[0]} is missing max_d/max_q')

        padded = self._padModelDataset(dataset, max_d=int(max_d), max_q=int(max_q))
        return collateGrouped([padded])

    @staticmethod
    def _returnPreparedStage(
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> dict[str, torch.Tensor]:
        if stage != 'batch':
            raise ValueError(f'cannot return stage={stage!r} from already-collated input')
        return batch

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
    def _parseFormula(formula: str | None) -> FormulaSpec:
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

    @staticmethod
    def _resolveFixedIndices(terms: Sequence[str], columns: Sequence[str]) -> list[int]:
        if not terms:
            return list(range(len(columns)))

        out: list[int] = []
        seen: set[int] = set()
        for term in terms:
            for idx in Router._resolveColumnTerm(term, columns):
                if idx not in seen:
                    out.append(idx)
                    seen.add(idx)
        return out

    @staticmethod
    def _resolveColumnTerm(term: str, columns: Sequence[str]) -> list[int]:
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

    @staticmethod
    def _buildRandomDesign(
        X_pre: np.ndarray,
        columns: Sequence[str],
        random_terms: Sequence[str],
    ) -> np.ndarray:
        parts = []
        for term in random_terms:
            term = term.strip().lower()
            if term == '1':
                parts.append(np.ones((X_pre.shape[0], 1), dtype=float))
                continue
            for idx in Router._resolveColumnTerm(term, columns):
                parts.append(X_pre[:, idx : idx + 1])
        if not parts:
            return np.empty((X_pre.shape[0], 0), dtype=float)
        return np.concatenate(parts, axis=1)

    @staticmethod
    def _resolveLikelihoodFamily(likelihood_family: int | str | None, y_type: str) -> int:
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

    @staticmethod
    def _coercePriors(
        priors: Mapping[str, Any] | None,
        *,
        d: int,
        q: int,
        likelihood_family: int,
    ) -> dict[str, np.ndarray]:
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

    @staticmethod
    def _preprocessedSdY(preprocessed: Mapping[str, Any]) -> float:
        if 'sd_y' in preprocessed:
            return float(np.asarray(preprocessed['sd_y']).item())
        return 1.0

    @staticmethod
    def _padVector(values: np.ndarray, size: int, fill: float = 0.0) -> np.ndarray:
        values = np.asarray(values)
        if values.shape == ():
            return values
        if values.shape[0] > size:
            raise ValueError(f'cannot pad vector with leading shape {values.shape[0]} to {size}')
        out = np.full((size, *values.shape[1:]), fill, dtype=values.dtype)
        out[: values.shape[0]] = values
        return out

    @staticmethod
    def _padModelDataset(
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
                out[key] = Router._padVector(out[key], max_d)

        if out['Z'].shape[-1] < max_q:
            Z = np.zeros((n, max_q), dtype=out['Z'].dtype)
            Z[:, :q] = out['Z'][:, :q]
            out['Z'] = Z
            for key in ('sigma_rfx', 'tau_rfx'):
                out[key] = Router._padVector(out[key], max_q)

            m = int(np.asarray(out['m']).item())
            rfx = np.zeros((m, max_q), dtype=out['rfx'].dtype)
            rfx[:, :q] = out['rfx'][:, :q]
            out['rfx'] = rfx

            corr = np.eye(max_q, dtype=out['corr_rfx'].dtype)
            corr[:q, :q] = out['corr_rfx'][:q, :q]
            out['corr_rfx'] = corr

        return out

    @staticmethod
    def _isPreprocessedDict(data: Mapping[str, Any]) -> bool:
        return (
            'X' in data
            and isinstance(data['X'], np.ndarray)
            and data['X'].ndim == 2
            and 'groups' in data
            and 'columns' in data
            and 'mask_n' not in data
            and 'Z' not in data
        )

    @staticmethod
    def _isModelDataset(data: Mapping[str, Any]) -> bool:
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
