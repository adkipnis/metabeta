"""Checkpoint packaging utilities for routed model inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from metabeta.datasets.preprocessor import DataPreprocessor, PreprocessReport
from metabeta.models.approximator import Approximator
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.dataloader import Dataloader, collateGrouped, toDevice
from metabeta.utils.evaluation import Proposal
from metabeta.utils.router import (
    JOINT_CHECKPOINT_VERSION,
    attachPreprocessorMetadata,
    buildRandomDesign,
    defaultRandomTerms,
    isCollatedBatch,
    isModelDataset,
    isPreprocessedDict,
    joinCheckpoints,
    padModelDataset,
    parseFormula,
    preprocessedSdY,
    resolveFixedIndices,
    resolveLikelihoodFamily,
    resolvePriors,
    routeValue,
    routingSortKey,
    returnPreparedStage,
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
        self.submodels.sort(key=routingSortKey)
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
        priors: Any = None,
        preprocessor: DataPreprocessor | str | Path | None = None,
        fit_preprocessor: bool = False,
        group_name: str | None = None,
        likelihood_family: int | str | None = None,
        q: int | None = None,
        stage: str = 'batch',
        dry_run: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, np.ndarray] | list[
        dict[str, np.ndarray]
    ] | PreprocessReport:
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
            return returnPreparedStage(batch, stage)

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == '.npz':
                loader = Dataloader(path, batch_size=self.batch_size)
                batch = loader.fullBatch()
                return returnPreparedStage(batch, stage)
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

        if isinstance(data, Mapping) and isCollatedBatch(data):
            return returnPreparedStage(dict(data), stage)

        if isinstance(data, Mapping) and isModelDataset(data):
            dataset = {str(k): v for k, v in data.items()}
            if stage == 'dataset':
                return dataset
            batch = collateGrouped([dataset])
            return returnPreparedStage(batch, stage)

        if isinstance(data, Mapping) and isPreprocessedDict(data):
            if stage == 'preprocessed':
                return {str(k): v for k, v in data.items()}
            datasets = self._buildModelDatasets(
                data,
                formula=formula,
                priors=priors,
                likelihood_family=likelihood_family,
                q=q,
            )
            if stage == 'dataset':
                return datasets[0] if len(datasets) == 1 else datasets
            batch = self._collateForSelectedSubmodel(datasets)
            return returnPreparedStage(batch, stage)

        if not isinstance(data, Mapping):
            raise TypeError('router input must be tabular, path-backed, preprocessed, or collated')
        raise TypeError('router input mapping does not match a supported data stage')

    def route(self, data: Any, **prepare_kwargs: Any) -> list[str]:
        """Return the selected submodel id for each dataset in ``data``."""

        batch = self.prepareData(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not isCollatedBatch(batch):
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
        if not isinstance(batch, dict) or not isCollatedBatch(batch):
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
        if not isinstance(batch, dict) or not isCollatedBatch(batch):
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
        spec = parseFormula(formula)
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
            return attachPreprocessorMetadata(prep.transform(df), prep)

        prep = DataPreprocessor(group_name=group_name)
        if dry_run:
            return prep.audit(df)
        if not fit_preprocessor:
            raise ValueError(
                'tabular router input requires a fitted preprocessor or fit_preprocessor=True'
            )
        return attachPreprocessorMetadata(prep.fit_transform(df), prep)

    def _buildModelDatasets(
        self,
        preprocessed: Mapping[str, Any],
        *,
        formula: str | None,
        priors: Any,
        likelihood_family: int | str | None,
        q: int | None,
    ) -> list[dict[str, np.ndarray]]:
        required = ('X', 'y', 'groups', 'columns', 'd', 'n', 'ns', 'm', 'y_type')
        missing = [key for key in required if key not in preprocessed]
        if missing:
            raise KeyError(f'preprocessed data is missing keys: {missing}')

        y_type = str(np.asarray(preprocessed['y_type']).item())
        if y_type == 'multiclass':
            raise ValueError('multiclass targets are not supported by the router')
        likelihood = resolveLikelihoodFamily(likelihood_family, y_type)

        X_pre = np.asarray(preprocessed['X'], dtype=float)
        y = np.asarray(preprocessed['y'])
        groups = np.asarray(preprocessed['groups'])
        columns = tuple(str(c) for c in np.asarray(preprocessed['columns']).tolist())
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError('preprocessed groups must be a 1D array aligned with y')
        if X_pre.ndim != 2 or X_pre.shape[0] != len(y):
            raise ValueError('preprocessed X must have shape (n, d) aligned with y')

        spec = parseFormula(formula)
        fixed_indices = resolveFixedIndices(spec.fixed_terms, columns)
        fixed_names = []
        if spec.intercept:
            X_parts = [np.ones((X_pre.shape[0], 1), dtype=float)]
            fixed_names.append('Intercept')
        else:
            X_parts = []
        X_parts += [X_pre[:, idx : idx + 1] for idx in fixed_indices]
        fixed_names += [columns[idx] for idx in fixed_indices]
        X = np.concatenate(X_parts, axis=1) if X_parts else np.empty((X_pre.shape[0], 0))

        random_terms = spec.random_terms
        if not random_terms:
            if q is None:
                random_terms = ('1',)
            else:
                random_terms = defaultRandomTerms(q, columns)
        Z = buildRandomDesign(X_pre, columns, random_terms)
        random_names = self._randomNames(random_terms, columns)

        actual_d = int(X.shape[-1])
        actual_q = int(Z.shape[-1])
        if actual_d < 1:
            raise ValueError('fixed-effect design must include at least one column')
        if actual_q < 1:
            raise ValueError('random-effect design must include at least one column')
        if len(random_names) != actual_q:
            raise ValueError('random-effect names must align with Z columns')

        prior_variants = resolvePriors(
            priors,
            fixed_names=fixed_names,
            random_names=random_names,
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

        base = {
            'X': X.astype(np.float64),
            'Z': Z.astype(np.float64),
            'y': y.astype(np.float64),
            'groups': groups,
            'd': np.array(actual_d),
            'q': np.array(actual_q),
            'n': np.array(n),
            'm': np.array(m),
            'ns': ns,
            'sd_y': np.array(preprocessedSdY(preprocessed)),
            'ffx': np.zeros(actual_d, dtype=np.float64),
            'sigma_rfx': np.ones(actual_q, dtype=np.float64),
            'corr_rfx': np.eye(actual_q, dtype=np.float64),
            'rfx': np.zeros((m, actual_q), dtype=np.float64),
        }
        if hasSigmaEps(likelihood):
            base['sigma_eps'] = np.array(1.0)

        datasets = []
        for prior_index, (prior_name, prior_values) in enumerate(prior_variants):
            dataset = {**base, **prior_values}
            dataset['_router_prior_index'] = np.array(prior_index, dtype=np.int64)
            dataset['_router_source_index'] = np.array(0, dtype=np.int64)
            if prior_name is not None:
                dataset['_router_prior_name'] = np.array(prior_name)
            datasets.append(dataset)
        return datasets

    def _collateForSelectedSubmodel(
        self, datasets: list[dict[str, np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        tentative = collateGrouped(datasets)
        routes, _ = self._routeBatch(tentative)
        if len(set(routes)) != 1:
            raise NotImplementedError(
                'mixed-submodel prior batches are not reassembled yet; use one compatible '
                'dataset family at a time'
            )

        selected = self._submodel_by_id[routes[0]]
        max_d = routeValue(selected, 'max_d')
        max_q = routeValue(selected, 'max_q')
        if max_d is None or max_q is None:
            raise ValueError(f'selected submodel {routes[0]} is missing max_d/max_q')

        padded = [
            padModelDataset(dataset, max_d=int(max_d), max_q=int(max_q)) for dataset in datasets
        ]
        batch = collateGrouped(padded)
        self._attachRouterMetadata(batch, padded)
        return batch

    @staticmethod
    def _randomNames(random_terms: tuple[str, ...], columns: tuple[str, ...]) -> list[str]:
        names = []
        for term in random_terms:
            if term == '1':
                names.append('Intercept')
            else:
                names.extend(columns[idx] for idx in resolveFixedIndices([term], columns))
        return names

    @staticmethod
    def _attachRouterMetadata(
        batch: dict[str, Any],
        datasets: list[dict[str, np.ndarray]],
    ) -> None:
        batch['_router_prior_index'] = torch.as_tensor(
            [int(np.asarray(dataset['_router_prior_index']).item()) for dataset in datasets],
            dtype=torch.int64,
        )
        batch['_router_source_index'] = torch.as_tensor(
            [int(np.asarray(dataset['_router_source_index']).item()) for dataset in datasets],
            dtype=torch.int64,
        )
        batch['_router_prior_name'] = [
            str(np.asarray(dataset['_router_prior_name']).item())
            if '_router_prior_name' in dataset
            else None
            for dataset in datasets
        ]

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
            report = {
                'index': i,
                'submodel_id': submodel_id,
                'dimensions': self._datasetDimensions(batch, i),
                'online_stats': 'stats' not in batch,
                'failures_by_submodel': failures,
            }
            if '_router_prior_index' in batch:
                report['prior_index'] = int(batch['_router_prior_index'][i].item())
            if '_router_source_index' in batch:
                report['source_index'] = int(batch['_router_source_index'][i].item())
            if '_router_prior_name' in batch:
                report['prior_name'] = batch['_router_prior_name'][i]
            validation.append(report)
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


CheckpointRouter = Router
