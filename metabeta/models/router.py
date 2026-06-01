"""Checkpoint packaging utilities for routed model inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import operator
import warnings

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from metabeta.datasets.preprocessor import DataPreprocessor, PreprocessReport
from metabeta.models.approximator import Approximator
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.constants import hasSigmaEps
from metabeta.utils.dataloader import Dataloader, collateGrouped, toDevice
from metabeta.utils.evaluation import Proposal
from metabeta.utils.router import (
    JOINT_CHECKPOINT_VERSION,
    ScaleInfo,
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
    diagnostics: dict[str, Any] | None = None
    param_names: dict[str, list[str]] | None = None
    formula: str | None = None
    priors_str: str | None = None
    group_names: list[str] | None = None
    prior_params: dict[str, np.ndarray] | None = None
    scale_info: 'ScaleInfo | None' = None
    batch: 'dict[str, torch.Tensor] | None' = None


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
        diagnostics: bool = False,
        **prepare_kwargs: Any,
    ) -> RouterResult:
        """Sample from the posterior, routing each dataset to the smallest compatible submodel."""

        self._param_names: dict[str, list[str]] | None = None
        self._scale_info: 'ScaleInfo | None' = None
        self._group_labels: list[str] | None = None
        batch = self.prepareData(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not isCollatedBatch(batch):
            raise TypeError('sample requires data prepared to batch stage')
        self._validateBatchFormat(batch)
        routes, validation = self._routeBatch(batch)
        batch = toDevice(batch, self.device)
        proposal = self._runRouted(batch, routes, n_samples=n_samples)
        diags = self._computeDiagnostics(proposal, batch) if diagnostics else None
        self._rescaleProposal(proposal, batch)
        prior_params: dict[str, np.ndarray] = {
            key: batch[key].float().cpu().numpy()
            for key in ('tau_ffx', 'nu_ffx', 'tau_rfx', 'eta_rfx', 'tau_eps')
            if key in batch
        }
        for key in ('family_ffx', 'family_sigma_rfx', 'family_sigma_eps'):
            if key in batch:
                prior_params[key] = batch[key].long().cpu().numpy()
        return RouterResult(
            proposal=proposal,
            routes=routes,
            validation=validation,
            diagnostics=diags,
            param_names=self._param_names,
            formula=prepare_kwargs.get('formula'),
            priors_str=_priorStr(prepare_kwargs.get('priors')),
            group_names=self._group_labels,
            prior_params=prior_params or None,
            scale_info=self._scale_info,
            batch=batch if diagnostics else None,
        )

    @torch.no_grad()
    def forward(self, data: Any, **prepare_kwargs: Any) -> RouterResult:
        """Evaluate the forward log-probability path for batches with parameters.

        Mixed-submodel batches are not yet supported for forward passes.
        """

        batch = self.prepareData(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not isCollatedBatch(batch):
            raise TypeError('forward requires data prepared to batch stage')
        self._validateBatchFormat(batch)
        routes, validation = self._routeBatch(batch)

        unique_ids = list(dict.fromkeys(routes))
        if len(unique_ids) != 1:
            raise NotImplementedError(
                'mixed-submodel forward pass is not yet supported; route one compatible '
                'dataset family at a time'
            )
        submodel_id = unique_ids[0]
        batch = toDevice(batch, self.device)
        model = self.model(submodel_id)
        self._validateBatchMatchesModel(batch, model)
        log_probs = model(batch)
        return RouterResult(
            proposal=None, routes=routes, validation=validation, log_probs=log_probs
        )

    @torch.no_grad()
    def log_prob(self, data: Any, **prepare_kwargs: Any) -> RouterResult:
        """Compute log-probability of parameters in ``data`` under the routed posterior.

        The batch must contain target parameter arrays (``ffx``, ``sigma_rfx``,
        ``rfx``).  These are included automatically when data enters through the
        formula/prior path; callers passing a raw collated batch must supply them.

        Returns a ``RouterResult`` with ``log_probs`` set to a dict containing
        ``'global'`` and ``'local'`` log-probability tensors, shape ``(B,)`` and
        ``(B, m)`` respectively.

        Mixed-submodel batches are not yet supported.
        """
        batch = self.prepareData(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not isCollatedBatch(batch):
            raise TypeError('log_prob requires data prepared to batch stage')
        self._validateBatchFormat(batch)
        self._validateParameterKeys(batch)
        routes, validation = self._routeBatch(batch)

        unique_ids = list(dict.fromkeys(routes))
        if len(unique_ids) != 1:
            raise NotImplementedError(
                'mixed-submodel log_prob is not yet supported; route one compatible '
                'dataset family at a time'
            )
        submodel_id = unique_ids[0]
        batch = toDevice(batch, self.device)
        model = self.model(submodel_id)
        self._validateBatchMatchesModel(batch, model)
        log_probs = model(batch)
        return RouterResult(
            proposal=None, routes=routes, validation=validation, log_probs=log_probs
        )

    def _computeDiagnostics(
        self, proposal: Proposal, batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        from metabeta.evaluation.predictive import (
            getPosteriorPredictive,
            posteriorPredictiveAUC,
            posteriorPredictiveDeviance,
            posteriorPredictiveNLL,
            posteriorPredictiveR2,
            psisLooNLL,
        )

        lf = int(batch['likelihood_family'].flatten()[0].item())
        pp = getPosteriorPredictive(proposal, batch, likelihood_family=lf)
        log_p = pp.log_prob(batch['y'].unsqueeze(-1))  # (B, m, n, S)
        ppc_nll = posteriorPredictiveNLL(pp, batch, mode='mixture', log_p=log_p)
        loo_nll_vals, pareto_k_vals = psisLooNLL(
            pp, batch, w=proposal.weights, reff=proposal.reff, log_p=log_p
        )

        b = int(batch['X'].shape[0])
        agg = (lambda t: t.item()) if b == 1 else (lambda t: t.mean().item())
        loo_nll = agg(loo_nll_vals)
        pareto_k = agg(pareto_k_vals)

        if lf == 0:
            vals = posteriorPredictiveR2(pp, batch, w=proposal.weights)
            fit = vals.item() if b == 1 else vals.mean().item()
            fit_label = 'R²' if b == 1 else 'Mean R²'
        elif lf == 1:
            vals = posteriorPredictiveAUC(pp, batch, w=proposal.weights)
            fit = vals.item() if b == 1 else vals.mean().item()
            fit_label = 'AUC' if b == 1 else 'Mean AUC'
        else:
            vals = posteriorPredictiveDeviance(pp, batch, w=proposal.weights)
            fit = vals.item() if b == 1 else vals.mean().item()
            fit_label = 'Deviance' if b == 1 else 'Mean Deviance'

        def _summarize(samples: torch.Tensor, dim: int) -> dict[str, torch.Tensor]:
            s = samples.float()
            return {
                'mean': s.mean(dim=dim),
                'median': torch.quantile(s, 0.5, dim=dim),
                'lower': torch.quantile(s, 0.025, dim=dim),
                'upper': torch.quantile(s, 0.975, dim=dim),
            }

        param_summary: dict[str, Any] = {
            'ffx': _summarize(proposal.ffx, dim=-2),
            'sigma_rfx': _summarize(proposal.sigma_rfx, dim=-2),
        }
        if proposal.has_sigma_eps:
            param_summary['sigma_eps'] = _summarize(proposal.sigma_eps, dim=-1)

        return {
            'ppc_nll': ppc_nll,
            'param_summary': param_summary,
            'fit': fit,
            'fit_label': fit_label,
            'loo_nll': loo_nll,
            'pareto_k': pareto_k,
            'pp': pp,
        }

    def _validateParameterKeys(self, batch: Mapping[str, Any]) -> None:
        missing = [k for k in ('ffx', 'sigma_rfx', 'rfx') if k not in batch]
        if missing:
            raise KeyError(f'batch is missing parameter keys for log_prob: {missing}')

        B, M, _, d = batch['X'].shape
        q = batch['Z'].shape[-1]
        if tuple(batch['ffx'].shape) != (B, d):
            raise ValueError(f'ffx must have shape ({B}, {d}), got {tuple(batch["ffx"].shape)}')
        if tuple(batch['sigma_rfx'].shape) != (B, q):
            raise ValueError(
                f'sigma_rfx must have shape ({B}, {q}), got {tuple(batch["sigma_rfx"].shape)}'
            )
        if batch['rfx'].ndim != 3 or batch['rfx'].shape[0] != B or batch['rfx'].shape[-1] != q:
            raise ValueError(
                f'rfx must have shape (B={B}, m, q={q}), got {tuple(batch["rfx"].shape)}'
            )

    @torch.no_grad()
    def _runRouted(
        self,
        batch: dict[str, torch.Tensor],
        routes: list[str],
        *,
        n_samples: int,
    ) -> Proposal:
        """Run inference for a single-submodel batch."""
        unique_ids = list(dict.fromkeys(routes))
        if len(unique_ids) != 1:
            raise NotImplementedError(
                'mixed-submodel sample is not yet supported; route one compatible '
                'dataset family at a time'
            )
        submodel_id = unique_ids[0]
        model = self.model(submodel_id)
        self._validateBatchMatchesModel(batch, model)
        proposal = model.estimate(batch, n_samples=n_samples)
        return proposal

    def _rescaleProposal(self, proposal: Proposal, batch: dict[str, torch.Tensor]) -> None:
        if 'sd_y' not in batch or 'likelihood_family' not in batch:
            return
        lf = int(batch['likelihood_family'].flatten()[0].item())
        if hasSigmaEps(lf):
            proposal.rescale(batch['sd_y'])

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
        if group_name and group_name in df.columns:
            self._group_labels = sorted(str(v) for v in df[group_name].dropna().unique())
        else:
            self._group_labels = None
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
        self._warnInconsistentColumns(X_pre, columns, fixed_indices)
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
        self._param_names = {'ffx': list(fixed_names), 'sigma_rfx': list(random_names)}

        # Build scale info for optional back-transformation to original X units
        x_means_meta = np.asarray(
            preprocessed.get('x_means', np.zeros(X_pre.shape[1])), dtype=float
        )
        x_stds_meta = np.asarray(preprocessed.get('x_stds', np.ones(X_pre.shape[1])), dtype=float)
        y_mean_meta = float(np.asarray(preprocessed.get('y_mean', 0.0)).item())
        y_std_meta = float(np.asarray(preprocessed.get('sd_y', 1.0)).item())

        si_means = (
            [0.0] + [float(x_means_meta[idx]) for idx in fixed_indices]
            if spec.intercept
            else [float(x_means_meta[idx]) for idx in fixed_indices]
        )
        si_stds = (
            [1.0] + [float(x_stds_meta[idx]) for idx in fixed_indices]
            if spec.intercept
            else [float(x_stds_meta[idx]) for idx in fixed_indices]
        )
        self._scale_info = ScaleInfo(
            y_mean=y_mean_meta,
            y_std=y_std_meta,
            x_means=np.array(si_means),
            x_stds=np.array(si_stds),
            param_names=list(fixed_names),
            has_intercept=spec.intercept,
        )

        actual_d = int(X.shape[-1])
        actual_q = int(Z.shape[-1])
        if actual_d < 1:
            raise ValueError('fixed-effect design must include at least one column')
        if actual_q < 1:
            raise ValueError('random-effect design must include at least one column')
        if len(random_names) != actual_q:
            raise ValueError('random-effect names must align with Z columns')

        sd_y = preprocessedSdY(preprocessed)
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
            'sd_y': np.array(sd_y),
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
    def _warnInconsistentColumns(
        X_pre: np.ndarray,
        columns: tuple[str, ...],
        fixed_indices: list[int],
    ) -> None:
        if not fixed_indices or X_pre.shape[0] < 2:
            return
        X_active = X_pre[:, fixed_indices]

        stds = X_active.std(axis=0)
        near_constant = [columns[idx] for idx, s in zip(fixed_indices, stds) if s < 0.01]
        if near_constant:
            warnings.warn(
                f'near-constant predictor columns in design (std < 0.01): {near_constant}. '
                'Verify DataPreprocessor was fitted before routing.',
                RuntimeWarning,
                stacklevel=4,
            )

        if X_active.shape[1] >= 2 and len(near_constant) == 0:
            corr = np.corrcoef(X_active.T)
            high_corr = [
                (columns[fixed_indices[i]], columns[fixed_indices[j]], float(corr[i, j]))
                for i in range(corr.shape[0])
                for j in range(i + 1, corr.shape[0])
                if abs(corr[i, j]) > 0.95
            ]
            if high_corr:
                pairs_str = ', '.join(f'{a}/{b} ({c:.2f})' for a, b, c in high_corr)
                warnings.warn(
                    f'highly correlated predictor columns in design (|r| > 0.95): {pairs_str}. '
                    'Consider dropping one from the formula or re-fitting DataPreprocessor.',
                    RuntimeWarning,
                    stacklevel=4,
                )

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

        for dim_key, route_key, cmp, sym in (
            ('d', 'max_d', operator.gt, '>'),
            ('q', 'max_q', operator.gt, '>'),
            ('m', 'max_m', operator.gt, '>'),
            ('n', 'max_n_total', operator.gt, '>'),
            ('d', 'min_d', operator.lt, '<'),
            ('q', 'min_q', operator.lt, '<'),
            ('m', 'min_m', operator.lt, '<'),
        ):
            value = dims[dim_key]
            bound = routing.get(route_key)
            if bound is not None and cmp(value, int(bound)):
                failures.append(f'{dim_key}={value} {sym} {route_key}={bound}')

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

        if batch['mask_d'].shape[-1] != batch['X'].shape[-1]:
            raise ValueError(
                f'mask_d width {batch["mask_d"].shape[-1]} must match X feature dim {batch["X"].shape[-1]}'
            )
        if batch['mask_q'].shape[-1] != batch['Z'].shape[-1]:
            raise ValueError(
                f'mask_q width {batch["mask_q"].shape[-1]} must match Z feature dim {batch["Z"].shape[-1]}'
            )
        active_m = batch['mask_m'].sum(dim=-1)
        if not torch.equal(active_m.to(batch['m'].dtype), batch['m']):
            raise ValueError('mask_m active count must equal m for each batch row')

        finite_keys = ('X', 'Z', 'y', 'nu_ffx', 'tau_ffx', 'tau_rfx', 'eta_rfx')
        for key in finite_keys:
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f'batch contains non-finite values in {key}')

        ns_sum = (batch['ns'] * batch['mask_m'].to(batch['ns'].dtype)).sum(dim=-1)
        if not torch.equal(ns_sum, batch['n']):
            raise ValueError('n must equal ns.sum() over active groups')

        for i in range(batch['ns'].shape[0]):
            mask_m_i = batch['mask_m'][i].bool()
            ns_i = batch['ns'][i]
            if mask_m_i.any() and (ns_i[mask_m_i] <= 0).any():
                raise ValueError(f'active groups must have ns > 0 (dataset {i})')
            if (~mask_m_i).any() and (ns_i[~mask_m_i] != 0).any():
                raise ValueError(f'inactive groups must have ns == 0 (dataset {i})')

        if 'mask_mq' in batch:
            expected_mq = batch['mask_m'].unsqueeze(-1) & batch['mask_q'].unsqueeze(-2)
            if not torch.equal(batch['mask_mq'].bool(), expected_mq):
                raise ValueError('mask_mq must equal mask_m[..., None] & mask_q[:, None, :]')

        if 'stats' in batch and isinstance(batch['stats'], dict):
            stats = batch['stats']
            d_batch = batch['X'].shape[-1]
            q_batch = batch['Z'].shape[-1]
            if 'beta_est' in stats and stats['beta_est'].shape[-1] != d_batch:
                raise ValueError(
                    f'stats.beta_est last dim {stats["beta_est"].shape[-1]} must match '
                    f'X feature dim {d_batch}'
                )
            if 'sigma_rfx_est' in stats and stats['sigma_rfx_est'].shape[-1] != q_batch:
                raise ValueError(
                    f'stats.sigma_rfx_est last dim {stats["sigma_rfx_est"].shape[-1]} must match '
                    f'Z feature dim {q_batch}'
                )
            for key, val in stats.items():
                if torch.is_tensor(val) and not torch.isfinite(val).all():
                    raise ValueError(f'stats.{key} contains non-finite values')

        if 'likelihood_family' in batch:
            lf = int(batch['likelihood_family'].flatten()[0].item())
            if lf == 0 and 'sd_y' in batch:
                mask_n = batch['mask_n']
                if mask_n.any():
                    active_y = batch['y'][mask_n].float()
                    y_mean = active_y.mean().item()
                    y_std = active_y.std().item() if active_y.numel() > 1 else 1.0
                    if abs(y_mean) > 0.5 or abs(y_std - 1.0) > 0.5:
                        warnings.warn(
                            f'continuous target y appears unstandardized '
                            f'(mean={y_mean:.2f}, std={y_std:.2f}); '
                            'verify DataPreprocessor was applied before routing',
                            RuntimeWarning,
                            stacklevel=2,
                        )

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

    def plotPPD(
        self,
        result: RouterResult,
        indices: Sequence[int] | None = None,
        plot_dir: Path | None = None,
    ) -> None:
        """Plot posterior predictive densities for datasets in result.

        Requires ``sample()`` to have been called with ``diagnostics=True``.
        y is shown on the standardized scale used internally by the model.
        """
        from metabeta.plotting.predictive import plotPPD as _plotPPD

        if result.diagnostics is None or result.batch is None:
            raise ValueError('call sample() with diagnostics=True first')
        pp = result.diagnostics.get('pp')
        if pp is None:
            raise ValueError('pp not found in diagnostics')
        b = int(result.batch['y'].shape[0])
        if indices is None:
            indices = list(range(b))
        _plotPPD(pp, result.batch, indices=indices, plot_dir=plot_dir)

    def posteriorSummary(
        self,
        result: RouterResult,
        *,
        ci: float = 0.95,
        batch_index: int = 0,
        x_scale: str = 'standardized',
    ) -> str:
        """Return a formatted posterior summary table for a sampled RouterResult.

        Fit metrics (R²/AUC/Deviance), LOO-NLL, and Pareto k are shown when
        ``sample()`` was called with ``diagnostics=True``.  For per-group random
        effects use ``rfxTable()``.
        """
        if result.proposal is None:
            raise ValueError('RouterResult has no proposal; call sample() first')
        d = result.diagnostics or {}
        dims = (
            result.validation[batch_index]['dimensions']
            if result.validation and batch_index < len(result.validation)
            else {}
        )
        return posteriorTable(
            result.proposal,
            result.param_names,
            formula=result.formula,
            priors_str=result.priors_str,
            prior_params=result.prior_params,
            fit=d.get('fit'),
            fit_label=d.get('fit_label', 'fit'),
            loo_nll=d.get('loo_nll'),
            pareto_k=d.get('pareto_k'),
            ci=ci,
            batch_index=batch_index,
            scale_info=result.scale_info,
            x_scale=x_scale,
            n=dims.get('n'),
            m=dims.get('m'),
        )

    def rfxTable(
        self,
        result: RouterResult,
        *,
        ci: float = 0.95,
        batch_index: int = 0,
    ) -> str:
        """Return a formatted per-group random effects table for a sampled RouterResult."""
        if result.proposal is None:
            raise ValueError('RouterResult has no proposal; call sample() first')
        fmt = 'github'
        alpha = (1 - ci) / 2
        lo_pct = f'{alpha * 100:g}%'
        hi_pct = f'{(1 - alpha) * 100:g}%'

        names = result.param_names or {}
        rfx = result.proposal.rfx[batch_index].float()  # (m, S, q)
        m, _, q_model = rfx.shape
        srfx_names = names.get('sigma_rfx') or [f'rfx_{i}' for i in range(q_model)]
        n_rfx = len(srfx_names)
        g_names = result.group_names or [str(i) for i in range(m)]

        headers = ['Group']
        for name in srfx_names:
            headers += [f'{name} Mean', f'{name} {lo_pct}', f'{name} {hi_pct}']

        rows = []
        for gi in range(m):
            row: list[Any] = [g_names[gi] if gi < len(g_names) else str(gi)]
            for qi in range(n_rfx):
                col = rfx[gi, :, qi]
                row += [
                    col.mean().item(),
                    col.quantile(alpha).item(),
                    col.quantile(1 - alpha).item(),
                ]
            rows.append(row)

        scale_note = f'Scale:    original y units' + (
            f'  (intercept deviations from population intercept β₀)'
            if any('intercept' in n.lower() for n in srfx_names)
            else ''
        )
        parts = [
            'Random Effects:',
            scale_note,
            tabulate(rows, headers=headers, floatfmt='.3f', tablefmt=fmt),
        ]
        return '\n'.join(parts)


def _priorStr(priors: Any) -> str:
    if priors is None:
        return 'default'
    if isinstance(priors, str):
        return priors
    if isinstance(priors, dict):
        keys = list(priors.keys())
        suffix = '…' if len(keys) > 3 else ''
        return f'custom ({", ".join(keys[:3])}{suffix})'
    return 'custom'


def _buildPriorLines(
    ffx_names: list[str],
    srfx_names: list[str],
    prior_params: dict[str, np.ndarray],
    batch_index: int,
    scale_info: 'ScaleInfo | None' = None,
    x_scale: str = 'standardized',
) -> list[str]:
    """Build per-parameter prior description lines for the summary header.

    Priors are stored on the fully-standardized (★) scale inside the model.
    This function rescales them to match whatever scale is being displayed:

    - Default (standardized X, original y): multiply τ by σ_y.
      This gives the effective prior on β★_k × σ_y (Δy per SD of predictor).
      Intercept location remains 0 (model is centred at mean y).

    - Original (x_scale='original'): multiply τ by σ_y/σ_{x_k}.
      This gives the effective prior on β_k (Δy per unit of predictor).
      Intercept location shifts to μ_y (since E[β_0] = μ_y when all priors
      are centred and predictors are mean-zero after standardization).

    σ_rfx are always displayed in original y units (rescaled by σ_y in the
    proposal), so τ_rfx is always multiplied by σ_y regardless of x_scale.
    """
    from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF

    lines: list[str] = []
    n_ffx, n_rfx = len(ffx_names), len(srfx_names)

    # In default mode priors stay in the fully standardized (★) space where the
    # model operates.  In original mode they are rescaled to match the original
    # y/X units shown for the estimates.
    rescale = x_scale == 'original' and scale_info is not None
    y_std = scale_info.y_std if rescale else 1.0
    y_mean = scale_info.y_mean if rescale else 0.0

    if 'tau_ffx' in prior_params and 'nu_ffx' in prior_params:
        tau = prior_params['tau_ffx'][batch_index, :n_ffx]
        mu = prior_params['nu_ffx'][batch_index, :n_ffx]
        fam_id = int(prior_params['family_ffx'][batch_index]) if 'family_ffx' in prior_params else 0
        fam = FFX_FAMILIES[fam_id] if fam_id < len(FFX_FAMILIES) else 'normal'
        for j, name in enumerate(ffx_names):
            tau_j = float(tau[j]) * y_std
            mu_j = float(mu[j]) * y_std
            if rescale and j < len(scale_info.x_stds):
                x_std_j = float(scale_info.x_stds[j])
                if x_std_j > 0:
                    tau_j /= x_std_j
                # Intercept prior location shifts to μ_y in original space when
                # ν★=0 and predictors are mean-zero after standardization.
                if scale_info.has_intercept and j == 0:
                    mu_j += y_mean

            loc = f'{mu_j:.4g}' if abs(mu_j) > 1e-9 else '0'
            if fam == 'normal':
                lines.append(f'  {name} ~ N({loc}, {tau_j:.4g})')
            else:
                lines.append(f'  {name} ~ t₅({loc}, {tau_j:.4g})')

    if 'tau_rfx' in prior_params and n_rfx > 0:
        tau_r = prior_params['tau_rfx'][batch_index, :n_rfx]
        fam_id = (
            int(prior_params['family_sigma_rfx'][batch_index])
            if 'family_sigma_rfx' in prior_params
            else 0
        )
        fam = SIGMA_FAMILIES[fam_id] if fam_id < len(SIGMA_FAMILIES) else 'halfnormal'
        for j, name in enumerate(srfx_names):
            # σ_rfx is always displayed in y units; rescale τ_rfx in original mode
            tau_j = float(tau_r[j]) * y_std
            if fam == 'halfnormal':
                lines.append(f'  σ_{name} ~ HN({tau_j:.4g})')
            elif fam == 'halfstudent':
                lines.append(f'  σ_{name} ~ HT₅({tau_j:.4g})')
            else:
                lines.append(f'  σ_{name} ~ Exp({tau_j:.4g})')
        if 'eta_rfx' in prior_params and n_rfx > 1:
            eta = float(prior_params['eta_rfx'][batch_index])
            if eta > 0:
                lines.append(f'  Corr ~ LKJ({eta:.2g})')

    if 'tau_eps' in prior_params:
        tau_e = float(np.asarray(prior_params['tau_eps']).ravel()[batch_index]) * y_std
        fam_id = (
            int(np.asarray(prior_params['family_sigma_eps']).ravel()[batch_index])
            if 'family_sigma_eps' in prior_params
            else 1
        )
        fam = SIGMA_FAMILIES[fam_id] if fam_id < len(SIGMA_FAMILIES) else 'halfstudent'
        if fam == 'halfnormal':
            lines.append(f'  σ_Residual ~ HN({tau_e:.4g})')
        elif fam == 'halfstudent':
            lines.append(f'  σ_Residual ~ HT₅({tau_e:.4g})')
        else:
            lines.append(f'  σ_Residual ~ Exp({tau_e:.4g})')

    return lines


def posteriorTable(
    proposal: 'Proposal',
    param_names: dict[str, list[str]] | None = None,
    *,
    formula: str | None = None,
    priors_str: str | None = None,
    prior_params: dict[str, np.ndarray] | None = None,
    fit: float | None = None,
    fit_label: str = 'fit',
    loo_nll: float | None = None,
    pareto_k: float | None = None,
    ci: float = 0.95,
    batch_index: int = 0,
    scale_info: 'ScaleInfo | None' = None,
    x_scale: str = 'standardized',
    n: int | None = None,
    m: int | None = None,
) -> str:
    """Format a posterior summary table in the style of lme4 summaries.

    Sections: Fixed Effects (mean, SD, CI, P(>0), contraction), Standard
    Deviations (sigma_rfx + residual), and Correlations (posterior mean, when
    q > 1).  For per-group random effects use ``Router.rfxTable()``.
    Contraction requires ``prior_params`` (populated automatically by
    ``Router.sample()``).  Fit metrics require ``diagnostics=True``.
    """
    fmt = 'pipe'
    alpha = (1 - ci) / 2
    lo_pct = f'{alpha * 100:g}%'
    hi_pct = f'{(1 - alpha) * 100:g}%'

    ffx = proposal.ffx[batch_index].float()  # (S, d)
    if x_scale == 'original':
        if scale_info is None:
            import warnings

            warnings.warn(
                'x_scale="original" requires scale_info; falling back to standardized. '
                'Call router.sample() with fit_preprocessor=True to enable back-transformation.',
                RuntimeWarning,
                stacklevel=2,
            )
            x_scale = 'standardized'
        else:
            ffx = scale_info.to_original_scale(ffx)
    srfx = proposal.sigma_rfx[batch_index].float()  # (S, q)
    S = ffx.shape[0]

    names = param_names or {}
    ffx_names = names.get('ffx') or [f'ffx_{i}' for i in range(ffx.shape[-1])]
    srfx_names = names.get('sigma_rfx') or [f'rfx_{i}' for i in range(srfx.shape[-1])]

    def _col_stats(samples: torch.Tensor, j: int) -> tuple[float, float, float, float]:
        col = samples[:, j]
        return (
            col.mean().item(),
            col.std().item(),
            col.quantile(alpha).item(),
            col.quantile(1 - alpha).item(),
        )

    # prior SD per ffx parameter for contraction = 1 - Var(post) / Var(prior)
    # τ is stored in standardized (★) space; rescale to match the displayed units.
    from metabeta.utils.constants import FFX_FAMILIES, STUDENT_DF

    tau_arr = None
    _ffx_prior_var_multiplier = 1.0  # 1.0 for Normal; df/(df-2) for Student-t
    if prior_params is not None and 'tau_ffx' in prior_params and 'nu_ffx' in prior_params:
        n_ffx_active = len(ffx_names)
        tau_arr = prior_params['tau_ffx'][batch_index, :n_ffx_active].copy().astype(float)
        if scale_info is not None:
            for j in range(n_ffx_active):
                x_std_j = float(scale_info.x_stds[j]) if j < len(scale_info.x_stds) else 1.0
                denom = x_std_j if x_scale == 'original' else 1.0
                tau_arr[j] *= scale_info.y_std / max(denom, 1e-12)
        if 'family_ffx' in prior_params:
            fam_id = int(prior_params['family_ffx'][batch_index])
            fam = FFX_FAMILIES[fam_id] if fam_id < len(FFX_FAMILIES) else 'normal'
            if fam == 'student' and STUDENT_DF > 2:
                _ffx_prior_var_multiplier = STUDENT_DF / (STUDENT_DF - 2)

    def _contraction(post_sd: float, j: int) -> float | None:
        if tau_arr is None:
            return None
        tau_j = float(tau_arr[j])
        if tau_j <= 0:
            return None
        prior_var = tau_j**2 * _ffx_prior_var_multiplier
        return float(np.clip(1.0 - post_sd**2 / prior_var, 0.0, 1.0))

    # ── header ────────────────────────────────────────────────────────────────
    parts: list[str] = []
    if formula:
        parts.append(f'Formula:  {formula}')
    scale_label = (
        'standardized X, original y  (slopes: Δy per SD predictor; σ and rfx: y units)'
        if x_scale == 'standardized'
        else 'original  (slopes: Δy per unit predictor; σ and rfx: y units)'
    )
    parts.append(f'Scale:    {scale_label}')
    prior_lines = (
        _buildPriorLines(
            ffx_names,
            srfx_names,
            prior_params,
            batch_index,
            scale_info=scale_info,
            x_scale=x_scale,
        )
        if prior_params is not None
        else []
    )
    prior_header = 'Priors:' if x_scale == 'original' else 'Priors (★):'
    if prior_lines:
        parts.append(prior_header)
        parts.extend(prior_lines)
    elif priors_str:
        parts.append(f'{prior_header}   {priors_str}')
    if parts:
        parts.append('')

    # ── Fixed Effects ─────────────────────────────────────────────────────────
    show_contr = tau_arr is not None
    ffx_headers = ['', 'Mean', 'SD', lo_pct, hi_pct, 'P(>0)']
    if show_contr:
        ffx_headers.append('Contr.')
    ffx_rows = []
    for j, name in enumerate(ffx_names):
        mean, sd, lo, hi = _col_stats(ffx, j)
        p_pos = (ffx[:, j] > 0).float().mean().item()
        row: list[Any] = [name, mean, sd, lo, hi, p_pos]
        if show_contr:
            row.append(_contraction(sd, j))
        ffx_rows.append(row)

    parts += [
        'Fixed Effects:',
        tabulate(ffx_rows, headers=ffx_headers, floatfmt='.3f', tablefmt=fmt),
        '',
    ]

    # ── Standard Deviations ───────────────────────────────────────────────────
    srfx_rows = []
    for j, name in enumerate(srfx_names):
        mean, sd, lo, hi = _col_stats(srfx, j)
        srfx_rows.append([name, mean, sd, lo, hi])

    if proposal.has_sigma_eps:
        seps = proposal.sigma_eps[batch_index].float()  # (S,)
        srfx_rows.append(
            [
                'Residual',
                seps.mean().item(),
                seps.std().item(),
                seps.quantile(alpha).item(),
                seps.quantile(1 - alpha).item(),
            ]
        )

    parts += [
        'Standard Deviations (y units):',
        tabulate(
            srfx_rows,
            headers=['', 'Mean', 'SD', lo_pct, hi_pct],
            floatfmt='.3f',
            tablefmt=fmt,
        ),
    ]

    # ── Correlations ──────────────────────────────────────────────────────────
    n_rfx = len(srfx_names)
    if n_rfx > 1:
        corr = proposal.corr_rfx
        if corr is not None:
            corr_b = corr[batch_index].float()  # (S, q, q) or (q, q)
            corr_mean = corr_b.mean(0) if corr_b.dim() == 3 else corr_b
            corr_rows = [
                [srfx_names[j]] + [corr_mean[j, k].item() for k in range(n_rfx)]
                for j in range(n_rfx)
            ]
            parts += [
                '',
                'Correlations:',
                tabulate(
                    corr_rows,
                    headers=[''] + srfx_names,
                    floatfmt='.3f',
                    tablefmt=fmt,
                ),
            ]

    # ── footer ────────────────────────────────────────────────────────────────
    footer_parts = []
    if n is not None:
        footer_parts.append(f'n = {n}')
    if m is not None:
        footer_parts.append(f'm = {m}')
    footer_parts.append(f'draws = {S}')
    if fit is not None:
        footer_parts.append(f'{fit_label} = {fit:.3f}')
    if loo_nll is not None:
        footer_parts.append(f'LOO-NLL = {loo_nll:.3f}')
    if pareto_k is not None:
        footer_parts.append(f'Pareto k = {pareto_k:.3f}')
    parts += ['', '   '.join(footer_parts)]

    return '\n'.join(parts)


CheckpointRouter = Router
