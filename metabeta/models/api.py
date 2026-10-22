"""Checkpoint packaging utilities for routed model inference."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, overload
import contextlib
import math
import operator
import os
import warnings

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm

import metabeta
from metabeta.datasets.preprocessor import DataPreprocessor, PreprocessReport
from metabeta.evaluation import predictive as _predictive
from metabeta.models.approximator import Approximator
from metabeta.posthoc.evidence import tEvidence
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler
from metabeta.posthoc.metropolis import (
    IS2_N_INNER,
    LocalMoves,
    MetropolisSampler,
    suggestPoolSize,
)
from metabeta.utils.config import ApproximatorConfig
from metabeta.utils.constants import LIKELIHOOD_FAMILIES, hasSigmaEps
from metabeta.utils.dataloader import (
    Dataloader,
    collateGrouped,
    sliceBatch,
    toDevice,
    trimBatchPadding,
)
from metabeta.utils.psis import psislw
from metabeta.utils.results import Proposal, concatProposalsBatch
from metabeta.utils.api import (
    JOINT_CHECKPOINT_VERSION,
    ScaleInfo,
    attachPreprocessorMetadata,
    buildRandomDesign,
    defaultRandomTerms,
    isCollatedBatch,
    isModelDataset,
    isPreprocessedDict,
    padModelDataset,
    parseFormula,
    preprocessedSdY,
    resolveFixedIndices,
    resolveLikelihoodFamily,
    resolvePriors,
    restoreColumnCase,
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

# Safeguard defaults. PSIS_K_THRESHOLD follows Yao et al. (2018): k̂ < 0.7 certifies the
# flow posterior as a usable importance proposal; above it the correction is unreliable
# and exact MCMC is recommended. MAP_Z_THRESHOLD flags posteriors whose mean drifts too
# far (in posterior SDs) from the flow-independent analytical MAP anchor; calibrate with
# experiments/evaluation/map_z_calibration.py on cached validation samples.
PSIS_K_THRESHOLD = 0.7
MAP_Z_THRESHOLD = 5.0
MIN_SAFEGUARD_SAMPLES = 64
# IMH refinement: chain layout and the acceptance level below which the flow-posterior
# gap is large
IMH_N_CHAINS = 4
IMH_BURNIN = 25
IMH_ACCEPT_WARN = 0.1
IMH_ESS_WARN = 150
# below it the random-walk moves narrow the gap but may not close it: exact MCMC is recommended
IMH_ESS_NUTS = 10
# sample(random_walk=True): random-walk moves after the IMH on the datasets IMH_ESS_WARN flags.
# A CPU step costs in proportion to chains × IS² draws per group, a GPU step barely, so the GPU
# runs more of both: K=4 and 500 chains (half the sequential steps of 250 for the same draws).
RANDOM_WALK_DEFAULT = {
    'cpu': LocalMoves(n_inner=2, gate_ess=IMH_ESS_WARN),
    'cuda': LocalMoves(n_chains=500, n_inner=4, gate_ess=IMH_ESS_WARN),
}
# warnings point at the user's call, the first frame outside metabeta and torch (torch's no_grad
# wrapper sits between Api.sample and its caller)
INTERNAL_PREFIXES = tuple(str(Path(m.__file__).parent) + os.sep for m in (metabeta, torch))

# rows (dataset × prior) per chunk of ``sample`` when ``batch_size`` is None; the CUDA value is
# measured in #42
CHUNK_ROWS = {'cpu': 64, 'cuda': 256}
# warnings list at most this many datasets or rows, then the count
LISTED_ROWS = 10

K = TypeVar('K', bound=Hashable)


def mapZScores(
    proposal: 'Proposal',
    stats: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> np.ndarray:
    """Per-dataset max |posterior mean − MAP| / posterior SD over valid parameter dims.

    The MAP estimates come from the (flow-independent) analytical GLMM fit that already
    conditions the summarizer, so a large max-z means the flow posterior drifted away
    from a likelihood-based anchor on this dataset — a symptom of data outside the
    training distribution.  Proposal and stats must live in the same (standardized)
    outcome scale.
    """
    eps = 1e-8

    def _z(samples: torch.Tensor, est: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = samples.float().mean(dim=-2)
        sd = samples.float().std(dim=-2).clamp_min(eps)
        return ((mean - est.float()).abs() / sd) * mask.float()

    z_parts = [_z(proposal.ffx, stats['beta_est'], batch['mask_d'])]
    if 'sigma_rfx_est' in stats:
        z_parts.append(_z(proposal.sigma_rfx, stats['sigma_rfx_est'], batch['mask_q']))
    if proposal.has_sigma_eps and 'sigma_eps_est' in stats:
        z_parts.append(
            _z(
                proposal.sigma_eps.unsqueeze(-1),
                stats['sigma_eps_est'],
                torch.ones_like(stats['sigma_eps_est']),
            )
        )
    return torch.cat(z_parts, dim=-1).max(dim=-1).values.cpu().numpy()


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
    ns: np.ndarray | None = None
    safeguards: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None


@dataclass
class _Metadata:
    """What a dataset's result carries beyond its rows: its parameter names, its scale
    information, its group labels, and the permutation from the model's fixed-effect layout
    (random slopes first) back to the formula's order."""

    param_names: dict[str, list[str]]
    scale_info: ScaleInfo
    group_names: list[str] | None
    ffx_order: list[int]


@dataclass
class _Prepared:
    """One input dataset of ``sample`` after preparation and routing: its batch at its own
    padding, whose rows are its priors, the padded model datasets the batch was collated from
    (None for collated input), and its routes, validation and metadata."""

    label: Hashable
    batch: dict[str, torch.Tensor]
    datasets: list[dict[str, np.ndarray]] | None
    routes: list[str]
    validation: list[dict[str, Any]]
    metadata: _Metadata | None

    @property
    def rows(self) -> int:
        return int(self.batch['X'].shape[0])


class Api:
    """Route dataloader-formatted datasets through a joint checkpoint.

    Models are instantiated lazily when their first compatible batch is run, or all at
    construction with ``warmup=True``.

    Tabular inputs are normalized through explicit stages: dataframe/parquet,
    preprocessed numpy dict, model dataset dict, then collated dataloader batch.
    Formula support is intentionally narrow and currently covers additive fixed
    terms plus one lme4-style random-effect term.  A categorical column enters as one
    indicator per level except its most frequent level, the reference; R's
    ``contr.treatment`` takes the first level instead.
    """

    @classmethod
    def from_pretrained(
        cls,
        family: str,
        *,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        device: str | torch.device = 'cpu',
        batch_size: int | None = None,
        warmup: bool = True,
        verbose: bool = True,
    ) -> 'Api':
        """Load a pretrained checkpoint for ``family`` from Hugging Face Hub.

        Downloads and caches the checkpoint automatically on first call.
        Subsequent calls reuse the cache unless the upstream tag has moved.

        Pretrained weights, downloaded by ``from_pretrained``: CC BY-NC-SA 4.0, non-commercial.

        Parameters
        ----------
        family:
            Likelihood family — ``"gaussian"`` (or ``"normal"``), ``"bernoulli"``, or
            ``"poisson"``, in any case.
        cache_dir:
            Override the default HF Hub cache (``~/.cache/huggingface/hub``).
        force_download:
            Re-download even if a cached copy exists.
        device:
            Torch device the models run on, e.g. ``"cuda"`` for a GPU.
        batch_size:
            Rows (one dataset under one prior) ``sample`` runs at a time through every
            stage, from the network pass to the diagnostics; ``None`` takes 64 rows on CPU
            and 256 on CUDA.  Lower it if large datasets run out of memory.
        warmup:
            Build every submodel and run one discarded sampling pass on dummy data while
            loading, so the first ``sample`` call is as fast as later ones.  ``False`` loads
            faster and moves that cost to the first call.
        verbose:
            Print concise checkpoint download/load status messages.
        """
        from metabeta.utils.hub import download_checkpoint

        path = download_checkpoint(
            family,
            cache_dir=cache_dir,
            force_download=force_download,
            verbose=verbose,
        )
        if verbose:
            print(f'Loading metabeta checkpoint on {device}...', flush=True)
        api = cls(path, device=device, batch_size=batch_size, warmup=warmup)
        if verbose:
            print('Metabeta checkpoint ready.', flush=True)
        return api

    def __init__(
        self,
        joint_checkpoint: str | Path,
        *,
        device: str | torch.device = 'cpu',
        batch_size: int | None = None,
        warmup: bool = True,
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

        if warmup:
            # the first call's cost is paid once per process, not per submodel, so one pass on
            # the smallest submodel suffices; building the others spares their first batch (#19)
            self._warmupSubmodel(self.submodels[0])
            for entry in self.submodels:
                self.model(str(entry['id']))

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
        fit_preprocessor: bool = True,
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
        return self._prepare(
            data,
            formula=formula,
            priors=priors,
            preprocessor=preprocessor,
            fit_preprocessor=fit_preprocessor,
            group_name=group_name,
            likelihood_family=likelihood_family,
            q=q,
            stage=stage,
            dry_run=dry_run,
        )[0]

    def _prepare(
        self,
        data: Any,
        *,
        formula: str | None = None,
        priors: Any = None,
        preprocessor: DataPreprocessor | str | Path | None = None,
        fit_preprocessor: bool = True,
        group_name: str | None = None,
        likelihood_family: int | str | None = None,
        q: int | None = None,
        stage: str = 'batch',
        dry_run: bool = False,
    ) -> tuple[Any, list[dict[str, np.ndarray]] | None, _Metadata | None]:
        """``prepareData``'s stage value, plus the padded model datasets a formula batch was
        collated from and the dataset's metadata (both None at earlier stages and for collated
        input)."""

        if stage not in {'preprocessed', 'dataset', 'batch'}:
            raise ValueError(f'unknown prepareData stage: {stage}')

        if isinstance(data, Dataloader):
            batch = data.fullBatch()
            return returnPreparedStage(batch, stage), None, None

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == '.npz':
                loader = Dataloader(path, batch_size=self.batch_size)
                batch = loader.fullBatch()
                return returnPreparedStage(batch, stage), None, None
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
                return preprocessed, None, None
            data = preprocessed

        if isinstance(data, Mapping) and isCollatedBatch(data):
            return returnPreparedStage(dict(data), stage), None, None

        if isinstance(data, Mapping) and isModelDataset(data):
            dataset = {str(k): v for k, v in data.items()}
            if stage == 'dataset':
                return dataset, None, None
            batch = collateGrouped([dataset])
            return returnPreparedStage(batch, stage), None, None

        if isinstance(data, Mapping) and isPreprocessedDict(data):
            if stage == 'preprocessed':
                return {str(k): v for k, v in data.items()}, None, None
            datasets, metadata = self._buildModelDatasets(
                data,
                formula=formula,
                priors=priors,
                likelihood_family=likelihood_family,
                q=q,
            )
            if stage == 'dataset':
                return datasets[0] if len(datasets) == 1 else datasets, None, None
            batch, padded = self._collateForSelectedSubmodel(datasets)
            return returnPreparedStage(batch, stage), padded, metadata

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

    @overload
    def sample(
        self, data: list[pd.DataFrame] | tuple[pd.DataFrame, ...], **kwargs: Any
    ) -> list[RouterResult | None]:
        ...

    @overload
    def sample(self, data: Mapping[K, pd.DataFrame], **kwargs: Any) -> dict[K, RouterResult | None]:
        ...

    @overload
    def sample(self, data: Any, **kwargs: Any) -> RouterResult:
        ...

    @torch.no_grad()
    def sample(
        self,
        data: Any,
        *,
        n_samples: int = 1000,
        loo: bool = False,
        refine: bool | str = True,
        k_threshold: float = PSIS_K_THRESHOLD,
        map_z_threshold: float = MAP_Z_THRESHOLD,
        random_walk: LocalMoves | bool = False,
        evidence: bool = False,
        seed: int | None = None,
        diagnostics: bool | None = None,  # deprecated alias of loo, removed in 0.7
        **prepare_kwargs: Any,
    ) -> RouterResult | list[RouterResult | None] | dict[Any, RouterResult | None]:
        """Sample from the posterior, routing each dataset to the smallest compatible submodel.

        ``data`` is one dataset (a DataFrame, a parquet or ``.npz`` path, a preprocessed dict or a
        collated batch) or many: a list, a tuple or a dict of DataFrames under one formula, one
        set of priors and one preprocessing.  Many datasets return a list, or a dict with the
        same keys, holding one result per dataset as ``sample(df)`` would return it, with its own
        parameter names, scale information and group labels; datasets that need different
        submodels run in one call.  A dataset that cannot be prepared (outside the checkpoint's
        limits, a collinear named column, a term or prior name that does not resolve) is skipped:
        ``None`` takes its place and one warning lists the skipped datasets with their reasons,
        and if every dataset fails the first error is raised with the dataset named.  Unsupported
        formula syntax and a wrong argument type raise at once.  Warnings name datasets by their
        key or index, and by the pair of dataset and prior under several priors.

        By default (``refine=True``, i.e. ``refine='imh'``) the flow posterior is refined
        by Independence Metropolis-Hastings: the flow serves as the proposal of a Markov
        chain targeting the exact rfx-marginalized posterior, in closed form (Normal) or
        through unbiased importance-sampling estimates of the marginal likelihood with a
        Laplace proposal (GLMMs, a pseudo-marginal chain), so the chain's stationary distribution does not
        depend on flow accuracy.  Both refinements redraw the rfx from their conditional at the
        accepted (or weighted) globals, so the local posterior network is not run at all:
        its cost is skipped and the analytical rfx estimate seeds the Laplace mode search
        instead (only the raw flow, ``refine=False``, samples it).  The proposal pool is
        over-drawn by the burn-in so exactly ``n_samples`` post-burn-in draws are returned.
        Per-dataset acceptance rates and a suggested pool size (the smallest ``n_samples``
        expected to reach the
        calibration-validated effective-draw target at the measured acceptance) are
        reported in ``RouterResult.safeguards``; datasets with mean acceptance below
        ``IMH_ACCEPT_WARN`` trigger a warning recommending the suggested size or exact
        MCMC.  The IMH can only revisit pool members, so where the flow lacks tail mass its
        chains repeat few states and the refined sample can be too narrow until n_samples is large;
        datasets whose minimum bulk ESS over the global parameters falls below
        ``IMH_ESS_WARN`` are flagged with a warning.  ``random_walk=True`` continues the IMH
        on the flagged datasets with pseudo-marginal random-walk moves on the same target
        (posthoc/metropolis.py, ``LocalMoves``; ``RANDOM_WALK_DEFAULT``, by device) instead of
        warning; below ``IMH_ESS_NUTS`` they may not repair it fully, and a warning recommends
        exact MCMC there.  A
        ``LocalMoves`` instance sets them, and its own gate, explicitly.  Per dataset, the safeguards record
        the flag ('ess_flag'), whether the moves ran ('random_walk_ran') and their acceptance
        ('random_walk_accept', NaN where they did not run).

        ``refine='is'`` selects the previous self-normalized importance-sampling
        correction instead, with the PSIS shape diagnostic k̂ (Yao et al., 2018) computed
        per dataset: datasets with k̂ ≤ ``k_threshold`` receive the correction, datasets
        above it keep the raw flow posterior and trigger a warning.  ``refine=False``
        returns the raw flow posterior.  Refinement (like the MAP cross-check) applies
        only when ``n_samples >= MIN_SAFEGUARD_SAMPLES`` and is skipped silently below.

        Independently of ``refine``, posterior means are cross-checked against the
        analytical MAP estimates that condition the model (a flow-independent anchor);
        datasets whose maximum |posterior mean − MAP| / posterior SD exceeds
        ``map_z_threshold`` trigger a warning.  All checks are reported in
        ``RouterResult.safeguards``.

        Every call also stores the posterior predictive and a fit metric (R², AUC or deviance)
        in ``RouterResult.diagnostics``.  ``loo=True`` adds the PSIS-LOO NLL and its Pareto k̂,
        which cost about as much as the rest of the diagnostics together.

        ``evidence=True`` also returns each dataset's log marginal likelihood log p(y) on the raw
        response scale in ``RouterResult.evidence``; the log Bayes factor between two models of
        the same data is the difference of their ``log_evidence``.  It costs one more likelihood
        pass (posthoc/evidence.py), needs ``n_samples >= MIN_SAFEGUARD_SAMPLES`` and warns where
        the PSIS k̂ of that pass or, with refinement, of the flow pool's weights exceeds
        ``k_threshold``.

        ``seed`` seeds torch for this call only: two calls with one seed return the same draws,
        as after ``torch.manual_seed(seed)``, and the caller's CPU and CUDA random streams stay
        as they were.  It reproduces the call as a whole, so a dataset's draws differ between a
        batched and a separate call, and between containers that order or size datasets
        differently.
        """

        if diagnostics is not None:
            warnings.warn(
                'diagnostics= is deprecated and will be removed in 0.7: every call now computes '
                'the fit metric, and loo=True adds the LOO-NLL.',
                FutureWarning,
                skip_file_prefixes=INTERNAL_PREFIXES,
            )
            loo = loo or diagnostics
        method = {True: 'imh', False: None}.get(refine, refine)
        if method not in (None, 'imh', 'is'):
            raise ValueError(f"refine must be True, False, 'imh' or 'is' (got {refine!r})")
        do_refine = method is not None and n_samples >= MIN_SAFEGUARD_SAMPLES
        if random_walk is not False and method != 'imh':
            raise ValueError("random_walk requires refine='imh'")
        if evidence and n_samples < MIN_SAFEGUARD_SAMPLES:
            raise ValueError(f'evidence=True needs n_samples >= {MIN_SAFEGUARD_SAMPLES}')
        moves = None
        if method == 'imh' and random_walk:
            preset = RANDOM_WALK_DEFAULT['cuda' if self.device.type == 'cuda' else 'cpu']
            moves = preset if random_walk is True else random_walk

        container = _asContainer(data)
        labels, frames = ([0], [data]) if container is None else container
        items = self._prepareDatasets(labels, frames, container is not None, prepare_kwargs)
        prepared = [item for item in items if item is not None]

        # every stage runs per chunk of at most batch_size rows; a chunk holds whole datasets of
        # one submodel, sorted by size, and is cut inside a dataset only when it alone exceeds
        # the chunk, so a result joins one slice in the common case
        chunk_rows = self.batch_size or CHUNK_ROWS['cuda' if self.device.type == 'cuda' else 'cpu']
        order = sorted(range(len(prepared)), key=lambda i: self._sizeKey(prepared[i]))
        chunks = _cutChunks(prepared, order, chunk_rows)
        pieces: list[list[dict[str, Any]]] = [[] for _ in prepared]
        n_rows = sum(item.rows for item in prepared)
        # a one-chunk call shows no bar; otherwise `disable` stays unset so TQDM_DISABLE applies
        silent = {'disable': True} if len(chunks) == 1 else {}
        with _seeded(seed), tqdm(total=n_rows, unit='dataset', leave=False, **silent) as bar:
            for chunk in chunks:
                batch, routes = self._chunkBatch(prepared, chunk)
                out = self._sampleChunk(
                    batch,
                    routes,
                    n_samples=n_samples,
                    method=method if do_refine else None,
                    k_threshold=k_threshold,
                    map_z_threshold=map_z_threshold,
                    moves=moves,
                    evidence=evidence,
                    loo=loo,
                )
                offset = 0
                for i, start, end in chunk:
                    pieces[i].append(_sliceRows(out, offset, offset + end - start))
                    offset += end - start
                bar.update(offset)
        results = [
            self._assembleResult(item, pieces[i], prepare_kwargs) for i, item in enumerate(prepared)
        ]

        self._warnSafeguards(
            _flagRows(results, 'safeguards'),
            _flagRows(results, 'evidence'),
            _rowLabels(prepared, with_dataset=container is not None),
        )
        if container is None:
            return results[0]
        found = iter(results)
        out = [None if item is None else next(found) for item in items]
        return dict(zip(labels, out)) if isinstance(data, Mapping) else out

    def _prepareDatasets(
        self,
        labels: list[Hashable],
        frames: list[Any],
        container: bool,
        prepare_kwargs: dict[str, Any],
    ) -> list[_Prepared | None]:
        """Prepare and route each dataset on its own.  A ``ValueError`` or ``KeyError`` raised
        for one dataset of a container skips it (None in its place, one warning per call);
        other errors raise at once, and so does the first error when every dataset fails.
        Preparation warnings are emitted once per message with the datasets that raised them."""
        items: list[_Prepared | None] = []
        skipped: list[tuple[Hashable, Exception]] = []
        noted: dict[tuple[type, str], list[Hashable]] = {}
        for label, frame in zip(labels, frames):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    items.append(self._prepareDataset(frame, label, prepare_kwargs))
                except (ValueError, KeyError) as exc:
                    items.append(None)
                    skipped.append((label, exc))
            for w in caught:
                noted.setdefault((w.category, str(w.message)), []).append(label)
        if not any(item is not None for item in items):
            label, exc = skipped[0]
            if container:
                exc.add_note(f'raised while preparing dataset {label!r}')
            raise exc
        for (category, message), who in noted.items():
            text = f'dataset(s) {_listed(who)}: {message}' if container else message
            warnings.warn(text, category, skip_file_prefixes=INTERNAL_PREFIXES)
        if skipped:
            reasons = [
                f'  {label!r}: {str(exc).splitlines()[0]}' for label, exc in skipped[:LISTED_ROWS]
            ]
            if len(skipped) > LISTED_ROWS:
                reasons.append(f'  … {len(skipped)} in all')
            warnings.warn(
                f'Skipped {len(skipped)} of {len(items)} dataset(s) that could not be prepared '
                '(None in their place):\n' + '\n'.join(reasons),
                skip_file_prefixes=INTERNAL_PREFIXES,
            )
        return items

    def _prepareDataset(
        self, data: Any, label: Hashable, prepare_kwargs: dict[str, Any]
    ) -> _Prepared:
        """One dataset prepared to a batch at its own padding, validated and routed."""
        batch, datasets, metadata = self._prepare(data, **prepare_kwargs)
        if not isinstance(batch, dict) or not isCollatedBatch(batch):
            raise TypeError('sample requires data prepared to batch stage')
        self._validateBatchFormat(batch)
        routes, validation = self._routeBatch(batch)
        for entry in validation:
            entry['dataset'] = label
        return _Prepared(
            label, toDevice(batch, self.device), datasets, routes, validation, metadata
        )

    def _sizeKey(self, item: _Prepared) -> tuple[tuple[float, str], int, int]:
        """Submodel, then number of groups, then largest group."""
        submodel = routingSortKey(self._submodel_by_id[item.routes[0]])
        return submodel, int(item.batch['m'].max()), int(item.batch['ns'].max())

    def _chunkBatch(
        self, items: list[_Prepared], chunk: list[tuple[int, int, int]]
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """A chunk's batch at its own padding and its routes: a slice of one dataset's batch, or
        the datasets of several collated together."""
        if len(chunk) == 1:
            i, start, end = chunk[0]
            return trimBatchPadding(sliceBatch(items[i].batch, start, end)), items[i].routes
        datasets = [dataset for i, _, _ in chunk for dataset in items[i].datasets]
        routes = [route for i, _, _ in chunk for route in items[i].routes]
        return toDevice(collateGrouped(datasets), self.device), routes

    def _assembleResult(
        self, item: _Prepared, pieces: list[dict[str, Any]], prepare_kwargs: dict[str, Any]
    ) -> RouterResult:
        """One dataset's result from its slices of the chunk outputs."""
        batch = item.batch
        proposal, safeguards, log_ev, diags = self._joinChunks(pieces, batch)
        self._rescaleProposal(proposal, batch)
        prior_params: dict[str, np.ndarray] = {
            key: batch[key].float().cpu().numpy()
            for key in ('tau_ffx', 'nu_ffx', 'tau_rfx', 'eta_rfx', 'tau_eps')
            if key in batch
        }
        for key in ('family_ffx', 'family_sigma_rfx', 'family_sigma_eps'):
            if key in batch:
                prior_params[key] = batch[key].long().cpu().numpy()
        meta = item.metadata
        self._restoreFixedOrder(
            proposal, prior_params, diags, safeguards, None if meta is None else meta.ffx_order
        )
        return RouterResult(
            proposal=proposal,
            routes=item.routes,
            validation=item.validation,
            diagnostics=diags,
            param_names=None if meta is None else meta.param_names,
            formula=prepare_kwargs.get('formula'),
            priors_str=_priorStr(prepare_kwargs.get('priors')),
            group_names=None if meta is None else meta.group_names,
            prior_params=prior_params or None,
            scale_info=None if meta is None else meta.scale_info,
            batch=batch,
            ns=batch['ns'].cpu().numpy() if 'ns' in batch else None,
            safeguards=safeguards,
            evidence=log_ev,
        )

    def _sampleChunk(
        self,
        batch: dict[str, torch.Tensor],
        routes: list[str],
        *,
        n_samples: int,
        method: str | None,
        k_threshold: float,
        map_z_threshold: float,
        moves: LocalMoves | None,
        evidence: bool,
        loo: bool,
    ) -> dict[str, Any]:
        """Today's pipeline on one chunk: network pass, refinement (``method`` None skips it),
        evidence, MAP check and per-row diagnostics.  Returns the chunk's proposal and its
        per-row numbers and flags; ``_joinChunks`` concatenates them and ``_warnSafeguards``
        turns the flags into warnings."""
        # IMH consumes the first `burnin` steps of each chain, so over-draw the flow
        # pool to hand back exactly n_samples post-burn-in draws; both refinements redraw
        # the rfx from their conditional, so the local posterior is skipped for them
        pool = n_samples
        if method == 'imh':
            pool = IMH_N_CHAINS * (math.ceil(n_samples / IMH_N_CHAINS) + IMH_BURNIN)
        proposal, stats = self._runRouted(batch, routes, n_samples=pool, local=method is None)
        safeguards: dict[str, Any] | None = None
        if method is not None:
            proposal, safeguards = self._refineAndCheck(
                proposal,
                batch,
                method=method,
                k_threshold=k_threshold,
                n_samples=n_samples,
                moves=moves,
            )
        log_ev = None
        if evidence:
            pool_k = safeguards['psis_k'] if safeguards else None
            log_ev = self._evidence(proposal, batch, k_threshold=k_threshold, pool_k=pool_k)
        if stats is not None and n_samples >= MIN_SAFEGUARD_SAMPLES:
            map_check = self._mapConsistency(proposal, stats, batch, threshold=map_z_threshold)
            safeguards = {**(safeguards or {}), **map_check}
        diags = self._computeDiagnostics(proposal, batch, loo=loo)
        return {'proposal': proposal, 'safeguards': safeguards, 'evidence': log_ev, 'diags': diags}

    def _joinChunks(
        self, chunks: list[dict[str, Any]], batch: dict[str, torch.Tensor]
    ) -> tuple[Proposal, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
        """Join the chunks' outputs in row order: the proposal at the batch's group count, the
        safeguards and evidence row by row, and the diagnostics with the posterior predictive
        and the fit aggregates of the whole."""
        proposals = [c['proposal'] for c in chunks]
        proposal = proposals[0] if len(proposals) == 1 else concatProposalsBatch(proposals)
        proposal = proposal.resizeGroups(batch['mask_n'].shape[1])
        joined = {
            key: None if chunks[0][key] is None else _joinRows([c[key] for c in chunks])
            for key in ('safeguards', 'evidence', 'diags')
        }
        diags = self._finishDiagnostics(proposal, batch, joined['diags'])
        return proposal, joined['safeguards'], joined['evidence'], diags

    @staticmethod
    def _warnSafeguards(
        safeguards: dict[str, Any] | None, evidence: dict[str, Any] | None, labels: list[Any]
    ) -> None:
        """At most one warning per type for the rows the safeguards flagged, named by their
        ``labels``; values are printed for the first ``LISTED_ROWS`` of them."""

        def warn(message: str) -> None:
            warnings.warn(message, skip_file_prefixes=INTERNAL_PREFIXES)

        def named(flagged: np.ndarray) -> tuple[np.ndarray, str]:
            return flagged[:LISTED_ROWS], _listed([labels[i] for i in flagged])

        sg = safeguards or {}
        if 'accept_flag' in sg:
            a_bar, ess_min, ran = sg['accept_rate'], sg['ess_min'], sg['random_walk_ran']
            # where the random-walk moves ran, they continued the chains these two warnings describe
            flagged, names = named(np.flatnonzero(sg['accept_flag'] & ~ran))
            if flagged.size:
                a_vals = ', '.join(f'{a_bar[i]:.2f}' for i in flagged)
                s_vals = ', '.join(f'{int(sg["suggested_n_samples"][i])}' for i in flagged)
                respectively = ' (respectively)' if flagged.size > 1 else ''
                warn(
                    f'Low IMH acceptance for dataset(s) {names}: {a_vals} < '
                    f'{sg["accept_threshold"]:g}.\n'
                    'The network is far from the posterior, so the intervals can be too narrow. '
                    'Remedies, in order:\n'
                    f'  1. more draws: n_samples of at least {s_vals}{respectively}\n'
                    '  2. exact MCMC, e.g. NUTS'
                )
            flagged, names = named(np.flatnonzero(sg['ess_flag'] & ~ran))
            if flagged.size:
                e_vals = ', '.join(f'{ess_min[i]:.0f}' for i in flagged)
                _, low = named(flagged[ess_min[flagged] < IMH_ESS_NUTS])
                caveat = f' (may not suffice for {low})' if low != '[]' else ''
                warn(
                    f'Low effective sample size (ESS) for dataset(s) {names}: min ESS '
                    f'{e_vals} < {sg["ess_threshold"]:g}.\n'
                    'The IMH chains repeat few draws, so the intervals can be too narrow. '
                    'Remedies, in order:\n'
                    '  1. more draws (n_samples)\n'
                    f'  2. random_walk=True{caveat}\n'
                    '  3. exact MCMC, e.g. NUTS'
                )
            # the moves replace the ESS warning, except where the IMH left them too little to work with
            low, names = named(np.flatnonzero(ran & (ess_min < IMH_ESS_NUTS)))
            if low.size:
                e_vals = ', '.join(f'{ess_min[i]:.0f}' for i in low)
                warn(
                    f'Very low IMH ESS before the random-walk moves for dataset(s) {names}: '
                    f'{e_vals} < {IMH_ESS_NUTS:g}.\n'
                    'The moves may not close the gap, so the intervals can still be too narrow.\n'
                    'Remedy: check against exact MCMC, e.g. NUTS.'
                )
        if 'psis_fallback' in sg:
            flagged, names = named(np.flatnonzero(sg['psis_fallback']))
            if flagged.size:
                k_vals = ', '.join(f'{sg["psis_k"][i]:.2f}' for i in flagged)
                warn(
                    f'PSIS reliability check failed for dataset(s) {names} '
                    f'(k̂ = {k_vals} > {sg["psis_k_threshold"]:g}): the posterior approximation '
                    'may be inaccurate and the importance correction is not applicable, so the '
                    'raw posterior was returned there. Treat those results with caution and '
                    'consider exact MCMC (e.g. NUTS) instead.'
                )
        if evidence is not None:
            flagged, names = named(np.flatnonzero(evidence['psis_flag']))
            if flagged.size:
                pool_k = evidence['pool_psis_k']
                k_max = (
                    evidence['psis_k'] if pool_k is None else np.maximum(evidence['psis_k'], pool_k)
                )
                k_vals = ', '.join(f'{k_max[i]:.2f}' for i in flagged)
                warn(
                    f'Evidence reliability check failed for dataset(s) {names} '
                    f'(PSIS k̂ of the evidence pass or the flow pool = {k_vals} > '
                    f'{evidence["psis_k_threshold"]:g}): '
                    'the log-evidence and any Bayes factor built on it may be inaccurate. Treat '
                    'those values with caution and consider bridge sampling on exact MCMC (e.g. '
                    'NUTS) draws instead.'
                )
        if 'map_z_flag' in sg:
            flagged, names = named(np.flatnonzero(sg['map_z_flag']))
            if flagged.size:
                z_vals = ', '.join(f'{sg["map_z"][i]:.1f}' for i in flagged)
                warn(
                    f'MAP check failed for dataset(s) {names}: max |posterior mean − '
                    f'MAP| / posterior SD = {z_vals} > {sg["map_z_threshold"]:g}.\n'
                    'The posterior disagrees with the analytical point estimates, so the data may '
                    'lie outside what the network was trained on.\n'
                    'Remedy: exact MCMC, e.g. NUTS.'
                )

    @staticmethod
    def _restoreFixedOrder(
        proposal: Proposal,
        prior_params: dict[str, np.ndarray],
        diags: dict[str, Any],
        safeguards: dict[str, Any] | None,
        order: list[int] | None,
    ) -> None:
        """Permute the fixed-effect columns from the model's layout (random slopes first) back
        to the user's formula order. Padding columns stay in place, and ``result.batch`` keeps
        the model's layout."""
        if order is None or order == sorted(order):
            return

        def lead(n: int) -> list[int]:
            return order + list(range(len(order), n))

        g = proposal.samples_g
        d = proposal.d
        proposal.data['global']['samples'] = torch.cat([g[..., lead(d)], g[..., d:]], dim=-1)
        for key in ('tau_ffx', 'nu_ffx'):
            if key in prior_params:
                prior_params[key] = prior_params[key][:, lead(prior_params[key].shape[1])]
        if safeguards is not None and 'ess' in safeguards:  # (B, D_g), ffx columns first
            safeguards['ess'] = safeguards['ess'][:, lead(safeguards['ess'].shape[1])]
        if 'param_summary' in diags:
            summary = diags['param_summary']['ffx']
            diags['param_summary']['ffx'] = {
                k: v[..., lead(v.shape[-1])] for k, v in summary.items()
            }

    @staticmethod
    def _dummyResponse(family: int | None, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` responses the family's checkpoint accepts: 0/1, counts, else Gaussian."""
        if family == 1:
            return rng.integers(0, 2, size=n).astype(float)
        if family == 2:
            return rng.integers(0, 10, size=n).astype(float)
        return rng.standard_normal(n)

    def _warmupSubmodel(self, entry: dict) -> None:
        """Warm the default sampling path on minimal dummy data routed to ``entry``."""
        r = entry.get('routing', {})
        d = max(int(r.get('min_d', 1)), 2)  # at least intercept + one covariate
        q = int(r.get('min_q', 1))
        fam = int(r.get('likelihood_family', 0))
        min_m = max(
            5, int(r.get('min_m', 0)) + 1, max(d, q * (q + 1) // 2) + int(r.get('min_bg_df', 0)) + 1
        )
        min_n = max(4, int(r.get('min_n', 0)) + 1, q + int(r.get('min_within_df', 0)) + 1)
        n = min_m * min_n

        rng = np.random.default_rng(0)
        dummy: dict[str, Any] = {'_y': self._dummyResponse(fam, n, rng)}
        for j in range(d - 1):
            dummy[f'_x{j}'] = rng.standard_normal(n)
        dummy['_g'] = np.repeat(np.arange(min_m), min_n)

        fixed = ' + '.join(f'_x{j}' for j in range(d - 1))
        rfx_slopes = (' + ' + ' + '.join(f'_x{j}' for j in range(q - 1))) if q > 1 else ''
        formula = f'_y ~ {fixed} + (1{rfx_slopes} | _g)'

        self.warmup(pd.DataFrame(dummy), formula=formula, likelihood_family=fam)

    def warmup(self, data: Any = None, **prepare_kwargs: Any) -> None:
        """Run the default sampling path once, IMH at ``MIN_SAFEGUARD_SAMPLES`` draws, so the
        first real call is as fast as later ones.  Its draws and warnings are discarded.

        ``warmup=True`` (the default) does this at ``__init__`` time; the cost it removes is
        paid once per process, so one pass serves every submodel.  Call this explicitly only
        if ``warmup=False`` was passed.

        ``data`` is optional when ``formula`` is passed as a keyword argument:
        a minimal dummy DataFrame is generated automatically from the formula
        so no real data is required.
        """
        if data is None:
            formula = prepare_kwargs.get('formula')
            if not formula:
                raise ValueError('warmup requires either data or a formula= kwarg')
            from metabeta.utils.api import parseFormula

            spec = parseFormula(formula)
            d = len(spec.fixed_terms) + (1 if spec.intercept else 0)
            q = len(spec.random_terms)

            min_m, min_n = 5, 4
            fam: int | None = None
            for entry in self.submodels:
                r = entry.get('routing', {})
                if r.get('min_d', 0) <= d <= r.get('max_d', 9999):
                    bg_df = int(r.get('min_bg_df', 0))
                    within_df = int(r.get('min_within_df', 0))
                    min_m = max(
                        min_m, int(r.get('min_m', 0)) + 1, max(d, q * (q + 1) // 2) + bg_df + 1
                    )
                    min_n = max(min_n, int(r.get('min_n', 0)) + 1, q + within_df + 1)
                    if fam is None and 'likelihood_family' in r:
                        fam = int(r['likelihood_family'])

            rng = np.random.default_rng(0)
            n = min_m * min_n
            dummy: dict[str, Any] = {}
            if spec.target:
                dummy[spec.target] = self._dummyResponse(fam, n, rng)
            for term in spec.fixed_terms:
                dummy[term] = rng.standard_normal(n)
            if spec.group_name:
                dummy[spec.group_name] = np.repeat(np.arange(min_m), min_n)
            data = pd.DataFrame(dummy)
        # the draws are discarded, and so are the warnings about them
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.sample(data, n_samples=MIN_SAFEGUARD_SAMPLES, **prepare_kwargs)

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
        self, proposal: Proposal, batch: dict[str, torch.Tensor], *, loo: bool
    ) -> dict[str, Any]:
        """Per-row diagnostics of one chunk; ``_finishDiagnostics`` adds the whole's aggregates."""
        lf = int(batch['likelihood_family'].flatten()[0].item())
        pp = _predictive.getPosteriorPredictive(proposal, batch, likelihood_family=lf)

        if lf == 0:
            vals = _predictive.posteriorPredictiveR2(pp, batch, w=proposal.weights)
        elif lf == 1:
            vals = _predictive.posteriorPredictiveAUC(pp, batch, w=proposal.weights)
        else:
            vals = _predictive.posteriorPredictiveDeviance(pp, batch, w=proposal.weights)
        diags = {'fit_vals': vals}
        if not loo:
            return diags

        # PSIS over every observation, the costly part, and the rest of what diagnostics=True gave
        log_p = pp.log_prob(batch['y'].unsqueeze(-1))  # (B, m, n, S)
        ppc_nll = _predictive.posteriorPredictiveNLL(pp, batch, mode='mixture', log_p=log_p)
        loo_nll_vals, pareto_k_vals = _predictive.psisLooNLL(
            pp, batch, w=proposal.weights, reff=proposal.reff, log_p=log_p
        )

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
        return diags | {
            'ppc_nll': ppc_nll,
            'param_summary': param_summary,
            'loo_nll_vals': loo_nll_vals,
            'pareto_k_vals': pareto_k_vals,
        }

    @staticmethod
    def _finishDiagnostics(
        proposal: Proposal, batch: dict[str, torch.Tensor], diags: dict[str, Any]
    ) -> dict[str, Any]:
        """The posterior predictive of the whole and the fit aggregates over its rows."""
        lf = int(batch['likelihood_family'].flatten()[0].item())
        b = int(batch['X'].shape[0])
        label = {0: 'R²', 1: 'AUC', 2: 'Deviance'}[lf]
        agg = (lambda t: t.item()) if b == 1 else (lambda t: t.mean().item())
        out = {
            'fit': agg(diags['fit_vals']),
            'fit_label': label if b == 1 else f'Mean {label}',
            'pp': _predictive.getPosteriorPredictive(proposal, batch, likelihood_family=lf),
        }
        if 'loo_nll_vals' in diags:
            out |= {'loo_nll': agg(diags['loo_nll_vals']), 'pareto_k': agg(diags['pareto_k_vals'])}
        return diags | out

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
        local: bool = True,
    ) -> tuple[Proposal, dict[str, torch.Tensor] | None]:
        """Run inference for a single-submodel batch; returns (proposal, analytical stats).

        The analytical GLMM statistics that condition the summarizer are computed here
        once and threaded through ``estimate`` so the MAP consistency safeguard can reuse
        them without a second fit.  With ``local=False`` the local posterior is skipped and
        the analytical rfx estimate stands in for its samples (the refinements redraw the
        rfx anyway).
        """
        unique_ids = list(dict.fromkeys(routes))
        if len(unique_ids) != 1:
            # datasets given as DataFrames never share a chunk across submodels; the rows of one
            # collated batch are padded alike and can only run on one
            raise ValueError(
                f'the rows of a collated batch route to different submodels {unique_ids}; pass '
                'the datasets as a list or dict of DataFrames instead'
            )
        submodel_id = unique_ids[0]
        model = self.model(submodel_id)
        self._validateBatchMatchesModel(batch, model)
        stats = model._dataStatistics(batch) if model.analytical_context else None
        proposal = model.estimate(batch, n_samples=n_samples, stats=stats, local=local)
        return proposal, stats

    def _refineAndCheck(
        self,
        proposal: Proposal,
        batch: dict[str, torch.Tensor],
        *,
        method: str,
        k_threshold: float,
        n_samples: int,
        moves: LocalMoves | None = None,
    ) -> tuple[Proposal, dict[str, Any]]:
        """Refine the flow posterior ('imh' or 'is') and compute the reliability flags.

        Both methods target the exact rfx-marginalized posterior for Normal. For GLMMs, 'imh'
        targets the exact posterior through a pseudo-marginal chain on unbiased IS² estimates
        of the marginal likelihood, and 'is' targets its Laplace approximation. Each redraws
        the rfx from the matching conditional, so the delivered draws form a consistent joint
        sample.

        'imh' (default) runs Independence Metropolis-Hastings with the flow pool as
        proposals: asymptotically exact in the pool size, no weight-degeneracy failure
        mode.  Per-dataset acceptance, a suggested pool size and the bulk ESS per global
        parameter are reported; mean acceptance below ``IMH_ACCEPT_WARN`` or a minimum ESS
        below ``IMH_ESS_WARN`` is flagged.  The post-burn-in
        draws are trimmed to exactly ``n_samples``.  ``moves`` (``LocalMoves``) runs random-walk
        moves where its gate says; the ESS stays the IMH chains' (the gate's statistic).  Both
        methods report the PSIS k̂ of the pool's weights ('psis_k'), which the evidence check
        reuses.

        'is' applies self-normalized importance sampling with PSIS smoothing; datasets
        with k̂ > ``k_threshold`` fall back to uniform weights (the raw flow globals, with
        conditionally redrawn rfx) and are flagged.  The weighted sample is systematically
        resampled back to ``n_samples`` equal-weight draws (an exact pass-through for
        fallback datasets).

        The flags are emitted as warnings by ``_warnSafeguards`` once every chunk is done.
        """
        lf = int(batch['likelihood_family'].flatten()[0].item())
        if method == 'imh':
            return self._refineIMH(proposal, batch, lf=lf, n_samples=n_samples, moves=moves)

        if lf == 0:
            sampler = ImportanceSampler(
                batch,
                marginal=True,
                rb_redraw=True,
                corr_prior=True,
                pareto=True,
                k_threshold=k_threshold,
                likelihood_family=lf,
            )
        else:
            sampler = LaplaceImportanceSampler(
                batch,
                corr_prior=True,
                pareto=True,
                k_threshold=k_threshold,
                likelihood_family=lf,
            )
        B = proposal.samples_g.shape[0]
        proposal = sampler(proposal)
        psis_k = proposal.is_results['pareto_k'].detach().float().cpu().numpy().copy()
        fallback = proposal.is_results['fallback'].detach().cpu().numpy().copy()

        # systematic resampling back to equal weights; uniform weights map to identity
        idx = torch.searchsorted(
            torch.cumsum(proposal.is_results['weights'], dim=-1),
            (
                torch.rand(B, 1, device=proposal.samples_g.device) / n_samples
                + torch.arange(n_samples, device=proposal.samples_g.device).view(1, -1) / n_samples
            ),
            right=True,
        ).clamp(max=proposal.samples_g.shape[-2] - 1)
        proposal.subset(idx)
        proposal.is_results = {}
        return proposal, {
            'refine_method': 'isMarginal' if lf == 0 else 'isLaplace',
            'psis_k': psis_k,
            'psis_k_threshold': k_threshold,
            'psis_fallback': fallback,
        }

    def _refineIMH(
        self,
        proposal: Proposal,
        batch: dict[str, torch.Tensor],
        *,
        lf: int,
        n_samples: int,
        moves: LocalMoves | None = None,
    ) -> tuple[Proposal, dict[str, Any]]:
        """IMH-refine the flow posterior; see ``_refineAndCheck`` for the contract."""
        sampler = MetropolisSampler(
            batch,
            n_chains=IMH_N_CHAINS,
            n_steps=proposal.n_samples // IMH_N_CHAINS,
            burnin=IMH_BURNIN,
            mode='marginal' if lf == 0 else 'laplace',
            likelihood_family=lf,
            # GLMMs: pseudo-marginal chain (unbiased IS² weights), i.e. the exact posterior
            n_inner=0 if lf == 0 else IS2_N_INNER,
            moves=moves,
        )
        proposal, diag = sampler(proposal)
        accept, ess, ran, rw_accept = (
            diag[k] for k in ('accept_rate', 'ess', 'moved', 'move_accept')
        )
        pool_k = psislw(diag['log_w'].cpu().double().numpy())[1]
        B = accept.shape[0]
        ran, ess = ran.numpy(), ess.numpy()  # ess (B, D_g): the IMH chains', NaN = padding
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows
            ess_min = np.nanmin(ess, axis=1)

        # the pool was over-drawn by the burn-in; trim to exactly n_samples
        idx = (
            torch.arange(n_samples, device=proposal.samples_g.device)
            .unsqueeze(0)
            .expand(B, n_samples)
        )
        proposal.subset(idx)

        a_bar = accept.mean(dim=1)
        suggested = suggestPoolSize(accept)
        flag = (a_bar < IMH_ACCEPT_WARN).cpu().numpy()
        ess_flag = ess_min < IMH_ESS_WARN
        return proposal, {
            'refine_method': ('imhMarginal' if lf == 0 else 'imhPM') + ('+moves' if moves else ''),
            'accept_rate': a_bar.detach().float().cpu().numpy().copy(),
            'accept_threshold': IMH_ACCEPT_WARN,
            'accept_flag': flag,
            'suggested_n_samples': suggested.detach().cpu().numpy().copy(),
            'ess': ess,
            'ess_min': ess_min,
            'ess_threshold': IMH_ESS_WARN,
            'ess_flag': ess_flag,
            'random_walk_ran': ran,
            'random_walk_accept': rw_accept.numpy(),
            'psis_k': pool_k,
        }

    def _mapConsistency(
        self,
        proposal: Proposal,
        stats: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        threshold: float,
    ) -> dict[str, Any]:
        """Cross-check posterior means against the analytical MAP anchor.

        See ``mapZScores`` for the statistic.  Datasets whose max-z exceeds
        ``threshold`` are flagged (``_warnSafeguards`` recommends exact MCMC there).
        """
        map_z = mapZScores(proposal, stats, batch)
        flag = map_z > threshold
        return {
            'map_z': map_z,
            'map_z_threshold': threshold,
            'map_z_flag': flag,
        }

    def _evidence(
        self,
        proposal: Proposal,
        batch: dict[str, torch.Tensor],
        *,
        k_threshold: float,
        pool_k: np.ndarray | None,
    ) -> dict[str, Any]:
        """log p(y) per dataset from the unrescaled posterior.

        A dataset is flagged where the PSIS k̂ of the t pass or of the flow pool's weights (pool_k,
        None without refinement) exceeds k_threshold: the t is fitted to draws that inherit the
        pool's gaps, which the t pass's own k̂ can miss.
        """
        lf = int(batch['likelihood_family'].flatten()[0].item())
        t_pass = tEvidence(batch, proposal, lf, proposal.n_samples)
        log_ev, psis_k, eff = (t_pass[k] for k in ('log_ev', 'k', 'eff'))
        k_max = psis_k if pool_k is None else np.maximum(psis_k, pool_k)
        flag = k_max > k_threshold
        return {
            'log_evidence': log_ev,
            'psis_k': psis_k,
            'pool_psis_k': pool_k,
            'efficiency': eff,
            'psis_k_threshold': k_threshold,
            'psis_flag': flag,
        }

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
        raw_columns = list(df.columns)
        df = df.copy()
        df.columns = df.columns.str.lower()

        target = spec.target.lower() if spec.target is not None else 'y'
        if target != 'y':
            if target not in df.columns:
                raise KeyError(f'formula target column not found: {target}')
            df = df.rename(columns={target: 'y'})

        group_name = (group_name or spec.group_name or '').lower()
        terms = {*spec.fixed_terms, *spec.random_terms}
        if preprocessor is not None:
            prep = (
                DataPreprocessor.load(preprocessor)
                if isinstance(preprocessor, (str, Path))
                else preprocessor
            )
            if dry_run:
                return prep.audit(df)
            preprocessed = prep.transform(df)
        else:
            if formula is not None and group_name:
                # fit on the formula's columns only, so an unrelated column cannot drop a named
                # one through the correlation filter; a term may name a dummy, '{column}_{level}'.
                # Without a group name the preprocessor needs every column to detect the group
                named = {c for c in df.columns for t in terms if t == c or t.startswith(f'{c}_')}
                df = df[[c for c in df.columns if c in {'y', group_name} | named]]
            prep = DataPreprocessor(group_name=group_name)
            if dry_run:
                return prep.audit(df)
            if not fit_preprocessor:
                raise ValueError(
                    'tabular router input requires a fitted preprocessor or fit_preprocessor=True'
                )
            preprocessed = prep.fit_transform(df)
        # a dropped column would resolve to another one sharing its prefix, or to none
        dropped = sorted(terms & prep._dropped_cols)
        if dropped:
            raise ValueError(
                f'the preprocessor dropped formula column(s) {dropped} as nearly collinear with '
                f'another predictor (|r| > {prep.corr_threshold:g}) or near-constant: keep one '
                'column of each collinear pair in the formula'
            )
        # every output name (parameters, summaries, plots) derives from these columns
        columns = restoreColumnCase(preprocessed['columns'].tolist(), raw_columns)
        preprocessed['columns'] = np.array(columns, dtype=str)
        return attachPreprocessorMetadata(preprocessed, prep)

    def _buildModelDatasets(
        self,
        preprocessed: Mapping[str, Any],
        *,
        formula: str | None,
        priors: Any,
        likelihood_family: int | str | None,
        q: int | None,
    ) -> tuple[list[dict[str, np.ndarray]], _Metadata]:
        """One model dataset per prior, and the dataset's metadata."""
        required = ('X', 'y', 'groups', 'columns', 'd', 'n', 'ns', 'm', 'y_type')
        missing = [key for key in required if key not in preprocessed]
        if missing:
            raise KeyError(f'preprocessed data is missing keys: {missing}')

        y_type = str(np.asarray(preprocessed['y_type']).item())
        if y_type == 'multiclass':
            raise ValueError('multiclass targets are not supported by the router')
        likelihood = resolveLikelihoodFamily(likelihood_family, y_type)
        if y_type == 'binary' and likelihood == 0:
            warnings.warn(
                "the outcome is binary (0/1) but fitted with the 'normal' family; for a binary "
                "outcome load Api.from_pretrained('bernoulli')",
                skip_file_prefixes=INTERNAL_PREFIXES,
            )

        X_pre = np.asarray(preprocessed['X'], dtype=float)
        y = np.asarray(preprocessed['y'])
        groups = np.asarray(preprocessed['groups'])
        columns = tuple(str(c) for c in np.asarray(preprocessed['columns']).tolist())
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError('preprocessed groups must be a 1D array aligned with y')
        if X_pre.ndim != 2 or X_pre.shape[0] != len(y):
            raise ValueError('preprocessed X must have shape (n, d) aligned with y')

        spec = parseFormula(formula)
        # no formula: every column is a fixed effect; a formula without fixed terms
        # (y ~ 1 + (1 | g)) fits the intercept only, as in lme4
        fixed_indices = (
            list(range(len(columns)))
            if formula is None
            else resolveFixedIndices(spec.fixed_terms, columns)
        )
        self._warnInconsistentColumns(X_pre, columns, fixed_indices)
        random_terms = spec.random_terms
        if not random_terms:
            if q is None:
                random_terms = ('1',)
            else:
                random_terms = defaultRandomTerms(q, columns)
        # the networks were trained with the random-effect design equal to the leading
        # fixed-effect columns, so a random intercept pairs with the fixed one and every
        # random slope is a fixed term
        if '1' in random_terms and not spec.intercept:
            added = '' if spec.random_terms else ', added because the formula has no random term,'
            raise ValueError(
                f'a random intercept{added} needs a fixed intercept: drop the 0 from the fixed '
                'terms or suppress the random intercept with (0 + ... | group)'
            )
        if spec.intercept and '1' not in random_terms:
            raise ValueError(
                'a fixed intercept needs a random intercept: write (1 + ... | group) or '
                'suppress the fixed intercept with 0 +'
            )
        # the model sees the random slopes first, in the random term's order; every output
        # is mapped back to the user's order (`fixed_names`) through `ffx_order`
        slope_indices = resolveFixedIndices([t for t in random_terms if t != '1'], columns)
        missing = [columns[idx] for idx in slope_indices if idx not in fixed_indices]
        if missing:
            raise ValueError(
                f'random slope {", ".join(missing)} is not a fixed term: add it to the fixed terms'
            )
        other_indices = [idx for idx in fixed_indices if idx not in slope_indices]
        intercept = [-1] if spec.intercept else []  # -1 stands for the column of ones
        user_columns = intercept + fixed_indices
        model_columns = intercept + slope_indices + other_indices
        ffx_order = [model_columns.index(idx) for idx in user_columns]
        fixed_names = ['Intercept' if idx < 0 else columns[idx] for idx in user_columns]
        X_parts = [np.ones(len(y)) if idx < 0 else X_pre[:, idx] for idx in model_columns]
        X = np.column_stack(X_parts) if X_parts else np.empty((X_pre.shape[0], 0))

        Z = buildRandomDesign(X_pre, columns, random_terms)
        random_names = self._randomNames(random_terms, columns)
        levels = preprocessed.get('group_levels')  # the preprocessor's label of each group code

        # Build scale info for optional back-transformation to original X units
        x_means_meta = np.asarray(
            preprocessed.get('x_means', np.zeros(X_pre.shape[1])), dtype=float
        )
        x_stds_meta = np.asarray(preprocessed.get('x_stds', np.ones(X_pre.shape[1])), dtype=float)
        y_mean_meta = float(np.asarray(preprocessed.get('y_mean', 0.0)).item())
        y_std_meta = float(np.asarray(preprocessed.get('sd_y', 1.0)).item())

        si_means = [0.0 if idx < 0 else float(x_means_meta[idx]) for idx in user_columns]
        si_stds = [1.0 if idx < 0 else float(x_stds_meta[idx]) for idx in user_columns]
        metadata = _Metadata(
            param_names={'ffx': list(fixed_names), 'sigma_rfx': list(random_names)},
            scale_info=ScaleInfo(
                y_mean=y_mean_meta,
                y_std=y_std_meta,
                x_means=np.array(si_means),
                x_stds=np.array(si_stds),
                param_names=list(fixed_names),
                random_names=list(random_names),
                has_intercept=spec.intercept,
            ),
            group_names=None if levels is None else [str(v) for v in levels],
            ffx_order=ffx_order,
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
        # priors are given in the user's order, by name or by position; the model takes its own
        to_model = [user_columns.index(idx) for idx in model_columns]
        prior_variants = [
            (name, values | {key: values[key][to_model] for key in ('nu_ffx', 'tau_ffx')})
            for name, values in resolvePriors(
                priors,
                fixed_names=fixed_names,
                random_names=random_names,
                likelihood_family=likelihood,
            )
        ]

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
            if prior_name is not None:
                dataset['_router_prior_name'] = np.array(prior_name)
            datasets.append(dataset)
        return datasets, metadata

    def _collateForSelectedSubmodel(
        self, datasets: list[dict[str, np.ndarray]]
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, np.ndarray]]]:
        """The batch of one dataset's priors, padded to the submodel they route to (the same
        for every prior, as routing reads the dimensions only), and the padded datasets."""
        tentative = collateGrouped(datasets[:1])
        routes, _ = self._routeBatch(tentative)
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
        return batch, padded

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
                skip_file_prefixes=INTERNAL_PREFIXES,
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
                    skip_file_prefixes=INTERNAL_PREFIXES,
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
        misfit = self._misfit(batch, i, failures_by_submodel)
        # a batch of one dataset names no row; sample labels its datasets itself
        which = 'the dataset' if batch['X'].shape[0] == 1 else f'dataset {i}'
        raise ValueError(f"{which} is outside the checkpoint's limits: {misfit}")

    def _misfit(
        self,
        batch: Mapping[str, torch.Tensor],
        i: int,
        failures_by_submodel: dict[str, list[str]],
    ) -> str:
        """Why dataset i fits no submodel: a family the checkpoint does not fit and the dimensions
        outside the range its submodels cover together, else the failures of the submodel closest
        to fitting."""
        dims = self._datasetDimensions(batch, i)
        misfits = []
        family = dims['likelihood_family']
        fitted = {
            int(f) for e in self.submodels if (f := routeValue(e, 'likelihood_family')) is not None
        }
        if family is not None and fitted and family not in fitted:
            name = LIKELIHOOD_FAMILIES[family]
            supported = ', '.join(repr(LIKELIHOOD_FAMILIES[f]) for f in sorted(fitted))
            misfits.append(
                f"family '{name}' (supported: {supported}); load Api.from_pretrained('{name}')"
            )
        for label, low, high, min_key, max_key in (
            ('groups', dims['m'], dims['m'], 'min_m', 'max_m'),
            ('rows per group', dims['min_n_i'], dims['max_n_i'], 'min_n', 'max_n'),
            ('rows', dims['n'], dims['n'], None, 'max_n_total'),
            ('fixed effects', dims['d'], dims['d'], 'min_d', 'max_d'),
            ('random effects', dims['q'], dims['q'], 'min_q', 'max_q'),
        ):
            lows = [int(v) for e in self.submodels if (v := routeValue(e, min_key)) is not None]
            highs = [int(v) for e in self.submodels if (v := routeValue(e, max_key)) is not None]
            lo, hi = min(lows, default=None), max(highs, default=None)
            if (lo is not None and low < lo) or (hi is not None and high > hi):
                value = low if low == high else f'{low} to {high}'
                supported = f'up to {hi}' if lo is None else f'{lo} to {hi}'
                misfits.append(f'{value} {label} (supported: {supported})')
        return '; '.join(misfits or min(failures_by_submodel.values(), key=len))

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
                            skip_file_prefixes=INTERNAL_PREFIXES,
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
        prior_ids: Sequence[int | str] | None = None,
        plot_dir: Path | None = None,
        *,
        seed: int | None = None,
        indices: Sequence[int] | None = None,  # deprecated alias of prior_ids, removed in 0.7
    ) -> None:
        """Plot posterior predictive densities for datasets in result.

        y is shown on the standardized scale used internally by the model.  ``seed`` works as in
        ``sample``.
        """
        from metabeta.plotting.predictive import plotPPD as _plotPPD

        if result.diagnostics is None or result.batch is None:
            raise ValueError('result has no posterior predictive; call sample() first')
        pp = result.diagnostics['pp']
        b = int(result.batch['y'].shape[0])
        prior_ids = _renamedArg('indices', indices, 'prior_ids', prior_ids)
        rows = range(b) if prior_ids is None else [_priorRow(result, p) for p in prior_ids]
        with _seeded(seed):
            _plotPPD(pp, result.batch, indices=list(rows), plot_dir=plot_dir)

    def plotParameters(
        self,
        result: RouterResult,
        prior_id: int | str = 0,
        with_prior: bool = True,
        *,
        index: int | None = None,  # deprecated alias of prior_id, removed in 0.7
        **kwargs: Any,
    ) -> Any:
        """Pair-grid of posterior parameter samples in original-scale units.

        Builds the original-scale posterior proposal from result and delegates
        to plotParameters.  If with_prior=True and prior_params are available,
        overlays prior marginal KDEs.
        """
        from metabeta.plotting.parameters import plotParameters as _plotParameters
        from metabeta.utils.evaluation import makePriorPDFs
        from metabeta.utils.results import Proposal

        if result.proposal is None:
            raise ValueError('RouterResult has no proposal; call sample() first')
        if result.scale_info is None:
            raise ValueError('RouterResult has no scale_info')
        index = _priorRow(result, _renamedArg('index', index, 'prior_id', prior_id))

        prop = result.proposal
        si = result.scale_info
        has_eps = prop.has_sigma_eps

        # (S, d), (S, q), (S, q, q)
        ffx_orig, srfx, corr_rfx = _globalDraws(prop, index, si, 'original')
        parts = [ffx_orig, srfx]
        if has_eps:
            parts.append(prop.sigma_eps[index].unsqueeze(-1))  # (S, 1)
        samples_g = torch.cat(parts, dim=-1).unsqueeze(0)      # (1, S, D)

        post_orig = Proposal(
            proposed={
                'global': {'samples': samples_g, 'log_prob': prop.log_prob_g[index : index + 1]},
                'local': {k: v[index : index + 1] for k, v in prop.data['local'].items()},
            },
            has_sigma_eps=has_eps,
            corr_rfx=corr_rfx.unsqueeze(0) if corr_rfx is not None else None,
        )

        param_names = result.param_names or {}
        n_ffx = len(param_names.get('ffx', []))
        n_srfx = len(param_names.get('sigma_rfx', []))
        names = (
            param_names.get('ffx', [])
            + [f'σ_{n}' for n in param_names.get('sigma_rfx', [])]
            + (['σ_ε'] if has_eps else [])
        )

        prior_pdfs = (
            makePriorPDFs(result, batch_index=index)
            if with_prior and result.prior_params is not None
            else None
        )

        return _plotParameters(
            post_orig,
            prior_pdfs=prior_pdfs,
            names=names,
            d_active=n_ffx,
            q_active=n_srfx,
            **kwargs,
        )

    def posteriorSummary(
        self,
        result: RouterResult,
        *,
        ci: float = 0.95,
        prior_id: int | str = 0,
        x_scale: str = 'original',
        batch_index: int | None = None,  # deprecated alias of prior_id, removed in 0.7
    ) -> str:
        """Return a formatted posterior summary table for a sampled RouterResult.

        ``prior_id`` picks the row: the name a prior had in ``sample(priors={name: ...})``,
        or its index.  The last line lists the safeguards, then the fit metric
        (R²/AUC/Deviance), then LOO-NLL and its LOO k̂ when ``sample()`` was called with
        ``loo=True``.  For per-group random effects use ``rfxSummary()``.
        """
        if result.proposal is None:
            raise ValueError('RouterResult has no proposal; call sample() first')
        batch_index = _priorRow(
            result, _renamedArg('batch_index', batch_index, 'prior_id', prior_id)
        )
        d = result.diagnostics or {}
        dims = (
            result.validation[batch_index]['dimensions']
            if result.validation and batch_index < len(result.validation)
            else {}
        )

        def at(values: Any, default: Any = None) -> Any:
            """This row's entry of a per-dataset array, as a Python scalar."""
            return values[batch_index].item() if values is not None else default

        fit_vals = d.get('fit_vals')
        fit = at(fit_vals, d.get('fit'))
        loo_nll = at(d.get('loo_nll_vals'), d.get('loo_nll'))
        pareto_k = at(d.get('pareto_k_vals'), d.get('pareto_k'))
        fit_label = d.get('fit_label', 'fit')
        if fit_vals is not None and fit_label.startswith('Mean '):
            fit_label = fit_label[5:]
        sg = result.safeguards or {}
        psis_k = at(sg.get('psis_k'))
        map_z = at(sg.get('map_z'))
        accept = at(sg.get('accept_rate'))
        ess_min = at(sg.get('ess_min'))
        random_walk = at(sg.get('random_walk_ran'), False)
        return posteriorTable(
            result.proposal,
            result.param_names,
            formula=result.formula,
            priors_str=result.priors_str,
            prior_params=result.prior_params,
            fit=fit,
            fit_label=fit_label,
            loo_nll=loo_nll,
            pareto_k=pareto_k,
            accept_rate=accept,
            ess_min=ess_min,
            random_walk=random_walk,
            psis_k=psis_k,
            psis_k_threshold=sg.get('psis_k_threshold', PSIS_K_THRESHOLD),
            map_z=map_z,
            map_z_threshold=sg.get('map_z_threshold', MAP_Z_THRESHOLD),
            ci=ci,
            batch_index=batch_index,
            scale_info=result.scale_info,
            x_scale=x_scale,
            n=dims.get('n'),
            m=dims.get('m'),
        )

    def posteriorDraws(
        self,
        result: RouterResult,
        *,
        prior_id: int | str = 0,
        x_scale: str = 'original',
    ) -> pd.DataFrame:
        """Return one dataset's posterior draws of the global parameters, one column each.

        The columns carry the summary's names, order and scale: the fixed effects, ``σ_<term>``
        per random effect, ``σ_Residual`` for the Gaussian family and ``Corr(<a>, <b>)`` per
        pair of random effects.  Their means match ``posteriorSummary`` with the same arguments.
        """
        if result.proposal is None:
            raise ValueError('RouterResult has no proposal; call sample() first')
        proposal, bi = result.proposal, _priorRow(result, prior_id)
        x_scale = _resolveXScale(x_scale, result.scale_info)
        ffx, srfx, corr = _globalDraws(proposal, bi, result.scale_info, x_scale)
        names = result.param_names or {}
        ffx_names = names.get('ffx') or [f'ffx_{i}' for i in range(ffx.shape[-1])]
        srfx_names = names.get('sigma_rfx') or [f'rfx_{i}' for i in range(srfx.shape[-1])]
        q = len(srfx_names)
        columns = {name: ffx[:, j] for j, name in enumerate(ffx_names)}
        columns |= {f'σ_{name}': srfx[:, j] for j, name in enumerate(srfx_names)}
        if proposal.has_sigma_eps:
            columns['σ_Residual'] = proposal.sigma_eps[bi].float()
        if corr is not None:
            columns |= {
                f'Corr({srfx_names[j]}, {srfx_names[k]})': corr[:, j, k]
                for j in range(q)
                for k in range(j + 1, q)
            }
        return pd.DataFrame({name: draws.cpu().numpy() for name, draws in columns.items()})

    def groupDraws(
        self,
        result: RouterResult,
        *,
        prior_id: int | str = 0,
        x_scale: str = 'original',
    ) -> pd.DataFrame:
        """Return each group's coefficient draws for the terms with random effects: the fixed
        effect plus the group's deviation, like ``coef`` in lme4.

        One row per draw and one column per (group, term), on the scale of ``rfxSummary``.
        """
        if result.proposal is None:
            raise ValueError('RouterResult has no proposal; call sample() first')
        bi = _priorRow(result, prior_id)
        x_scale = _resolveXScale(x_scale, result.scale_info)
        ffx, _, _ = _globalDraws(result.proposal, bi, result.scale_info, x_scale)  # (S, d)
        rfx = result.proposal.rfx[bi].float()  # (m, S, q)
        if x_scale == 'original':
            rfx = result.scale_info.rfxToOriginal(rfx)
        names = result.param_names or {}
        terms = names['sigma_rfx']  # each has a fixed effect: prepareData refuses others
        groups = result.group_names or [str(i) for i in range(rfx.shape[0])]
        fixed = ffx[:, [names['ffx'].index(term) for term in terms]]  # (S, q)
        coef = fixed + rfx[: len(groups), :, : len(terms)]  # (m, S, q)
        columns = pd.MultiIndex.from_product([groups, terms], names=['group', 'term'])
        return pd.DataFrame(coef.permute(1, 0, 2).flatten(1).cpu().numpy(), columns=columns)

    def posteriorPredictive(
        self,
        result: RouterResult,
        *,
        prior_id: int | str = 0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the observed responses (n,) and one replicate per posterior draw (n, S).

        Both are in the data's units, with the rows grouped as the model sees them rather than
        in the data's order.  ``seed`` works as in ``sample``.
        """
        if result.diagnostics is None or result.batch is None:
            raise ValueError('result has no posterior predictive; call sample() first')
        bi = _priorRow(result, prior_id)
        with _seeded(seed):
            y_rep = result.diagnostics['pp'].sample()[bi][_rowMask(result, bi)]  # (n, S)
        return _observedY(result, bi).cpu().numpy(), _toDataUnits(result, y_rep).cpu().numpy()

    def bayesFactor(self, result: RouterResult, reference: RouterResult) -> pd.DataFrame:
        """Return the Bayes factor (BF) of ``result``'s model over ``reference``'s, per prior.

        Both results come from ``sample(evidence=True)`` on the same data; a result with one
        row pairs with every row of the other.  ``strength`` grades the BF either way after
        Kass and Raftery (1995), and ``PSIS k̂`` is the worse of the two evidence estimates'
        diagnostics.
        """
        if result.evidence is None or reference.evidence is None:
            raise ValueError('bayesFactor needs results of sample(evidence=True)')
        y, y_ref = _observedY(result, 0), _observedY(reference, 0)
        if y.shape != y_ref.shape or not torch.allclose(y.sort().values, y_ref.sort().values):
            raise ValueError(
                'a Bayes factor compares models of the same data; these results differ'
            )
        log_ev, log_ev_ref = result.evidence['log_evidence'], reference.evidence['log_evidence']
        log_bf = log_ev - log_ev_ref  # a one-row result broadcasts
        rows = result if len(log_ev) >= len(log_ev_ref) else reference
        names = [entry.get('prior_name') for entry in rows.validation]
        return pd.DataFrame(
            {
                'log BF': log_bf,
                'BF': np.exp(log_bf),
                'strength': [_bfStrength(b) for b in log_bf],
                'PSIS k̂': np.maximum(result.evidence['psis_k'], reference.evidence['psis_k']),
            },
            index=pd.Index(range(len(names)) if None in names else names, name='prior'),
        )

    def rfxSummary(
        self,
        result: RouterResult,
        *,
        ci: float = 0.95,
        prior_id: int | str = 0,
        x_scale: str = 'original',
        batch_index: int | None = None,  # deprecated alias of prior_id, removed in 0.7
    ) -> str:
        """Return a formatted per-group random effects summary for a sampled RouterResult.

        Produces one table per random-effect component (intercept, slope, …), each under
        its own header, rather than a single wide table with repeated column groups.
        ``x_scale`` has the meaning it has in ``posteriorSummary``: on the original scale a
        slope is per unit of its predictor and the intercept is the group's deviation at x = 0.
        """
        if result.proposal is None:
            raise ValueError('RouterResult has no proposal; call sample() first')
        batch_index = _priorRow(
            result, _renamedArg('batch_index', batch_index, 'prior_id', prior_id)
        )
        fmt = 'github'
        alpha = (1 - ci) / 2
        lo_pct = f'{alpha * 100:g}%'
        hi_pct = f'{(1 - alpha) * 100:g}%'

        names = result.param_names or {}
        x_scale = _resolveXScale(x_scale, result.scale_info)
        _, sigma_rfx, _ = _globalDraws(result.proposal, batch_index, result.scale_info, x_scale)
        rfx = result.proposal.rfx[batch_index].float()  # (m, S, q)
        if x_scale == 'original':
            rfx = result.scale_info.rfxToOriginal(rfx)
        m, _, q_model = rfx.shape
        srfx_names = names.get('sigma_rfx') or [f'rfx_{i}' for i in range(q_model)]
        n_rfx = len(srfx_names)
        g_names = result.group_names or [str(i) for i in range(m)]

        # per-group observation counts from the stored ns array
        ns_row: list[int] | None = None
        if result.ns is not None:
            ns_batch = result.ns[batch_index]  # (max_m,)
            ns_row = [int(ns_batch[gi]) for gi in range(m)]

        # posterior mean σ_rfx per component, used for z-scores
        sigma_rfx_mean = sigma_rfx.mean(dim=0)  # (q,)

        # posterior means per group per component — used for SD ratio
        rfx_means = rfx.mean(dim=1)  # (m, q)

        headers = ['Group']
        if ns_row is not None:
            headers += ['n']
        headers += ['Mean', 'SD', lo_pct, '50.0%', hi_pct, 'z']
        floatfmt: list[str] = (
            ['', 'g', '.3f', '.3f', '.3f', '.3f', '.3f', '.2f']
            if ns_row is not None
            else ['', '.3f', '.3f', '.3f', '.3f', '.3f', '.2f']
        )

        parts = ['Random Effects:']

        for qi, name in enumerate(srfx_names):
            rows = []
            for gi in range(m):
                col = rfx[gi, :, qi]
                mean_val = col.mean().item()
                z = (
                    mean_val / sigma_rfx_mean[qi].item()
                    if sigma_rfx_mean[qi].item() != 0
                    else float('nan')
                )
                row: list[Any] = [g_names[gi] if gi < len(g_names) else str(gi)]
                if ns_row is not None:
                    row.append(ns_row[gi])
                row += [
                    mean_val,
                    col.std().item(),
                    col.quantile(alpha).item(),
                    col.quantile(0.5).item(),
                    col.quantile(1 - alpha).item(),
                    z,
                ]
                rows.append(row)

            emp_sd = rfx_means[:, qi].std().item()
            est_sd = sigma_rfx_mean[qi].item()
            ratio = emp_sd / est_sd if est_sd > 0 else float('nan')

            parts += [
                '',
                f'{name}:',
                tabulate(rows, headers=headers, floatfmt=floatfmt, tablefmt=fmt),
                f'SD ratio = {ratio:.2f}',
            ]

        return '\n'.join(parts)


def _resolveXScale(x_scale: str, scale_info: 'ScaleInfo | None') -> str:
    """The scale a summary can print: 'original' needs the preprocessing's scale info."""
    if x_scale == 'original' and scale_info is None:
        warnings.warn(
            'x_scale="original" requires scale_info; falling back to standardized. '
            'Call router.sample() with fit_preprocessor=True to enable back-transformation.',
            RuntimeWarning,
            skip_file_prefixes=INTERNAL_PREFIXES,
        )
        return 'standardized'
    return x_scale


def _renamedArg(old: str, old_value: Any, new: str, new_value: Any) -> Any:
    """The value of an argument renamed to ``new``, warning while callers still pass ``old``
    (removed in 0.7, #32)."""
    if old_value is None:
        return new_value
    warnings.warn(
        f'{old}= is deprecated and will be removed in 0.7: use {new}=, which also takes a '
        'prior name.',
        FutureWarning,
        skip_file_prefixes=INTERNAL_PREFIXES,
    )
    return old_value


def _asContainer(data: Any) -> tuple[list[Hashable], list[pd.DataFrame]] | None:
    """The labels and frames of a list, tuple or dict of DataFrames; None for one dataset."""
    if isinstance(data, (list, tuple)):
        labels, frames = list(range(len(data))), list(data)
    elif isinstance(data, Mapping) and (
        not data or any(isinstance(v, pd.DataFrame) for v in data.values())
    ):
        labels, frames = list(data), list(data.values())
    else:
        return None
    if not frames:
        raise ValueError('sample needs at least one dataset')
    other = [label for label, frame in zip(labels, frames) if not isinstance(frame, pd.DataFrame)]
    if other:
        raise TypeError(
            f'every dataset of a list, tuple or dict must be a DataFrame; dataset(s) '
            f'{_listed(other)} are not'
        )
    return labels, frames


def _cutChunks(
    items: list[_Prepared], order: list[int], chunk_rows: int
) -> list[list[tuple[int, int, int]]]:
    """Chunks of at most ``chunk_rows`` rows over the items in ``order``, as (item, start, end)
    slices.  A chunk never mixes submodels, and a cut falls on an item boundary unless one
    item alone has more rows than a chunk."""
    chunks: list[list[tuple[int, int, int]]] = []
    rows = chunk_rows  # the open chunk's rows: full, so the first item opens a new one
    for i in order:
        item = items[i]
        if item.rows > chunk_rows:
            chunks += [
                [(i, start, min(start + chunk_rows, item.rows))]
                for start in range(0, item.rows, chunk_rows)
            ]
            rows = chunk_rows
        elif rows + item.rows > chunk_rows or items[chunks[-1][0][0]].routes[0] != item.routes[0]:
            chunks.append([(i, 0, item.rows)])
            rows = item.rows
        else:
            chunks[-1].append((i, 0, item.rows))
            rows += item.rows
    return chunks


def _sliceRows(out: Any, start: int, end: int) -> Any:
    """Rows [start, end) of a chunk's output: the proposal and its per-row arrays (numpy or
    torch, nested dicts included); scalars and None pass through."""
    if isinstance(out, Proposal):
        return out.slice_b(start, end)
    if isinstance(out, dict):
        return {key: _sliceRows(value, start, end) for key, value in out.items()}
    if (torch.is_tensor(out) or isinstance(out, np.ndarray)) and out.ndim:
        return out[start:end]
    return out


def _joinRows(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate the chunks' per-row arrays (numpy or torch, nested dicts included) along
    the row axis; scalars and None are the same in every chunk and come from the first."""
    out = {}
    for key, first in parts[0].items():
        values = [part[key] for part in parts]
        if isinstance(first, dict):
            out[key] = _joinRows(values)
        elif torch.is_tensor(first) and first.ndim:
            out[key] = torch.cat(values)
        elif isinstance(first, np.ndarray) and first.ndim:
            out[key] = np.concatenate(values)
        else:
            out[key] = first
    return out


def _rowLabels(items: list[_Prepared], *, with_dataset: bool) -> list[Any]:
    """How the warnings name each row, in the items' order: by its dataset label, or by the
    pair of dataset label and prior (its name, else its index) when a dataset has several
    priors.  A single dataset drops the dataset half, so its rows read as its priors."""
    several = any(item.rows > 1 for item in items)
    labels = []
    for item in items:
        for entry in item.validation:
            prior = entry.get('prior_name')
            prior = entry['index'] if prior is None else prior
            if not with_dataset:
                labels.append(prior)
            else:
                labels.append((item.label, prior) if several else item.label)
    return labels


def _flagRows(results: list[RouterResult], attr: str) -> dict[str, Any] | None:
    """The results' safeguards or evidence joined row by row for the warnings; ``ess`` is
    (rows, globals) and differs in width across submodels, and no warning reads it."""
    parts = [getattr(result, attr) for result in results]
    if parts[0] is None:
        return None
    return _joinRows([{k: v for k, v in part.items() if k != 'ess'} for part in parts])


def _listed(labels: list[Any]) -> str:
    """A list of labels, cut after ``LISTED_ROWS`` with the count."""
    if len(labels) <= LISTED_ROWS:
        return str(labels)
    return f'{str(labels[:LISTED_ROWS])[:-1]}, … {len(labels)} in all]'


def _priorRow(result: RouterResult, prior_id: int | str) -> int:
    """The row of ``prior_id`` in ``result``: its index, or the name the prior had in
    ``sample(priors={name: ...})`` (#32)."""
    if not isinstance(prior_id, str):
        return int(prior_id)
    names = [entry.get('prior_name') for entry in result.validation]
    if prior_id not in names:
        known = [name for name in names if name is not None]
        raise KeyError(f'no prior named {prior_id!r} in this result; its priors are {known}')
    return names.index(prior_id)


@contextlib.contextmanager
def _seeded(seed: int | None) -> Iterator[None]:
    """Seed torch inside the block, then give the caller back its CPU and CUDA random state
    (#32)."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
        torch.manual_seed(seed)
        yield


def _rowMask(result: RouterResult, row: int) -> torch.Tensor:
    """Which (group, slot) cells of a batch row hold data, as a boolean (m, n_i) mask."""
    return result.batch['mask_n'][row].bool()


def _toDataUnits(result: RouterResult, y: torch.Tensor) -> torch.Tensor:
    """Responses on the data's scale: the Gaussian response is standardized in the batch, and
    the scale info of the other families holds mean 0 and SD 1."""
    si = result.scale_info
    return y if si is None else y * si.y_std + si.y_mean


def _observedY(result: RouterResult, row: int) -> torch.Tensor:
    """A batch row's observed responses (n,) in the data's units, grouped as in the batch."""
    return _toDataUnits(result, result.batch['y'][row][_rowMask(result, row)].float())


# Kass and Raftery (1995), JASA 90:773, p. 777: upper BF bounds of each grade of evidence;
# 'weak' shortens their "not worth more than a bare mention"
BF_STRENGTH = ((3, 'weak'), (20, 'positive'), (150, 'strong'))


def _bfStrength(log_bf: float) -> str:
    """How strongly a Bayes factor favors one of its two models, either way (#33)."""
    bf = math.exp(abs(log_bf))
    return next((label for bound, label in BF_STRENGTH if bf < bound), 'very strong')


def _globalDraws(
    proposal: 'Proposal', batch_index: int, scale_info: 'ScaleInfo | None', x_scale: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One dataset's fixed-effect, random-effect SD and correlation draws on the scale a summary
    prints, still padded as in the proposal (#31)."""
    ffx = proposal.ffx[batch_index].float()  # (S, d)
    srfx = proposal.sigma_rfx[batch_index].float()  # (S, q)
    corr = proposal.corr_rfx
    corr_b = corr[batch_index].float() if corr is not None else None  # (S, q, q)
    if x_scale == 'original':
        ffx = scale_info.to_original_scale(ffx)
        srfx, corr_b = scale_info.sdCorrToOriginal(srfx, corr_b)
    return ffx, srfx, corr_b


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


SIGMA_PRIOR_SYMBOL = {'halfnormal': 'HN', 'halfstudent': 'HT₅', 'exponential': 'Exp'}


def _buildPriorLines(
    ffx_names: list[str],
    srfx_names: list[str],
    prior_params: dict[str, np.ndarray],
    batch_index: int,
    scale_info: 'ScaleInfo | None' = None,
    x_scale: str = 'original',
) -> list[str]:
    """Build per-parameter prior description lines for the summary header.

    Priors are stored on the fully-standardized (★) scale inside the model.
    This function rescales them to match whatever scale is being displayed:

    - Standardized (x_scale='standardized', standardized X, original y): multiply ν, τ by σ_y.
      This gives the effective prior on β★_k × σ_y (Δy per SD of predictor).
      Intercept location remains 0 (model is centred at mean y).

    - Original (x_scale='original', the default): each line describes the displayed parameter
      (ScaleInfo.ffxPriorToOriginal): a slope's prior is scaled by σ_y/σ_{x_k}, and the
      intercept at X = 0 collects the slopes' priors.  A random slope's SD prior is scaled the
      same way.  Where the pushforward has no closed form (a Student-t intercept; the random
      intercept's SD and the correlations once a random slope sits on a centered predictor),
      the line states the specified prior and the predictors at whose means it applies.
    """
    from metabeta.utils.constants import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF

    lines: list[str] = []
    n_ffx, n_rfx = len(ffx_names), len(srfx_names)

    # the summary prints the draws in y units on both scales (Api._rescaleProposal), so the priors
    # follow; the original scale also maps them to X units and moves the intercept to X = 0
    rescale = x_scale == 'original' and scale_info is not None
    y_std = scale_info.y_std if scale_info is not None else 1.0

    if 'tau_ffx' in prior_params and 'nu_ffx' in prior_params:
        tau = prior_params['tau_ffx'][batch_index, :n_ffx].astype(float)
        nu = prior_params['nu_ffx'][batch_index, :n_ffx].astype(float)
        fam_id = int(prior_params['family_ffx'][batch_index]) if 'family_ffx' in prior_params else 0
        fam = FFX_FAMILIES[fam_id] if fam_id < len(FFX_FAMILIES) else 'normal'
        locs, scales, notes = nu * y_std, tau * y_std, [''] * n_ffx
        if rescale:
            locs, scales = scale_info.ffxPriorToOriginal(nu, tau)
            centered = [n for n, mu in zip(ffx_names[1:], scale_info.x_means[1:]) if mu != 0]
            if scale_info.has_intercept and fam != 'normal' and centered:
                # a sum of Student-t terms is no Student-t: state the specified intercept prior
                locs[0], scales[0] = nu[0] * y_std + scale_info.y_mean, tau[0] * y_std
                notes[0] = _atMean(centered)
        symbol = 'N' if fam == 'normal' else 't₅'
        for name, loc_j, scale_j, note in zip(ffx_names, locs, scales, notes):
            loc = f'{loc_j:.4g}' if abs(loc_j) > 1e-9 else '0'
            lines.append(f'  {name} ~ {symbol}({loc}, {scale_j:.4g}){note}')

    if 'tau_rfx' in prior_params and n_rfx > 0:
        tau_r = prior_params['tau_rfx'][batch_index, :n_rfx]
        fam_id = (
            int(prior_params['family_sigma_rfx'][batch_index])
            if 'family_sigma_rfx' in prior_params
            else 0
        )
        fam = SIGMA_FAMILIES[fam_id] if fam_id < len(SIGMA_FAMILIES) else 'halfnormal'
        # only the random intercept's row of ScaleInfo.rfxMap mixes effects: its SD and the
        # correlations keep the specified prior, at the means of the random slopes' predictors
        at_mean = ''
        if rescale and 'Intercept' in scale_info.random_names:
            i = scale_info.random_names.index('Intercept')
            row = scale_info.rfxMap()[i]
            at_mean = _atMean(
                [n for k, n in enumerate(scale_info.random_names) if k != i and row[k] != 0]
            )
        for j, name in enumerate(srfx_names):
            # σ_rfx is always displayed in y units; rescale τ_rfx in original mode, where a
            # slope SD is divided by its predictor's SD (the intercept's x_std is 1)
            tau_j = float(tau_r[j]) * y_std
            if rescale:
                tau_j /= float(scale_info.x_stds[scale_info.param_names.index(name)])
            note = at_mean if name == 'Intercept' else ''
            lines.append(f'  σ_{name} ~ {SIGMA_PRIOR_SYMBOL[fam]}({tau_j:.4g}){note}')
        if 'eta_rfx' in prior_params and n_rfx > 1:
            eta = float(prior_params['eta_rfx'][batch_index])
            if eta > 0:
                lines.append(f'  Corr ~ LKJ({eta:.2g}){at_mean}')

    if 'tau_eps' in prior_params:
        tau_e = float(np.asarray(prior_params['tau_eps']).ravel()[batch_index]) * y_std
        fam_id = (
            int(np.asarray(prior_params['family_sigma_eps']).ravel()[batch_index])
            if 'family_sigma_eps' in prior_params
            else 1
        )
        fam = SIGMA_FAMILIES[fam_id] if fam_id < len(SIGMA_FAMILIES) else 'halfstudent'
        lines.append(f'  σ_Residual ~ {SIGMA_PRIOR_SYMBOL[fam]}({tau_e:.4g})')

    return lines


def _atMean(predictors: list[str]) -> str:
    """Where a prior line without closed form applies: at the means of `predictors`."""
    return f' at mean {", ".join(predictors)}' if predictors else ''


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
    accept_rate: float | None = None,
    ess_min: float | None = None,
    random_walk: bool = False,
    psis_k: float | None = None,
    psis_k_threshold: float = PSIS_K_THRESHOLD,
    map_z: float | None = None,
    map_z_threshold: float = MAP_Z_THRESHOLD,
    ci: float = 0.95,
    batch_index: int = 0,
    scale_info: 'ScaleInfo | None' = None,
    x_scale: str = 'original',
    n: int | None = None,
    m: int | None = None,
) -> str:
    """Format a posterior summary table in the style of lme4 summaries.

    Sections: Fixed Effects (mean, SD, CI, P(>0), contraction), Standard
    Deviations (sigma_rfx + residual), and Correlations (posterior mean, when
    q > 1).  For per-group random effects use ``Api.rfxSummary()``.
    Contraction requires ``prior_params`` (populated automatically by
    ``Api.sample()``).  LOO-NLL and LOO k̂ require ``loo=True``.
    """
    fmt = 'github'
    alpha = (1 - ci) / 2
    lo_pct = f'{alpha * 100:g}%'
    hi_pct = f'{(1 - alpha) * 100:g}%'

    x_scale = _resolveXScale(x_scale, scale_info)
    ffx, srfx, corr_b = _globalDraws(proposal, batch_index, scale_info, x_scale)
    S = ffx.shape[0]

    names = param_names or {}
    ffx_names = names.get('ffx') or [f'ffx_{i}' for i in range(ffx.shape[-1])]
    srfx_names = names.get('sigma_rfx') or [f'rfx_{i}' for i in range(srfx.shape[-1])]

    def _col_stats(samples: torch.Tensor, j: int) -> tuple[float, float, float, float, float]:
        col = samples[:, j]
        return (
            col.mean().item(),
            col.std().item(),
            col.quantile(alpha).item(),
            col.quantile(0.5).item(),
            col.quantile(1 - alpha).item(),
        )

    # prior SD per ffx parameter for contraction = 1 - Var(post) / Var(prior)
    # τ is stored in standardized (★) space; rescale to match the displayed units.
    from metabeta.utils.constants import FFX_FAMILIES, STUDENT_DF

    tau_arr = None
    _ffx_prior_var_multiplier = 1.0  # 1.0 for Normal; df/(df-2) for Student-t
    if prior_params is not None and 'tau_ffx' in prior_params and 'nu_ffx' in prior_params:
        n_ffx_active = len(ffx_names)
        tau_arr = prior_params['tau_ffx'][batch_index, :n_ffx_active].astype(float)
        if x_scale == 'original':
            nu_arr = prior_params['nu_ffx'][batch_index, :n_ffx_active]
            tau_arr = scale_info.ffxPriorToOriginal(nu_arr, tau_arr)[1]
        elif scale_info is not None:
            tau_arr = tau_arr * scale_info.y_std
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
    if n is not None or m is not None:
        size_parts = []
        if n is not None:
            size_parts.append(f'n = {n}')
        if m is not None:
            size_parts.append(f'm = {m}')
        parts.append('   '.join(size_parts))
    if x_scale == 'standardized':
        parts.append('Scale:    standardized covariates')
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
    prior_header = 'Priors:'
    if prior_lines:
        parts.append(prior_header)
        parts.extend(prior_lines)
    elif priors_str:
        parts.append(f'{prior_header}   {priors_str}')
    if parts:
        parts.append('')

    # ── Fixed Effects ─────────────────────────────────────────────────────────
    show_contr = tau_arr is not None
    ffx_headers = ['', 'Mean', 'SD', lo_pct, '50.0%', hi_pct, 'P(>0)']
    if show_contr:
        ffx_headers.append('Contr.')
    ffx_rows = []
    for j, name in enumerate(ffx_names):
        mean, sd, lo, med, hi = _col_stats(ffx, j)
        p_pos = (ffx[:, j] > 0).float().mean().item()
        row: list[Any] = [name, mean, sd, lo, med, hi, p_pos]
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
        mean, sd, lo, med, hi = _col_stats(srfx, j)
        srfx_rows.append([name, mean, sd, lo, med, hi])

    if proposal.has_sigma_eps:
        seps = proposal.sigma_eps[batch_index].float()  # (S,)
        srfx_rows.append(
            [
                'Residual',
                seps.mean().item(),
                seps.std().item(),
                seps.quantile(alpha).item(),
                seps.quantile(0.5).item(),
                seps.quantile(1 - alpha).item(),
            ]
        )

    parts += [
        'Standard Deviations:',
        tabulate(
            srfx_rows,
            headers=['', 'Mean', 'SD', lo_pct, '50.0%', hi_pct],
            floatfmt='.3f',
            tablefmt=fmt,
        ),
    ]

    # ── Correlations ──────────────────────────────────────────────────────────
    n_rfx = len(srfx_names)
    if n_rfx > 1:
        if corr_b is not None:
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
    # safeguards first, then the fit metrics
    footer_parts = [f'draws = {S}']
    if accept_rate is not None:
        footer_parts.append(f'acceptance = {accept_rate:.2f}')
    if ess_min is not None:
        note = ' (random-walk moves ran)' if random_walk else ''
        footer_parts.append(f'min ESS = {ess_min:.0f}{note}')
    if psis_k is not None:
        note = '' if psis_k <= psis_k_threshold else ' (⚠ unreliable — consider MCMC)'
        footer_parts.append(f'PSIS k̂ = {psis_k:.3f}{note}')
    if map_z is not None:
        note = '' if map_z <= map_z_threshold else ' (⚠ disagrees with MAP — consider MCMC)'
        footer_parts.append(f'MAP z = {map_z:.1f}{note}')
    if fit is not None:
        footer_parts.append(f'{fit_label} = {fit:.3f}')
    if loo_nll is not None:
        footer_parts.append(f'LOO-NLL = {loo_nll:.3f}')
    if pareto_k is not None:
        footer_parts.append(f'LOO k̂ = {pareto_k:.3f}')
    parts += ['', '   '.join(footer_parts)]

    return '\n'.join(parts)


CheckpointRouter = Api
