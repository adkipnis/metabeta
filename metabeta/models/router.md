# Router Plan

## Summary

Build a model router in `metabeta/models/router.py` that packages multiple trained
checkpoints into a weights-only joint checkpoint and routes incoming datasets to the
smallest compatible submodel.

The implementation should start without MoE. A dataset is handled by exactly one
compatible model, selected from checkpoint metadata. Multi-checkpoint ensembles can
be layered on later using the existing proposal-joining utilities.

## Public Interface

- `joinCheckpoints(checkpoints, output_path, *, prefixes=None, ids=None) -> Path`
  - Accept checkpoint directories, checkpoint files, or named checkpoint records.
  - Load only checkpoint metadata and `model_state`.
  - Write a versioned joint checkpoint containing submodel entries.
  - Exclude optimizer state, RNG state, epoch counters, and trainer runtime state.

- `CheckpointRouter(joint_checkpoint, *, device='cpu', batch_size=None, compile_model=False)`
  - Load all submodels from the joint checkpoint.
  - Build each `Approximator` from stored `model_cfg`.
  - Move models to device, call `eval()`, and optionally compile.
  - Maintain a routing table derived from stored training/data config metadata.

- `CheckpointRouter.sample(data, *, n_samples, diagnostics=False, **kwargs)`
  - Normalize input data.
  - Validate dimensions, preprocessing, and analytical stats availability.
  - Route each dataset to the smallest compatible submodel.
  - Run posterior sampling via `Approximator.estimate`.
  - Return a router result containing the `Proposal`, routing report, validation
    report, and optional diagnostics.

- `CheckpointRouter.log_prob(data, params, **kwargs)`
  - Run the forward/evaluation path for supplied parameters.
  - Validate parameter shape against the selected submodel and active masks.

## Joint Checkpoint Format

Use a plain `torch.save` payload:

- `_version`: integer format version.
- `created_at`: ISO timestamp.
- `submodels`: list of entries with:
  - `id`: stable submodel id.
  - `source`: original checkpoint path and prefix.
  - `trainer_cfg`: saved training config, if present.
  - `data_cfg`: saved training data config.
  - `model_cfg`: full `ApproximatorConfig.to_dict()` payload.
  - `model_state`: weights only.
  - `routing`: normalized routing metadata.

Routing metadata should include `likelihood_family`, `max_d`, `max_q`, `min_d`,
`min_q`, `min_m`, `max_m`, `min_n`, `max_n`, `max_n_total`, `min_bg_df`,
`min_within_df`, `shape_profile`, and `model_id` when available.

If older checkpoints lack some fields, infer what is possible from `data_cfg`,
`trainer_cfg`, and `model_cfg`, and leave missing non-critical bounds as `None`.
Hard routing limits must never be guessed.

## Input Normalization

Accepted inputs:

- Existing `Dataloader`: use directly.
- `.npz` path: wrap in `Dataloader`, passing model-compatible padding dimensions
  after routing bounds are known.
- Collated batch dict: accept when it has the dataloader schema.
- Raw preprocessed dict: defer for v1 unless a safe conversion helper already exists.
- Raw `DataFrame`: out of scope for v1 unless the caller supplies a fitted
  `DataPreprocessor`.

The router should not silently preprocess arbitrary raw data. It should report
whether data is already dataloader-ready, path-backed, or unsupported.

## Validation And Routing

Validation should run before inference and produce a structured report. Raise
`ValueError` by default for hard failures.

Dimension coverage checks:

- Fixed effects: active `d <= max_d`.
- Random effects: active `q <= max_q`.
- Groups: `min_m <= m <= max_m` when bounds are known.
- Group sizes: every active `n_i` satisfies `min_n <= n_i <= max_n` when bounds
  are known.
- Total rows: `n <= max_n_total` and `n == ns.sum()` for active groups.
- Between-group degrees of freedom: `m >= max(d, q * (q + 1) // 2) + min_bg_df`
  when `min_bg_df` is known.
- Within-group degrees of freedom: every active `n_i >= q + min_within_df` when
  `min_within_df` is known.
- Likelihood: `likelihood_family` must match the submodel.

Preprocessing/schema checks:

- Required dataloader fields are present: `X`, `Z`, `y`, `groups` or grouped
  tensors, `ns`, `m`, `n`, masks, priors, and likelihood/family fields.
- Group indices are contiguous and ordered for path-backed collections.
- Masks agree with active dimensions and counts.
- Numeric tensors are finite where masks are active.
- Continuous targets are standardized only when the dataset metadata indicates a
  continuous likelihood; binary/count targets must remain on their natural scale.
- Predictor preprocessing should be consistent with `DataPreprocessor` constraints:
  no heavy missingness after preprocessing, no active constant columns, no active
  high-correlation columns beyond the configured threshold, rare categorical
  levels already handled, and count-like numeric predictors transformed according
  to the fitted preprocessing state.

Analytical stats checks:

- If `batch['stats']` exists, use it.
- If stats are absent and the selected model uses analytical context, let
  `Approximator.summarize()` compute stats online.
- If stats are malformed or dimensionally incompatible, raise before sampling.

Routing policy:

- Evaluate all submodels compatible with the dataset.
- Select the smallest compatible model by `(max_d, max_q, max_m, max_n_total)`.
- Split mixed batches by selected submodel.
- Run each sub-batch independently.
- Reassemble outputs in original dataset order.

## Inference Results

Return a small result object with:

- `proposal`: posterior samples.
- `routes`: dataset index to submodel id.
- `validation`: structured validation report.
- `diagnostics`: optional diagnostics payload.

Posterior samples should keep the existing `Proposal` shape conventions. If
submodels have different padded `d/q`, outputs must be unpadded to active dataset
dimensions before reassembly or padded to a common result shape with masks.

## Diagnostics

Diagnostics are opt-in:

- Posterior predictive checks using existing `metabeta.evaluation.predictive` and
  `metabeta.evaluation.summary` utilities.
- Posterior sample plots via `metabeta.plotting.parameters`.
- Parameter summary table with marginal means, medians, and 95% credible intervals.

Diagnostics should run after routing and sampling, and should respect original
dataset order.

## Tests

- Joint checkpoint writer excludes optimizer and RNG state.
- Joint checkpoint preserves model config, data config, and routing metadata.
- Router loads all submodels and leaves them in eval mode.
- Router selects the smallest compatible model for `d/q/m/n_i/n` bands.
- Router rejects datasets outside `max_d`, `max_q`, `max_m`, `max_n_total`, or
  likelihood family.
- Router rejects datasets violating `min_bg_df` and `min_within_df`.
- `.npz` input is wrapped in `Dataloader`; existing `Dataloader` is reused.
- Batches with precomputed `stats` and batches without `stats` both run.
- `sample()` returns valid `Proposal` shapes and restores original dataset order.
- `log_prob()` accepts correctly shaped parameter inputs and rejects malformed ones.

## Assumptions

- V1 routes each dataset to one model; MoE is deferred.
- Routing uses checkpoint metadata from training configs and model configs.
- Missing soft bounds are warnings; missing hard bounds required for compatibility
  are errors.
- Raw dataframe preprocessing is not part of v1.
- The joint checkpoint format is internal and versioned.
