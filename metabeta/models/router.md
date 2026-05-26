# Router Plan

## Summary

Build a model router that packages multiple trained checkpoints into a
weights-only joint checkpoint and routes incoming datasets to the smallest
compatible submodel.

Current status:

- Core router lives in `metabeta/models/router.py`.
- Shared router utilities and checkpoint joining live in `metabeta/utils/router.py`.
- A dataset/prior row is handled by exactly one compatible model, selected from
  checkpoint metadata.
- MoE and multi-checkpoint ensembles remain deferred.

## Public Interface

- `[done] joinCheckpoints(checkpoints, output_path, *, prefixes=None, ids=None) -> Path`
  - Accept checkpoint directories, checkpoint files, or named checkpoint records.
  - Load only checkpoint metadata and `model_state`.
  - Write a versioned joint checkpoint containing submodel entries.
  - Exclude optimizer state, RNG state, epoch counters, and trainer runtime state.
  - Implemented in `metabeta/utils/router.py` and re-exported from
    `metabeta/models/router.py` for compatibility.

- `[partial] Router(joint_checkpoint, *, device='cpu', batch_size=None)`
  - Load submodel metadata from the joint checkpoint.
  - Lazily build each `Approximator` from stored `model_cfg`.
  - Move models to device, call `eval()`, and optionally compile.
  - Maintain a routing table derived from stored training/data config metadata.
  - TODO: add `compile_model` support if needed.

- `[partial] Router.sample(data, *, n_samples, **prepare_kwargs)`
  - Normalize input data.
  - Validate dimensions and batch schema.
  - Route each dataset/prior row to the smallest compatible submodel.
  - Run posterior sampling via `Approximator.estimate`.
  - Return `RouterResult(proposal, routes, validation)`.
  - TODO: diagnostics.
  - TODO: mixed-submodel reassembly.

- `[partial] Router.forward(data, **prepare_kwargs)`
  - Run the forward/evaluation path for supplied parameters.
  - Validate parameter shape against the selected submodel and active masks.
  - TODO: add a clearer public `log_prob()` wrapper if this should be exposed as
    inference API rather than training-style `forward()`.

## Joint Checkpoint Format

`[done]` Uses a plain `torch.save` payload:

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

`[done]` The router now has explicit data stages:

1. Tabular input: `pandas.DataFrame` or `.parquet`.
2. Preprocessed numpy dict: output from `DataPreprocessor`.
3. Model dataset dict: dataloader-ready numpy arrays with formula-derived
   `X/Z`, priors, placeholders, and metadata.
4. Collated dataloader batch: tensor dict consumed by `Approximator`.

Accepted inputs:

- Existing `Dataloader`: use directly.
- `.npz` path: wrap in `Dataloader`.
- `.parquet` path: load with `pandas.read_parquet`, then run the tabular path.
- Raw `DataFrame`: accepted only when caller supplies a fitted
  `DataPreprocessor` or opts into fitting with `fit_preprocessor=True`.
- Raw preprocessed dict: converted through formula/prior handling into a model
  dataset and then collated.
- Model dataset dict: collated directly.
- Collated batch dict: accept when it has the dataloader schema.

The router does not silently fit preprocessing for raw tabular data. It raises
unless the caller passes a fitted preprocessor or `fit_preprocessor=True`.

Implemented `prepareData(..., stage=...)` stages:

- `stage='preprocessed'`: stop after tabular preprocessing.
- `stage='dataset'`: stop after model dataset construction.
- `stage='batch'`: return a collated tensor batch.

TODO:

- Support multiple raw/preprocessed datasets in one call while preserving
  `source_index`.
- Consider a first-class data-stage result object instead of returning different
  types by `stage`.

## Formula And Priors

`[done]` Formula handling is implemented in the model-dataset stage.

Supported formula subset:

- Additive fixed terms.
- Optional intercept via `1`, `0`, or `-1`.
- One lme4-style random-effect block: `(1 + x1 + x2 | group)`.
- `q` up to 5.
- If no random block is supplied:
  - `q=None` defaults to `(1 | group)`.
  - `q=k` defaults to `(1 + first k-1 preprocessed predictors | group)`.

Formula effects:

- Fixed terms select and reorder columns in `X`.
- Random terms select and reorder columns in `Z`.
- Categorical term names can match all dummy columns by prefix.

`[done]` Priors are resolved after formula construction, so they validate against
the final ordered `X/Z` names.

Supported prior inputs:

- `None`: Bambi-style defaults via `bambiDefaultPriors`.
- Canonical fit-style keys:
  - `nu_ffx`
  - `tau_ffx`
  - `family_ffx`
  - `tau_rfx`
  - `family_sigma_rfx`
  - `tau_eps`
  - `family_sigma_eps`
  - `eta_rfx`
- Term-based schema:
  - `fixed`
  - `random_sd`
  - `sigma_eps`
  - `corr_rfx`
- Multiple priors per dataset:
  - sequence of prior mappings
  - named mapping, e.g. `{'weak': {...}, 'tight': {...}}`

Multiple priors are implemented by expanding one dataset into one collated batch
row per prior. The model sees an ordinary batch; no new prior dimension is added.

Current prior limitations:

- Per-term fixed/random scales and locations are supported.
- Per-term prior families are rejected because the current model has one
  `family_ffx` and one `family_sigma_rfx` per batch row.
- Multiple datasets with dataset-specific prior sets are not implemented yet.

## Validation And Routing

`[partial]` Validation runs before inference and produces a structured report. Raise
`ValueError` by default for hard failures.

Implemented dimension coverage checks:

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

Implemented schema checks:

- Required dataloader fields are present: `X`, `Z`, `y`, `groups` or grouped
  tensors, `ns`, `m`, `n`, masks, priors, and likelihood/family fields.
- Numeric tensors are finite where masks are active.
- `n == ns.sum()` over active groups.
- Batch padding matches selected submodel before model execution.

Still TODO:

- Group indices are contiguous and ordered for all path-backed collections.
- Masks agree with active dimensions and counts beyond the current basic checks.
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
- TODO: If stats are malformed or dimensionally incompatible, raise before sampling.

Routing policy:

- `[done]` Evaluate all submodels compatible with the dataset/prior row.
- Select the smallest compatible model by `(max_d, max_q, max_m, max_n_total)`.
- `[todo]` Split mixed batches by selected submodel.
- `[todo]` Run each sub-batch independently.
- `[todo]` Reassemble outputs in original dataset order.

Current limitation:

- Mixed-submodel batches still raise `NotImplementedError`.

## Inference Results

`[partial]` Return a small result object with:

- `proposal`: posterior samples.
- `routes`: dataset index to submodel id.
- `validation`: structured validation report.
- `[todo] diagnostics`: optional diagnostics payload.

Validation reports now include prior metadata when available:

- `prior_index`
- `source_index`
- `prior_name`

Posterior samples should keep the existing `Proposal` shape conventions. If
submodels have different padded `d/q`, outputs must be unpadded to active dataset
dimensions before reassembly or padded to a common result shape with masks.

## Diagnostics

`[todo]` Diagnostics are opt-in:

- Posterior predictive checks using existing `metabeta.evaluation.predictive` and
  `metabeta.evaluation.summary` utilities.
- Posterior sample plots via `metabeta.plotting.parameters`.
- Parameter summary table with marginal means, medians, and 95% credible intervals.

Diagnostics should run after routing and sampling, and should respect original
dataset order.

## Tests

- `[done]` Joint checkpoint writer excludes optimizer and RNG state.
- `[done]` Joint checkpoint preserves model config, data config, and routing metadata.
- `[partial]` Router loads submodels lazily and leaves them in eval mode.
- `[done]` Router selects the smallest compatible model for `d/q/m/n_i/n` bands.
- `[done]` Router rejects datasets outside `max_d`, `max_q`, `max_m`, `max_n_total`, or
  likelihood family.
- `[done]` Router rejects datasets violating `min_bg_df` and `min_within_df`.
- `[done]` Preprocessed numpy dict from `math__grp_group.npz` converts through formula
  and default priors.
- `[done]` `math.parquet` converts through tabular preprocessing.
- `[done]` Formula random effects work up to `q=5`.
- `[done]` Canonical fit-style priors are accepted.
- `[done]` Multiple named term-based priors expand one dataset into multiple batch rows.
- `[todo]` `.npz` input is wrapped in `Dataloader`; existing `Dataloader` is reused.
- `[todo]` Batches with precomputed `stats` and batches without `stats` both run.
- `[todo]` `sample()` returns valid `Proposal` shapes and restores original dataset order.
- `[todo]` `log_prob()` accepts correctly shaped parameter inputs and rejects malformed ones.
- `[todo]` Invalid priors, named prior collections, `q > 5`, and term-family mismatch errors.
- `[todo]` Broader test run once required local fixture data exists.

## Assumptions

- V1 routes each dataset/prior row to one model; MoE is deferred.
- Routing uses checkpoint metadata from training configs and model configs.
- Missing soft bounds are warnings; missing hard bounds required for compatibility
  are errors.
- Raw dataframe preprocessing is explicit and opt-in; the router does not silently
  fit preprocessing state.
- The joint checkpoint format is internal and versioned.
