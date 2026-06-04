# metabeta

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/env-uv-6b47ff)](https://docs.astral.sh/uv/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey)](LICENSE)

**Amortized Bayesian inference for hierarchical regression models.**

`metabeta` learns to approximate posterior distributions for generalized linear mixed
models (GLMMs). Instead of fitting a new Markov chain or variational approximation from
scratch for every dataset, it trains neural posterior estimators on many simulated
hierarchical datasets and then reuses the learned inference procedure at test time.

The project sits between statistical modeling and foundation-model-style amortization:
it exposes a practical API for tabular grouped data, while keeping the simulator,
architecture, training loop, and evaluation code available for inspection and research.

## Why it exists

Hierarchical models are expressive and widely used, but posterior inference can be slow,
sensitive to geometry, and hard to scale across many similar analyses. `metabeta` targets
the repeated-analysis setting: train once over a broad prior and data-generating process,
then produce posterior samples for new grouped datasets with a neural forward pass.

The repository is designed to support three use cases:

- **Fast applied modeling** for users who want approximate Bayesian GLMM posteriors from
  a dataframe and a formula-like specification.
- **Research on amortized inference** for reviewers and researchers who want to inspect
  the simulator, priors, architectures, calibration diagnostics, and benchmark scripts.
- **Reusable engineering components** for synthetic data generation, routed checkpoint
  packaging, posterior evaluation, and post-hoc refinement.

## At a glance

- Supports normal, Bernoulli, and Poisson hierarchical likelihood families.
- Handles fixed effects, random effects, group-level scales, correlations, and Gaussian
  residual scale parameters where applicable.
- Uses permutation-aware set summaries over observations and groups.
- Uses conditional normalizing flows for global and local posterior factors.
- Can condition on weakly informative prior choices for fixed effects and variance terms.
- Includes simulation, training, evaluation, plotting, analytical GLMM baselines, and
  post-hoc correction utilities.
- Provides routed Hugging Face checkpoint loading via `Api.from_pretrained(...)`.

## Quick start

Install from source with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/adkipnis/metabeta.git
cd metabeta
uv sync
uv pip install -e .
```

Run posterior inference on a grouped dataframe:

```python
import pandas as pd

from metabeta.models.api import Api

df = pd.read_parquet("metabeta/datasets/from-r/parquet/math.parquet")

mb = Api.from_pretrained("normal")
result = mb.sample(
    df,
    formula="y ~ meanses + ses + minority + sex + (1 + ses | group)",
    n_samples=1000,
    diagnostics=True,
)

print(mb.posteriorSummary(result, x_scale="original"))
print(mb.rfxSummary(result))
```

`Api.from_pretrained(...)` downloads a routed checkpoint from the Hugging Face Hub on
first use and caches it locally. The available likelihood families are `"normal"`,
`"bernoulli"`, and `"poisson"`.

## Modeling interface

The high-level API accepts several input stages:

- pandas dataframes and parquet files,
- preprocessed numpy dictionaries,
- model-ready dataset dictionaries,
- collated tensor batches from `metabeta.utils.dataloader.Dataloader`.

Formula support is intentionally narrow and explicit. The current router supports
additive fixed effects and one lme4-style random-effect block, for example:

```text
y ~ x1 + x2 + (1 + x1 | group)
```

Random-effect dimension is currently limited to `q <= 5` in the formula path. Priors can
be supplied either as canonical arrays or as named per-term dictionaries, which makes it
possible to compare multiple prior settings for the same dataset.

## How it works

1. **Simulate hierarchical datasets.** `metabeta/simulation/` generates GLMM datasets
   across likelihood families, dimensions, group counts, group sizes, prior families,
   and data-source styles.
2. **Summarize locally and globally.** Set-transformer modules summarize observations
   within groups and then summarize groups across the dataset.
3. **Estimate posterior factors.** Conditional coupling flows model global parameters
   and per-group random effects, with masks for variable dimension and group structure.
4. **Route by problem size.** Joint checkpoints can contain several submodels. The API
   selects the smallest compatible submodel for a new dataset.
5. **Evaluate calibration and prediction.** Evaluation modules compute point recovery,
   interval coverage, simulation-based calibration, posterior predictive metrics, LOO
   diagnostics, and comparison plots against NUTS or ADVI fits.

## Repository map

```text
metabeta/
  analytical/     GLMM/LMM analytical fits, MAP/PQL helpers, and precomputation
  datasets/       preprocessing and source-specific dataset fetchers
  evaluation/     posterior quality, predictive, coverage, SBC, and summary metrics
  models/         approximator API, set transformers, normalizing flows, checkpoint docs
  plotting/       posterior, calibration, recovery, runtime, and comparison plots
  posthoc/        importance weighting, conformal, Laplace, SVGD, and warm-start methods
  simulation/     synthetic hierarchical data generation and reference fitting
  training/       training entry point and checkpoint loop
  utils/          config, dataloading, routing, IO, templates, logging, and device helpers
experiments/      reproducible experiment scripts grouped by package area
tests/            pytest coverage for models, simulation, evaluation, datasets, and utils
demos/            notebooks for introductory usage and prior sensitivity
scripts/          data generation, training, fitting, precompute, and checkpoint scripts
```

## Common workflows

Generate synthetic data:

```bash
uv run metabeta-generate --size small --family 0 --ds_type mixed --partition all
```

Train a model:

```bash
uv run metabeta-train --size small --family 0 --ds_type mixed --max_epochs 20
```

Evaluate a checkpoint:

```bash
uv run metabeta-evaluate --checkpoint metabeta/outputs/checkpoints/<run-name> --models all
```

Build routed checkpoint files for distribution:

```bash
uv run python scripts/build_ckpt.py --build
```

## Research status and limitations

`metabeta` is an active research codebase. It is meant to make amortized inference for
hierarchical regression inspectable and reproducible, not to hide statistical assumptions
behind a black-box interface.

Current limitations include:

- the formula parser supports one grouped random-effect term;
- routed sampling currently expects datasets in a batch to select the same submodel;
- checkpoint coverage depends on which likelihood-size combinations have been trained
  and published;
- posterior quality should be checked with calibration, predictive diagnostics, and
  reference fits when results matter.

## Development

```bash
uv sync
uv run python --version
uv run python -c "import metabeta"
uv run pytest
uv run blue --check --diff metabeta tests
```

Run the smallest relevant test first while developing, then expand to nearby modules and
the full suite when practical.

## License

This repository is licensed under [CC BY-NC-SA 4.0](LICENSE).
