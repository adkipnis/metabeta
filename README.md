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

TODO: point to intro ipynb  

## Modeling interface

The high-level API accepts several input stages:

- pandas dataframes and parquet files
- preprocessed and batched numpy and pytorch dictionaries

Formulas are supported in lme4-style, for example:

```text
y ~ x1 + x2 + (1 + x1 | group)
```

TODO: elaborate on the options for Prior specification

## How the model was built

1. **Simulate hierarchical datasets.** `metabeta/simulation/` generates GLMM datasets
   across likelihood families, GLMM dimensions, group counts, group sizes, prior families,
   and data-source styles.
2. **Summarize locally and globally.** Set-transformer modules summarize observations
   within groups and then summarize groups across the dataset.
3. **Estimate posterior factors.** Conditional coupling flows model global parameters
   and per-group random effects, with masks for variable GLMM dimensions.
4. **Evaluate calibration and prediction.** Evaluation modules compute parameter recovery,
   credible interval coverage, simulation-based calibration checks, posterior predictive metrics, LOO
   diagnostics, and comparison plots against NUTS or ADVI fits.

## Repository map

```text
metabeta/
  analytical/     GLMM analytical fits and helpers
  datasets/       preprocessing and source-specific dataset fetchers
  evaluation/     posterior quality, predictive, coverage, SBC, and summary metrics
  models/         approximator API, set transformers, normalizing flows
  plotting/       plot functions for posterior, calibration, recovery, and runtime
  posthoc/        optional post-hoc refinement methods
  simulation/     synthetic hierarchical data generation and reference fitting with PyMC
  training/       training entry point and checkpoint loop
experiments/      reproducible experiment scripts grouped by package area
tests/            pytest coverage for models, simulation, evaluation, datasets, and utils
demos/            notebooks for introductory usage and prior sensitivity
```
