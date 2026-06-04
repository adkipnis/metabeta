# metabeta

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/env-uv-6b47ff)](https://docs.astral.sh/uv/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey)](LICENSE)

**Amortized Bayesian inference for hierarchical regression models.**

`metabeta` is a PyTorch implementation and set of pretrained checkpoints for
prior-amortized Bayesian inference in generalized linear mixed-effects models (GLMMs).
Given a grouped dataset, a model formula, and optional prior hyperparameters, it returns
posterior samples for fixed effects, random effects, variance components, and
correlations with a forward pass instead of a fresh MCMC run.

- Supports Normal, Bernoulli, and Poisson outcomes.
- Accepts lme4-style formulas with one grouping factor.
- Conditions on prior family and hyperparameters at inference time.
- Supports batched prior-sensitivity analysis in one `sample()` call.
- Includes diagnostics for calibration, prediction, and comparison against NUTS or ADVI.
- Provides the simulator, neural architecture, training loop, evaluation code, demos, and
  experiment scripts.

## Quick start

Install from source with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/adkipnis/metabeta.git
cd metabeta
uv sync
uv pip install -e .
```

Run posterior inference on a grouped dataframe:

See [demos/intro.ipynb](demos/intro.ipynb) for a complete walkthrough on the
classic `sleepstudy` dataset. It loads a pretrained Normal checkpoint, fits an
lme4-style random-slope model, prints population-level and per-subject posterior
summaries, visualizes the parameter posterior with prior overlays, and checks the
fit against the observed trajectories.

## Modeling interface

The high-level API accepts several input stages:

- pandas dataframes and parquet files
- preprocessed and batched numpy and pytorch dictionaries

Formulas are supported in lme4-style, for example:

```text
y ~ x1 + x2 + (1 + x1 | group)
```

Prior specifications can be left as `None` to use the model defaults, passed as
explicit canonical arrays, or written as per-term dictionaries for fixed effects,
random-effect standard deviations, residual scale, and random-effect correlations.
Named dictionaries are batched automatically, so several prior choices can be
compared in one `sample()` call:

```python
priors = {
    "default": None,
    "conservative": {
        "fixed": {
            "family": "student",
            "Intercept": {"tau": 0.5},
            "x1": {"tau": 0.3},
        },
        "random_sd": {
            "family": "exponential",
            "Intercept": {"tau": 0.3},
            "x1": {"tau": 0.3},
        },
        "corr_rfx": {"eta": 2.0},
    },
}

result = mb.sample(df, formula="y ~ x1 + (1 + x1 | group)", priors=priors)
```

See [demos/priors.ipynb](demos/priors.ipynb) for a prior-sensitivity workflow
with simulated ground truth and posterior plots with analytical prior overlays.

## How the model was built

1. **Simulate hierarchical datasets.** `metabeta/simulation/` generates GLMM datasets
   across likelihood families, GLMM dimensions, group counts, group sizes, prior families,
   and data-source styles.
2. **Summarize locally and globally.** Set-transformer modules summarize observations
   within groups and then summarize groups across the dataset.
3. **Estimate posterior factors.** Conditional coupling flows model global parameters
   and per-group random effects, with masks for variable GLMM dimensions.
4. **Evaluate calibration and prediction.** Evaluation modules compute parameter recovery,
   credible interval coverage, simulation-based calibration checks, posterior predictive
   metrics, LOO diagnostics, and comparison plots against NUTS or ADVI fits.

## Repository map

| Path | Contents |
|---|---|
| [metabeta/](metabeta/) | Python package root |
| [metabeta/analytical/](metabeta/analytical/) | GLMM analytical fits and helpers |
| [metabeta/configs/](metabeta/configs/) | model and preset configuration files |
| [metabeta/datasets/](metabeta/datasets/) | preprocessing and source-specific dataset fetchers |
| [metabeta/evaluation/](metabeta/evaluation/) | posterior quality, predictive, coverage, SBC, and summary metrics |
| [metabeta/models/](metabeta/models/) | approximator API, set transformers, normalizing flows |
| [metabeta/plotting/](metabeta/plotting/) | plot functions for posterior, calibration, recovery, and runtime |
| [metabeta/posthoc/](metabeta/posthoc/) | optional post-hoc refinement methods |
| [metabeta/simulation/](metabeta/simulation/) | synthetic hierarchical data generation and reference fitting with PyMC |
| [metabeta/training/](metabeta/training/) | training entry point and checkpoint loop |
| [metabeta/utils/](metabeta/utils/) | config, dataloading, routing, IO, and shared helper code |
| [experiments/](experiments/) | reproducible experiment scripts grouped by package area |
| [scripts/](scripts/) | shell and Python scripts for data, fitting, training, and checkpoint packaging |
| [tests/](tests/) | pytest coverage for models, simulation, evaluation, datasets, and utils |
| [demos/](demos/) | notebooks for introductory usage and prior sensitivity |
