# metabeta

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/env-uv-6b47ff)](https://docs.astral.sh/uv/)
[![Release checks](https://github.com/adkipnis/metabeta/actions/workflows/release-checks.yml/badge.svg)](https://github.com/adkipnis/metabeta/actions/workflows/release-checks.yml)
[![Hugging Face](https://img.shields.io/badge/🤗-checkpoints-yellow)](https://huggingface.co/adkipnis/metabeta)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey)](LICENSE)

`metabeta` is an amortized model for Bayesian generalized linear mixed-effects models (GLMMs).
Given a grouped dataset, a model formula, and an optional prior specification, it returns
posterior samples for fixed effects, random effects, variance components, and
correlations with the speed and batching capabilities of a PyTorch forward pass.

- Supports Normal, Bernoulli, and Poisson outcomes.
- Accepts convenient lme4-style formulas.
- Conditions on prior family and hyperparameters at inference time.
- Supports batched prior-sensitivity analysis in one `sample()` call.
- Includes diagnostics for posterior contraction and prediction accuracy.
- Ships the pretrained inference API, plotting helpers, diagnostics, and demo notebooks.
- Keeps simulation, training, reference-method evaluation, and experiment scripts in the
  source repository for research workflows.

Pretrained checkpoints are hosted on [Hugging Face](https://huggingface.co/adkipnis/metabeta)
and are downloaded automatically on first use.

## Quick start

For local development, install from source with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/adkipnis/metabeta.git
cd metabeta
uv sync
uv pip install -e .
```

Run posterior inference on a grouped dataframe:

```python
import statsmodels.api as sm
from metabeta.models.api import Api

mb = Api.from_pretrained("normal")
df = sm.datasets.get_rdataset("sleepstudy", "lme4").data
result = mb.sample(df, formula="Reaction ~ Days + (Days | Subject)", n_samples=1000)
print(mb.posteriorSummary(result))
```

See [demos/intro.ipynb](demos/intro.ipynb) for the full `sleepstudy`
walkthrough and [demos/priors.ipynb](demos/priors.ipynb) for an exemplary prior-sensitivity
analysis.

## Development and research workflows

The default PyPI install is for pretrained-model inference. To reproduce training runs,
generate synthetic datasets, benchmark against reference methods, or extend the package,
clone the repository and install the optional research dependencies:

```bash
git clone https://github.com/adkipnis/metabeta.git
cd metabeta
uv sync --extra research --group simulation --dev
uv pip install -e ".[research]"
```

This enables the repository-only paths for synthetic data generation, model training,
benchmarking against reference methods, simulation-based calibration studies, and experiment
scripts. The `simulation` dependency group installs the GitHub-only `scamd` dependency used
by SCM dataset generation. The `metabeta.posthoc` package is included for experimental
posterior refinement, but it is not part of the production pretrained API and may change
without a deprecation window.

## From simulation to deployment

Each pretrained model is built through the same pipeline, from a dataset simulator to a checkpoint that can be loaded by the public API.

1. **Define the model family.** Choose the likelihood, GLMM dimensions, group structure,
   covariate styles, and prior families covered by a checkpoint.
2. **Generate training data.** Simulate hierarchical datasets and posterior reference targets
   across the configured design space.
3. **Format inputs.** Convert grouped tabular data into padded tensors, masks, and prior encodings shared by training and inference.
4. **Train amortized posteriors.** Use set-transformers (to learn low-dimensional permutation-invariant summaries of the datasets) and conditional coupling
   flows to approximate posteriors for fixed effects, variance parameters, correlations, and group-wise random effects.
5. **Validate behavior.** Check parameter recovery, credible interval coverage, simulation-based calibration, posterior
   predictive accuracy and run comparisons against reference methods.
6. **Package checkpoints.** Bundle weights, configs, routing metadata, and
   preprocessing expectations into a joint checkpoint.
7. **Deploy through the API.** Load joint checkpoints with
   `Api.from_pretrained(...)` for convenient posterior estimation and diagnostics.

## Repository map

| Path | Contents |
|---|---|
| [metabeta/analytical/](metabeta/analytical/) | GLMM analytical fits and helpers |
| [metabeta/configs/](metabeta/configs/) | model and preset configuration files |
| [metabeta/datasets/](metabeta/datasets/) | preprocessing and source-specific dataset fetchers |
| [metabeta/evaluation/](metabeta/evaluation/) | parameter recovery, coverage, SBC, posterior predictive checks and summary metrics |
| [metabeta/models/](metabeta/models/) | model API, set transformers, normalizing flows |
| [metabeta/plotting/](metabeta/plotting/) | plot functions for posterior samples, recovery, calibration, and runtime |
| [metabeta/posthoc/](metabeta/posthoc/) | experimental post-hoc posterior refinement helpers |
| [metabeta/simulation/](metabeta/simulation/) | synthetic hierarchical data generation and reference fitting with PyMC |
| [metabeta/training/](metabeta/training/) | training entry point and checkpoint loop |
| [metabeta/utils/](metabeta/utils/) | config, dataloading, routing, IO, and shared helper code |
| [experiments/](experiments/) | reproducible experiment scripts grouped by package area |
| [tests/](tests/) | pytest suite for models, simulation, evaluation, datasets, and utils |
| [demos/](demos/) | demo notebooks |
