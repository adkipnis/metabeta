# metabeta

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/env-uv-6b47ff)](https://docs.astral.sh/uv/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey)](LICENSE)

`metabeta` is an amortized model for Bayesian generalized linear mixed-effects models (GLMMs).
Given a grouped dataset, a model formula, and an optional prior specification, it returns
posterior samples for fixed effects, random effects, variance components, and
correlations with the speed and batching capabilities of a PyTorch forward pass.

- Supports Normal, Bernoulli, and Poisson outcomes.
- Accepts convenient lme4-style formulas.
- Conditions on prior family and hyperparameters at inference time.
- Supports batched prior-sensitivity analysis in one `sample()` call.
- Includes diagnostics for calibration, posterior prediction, and comparison against NUTS or ADVI.
- Provides the data simulator, neural architecture, training loop, evaluation code, demos, and
  experiment scripts.

Pretrained checkpoints are hosted on [Hugging Face](https://huggingface.co/adkipnis/metabeta)
and are downloaded automatically on first use.

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
import statsmodels.api as sm
from metabeta.models.api import Api

mb = Api.from_pretrained("normal")
df = sm.datasets.get_rdataset("sleepstudy", "lme4")
result = mb.sample(df, formula="Reaction ~ Days + (Days | Subject)", n_samples=1000)
print(mb.posteriorSummary(result, x_scale="original"))
```

See [demos/intro.ipynb](demos/intro.ipynb) for the full `sleepstudy`
walkthrough and [demos/priors.ipynb](demos/priors.ipynb) for an exemplary prior-sensitivity
analysis.

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
| [metabeta/analytical/](metabeta/analytical/) | GLMM analytical fits and helpers |
| [metabeta/configs/](metabeta/configs/) | model and preset configuration files |
| [metabeta/datasets/](metabeta/datasets/) | preprocessing and source-specific dataset fetchers |
| [metabeta/evaluation/](metabeta/evaluation/) | parameter recovery, coverage, SBC, posterior predictive checks and summary metrics |
| [metabeta/models/](metabeta/models/) | model API, set transformers, normalizing flows |
| [metabeta/plotting/](metabeta/plotting/) | plot functions for posterior samples, recovery, calibration, and runtime |
| [metabeta/simulation/](metabeta/simulation/) | synthetic hierarchical data generation and reference fitting with PyMC |
| [metabeta/training/](metabeta/training/) | training entry point and checkpoint loop |
| [metabeta/utils/](metabeta/utils/) | config, dataloading, routing, IO, and shared helper code |
| [experiments/](experiments/) | reproducible experiment scripts grouped by package area |
| [tests/](tests/) | pytest suite for models, simulation, evaluation, datasets, and utils |
| [demos/](demos/) | demo notebooks |
