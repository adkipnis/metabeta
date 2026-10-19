[![PyPI](https://img.shields.io/pypi/v/metabeta)](https://pypi.org/project/metabeta/)
[![CI](https://github.com/adkipnis/metabeta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/metabeta/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2609.24422-b31b1b)](https://arxiv.org/abs/2609.24422)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23458113.svg)](https://doi.org/10.5281/zenodo.23458113)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-yellow)](https://huggingface.co/adkipnis/metabeta)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb)

# MetaBeta

Bayesian inference for GLMMs (generalized linear mixed-effects models) in under a second:

- **Full posterior in one call** for Gaussian, Bernoulli and Poisson outcomes: fixed effects,
  random effects, and their standard deviations (SDs) and correlations.
- **Priors as input**: you can change them without retraining, fit several at once, and compare models using
  Bayes factors.
- **Asymptotically exact**: Metropolis–Hastings refines the networks' posterior samples against the exact unnormalized posterior.
- **Diagnostics included**: every fit reports diagnostics and suggests a remedy.
- **A familiar workflow**: lme4-style tables, per-group effects, pair plots and
  posterior predictive checks.

## Installation

```bash
pip install metabeta
```

The pretrained checkpoints download from [Hugging Face](https://huggingface.co/adkipnis/metabeta) on first use.

For development:
```bash
git clone https://github.com/adkipnis/metabeta && cd metabeta && uv sync
```

## Quick start

Fit lme4's `sleepstudy`: reaction times of 18 subjects over 10 days of sleep restriction.

```python
import pandas as pd
from metabeta import Api

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv')
mb = Api.from_pretrained('gaussian')
result = mb.sample(df, formula='Reaction ~ Days + (Days | Subject)')
print(mb.posteriorSummary(result))
```

```text
Formula:  Reaction ~ Days + (Days | Subject)
n = 180   m = 18
...
Fixed Effects:
|           |    Mean |    SD |    2.5% |   50.0% |   97.5% |   P(>0) |   Contr. |
|-----------|---------|-------|---------|---------|---------|---------|----------|
| Intercept | 251.276 | 8.046 | 235.865 | 251.166 | 267.935 |   1.000 |    0.999 |
| Days      |  10.434 | 1.653 |   7.267 |  10.485 |  13.676 |   1.000 |    0.999 |

Standard Deviations:
|           |   Mean |    SD |   2.5% |   50.0% |   97.5% |
|-----------|--------|-------|--------|---------|---------|
| Intercept | 29.938 | 8.059 | 17.744 |  29.099 |  47.685 |
| Days      |  6.368 | 1.472 |  3.936 |   6.194 |   9.584 |
| Residual  | 25.752 | 1.535 | 22.652 |  25.714 |  28.829 |
...
draws = 1000   acceptance = 0.78   min ESS = 456   PSIS k̂ = 0.240   MAP z = 0.5   R² = 0.827
```

- **Units**: ms, and ms per day for `Days` (lme4: 251.4 and 10.47).
- **Last line**: the diagnostics are explained in the [FAQ](#faq).
- **More**: `mb.rfxSummary(result)` prints each subject's random effects, and
  `mb.plotParameters(result)` visualizes posterior draws. The [tutorials](#tutorials) cover
  `posteriorDraws`, `groupDraws`, `posteriorPredictive` and `bayesFactor`.

## Tutorials

| Notebook | Content | |
|---|---|---|
| [01-quickstart](https://github.com/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb) | First fit: summaries, plots and diagnostics | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb) |
| [02-families](https://github.com/adkipnis/metabeta/blob/main/demos/02-families.ipynb) | Bernoulli and Poisson outcomes | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/02-families.ipynb) |
| [03-priors](https://github.com/adkipnis/metabeta/blob/main/demos/03-priors.ipynb) | Custom priors and Bayes factors | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/03-priors.ipynb) |

## How it was built

<img src="https://raw.githubusercontent.com/adkipnis/metabeta/main/assets/overview.svg" width="100%" alt="Top: training datasets are simulated from priors over the fixed effects, the random-effect SDs and their correlations. Bottom: summary networks condense each group, then all groups; flows map the summary and the prior to posterior draws.">

**Simulation** (figure, top)

- **Sizes**: the numbers of predictors d, random effects q, groups m and rows per group
  n<sub>i</sub> are drawn first.
- **Priors**: the prior families and their scales are drawn next.
- **Parameters**: the fixed effects β, random-effect SDs σ and correlations R are drawn from these
  priors.
- **Predictors**: X is sampled from synthetic tables, built from random structural causal models as in TabPFN/TabICL, or subsampled from hundreds of real tabular datasets.
- **Outcomes**: y is drawn from the GLMM, given X and the parameters.

**Training** (figure, bottom)

- **Summaries**: Σ<sub>l</sub> condenses all rows per group into a vector, and Σ<sub>g</sub>
  condenses these vectors into one summary s. Both ignore the order of rows and groups.
- **Augmentation**: efficient analytical estimates and the prior join the summaries.
- **Posterior**: normalizing flows turn random noise u into posterior draws. Π<sub>g</sub> draws
  β, σ and R from s; Π<sub>l</sub> draws each group's α<sub>i</sub> from its summary and a draw
  of Π<sub>g</sub>.
- **Objective**: all networks are trained end-to-end to optimize the probability of the true
  parameters under the proposed posterior.

A GLMM fit is then a forward pass through the trained networks. Metropolis–Hastings refines its draws to grant the estimated posterior asymptotic exactness.

## FAQ

<details>
<summary><b>What can MetaBeta fit?</b></summary>

- **Models**: Gaussian, Bernoulli (0/1) and Poisson GLMMs with one grouping factor.
- **Data**: 5 to 200 groups, 5 to 150 rows per group, at most 3,000 rows.
- **Effects**: at most 16 fixed and 5 random effects, intercepts included.
- **Random slopes**: each must also be a fixed effect.
- **Enough groups**: at least 4 more groups than fixed effects, and than random-effect variances
  and covariances.
- **Enough rows**: at least 2 more rows per group than random effects.
- **Formulas**: additive lme4 syntax, such as `y ~ x1 + x2 + (1 + x1 | g)`.
- **Categories**: string and categorical columns are dummy-coded, with the most frequent level as
  the reference (R uses the first).
- **Currently not supported**: `||`, `/`, interactions (`:`, `*`), functions (`log()`, `factor()`, `C()`),
  offsets and binomial trials. Transform the columns in the DataFrame instead; the error message
  says how.

</details>

<details>
<summary><b>Do I need a GPU?</b></summary>

No. A fit takes 0.2 to 1 second on a laptop CPU, and about 0.1 seconds on a fast GPU. Colab's
free runtimes are slower: one to a few seconds on CPU or T4. To use a GPU, pass `device='cuda'` to
`Api.from_pretrained`.

</details>

<details>
<summary><b>What do the diagnostics mean?</b></summary>

They are in the summary's last line and in `result.safeguards`:

- `acceptance`: the share of the networks' draws that Metropolis–Hastings accepts.
- `min ESS`: the smallest effective sample size (ESS).
- `PSIS k̂`: how well the networks' draws match the posterior, by Pareto-smoothed importance
  sampling (PSIS); lower is better.
- `MAP z`: the distance between the posterior mean and the maximum a posteriori (MAP) estimate, in
  posterior SDs.
- `R²`, `AUC` or `Deviance`, by family: how well the posterior predicts the data (in
  `result.diagnostics`).

`sample` warns when the acceptance, the ESS or `MAP z` crosses its threshold.

</details>

<details>
<summary><b>When should I fall back to NUTS?</b></summary>

When `sample` still warns after you apply its suggested remedy, e.g. more draws or
random-walk moves (`random_walk=True`). `02-families` compares a Poisson fit with NUTS.

</details>





## Contributing

Issues and pull requests are welcome; see
[`CONTRIBUTING.md`](https://github.com/adkipnis/metabeta/blob/main/CONTRIBUTING.md).

## Citing

```bibtex
@misc{kipnis.2026,
  title         = {Prior-Amortized In-Context {Bayesian} Inference for Generalized Linear
                   Mixed-Effects Models},
  author        = {Kipnis, Alex and Binz, Marcel and Schulz, Eric},
  year          = {2026},
  eprint        = {2609.24422},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.24422}
}
```

All versions share the DOI [10.5281/zenodo.23458113](https://doi.org/10.5281/zenodo.23458113).

## License

- Code: Apache-2.0.
- Weights: CC BY-NC-SA 4.0.
