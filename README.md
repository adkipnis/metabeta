[![PyPI](https://img.shields.io/pypi/v/metabeta)](https://pypi.org/project/metabeta/)
[![CI](https://github.com/adkipnis/metabeta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/metabeta/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2609.24422-b31b1b)](https://arxiv.org/abs/2609.24422)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-yellow)](https://huggingface.co/adkipnis/metabeta)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb)
[![License](https://img.shields.io/badge/license-code%20Apache--2.0%20%C2%B7%20weights%20CC%20BY--NC--SA%204.0-blue)](https://github.com/adkipnis/metabeta#license)

# metabeta

Amortized Bayesian inference for generalized linear mixed-effects models (GLMMs). Pretrained
networks turn a pandas DataFrame and an lme4-style formula into posterior draws in under a second.

<img src="https://raw.githubusercontent.com/adkipnis/metabeta/main/assets/overview.svg" width="100%" alt="Top: training datasets are simulated from priors over the fixed effects, the random-effect SDs and their correlations. Bottom: summary networks condense each group, then all groups; flows map the summary and the prior to posterior draws.">

- **Full posterior in one call** for normal, Bernoulli and Poisson outcomes: fixed effects,
  random effects, and their standard deviations (SDs) and correlations.
- **Priors as input**: change them without retraining, fit several at once, compare models by
  Bayes factors.
- **Corrected**: Metropolis–Hastings refines the networks' draws against the likelihood.
- **Checked**: every fit reports diagnostics and warns with a remedy when one fails.
- **Results in your data's units**: lme4-style tables, per-group effects, pair plots and
  posterior predictive checks.

## Installation

```bash
pip install metabeta
```

The pretrained checkpoints download from [Hugging Face](https://huggingface.co/adkipnis/metabeta)
on first use.

To work on the code, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/adkipnis/metabeta && cd metabeta && uv sync
```

## Quick start

Fit lme4's `sleepstudy`: reaction times of 18 subjects over 10 days of sleep restriction.

```python
import pandas as pd
from metabeta import Api

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv')
mb = Api.from_pretrained('normal')
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
| Intercept | 251.285 | 8.086 | 235.713 | 251.176 | 268.004 |   1.000 |    0.999 |
| Days      |  10.414 | 1.676 |   7.261 |  10.476 |  13.676 |   1.000 |    0.999 |

Standard Deviations:
|           |   Mean |    SD |   2.5% |   50.0% |   97.5% |
|-----------|--------|-------|--------|---------|---------|
| Intercept | 29.980 | 8.054 | 17.882 |  29.076 |  47.853 |
| Days      |  6.394 | 1.481 |  3.946 |   6.195 |   9.584 |
| Residual  | 25.713 | 1.515 | 22.652 |  25.686 |  28.791 |
...
draws = 1000   acceptance = 0.77   min ESS = 430   PSIS k̂ = 0.245 (refined)   MAP z = 0.5
```

- **Units**: ms, and ms per day for `Days` (lme4: 251.4 and 10.47).
- **Elided lines**: the priors and the random-effect correlation.
- **Last line**: the diagnostics, explained in [How it works](#how-it-works).
- **More**: `mb.rfxSummary(result)` prints each subject's random effects, and
  `mb.plotParameters(result)` draws the pair plot.

## Tutorials

All notebooks run on a CPU runtime.

| Notebook | What it shows | |
|---|---|---|
| [01-quickstart](https://github.com/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb) | sleepstudy (normal): summary, per-subject random effects, pair plot, diagnostics, lme4 comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb) |
| [02-families](https://github.com/adkipnis/metabeta/blob/main/demos/02-families.ipynb) | cbpp (Bernoulli) and Salamanders (Poisson): submodel routing, posterior predictive checks, random-walk moves against NUTS | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/02-families.ipynb) |
| [03-priors](https://github.com/adkipnis/metabeta/blob/main/demos/03-priors.ipynb) | specifying priors, fitting three in one call, Bayes factors | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/03-priors.ipynb) |

## How it works

The networks (figure):

1. **Training**: they learn from simulated GLMM datasets with random sizes, priors and parameters
   (top).
2. **Summary**: Σ<sub>l</sub> summarizes each group, and Σ<sub>g</sub> summarizes across groups
   (bottom).
3. **Posterior**: normalizing flows turn the summary, analytical estimates and the prior into
   draws. Π<sub>g</sub> draws the fixed effects β, random-effect SDs σ and correlations R;
   Π<sub>l</sub> draws each group's random effects α<sub>i</sub>.
4. **Routing**: each checkpoint holds four network sizes, and `sample` uses the smallest that fits
   the data.

Refinement, with `n_samples` ≥ 64 (default 1000):

1. The global flow proposes draws for independence Metropolis–Hastings (IMH).
2. The chain targets the posterior with the random effects integrated out: exactly (normal) or by
   a Laplace approximation (Bernoulli, Poisson).
3. The random effects are then redrawn from their conditional posterior.

Diagnostics, in `result.safeguards` and the summary's last line:

- `acceptance`: the IMH acceptance rate.
- `min ESS`: the smallest effective sample size (ESS).
- `PSIS k̂`: how well the flow's draws match the posterior, by Pareto-smoothed importance sampling
  (PSIS); lower is better.
- `MAP z`: the distance between the posterior mean and the maximum a posteriori (MAP) estimate, in
  posterior SDs.

`sample` warns, naming a remedy, when the acceptance, the ESS or `MAP z` crosses its threshold.

An ESS below 150 can leave intervals too narrow, typically because the flow lacks tail mass.
`random_walk=True` then adds random-walk moves on the same target, for those datasets only.

## Research workflow

The repository also holds the code that builds the checkpoints.

- `uv sync` installs its dependencies, including the research extra (PyMC, bambi, statsmodels
  and others).
- For a pip install, add the research extra with `pip install "metabeta[research]"`.

The pipeline, one script per step (each lists its options with `--help`):

1. `metabeta/simulation/generate.py` simulates training, validation and test datasets.
2. `metabeta/training/train.py` trains the networks for one family and size.
3. `metabeta/evaluation/evaluate.py` measures recovery, coverage, calibration and predictive fit,
   also against reference fits such as NUTS.
4. `scripts/build_ckpt.py` bundles the sizes into one checkpoint per family.

| Path | Contents |
|---|---|
| `metabeta/models/` | the `Api`, summary networks and normalizing flows |
| `metabeta/posthoc/` | IMH refinement, random-walk moves, marginal likelihood |
| `metabeta/analytical/` | analytical GLMM estimates that condition the networks |
| `metabeta/evaluation/`, `metabeta/plotting/` | metrics and plots |
| `metabeta/utils/`, `metabeta/configs/` | routing, preprocessing, priors and configs |
| `experiments/`, `scripts/` | experiment scripts and cluster job wrappers |

## FAQ

<details>
<summary><b>Do I need a GPU?</b></summary>

No. A fit takes 0.2 to 1 second on a laptop CPU, and about 0.1 seconds on a GPU
(`Api.from_pretrained(family, device='cuda')`).

</details>

<details>
<summary><b>When should I check a fit against NUTS?</b></summary>

When a warning from `sample` persists after its remedy:

- IMH acceptance below 0.1 at the suggested `n_samples`;
- minimum ESS below 150 with more draws and `random_walk=True`;
- posterior mean more than 5 posterior SDs from the MAP estimate.

After random-walk moves, `sample` suggests NUTS itself if the IMH ESS was below 10.
`02-families` compares a Poisson fit with NUTS.

</details>

<details>
<summary><b>What can metabeta fit?</b></summary>

- **Models**: normal, Bernoulli (0/1) and Poisson GLMMs with one grouping factor.
- **Data**: 5 to 200 groups, 5 to 150 rows per group, at most 3,000 rows.
- **Effects**: at most 16 fixed and 5 random effects, intercepts included.
- **Random slopes**: each must also be a fixed effect.
- **Enough groups**: at least 4 more groups than fixed effects, and than random-effect variances
  and covariances (q(q+1)/2 for q random effects).
- **Enough rows**: at least 2 more rows per group than random effects.
- **Formulas**: additive lme4 syntax, such as `y ~ x1 + x2 + (1 + x1 | g)`.
- **Categories**: string and categorical columns are dummy-coded, with the most frequent level as
  the reference (R uses the first).
- **Not supported**: `||`, `/`, interactions (`:`, `*`), functions (`log()`, `factor()`, `C()`),
  offsets and binomial trials. Transform the columns in the DataFrame instead; the error message
  says how.

Beyond these limits, `sample` raises an error naming the limit.

</details>

<details>
<summary><b>On which scale are the estimates?</b></summary>

- `posteriorSummary`, `rfxSummary` and `plotParameters` report on the data's scale: slopes per
  unit of each predictor.
- That means the response's units for normal, log-odds for Bernoulli and log rates for Poisson.
- `x_scale='standardized'` reports slopes per SD of the predictor instead.
- The networks, the draws in `result.proposal` and the priors passed to `sample(priors=...)` use a
  standardized scale: numeric predictors, and the normal response, at mean 0 and SD 1.
- `result.scale_info` maps draws to the data's scale, as the tutorials show.

</details>

## Citing

```bibtex
@misc{kipnis2026prioramortized,
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

## License

- Code: Apache-2.0.
- Pretrained weights, downloaded by `from_pretrained`: CC BY-NC-SA 4.0, non-commercial.
- For commercial use contact alexander.kipnis@helmholtz-munich.de.
