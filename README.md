[![PyPI](https://img.shields.io/pypi/v/metabeta)](https://pypi.org/project/metabeta/)
[![CI](https://github.com/adkipnis/metabeta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/metabeta/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2609.24422-b31b1b)](https://arxiv.org/abs/2609.24422)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-yellow)](https://huggingface.co/adkipnis/metabeta)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb)
[![License](https://img.shields.io/badge/license-code%20Apache--2.0%20%C2%B7%20weights%20CC%20BY--NC--SA%204.0-blue)](https://github.com/adkipnis/metabeta#license)

# metabeta

Amortized Bayesian inference for generalized linear mixed-effects models (GLMMs): pretrained
networks turn a grouped dataset and an lme4-style formula into posterior draws in a fraction of
a second.

<img src="https://raw.githubusercontent.com/adkipnis/metabeta/main/assets/overview.svg" width="100%" alt="Top: training datasets are simulated from priors over the fixed effects, the random-effect SDs and their correlations. Bottom: summary networks condense each group and then all groups; flows map the summary and the prior to posterior draws.">

- **Posterior draws in one call** for the fixed effects, the random effects and their standard
  deviations (SDs) and correlations, from a pandas DataFrame and a formula, for normal,
  Bernoulli and Poisson outcomes.
- **Priors are an input**, not baked in: change the prior without retraining, fit several priors
  in one call, and compare models by Bayes factors.
- **Refined and checked**: Metropolis–Hastings corrects the networks' draws against the
  model's likelihood, and every fit reports diagnostics that warn, naming a remedy, when a check
  fails.
- **Summaries in your data's units**: lme4-style tables, random effects per group, pair plots
  and posterior predictive checks.

## Installation

```bash
pip install metabeta
```

metabeta runs on a CPU; pass `device='cuda'` to `Api.from_pretrained` to use a GPU. The
pretrained checkpoints, one per outcome family with four sizes of network each, are on
[Hugging Face](https://huggingface.co/adkipnis/metabeta) and download on first use. To work on
the code: `git clone https://github.com/adkipnis/metabeta && cd metabeta && uv sync` (with
[uv](https://docs.astral.sh/uv/)).

## Quick start

lme4's `sleepstudy`: the reaction times of 18 subjects on 10 days of a sleep-restriction study.

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

The estimates are in the data's units: ms for the intercept and the SDs, ms per day for `Days`
(lme4: 251.4 and 10.47). The elided lines state the priors and the correlation of the random
effects; the last line reports the diagnostics described under [How it works](#how-it-works).
`mb.rfxSummary(result)` prints each subject's random effects, and `mb.plotParameters(result)`
draws the posterior pair plot.

## Tutorials

Each notebook runs on a CPU runtime.

| Notebook | What it shows | |
|---|---|---|
| [01-quickstart](https://github.com/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb) | sleepstudy (normal): the summary, random effects per subject, the pair plot, lme4's estimates and the diagnostics | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/01-quickstart.ipynb) |
| [02-families](https://github.com/adkipnis/metabeta/blob/main/demos/02-families.ipynb) | cbpp (Bernoulli) and Salamanders (Poisson): routing to submodels, posterior predictive checks, and random-walk moves that widen too-narrow intervals toward NUTS | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/02-families.ipynb) |
| [03-priors](https://github.com/adkipnis/metabeta/blob/main/demos/03-priors.ipynb) | how to specify priors, three priors fitted in one call, and Bayes factors | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adkipnis/metabeta/blob/main/demos/03-priors.ipynb) |

## How it works

The networks are trained on simulated GLMM datasets whose sizes, priors and parameters are drawn
at random (figure, top). On a dataset (figure, bottom), a local summary network Σ<sub>l</sub>
summarizes each group's rows, a global one Σ<sub>g</sub> summarizes the groups, and conditional
normalizing flows map the summary, analytical estimates and the prior to posterior draws:
Π<sub>g</sub> for the global parameters (fixed effects β, random-effect SDs σ and their
correlations R), Π<sub>l</sub> for each group's random effects α<sub>i</sub>. Each checkpoint
holds four sizes of these networks, and `sample` routes every dataset to the smallest one that
fits its design.

With `n_samples` of 64 or more (the default is 1000), `sample` refines the draws by independence
Metropolis–Hastings (IMH): the global flow proposes, and the chain targets the exact posterior
with the random effects integrated out (normal) or its Laplace approximation (Bernoulli,
Poisson), so the result does not hinge on the networks' accuracy alone. The random effects are
then redrawn from their conditional posterior. `result.safeguards` and the summary's last line
report four checks: the IMH acceptance rate, the smallest effective sample size (ESS), the
Pareto k̂ of the flow's draws as importance weights (Pareto-smoothed importance sampling, PSIS),
and the distance of the posterior mean from the maximum a posteriori (MAP) estimate in
posterior SDs. `sample` warns and names a remedy when the acceptance, the ESS or the MAP
distance crosses its threshold. Where the ESS falls below 150, typically because the flow lacks
tail mass, the IMH alone can leave intervals too narrow; `random_walk=True` then continues the
chain with random-walk moves on the same target, on those datasets only.

## Research workflow

The repository also holds the code that builds the checkpoints. The development install above
(`uv sync`) includes the research extra (PyMC, bambi, statsmodels and others) and the simulation
dependencies; `pip install "metabeta[research]"` adds the research extra to a pip install. The
pipeline, one script per step (each lists its options with `--help`):

1. `metabeta/simulation/generate.py` simulates training, validation and test datasets.
2. `metabeta/training/train.py` trains the summary networks and flows for one family and size.
3. `metabeta/evaluation/evaluate.py` measures parameter recovery, coverage, calibration and
   predictive fit, also against reference fits such as NUTS.
4. `scripts/build_ckpt.py` bundles the sizes into one checkpoint per family for `Api`.

| Path | Contents |
|---|---|
| `metabeta/models/` | the `Api`, summary networks and normalizing flows |
| `metabeta/posthoc/` | IMH refinement, random-walk moves, marginal likelihood |
| `metabeta/analytical/` | analytical GLMM estimates that condition the networks |
| `metabeta/simulation/`, `metabeta/training/` | dataset simulation and training |
| `metabeta/evaluation/`, `metabeta/plotting/` | metrics and plots |
| `metabeta/utils/`, `metabeta/configs/` | routing, preprocessing, priors and configs |
| `experiments/`, `scripts/` | experiment scripts and cluster job wrappers |
| `demos/`, `tests/` | the tutorials and the test suite |

## FAQ

<details>
<summary><b>Do I need a GPU?</b></summary>

No. On a laptop CPU each fit in the tutorials takes between 0.2 and 1 second. On a GPU
(`Api.from_pretrained(family, device='cuda')`) a fit takes about a tenth of a second.

</details>

<details>
<summary><b>When should I check a fit against NUTS?</b></summary>

When a warning from `sample` persists after its remedies: an IMH acceptance below 0.1 at the
suggested `n_samples`, a minimum ESS below 150 with more draws and `random_walk=True`, or a
posterior mean more than 5 posterior SDs from the MAP estimate. Where the ESS was below 10, the
line reporting the random-walk moves says so itself. `02-families` compares a Poisson fit with
NUTS.

</details>

<details>
<summary><b>What can metabeta fit?</b></summary>

- Normal, Bernoulli (0/1) and Poisson outcomes with one grouping factor.
- 5 to 200 groups, 5 to 150 rows per group and at most 3,000 rows.
- At most 16 fixed effects and 5 random effects, intercepts included; every random slope must
  also be a fixed term.
- At least 4 more groups than fixed effects, and than the q(q+1)/2 variances and covariances of
  q random effects; at least 2 more rows per group than random effects.
- Additive lme4 formulas, such as `y ~ x1 + x2 + (1 + x1 | g)`. String and categorical columns are
  dummy-coded, with the most frequent level as the reference (R takes the first).
- Not supported: `||`, `/`, interactions with `:` or `*`, function calls such as `log()`,
  `factor()` or `C()`, offsets and binomial trials. Transform the columns in the DataFrame
  instead; the error message says how.

`sample` routes each dataset to the smallest submodel that fits it and raises an error naming
the limit a dataset exceeds.

</details>

<details>
<summary><b>On which scale are the estimates?</b></summary>

`posteriorSummary`, `rfxSummary` and `plotParameters` report on the data's original scale:
slopes per unit of each predictor, in the response's units for the normal family, as log-odds
for Bernoulli and as log rates for Poisson. `x_scale='standardized'` reports slopes per SD of
the predictor instead. The networks work on the standardized scale, with numeric predictors
and, for the normal family, the response at mean 0 and SD 1. The draws in `result.proposal` and
the priors passed to `sample(priors=...)` are on that scale; `result.scale_info` maps draws to
the original scale, as the tutorials show.

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

Code: Apache-2.0. Pretrained weights, downloaded by `from_pretrained`: CC BY-NC-SA 4.0, non-commercial. For commercial use contact alexander.kipnis@helmholtz-munich.de.
