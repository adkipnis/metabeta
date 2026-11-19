# metabeta for R: a drop-in replacement for brms

This R package exposes metabeta through the interface of
[brms](https://paul-buerkner.github.io/brms/). Replace `library(brms)` with
`library(metabeta)` and existing code keeps working. `brm()`, `summary()`,
`fixef()`, `ranef()`, `coef()`, `VarCorr()`, `posterior_predict()`,
`posterior_epred()`, `loo()`, `kfold()`, `hypothesis()`, `pp_check()`,
`conditional_effects()`, `as_draws_df()` and the rest return objects of the same
shape, with the same variable names (`b_Days`, `sd_Subject__Intercept`,
`r_Subject[308,Days]`, ...). Every function exported by brms 2.23 exists with
identical arguments.

The difference is speed. Instead of compiling and running Stan, `brm()` calls the
pretrained amortized posterior network, then refines its proposal with independence
Metropolis-Hastings chains that target the exact posterior. A sleepstudy fit takes
about one second.

```r
library(metabeta)
data("sleepstudy", package = "lme4")

fit <- brm(Reaction ~ Days + (Days | Subject), data = sleepstudy,
           prior = prior(normal(10, 5), class = b))
summary(fit)
#>  Family: gaussian
#>   Links: mu = identity; sigma = identity
#> Formula: Reaction ~ Days + (Days | Subject)
#>    Data: sleepstudy (Number of observations: 180)
#>   Draws: 4 chains, each with iter = 1025; warmup = 25; thin = 1;
#>          total post-warmup draws = 4000
#>
#> Multilevel Hyperparameters:
#> ~Subject (Number of levels: 18)
#>                     Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> sd(Intercept)          29.94      7.61    17.79    48.61 1.00     2073     1949
#> sd(Days)                6.38      1.47     4.09     9.83 1.00     2098     2142
#> cor(Intercept,Days)    -0.02      0.30    -0.57     0.57 1.00     2832     3063
#>
#> Regression Coefficients:
#>           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> Intercept   250.28      7.91   234.82   265.27 1.00     2291     2027
#> Days         10.26      1.58     7.07    13.28 1.00     2142     2189
#> ...
```

## Installation

```r
install.packages("pak")
pak::pak("adkipnis/metabeta/R-package")
```

That is all; R 4.1 or later is needed, and neither Stan nor a C++ toolchain (the
package is written in R only). The first fit sets up the rest by itself:
[reticulate](https://rstudio.github.io/reticulate/) installs Python (3.12 or later) and
the `metabeta` Python package with its dependencies (PyTorch among them) into a cached
environment, and the pretrained network (about 135 MB per family) downloads from
Hugging Face. This needs an internet connection and takes a few minutes once;
afterwards a fit takes one to a few seconds (the first fit of each R session about
20 seconds more, while the network loads). To do the setup upfront, e.g. before working
offline, run `metabeta::install_metabeta()`.

To use a Python environment of your own instead, point reticulate to it before loading
the package (`Sys.setenv(RETICULATE_PYTHON = "path/to/python")`) and install the
Python package there (`pip install "metabeta>=0.6.0"`).

## What can be fitted

| | |
|---|---|
| Families | `gaussian()`, `bernoulli()` (or `binomial()` with one trial), `poisson()`, each with its canonical link |
| Population-level terms | anything `model.matrix()` builds: factors, interactions, `I()`, `poly()`, ...; up to 16 coefficients |
| Group-level terms | one grouping factor (or an interaction like `g1:g2`); correlated (`|`, `|ID|`) or uncorrelated (`||`, `gr(g, cor = FALSE)`); up to 2/3/4/5 coefficients for up to 4/8/12/16 population-level coefficients |
| Data | 5–200 groups with 5–150 observations each, at most 3000 observations |
| Priors | `normal()` or `student_t(5, ...)` on `b`/`Intercept` (one family for all); `normal(0, s)`, `student_t(5, 0, s)` or `exponential()` on `sd` and `sigma`; `lkj()` on `cor` |

Anything else (other families, smooths, autocorrelation, several grouping factors,
distributional or non-linear formulas, flat priors, ...) fails with an error that
explains why. Use brms for those models.

## How it maps onto brms

* **Same model.** The design comes from `model.matrix()`, so names, contrasts and
  transformations match brms. It is rescaled linearly to the network's standardized
  scale. Priors are mapped to that scale and draws are mapped back exactly. All
  parameters mean what they mean in brms, including the centered-predictor
  `Intercept` prior class. One exception: for correlated group-level slopes, the
  network needs centered predictors, so the priors on `sd(Intercept)` and on the
  correlations apply to the group-level intercept at the predictor means. The
  reported draws are transformed back exactly to brms' meaning (intercept at zero).
* **Real MCMC diagnostics.** With `algorithm = "sampling"` (the default), `chains`
  independent Metropolis-Hastings chains run on top of the network's proposal, so
  `Rhat`, `Bulk_ESS` and `Tail_ESS` are genuine. A low acceptance rate or chains
  that repeat few draws are reported as warnings, with remedies;
  `control = list(random_walk = TRUE)` continues such chains with random-walk moves.
  `algorithm = "meanfield"` (or `"fullrank"`, `"pathfinder"`, `"laplace"`) returns
  the raw network posterior.
* **Default priors** are weakly informative and scaled to the data rather than flat;
  see `get_prior()`.
* **Bayes factors without bridge sampling.** `bridge_sampler()`, `bayes_factor()`
  and `post_prob()` estimate the marginal likelihood by importance sampling from a
  Student-t fitted to the posterior, with a warning when the Pareto k of the weights
  exceeds 0.7. As in brms, fit with `save_pars = save_pars(all = TRUE)` to have it
  ready; otherwise `bridge_sampler()` repeats the (seeded) fit to compute it.
* **Stan-only features** (`stancode()`, `nuts_params()`, `log_posterior()`, ...)
  raise an error. `lp__` is not stored; `lprior` is. Stan tuning arguments
  (`cores`, `backend`, `init`, entries of `control` other than `random_walk`, ...)
  are accepted and ignored.
* **Refits are cheap.** `kfold()` and `loo(reloo = TRUE)` actually refit the model.
* Fitted objects have class `c("metabetafit", "brmsfit")`. Packages that work
  through generics (posterior, bayesplot, loo, tidybayes) accept them. Load
  metabeta *instead of* brms rather than alongside it.

Options: `metabeta.device` (`"cpu"` or e.g. `"cuda"`), `metabeta.verbose`, and
`metabeta.python_requirement`.

## What to pay attention to

* **Read the warnings.** metabeta checks every fit. A low Metropolis-Hastings
  acceptance rate (below 0.1) means the network's proposal fits the posterior
  poorly; the warning suggests how many draws (`iter`) restore calibration. Chains
  that repeat few draws (an effective sample size below 150) can make intervals too
  narrow; more draws or `control = list(random_walk = TRUE)` help. A posterior far
  from the analytical point estimates means the data may lie outside what the
  network was trained on. If the warnings persist, fit the model with brms. `Rhat`
  and the ESS columns of `summary()` come from real chains and are meaningful as
  usual.
* **Stay within the trained range.** Models outside the table above fail with an
  explanation, e.g. groups with fewer than 5 observations (drop or merge them) or
  more than 3000 observations. Priors much tighter or wider than weakly informative
  ones on the standardized scale also lie outside the training range; the
  acceptance rate then usually drops, and the warnings say so.
* **Priors differ from brms' defaults.** metabeta's defaults are weakly informative
  and scaled to the data (those of Bambi), while brms puts flat priors on the
  regression coefficients. To compare with a brms fit, set the priors explicitly in
  both; `prior_summary(fit)` lists the ones used.
* **Correlated random slopes** are fitted with the group-level intercept at the
  predictor means (see above). Priors on `sd(Intercept)` and on the correlations
  refer to that intercept, and so does the marginal likelihood. Reported draws are
  on brms' scale.
* **Load metabeta instead of brms**, not both: they export the same functions.
  Like brms, metabeta masks `stats::ar()`.
* **Reproducibility.** Fits are reproducible with `seed` (or `set.seed()` before
  `brm()`), for the same versions of the R and Python packages and the networks.
* **Disk space and platforms.** The Python environment takes about 1 GB. On Linux,
  the default PyTorch build from PyPI includes CUDA libraries (several GB); to avoid
  them, install metabeta into your own environment with the CPU build of PyTorch
  and point `RETICULATE_PYTHON` to it. A GPU is used with
  `options(metabeta.device = "cuda")`; it is not needed.
* **Windows with conda:** if R and Python share one conda environment, R's
  MKL-linked LAPACK and PyTorch ship different OpenMP runtimes, and whichever loads
  first wins. Fit a model (or run `install_metabeta()`) before other numerical code
  in a fresh session, or use a Python environment outside R's conda environment.
* **License.** The code is licensed Apache-2.0. The pretrained networks it downloads
  are licensed CC BY-NC-SA 4.0, which excludes commercial use.

## Validation

`validation/` holds two checks of the package. Neither is part of the installed package.

* [`reproduce-paper/`](validation/reproduce-paper/): a notebook that reruns the
  metabeta results of the paper with this package: the prior sensitivity analysis and
  Bayes factors on real data with the paper's data, models and priors, and the
  simulation studies (oracle benchmark, raw flow against the IMH head, stress tests) on
  freshly simulated datasets that follow the paper's protocol. Render it with
  `rmarkdown::render("validation/reproduce-paper/reproduce-paper.Rmd")`; it needs the
  packages lme4, glmmTMB, MASS and rmarkdown, and takes about two hours with its
  default sizes.
* [`brms-projects/`](validation/brms-projects/): eleven published analyses that use
  brms (books, tutorials and papers with open code and data), run with
  `library(metabeta)` in place of `library(brms)` and compared with the numbers they
  report. `Rscript validation/brms-projects/run.R` downloads and reruns them; see its
  README for the results.

## Development

```sh
Rscript -e 'roxygen2::roxygenise("R-package")'
Rscript -e 'testthat::test_local("R-package")'
R CMD build R-package && R CMD check --no-manual metabeta_*.tar.gz
```

`R/generated.R` (family constructors and the brms functions metabeta does not support,
with brms' exact arguments) is written by `Rscript tools/generate.R`; edit the lists there.
Most tests replace the network with a mock sampler (`options(metabeta.sampler = ...)`),
which checks that the reparametrization is exact draw by draw. `test-python.R` runs
the pretrained networks when the Python package is available. `test-compat.R`
compares every exported function's arguments with brms when it is installed.
