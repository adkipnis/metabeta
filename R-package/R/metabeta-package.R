#' metabeta: amortized Bayesian mixed models with a 'brms' interface
#'
#' `metabeta` provides the user interface of \pkg{brms} on top of the pretrained
#' amortized posterior estimator of the `metabeta` Python package. Replace
#' `library(brms)` with `library(metabeta)`, and calls such as `brm()`,
#' `summary()`, `fixef()`, `ranef()`, `posterior_predict()`, `loo()` or
#' `hypothesis()` keep working and return objects of the same shape.
#'
#' @section Supported models:
#' * Likelihoods: gaussian (identity link), bernoulli or binomial with one trial
#'   (logit link), and poisson (log link).
#' * Population-level effects: anything [stats::model.matrix()] can build, with
#'   up to 16 coefficients including the intercept.
#' * Group-level effects on one grouping factor (possibly an interaction such
#'   as `g1:g2`), either all correlated or all uncorrelated. The number q of
#'   group-level coefficients is limited by the number d of population-level
#'   coefficients: q <= 2, 3, 4, 5 for d <= 4, 8, 12, 16.
#' * Data: 5 to 200 groups with 5 to 150 observations each, at most 3000 in total.
#' * Priors: `normal()` or `student_t(5, ...)` on population-level effects (one
#'   family for all); `normal(0, ...)`, `student_t(5, 0, ...)` or
#'   `exponential()` on standard deviations and `sigma`; `lkj()` on
#'   correlations. Default priors are weakly informative instead of flat.
#'
#' Other models raise an error explaining why; fit them with \pkg{brms}.
#'
#' @section How the posterior is computed:
#' The design is built with [stats::model.matrix()] and mapped linearly onto the
#' standardized scale the network was trained on. Priors are mapped to that
#' scale and draws back from it exactly, so parameters mean what they mean in
#' \pkg{brms}. One exception: with correlated group-level slopes, the network
#' needs centered predictors, so the priors on `sd(Intercept)` and the
#' correlations apply to the group-level intercept at the predictor means. The
#' reported draws are still those of the intercept at zero.
#'
#' With `algorithm = "sampling"` (the default), the network's posterior is the
#' proposal of independence Metropolis-Hastings chains that target the exact
#' posterior, so `Rhat` and the effective sample sizes are genuine MCMC
#' diagnostics. A low acceptance rate is reported as a warning.
#'
#' @section Differences to brms:
#' * Stan tuning arguments (`cores`, `backend`, `init`, entries of `control`
#'   other than `random_walk`, ...) are ignored. Functions that need Stan, and brms functions for features metabeta
#'   cannot fit, raise an error that says so.
#' * The draws contain `lprior` but not `lp__`.
#' * Fitted models have class `c("metabetafit", "brmsfit")`, so packages that use
#'   generic functions (\pkg{posterior}, \pkg{bayesplot}, \pkg{loo},
#'   \pkg{tidybayes}, ...) accept them. Load metabeta instead of \pkg{brms}, not
#'   alongside it.
#'
#' @section Options:
#' * `metabeta.device`: torch device of the network, `"cpu"` (default) or e.g. `"cuda"`.
#' * `metabeta.python_requirement`: Python requirement installed by reticulate
#'   (default `"metabeta>=0.6.0"`); set it before `library(metabeta)`.
#' * `metabeta.verbose`: show checkpoint download progress.
#' * `metabeta.random_walk`: the default of `control = list(random_walk = ...)` in
#'   [brm()] (default `FALSE`).
#'
#' @keywords internal
"_PACKAGE"

#' @importFrom stats coef fitted formula family model.frame nobs predict residuals update
#'   gaussian binomial poisson rnorm rbinom rpois dnorm dbinom dpois dt pt plogis qlogis
#' @importFrom ggplot2 .data
NULL
