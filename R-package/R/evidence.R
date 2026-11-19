# Marginal likelihoods, Bayes factors and posterior model probabilities.
#
# metabeta's evidence pass estimates log p(y) by importance sampling from a Student-t
# fitted to the posterior draws. Python returns it for the design on the network's
# scale; R adds the Jacobian of the outcome scaling.

#' Log marginal likelihood of a model
#'
#' Estimates the log marginal likelihood by importance sampling from a
#' multivariate Student-t distribution fitted to the posterior draws (metabeta's
#' evidence pass), instead of bridge sampling. As in \pkg{brms}, the result works
#' with [bayes_factor()] and [post_prob()].
#'
#' A model fitted with `save_pars = save_pars(all = TRUE)` computed the estimate
#' during the fit. Otherwise the fit is repeated with its seed, which reproduces
#' its draws, and the estimate is computed on the way; this takes about as long as
#' the fit. `add_criterion(fit, "marglik")` stores the result.
#'
#' The estimate is unreliable when a few draws carry most of the importance
#' weight, in the evidence pass or among the network's proposals. The larger of
#' their Pareto k values measures this (as in [loo()]); above 0.7 a warning is
#' given.
#'
#' For correlated group-level effects with an intercept, the priors on
#' `sd(Intercept)` and the correlations refer to the intercept at the predictor
#' means (see [brm()]), so the marginal likelihood is that of this model.
#'
#' @param samples A fitted model.
#' @param ... Ignored.
#' @return An object of class `bridge` with the estimate in element `logml`
#'   and the Pareto k of the importance weights in `pareto_k`.
#' @examples
#' \dontrun{
#' fit1 <- brm(Reaction ~ Days + (Days | Subject), data = lme4::sleepstudy)
#' fit2 <- brm(Reaction ~ Days + (1 | Subject), data = lme4::sleepstudy)
#' bridge_sampler(fit1)
#' bayes_factor(fit1, fit2)
#' post_prob(fit1, fit2)
#' }
#' @export
bridge_sampler <- function(samples, ...) UseMethod("bridge_sampler")

#' @rdname bridge_sampler
#' @export
bridge_sampler.metabetafit <- function(samples, ...) {
  stored <- samples$criteria$marglik
  if (inherits(stored, "bridge")) return(stored)
  mb <- samples$metabeta
  args <- samples$stan_args
  n_samples <- args$chains * (args$iter - args$warmup)
  if (identical(mb$method, "prior") || n_samples < 64) {
    stop2("The marginal likelihood needs a model fitted to data with at least 64 draws.")
  }
  evidence <- mb$evidence
  if (is.null(evidence)) {
    # the same call as the fit, so the same draws, with the evidence pass
    evidence <- tryCatch(
      sample_network(mb$design$model, mb$hyper, mb$design$likelihood, n_samples, args$chains,
        refine = samples$algorithm == "sampling", seed = mb$seed, evidence = TRUE,
        random_walk = isTRUE(mb$random_walk)),
      error = explain_python_error
    )$evidence
  }
  k <- max(evidence$psis_k, evidence$pool_psis_k, na.rm = TRUE)
  if (!is.finite(k) || k > 0.7) {
    warning2(
      sprintf("The Pareto k of the importance weights is %.2f > 0.7, ", k),
      "so the estimate of the marginal likelihood is unreliable. More draws (iter) may help."
    )
  }
  # Gaussian outcomes were divided by y_sd; the Jacobian gives the density of y itself
  design <- mb$design
  logml <- evidence$log_evidence - length(design$y) * log(design$y_sd)
  structure(
    list(logml = logml, niter = n_samples, method = "importance sampling", repetitions = 1, pareto_k = k),
    class = c("metabeta_bridge", "bridge")
  )
}

#' @export
print.metabeta_bridge <- function(x, ...) {
  cat(
    "Importance sampling estimate of the log marginal likelihood: ", round(x$logml, 5),
    "\nEstimate obtained from ", x$niter, " draws of a Student-t fitted to the posterior ",
    "(Pareto k of the weights: ", round(x$pareto_k, 2), ").\n",
    sep = ""
  )
  invisible(x)
}

# The fits must model the same response for their marginal likelihoods to be comparable.
check_same_response <- function(models) {
  y <- lapply(models, function(m) m$metabeta$design$y)
  if (!all(vapply(y[-1], function(v) isTRUE(all.equal(v, y[[1]], check.attributes = FALSE)), TRUE))) {
    stop2("Responses do not match. Model comparisons are only valid if the response is the same in all models.")
  }
}

logml_of <- function(x) if (inherits(x, "bridge")) x$logml else bridge_sampler(x)$logml

#' Bayes factors and posterior model probabilities
#'
#' Bayes factors between models and posterior model probabilities, from the
#' marginal likelihoods estimated by [bridge_sampler()].
#'
#' @param x1,x2,x Fitted models (or results of [bridge_sampler()]).
#' @param log Return the log Bayes factor?
#' @param ... More fitted models (for `post_prob()`); otherwise ignored.
#' @param prior_prob Prior model probabilities; equal by default.
#' @param model_names Names of the models; by default, the argument expressions.
#' @return `bayes_factor()`: an object of class `bf_bridge` with the Bayes
#'   factor of `x1` over `x2` in element `bf`. `post_prob()`: a named vector of
#'   posterior model probabilities.
#' @seealso [bridge_sampler()]
#' @export
bayes_factor <- function(x1, x2, log = FALSE, ...) UseMethod("bayes_factor")

#' @rdname bayes_factor
#' @export
bayes_factor.metabetafit <- function(x1, x2, log = FALSE, ...) {
  names <- c(deparse1(substitute(x1)), deparse1(substitute(x2)))
  if (is.brmsfit(x2)) check_same_response(list(x1, x2))
  bayes_factor_result(logml_of(x1), logml_of(x2), log, names)
}

#' @rdname bayes_factor
#' @export
bayes_factor.bridge <- function(x1, x2, log = FALSE, ...) {
  bayes_factor_result(logml_of(x1), logml_of(x2), log, c(deparse1(substitute(x1)), deparse1(substitute(x2))))
}

bayes_factor_result <- function(logml1, logml2, log, names) {
  bf <- logml1 - logml2
  structure(list(bf = if (log) bf else exp(bf), log = log), model_names = names, class = c("metabeta_bf", "bf_bridge"))
}

#' @export
print.metabeta_bf <- function(x, ...) {
  names <- attr(x, "model_names") %||% c("x1", "x2")
  cat(
    "Estimated ", if (x$log) "log ", "Bayes factor in favor of ", names[1], " over ", names[2],
    ": ", formatC(x$bf, digits = 5, format = "f"), "\n",
    sep = ""
  )
  invisible(x)
}

#' @rdname bayes_factor
#' @export
post_prob <- function(x, ..., prior_prob = NULL, model_names = NULL) UseMethod("post_prob")

#' @rdname bayes_factor
#' @export
post_prob.metabetafit <- function(x, ..., prior_prob = NULL, model_names = NULL) {
  models <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)$models
  if (length(models) < 2) stop2("At least two models are required to compute posterior model probabilities.")
  check_same_response(models)
  prior_prob <- prior_prob %||% rep(1 / length(models), length(models))
  if (length(prior_prob) != length(models)) {
    stop2("Number of objects/logml-values needs to match number of elements in prior_prob.")
  }
  if (!isTRUE(all.equal(sum(prior_prob), 1))) stop2("Prior model probabilities do not sum to one.")
  logml <- vapply(models, logml_of, 0)
  w <- prior_prob * exp(logml - max(logml))
  stats::setNames(w / sum(w), names(models))
}
