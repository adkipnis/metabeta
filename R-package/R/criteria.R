# Model comparison and fit: loo(), waic(), kfold(), add_criterion(), loo_compare(),
# loo_model_weights(), bayes_R2() and loo_R2().

# The fitted models among x and '...', named like brms names them (by the argument
# expressions, or 'model_names'), and the remaining arguments.
fitted_models <- function(x, dots, names, model_names = NULL) {
  is_fit <- vapply(dots, is.brmsfit, TRUE)
  models <- c(list(x), dots[is_fit])
  names(models) <- model_names %||% names[c(TRUE, is_fit)]
  list(models = models, args = dots[!is_fit])
}

# The expressions passed as '...', as text (forwarded dots keep the caller's expressions).
dots_names <- function(...) vapply(as.list(substitute(list(...)))[-1], deparse1, "")

# Chain of each draw, for loo's relative efficiencies.
chain_ids <- function(x) rep(seq_len(nchains(x)), each = niterations(x))

compute_loo <- function(x, model_name, criterion = c("loo", "waic"), save_psis = FALSE) {
  ll <- log_lik(x)
  out <- if (match.arg(criterion) == "loo") {
    loo::loo(ll, r_eff = loo::relative_eff(exp(ll), chain_id = chain_ids(x)), save_psis = save_psis)
  } else {
    suppressWarnings(loo::waic(ll))
  }
  attr(out, "model_name") <- model_name
  out
}

# One criterion: returned as is. Several: a "loolist" with their comparison (as in brms).
loo_result <- function(loos, compare) {
  if (length(loos) == 1) return(loos[[1]])
  structure(list(loos = loos, diffs = if (compare) loo::loo_compare(loos)), class = "loolist")
}

#' @export
print.loolist <- function(x, ...) {
  for (name in names(x$loos)) {
    cat(paste0("Output of model '", name, "':\n"))
    print(x$loos[[name]], ...)
    cat("\n")
  }
  if (!is.null(x$diffs)) {
    cat("Model comparisons:\n")
    print(x$diffs, ...)
  }
  invisible(x)
}

#' Efficient approximate leave-one-out cross-validation
#'
#' PSIS-LOO and WAIC as in \pkg{brms}. With `reloo = TRUE`, observations whose
#' Pareto k exceeds `k_threshold` are left out by refitting the model (fast with
#' metabeta).
#'
#' @param x A fitted model.
#' @param ... More fitted models to compare.
#' @param compare Compare several models?
#' @param reloo Refit the model for observations with high Pareto k?
#' @param k_threshold Pareto k threshold for `reloo`.
#' @param save_psis Save the PSIS object?
#' @param moment_match Not supported by metabeta.
#' @param resp,pointwise,moment_match_args,reloo_args,model_names See \pkg{brms}.
#' @return An object of class `loo`, or `loolist` for several models.
#' @name loo-brms
NULL

#' @rdname loo-brms
#' @export
loo.metabetafit <- function(x, ..., compare = TRUE, resp = NULL, pointwise = FALSE, moment_match = FALSE,
                            reloo = FALSE, k_threshold = 0.7, save_psis = FALSE,
                            moment_match_args = list(), reloo_args = list(), model_names = NULL) {
  if (moment_match) stop2("Moment matching needs the Stan model and is not supported by metabeta; use reloo = TRUE.")
  fits <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)$models
  loos <- lapply(names(fits), function(name) {
    out <- compute_loo(fits[[name]], name, "loo", save_psis)
    if (reloo) out <- reloo_observations(fits[[name]], out, k_threshold)
    out
  })
  loo_result(stats::setNames(loos, names(fits)), compare)
}

#' @rdname loo-brms
#' @export
waic.metabetafit <- function(x, ..., compare = TRUE, resp = NULL, pointwise = FALSE, model_names = NULL) {
  fits <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)$models
  loos <- lapply(names(fits), function(name) compute_loo(fits[[name]], name, "waic"))
  loo_result(stats::setNames(loos, names(fits)), compare)
}

# Exact leave-one-out for observations with high Pareto k, by refitting without them.
reloo_observations <- function(x, loo, k_threshold) {
  bad <- which(loo::pareto_k_values(loo) > k_threshold)
  if (!length(bad)) return(loo)
  message(length(bad), " problematic observation(s) found.\nThe model will be refit ", length(bad), " times.")
  for (i in bad) {
    fit <- suppressWarnings(update(x, newdata = x$data[-i, , drop = FALSE]))
    elpd <- log_mean_exp(log_lik(fit, newdata = x$data[i, , drop = FALSE], allow_new_levels = TRUE)[, 1])
    lpd <- loo$pointwise[i, "elpd_loo"] + loo$pointwise[i, "p_loo"]  # in-sample density
    loo$pointwise[i, c("elpd_loo", "p_loo", "looic")] <- c(elpd, lpd - elpd, -2 * elpd)
    loo$diagnostics$pareto_k[i] <- 0
  }
  n <- nrow(loo$pointwise)
  for (col in c("elpd_loo", "p_loo", "looic")) {
    loo$estimates[col, ] <- c(sum(loo$pointwise[, col]), sqrt(n * stats::var(loo$pointwise[, col])))
  }
  loo
}

#' K-fold cross-validation
#'
#' Refits the model K times, each time without one fold of the data, as in
#' \pkg{brms}. Refits are fast with metabeta.
#'
#' @param x A fitted model.
#' @param ... More fitted models to compare.
#' @param K Number of folds.
#' @param folds `NULL` (random folds), `"stratified"`, `"grouped"`, `"loo"`, or a
#'   vector with the fold of each observation.
#' @param group Variable to stratify or group the folds by.
#' @param compare Compare several models?
#' @param model_names Names of the models.
#' @param save_fits Keep the fits (needed for [kfold_predict()])?
#' @param Ksub,joint,resp,recompile,future_args Accepted for compatibility.
#' @return An object of class `kfold`.
#' @export
kfold.metabetafit <- function(x, ..., K = 10, Ksub = NULL, folds = NULL, group = NULL, joint = FALSE,
                              compare = TRUE, resp = NULL, model_names = NULL, save_fits = FALSE,
                              recompile = NULL, future_args = list()) {
  fits <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)$models
  out <- lapply(names(fits), function(name) kfold_model(fits[[name]], name, K, folds, group, save_fits))
  loo_result(stats::setNames(out, names(fits)), compare)
}

kfold_model <- function(x, model_name, K, folds, group, save_fits) {
  data <- x$data
  N <- nrow(data)
  if (is.null(folds)) {
    fold <- loo::kfold_split_random(K, N)
  } else if (is.numeric(folds)) {
    fold <- folds
  } else {
    fold <- switch(match.arg(folds, c("stratified", "grouped", "loo")),
      stratified = loo::kfold_split_stratified(K, data[[group]]),
      grouped = loo::kfold_split_grouped(K, data[[group]]),
      loo = seq_len(N)
    )
  }
  # out-of-fold log-likelihood of each observation, then its log predictive density
  ll <- matrix(NA_real_, ndraws(x), N)
  fits <- list()
  for (k in sort(unique(fold))) {
    held_out <- fold == k
    fit <- suppressWarnings(update(x, newdata = data[!held_out, , drop = FALSE]))
    ll_k <- log_lik(fit, newdata = data[held_out, , drop = FALSE], allow_new_levels = TRUE)
    ll[seq_len(nrow(ll_k)), held_out] <- ll_k
    if (save_fits) fits[[length(fits) + 1]] <- fit
  }
  elpd <- apply(ll, 2, function(v) log_mean_exp(v[!is.na(v)]))
  lpd <- apply(log_lik(x), 2, log_mean_exp)
  pointwise <- cbind(elpd_kfold = elpd, p_kfold = lpd - elpd, kfoldic = -2 * elpd)
  estimates <- cbind(Estimate = colSums(pointwise), SE = sqrt(N * apply(pointwise, 2, stats::var)))
  out <- list(estimates = estimates, pointwise = pointwise, folds = fold)
  if (save_fits) out[c("fits", "data", "y")] <- list(fits, data, get_y(x))
  structure(out, class = c("kfold", "loo"), K = length(unique(fold)), model_name = model_name)
}

#' Add model fit criteria to model objects
#'
#' @param x A fitted model.
#' @param criterion `"loo"`, `"waic"`, `"kfold"`, `"bayes_R2"`, `"loo_R2"` and/or
#'   `"marglik"` (see [bridge_sampler()]).
#' @param model_name Name of the model in comparisons.
#' @param overwrite Recompute criteria that are already stored?
#' @param file File to save the updated model to.
#' @param force_save Save even if nothing changed?
#' @param ... Passed to the criterion functions.
#' @return The model, with the criteria in `x$criteria`.
#' @export
add_criterion <- function(x, ...) UseMethod("add_criterion")

#' @rdname add_criterion
#' @export
add_criterion.metabetafit <- function(x, criterion, model_name = NULL, overwrite = FALSE, file = NULL,
                                      force_save = FALSE, ...) {
  unknown <- setdiff(criterion, c("loo", "waic", "kfold", "bayes_R2", "loo_R2", "marglik"))
  if (length(unknown)) stop2("Criteria ", collapse_comma(unknown), " are not supported by metabeta.")
  model_name <- model_name %||% deparse1(substitute(x))
  new <- setdiff(criterion, if (!overwrite) names(x$criteria))
  for (crit in new) {
    x$criteria[[crit]] <- switch(crit,
      loo = compute_loo(x, model_name, "loo"),
      waic = compute_loo(x, model_name, "waic"),
      kfold = kfold(x, ..., model_names = model_name),
      bayes_R2 = bayes_R2(x, summary = FALSE),
      loo_R2 = loo_R2(x, summary = FALSE),
      marglik = bridge_sampler(x)
    )
  }
  file <- file %||% x$file
  if (!is.null(file) && (length(new) || force_save)) saveRDS(x, file = file)
  x
}

#' @export
loo_compare.metabetafit <- function(x, ..., criterion = c("loo", "waic", "kfold"), model_names = NULL) {
  criterion <- match.arg(criterion)
  fits <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)$models
  loos <- lapply(names(fits), function(name) {
    fits[[name]]$criteria[[criterion]] %||%
      stop2("Model '", name, "' does not contain a precomputed '", criterion, "' criterion. See ?loo_compare.brmsfit for help.")
  })
  loo::loo_compare(stats::setNames(loos, names(fits)))
}

#' @export
loo_model_weights.metabetafit <- function(x, ..., model_names = NULL) {
  split <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)
  lls <- lapply(split$models, log_lik)
  r_effs <- lapply(seq_along(lls), function(i) loo::relative_eff(exp(lls[[i]]), chain_id = chain_ids(split$models[[i]])))
  do.call(loo::loo_model_weights, c(list(lls, r_eff_list = r_effs), split$args))
}

#' Bayesian R2
#'
#' The Bayesian R-squared of Gelman et al. (2019) per draw, as in \pkg{brms};
#' `loo_R2()` is its leave-one-out version.
#'
#' @param object A fitted model.
#' @param summary Summarize the draws?
#' @param robust Use median and MAD instead of mean and SD?
#' @param probs Quantiles of the summary.
#' @param seed Seed for the Bayesian bootstrap of `loo_R2()`.
#' @param resp,args_epred,args_loglik Accepted for compatibility.
#' @param ... Passed to [posterior_epred()].
#' @return A one-row summary matrix, or a draws x 1 matrix.
#' @name R2-brms
NULL

#' @rdname R2-brms
#' @export
bayes_R2.metabetafit <- function(object, resp = NULL, summary = TRUE, robust = FALSE,
                                 probs = c(0.025, 0.975), ...) {
  prediction <- posterior_epred(object, ...)
  error <- -sweep(prediction, 2, get_y(object))
  var_prediction <- apply(prediction, 1, stats::var)
  R2 <- matrix(var_prediction / (var_prediction + apply(error, 1, stats::var)), dimnames = list(NULL, "R2"))
  if (summary) summarise_matrix(R2, probs, robust) else R2
}

#' @rdname R2-brms
#' @export
loo_R2.metabetafit <- function(object, resp = NULL, summary = TRUE, robust = FALSE,
                               probs = c(0.025, 0.975), seed = NULL, args_epred = list(),
                               args_loglik = list(), ...) {
  if (!is.null(seed)) set.seed(seed)
  y <- get_y(object)
  ll <- log_lik(object)
  psis <- suppressWarnings(loo::psis(-ll, r_eff = loo::relative_eff(exp(ll), chain_id = chain_ids(object))))
  error <- loo::E_loo(posterior_epred(object), psis, log_ratios = -ll)$value - y
  # Bayesian bootstrap: variances under Dirichlet weights over the observations
  N <- length(y)
  w <- matrix(stats::rexp(ndraws(object) * N), ncol = N)
  w <- w / rowSums(w)
  weighted_var <- function(v) (N / (N - 1)) * (w %*% v^2 - (w %*% v)^2)
  R2 <- matrix(1 - weighted_var(error) / weighted_var(y), dimnames = list(NULL, "R2"))
  if (summary) summarise_matrix(R2, probs, robust) else R2
}

#' Leave-one-out predictions
#'
#' PSIS-weighted leave-one-out predictive means, variances or quantiles, as in
#' \pkg{brms}.
#'
#' @param object A fitted model.
#' @param type `"mean"`, `"var"` or `"quantile"`.
#' @param probs Quantiles for `type = "quantile"`.
#' @param psis_object A precomputed PSIS object.
#' @param prob Probability mass of the interval.
#' @param resp,... Accepted for compatibility.
#' @return A vector or matrix of leave-one-out predictions.
#' @name loo_predict
NULL

loo_expectation <- function(object, draws, type, probs, psis_object) {
  ll <- log_lik(object)
  psis_object <- psis_object %||% suppressWarnings(loo::psis(-ll, r_eff = loo::relative_eff(exp(ll), chain_id = chain_ids(object))))
  loo::E_loo(draws, psis_object, type = match.arg(type, c("mean", "var", "quantile")), probs = probs, log_ratios = -ll)$value
}

#' @rdname loo_predict
#' @export
loo_predict <- function(object, ...) UseMethod("loo_predict")

#' @rdname loo_predict
#' @export
loo_predict.metabetafit <- function(object, type = c("mean", "var", "quantile"), probs = 0.5,
                                    psis_object = NULL, resp = NULL, ...) {
  loo_expectation(object, posterior_predict(object), type[1], probs, psis_object)
}

#' @rdname loo_predict
#' @export
loo_epred <- function(object, ...) UseMethod("loo_epred")

#' @rdname loo_predict
#' @export
loo_epred.metabetafit <- function(object, type = c("mean", "var", "quantile"), probs = 0.5,
                                  psis_object = NULL, resp = NULL, ...) {
  loo_expectation(object, posterior_epred(object), type[1], probs, psis_object)
}

#' @rdname loo_predict
#' @export
loo_linpred <- function(object, ...) UseMethod("loo_linpred")

#' @rdname loo_predict
#' @export
loo_linpred.metabetafit <- function(object, type = c("mean", "var", "quantile"), probs = 0.5,
                                    psis_object = NULL, resp = NULL, ...) {
  loo_expectation(object, posterior_linpred(object), type[1], probs, psis_object)
}

#' @rdname loo_predict
#' @export
loo_predictive_interval <- function(object, ...) UseMethod("loo_predictive_interval")

#' @rdname loo_predict
#' @export
loo_predictive_interval.metabetafit <- function(object, prob = 0.9, psis_object = NULL, ...) {
  probs <- c((1 - prob) / 2, (1 + prob) / 2)
  out <- t(loo_expectation(object, posterior_predict(object), "quantile", probs, psis_object))
  colnames(out) <- paste0(probs * 100, "%")
  out
}
