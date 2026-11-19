# Further functions of brms: deprecated aliases, model averaging, refitting helpers, small
# utilities and the Student-t distribution.

# Deprecated aliases -------------------------------------------------------------------

#' Deprecated aliases of brms
#'
#' `LOO()`, `WAIC()`, `add_loo()`, `add_waic()`, `add_ic()`, `compare_ic()`,
#' `parnames()` and `pp_expect()` are kept for scripts written for older
#' versions of \pkg{brms}.
#'
#' @param x,object A fitted model (`compare_ic()`: a list of models or criteria).
#' @param ... Further models or arguments of the current functions.
#' @param ic,value Criteria to add.
#' @param model_name Name of the model.
#' @inheritParams loo-brms
#' @return See [loo()], [waic()], [add_criterion()], [variables()] and
#'   [posterior_epred()].
#' @name deprecated-brms
NULL

#' @rdname deprecated-brms
#' @export
LOO <- function(x, ...) UseMethod("LOO")

#' @rdname deprecated-brms
#' @export
LOO.metabetafit <- function(x, ..., compare = TRUE, resp = NULL, pointwise = FALSE, moment_match = FALSE,
                            reloo = FALSE, k_threshold = 0.7, save_psis = FALSE,
                            moment_match_args = list(), reloo_args = list(), model_names = NULL) {
  # call loo() with the caller's argument expressions, which become the model names
  call <- match.call()
  call[[1]] <- loo.metabetafit
  eval(call, parent.frame())
}

#' @rdname deprecated-brms
#' @export
WAIC <- function(x, ...) UseMethod("WAIC")

#' @rdname deprecated-brms
#' @export
WAIC.metabetafit <- function(x, ..., compare = TRUE, resp = NULL, pointwise = FALSE, model_names = NULL) {
  call <- match.call()
  call[[1]] <- waic.metabetafit
  eval(call, parent.frame())
}

#' @rdname deprecated-brms
#' @export
add_loo <- function(x, model_name = NULL, ...) {
  add_criterion(x, "loo", model_name = model_name %||% deparse1(substitute(x)), ...)
}

#' @rdname deprecated-brms
#' @export
add_waic <- function(x, model_name = NULL, ...) {
  add_criterion(x, "waic", model_name = model_name %||% deparse1(substitute(x)), ...)
}

#' @rdname deprecated-brms
#' @export
add_ic <- function(x, ...) UseMethod("add_ic")

#' @rdname deprecated-brms
#' @export
add_ic.metabetafit <- function(x, ic = "loo", model_name = NULL, ...) {
  add_criterion(x, ic, model_name = model_name %||% deparse1(substitute(x)), ...)
}

#' @rdname deprecated-brms
#' @export
"add_ic<-" <- function(x, ..., value) add_ic(x, ic = value, ...)

#' @rdname deprecated-brms
#' @export
compare_ic <- function(..., x = NULL, ic = c("loo", "waic", "kfold")) {
  ic <- match.arg(ic)
  objects <- c(list(...), x)
  names(objects) <- names(objects) %||% paste0("model", seq_along(objects))
  criteria <- lapply(names(objects), function(name) {
    object <- objects[[name]]
    if (!is.brmsfit(object)) return(object)
    object$criteria[[ic]] %||% compute_loo(object, name, if (ic == "waic") "waic" else "loo")
  })
  loo::loo_compare(stats::setNames(criteria, names(objects)))
}

#' @rdname deprecated-brms
#' @export
parnames <- function(x, ...) UseMethod("parnames")

#' @rdname deprecated-brms
#' @export
parnames.metabetafit <- function(x, ...) variables(x)

#' @rdname deprecated-brms
#' @export
pp_expect <- function(object, ...) UseMethod("pp_expect")

#' @rdname deprecated-brms
#' @export
pp_expect.metabetafit <- function(object, ...) posterior_epred(object, ...)

#' Restructure old fitted models
#'
#' metabeta fits never need restructuring; they are returned unchanged.
#'
#' @param x A fitted model.
#' @param ... Ignored.
#' @return `x`.
#' @export
restructure <- function(x, ...) UseMethod("restructure")

#' @rdname restructure
#' @export
restructure.metabetafit <- function(x, ...) x

# Refitting -------------------------------------------------------------------------

#' Exact cross-validation for problematic observations
#'
#' Refits the model without each observation whose Pareto k exceeds
#' `k_threshold`, as in \pkg{brms}.
#'
#' @param x A fitted model, or a `loo` object for `reloo.loo()`.
#' @param loo A `loo` object of `x`; computed if `NULL`.
#' @param k_threshold Pareto k threshold.
#' @param fit The fitted model, for `reloo.loo()`.
#' @param newdata,resp,check,recompile,future_args Accepted for compatibility.
#' @param ... Ignored.
#' @return The updated `loo` object.
#' @export
reloo <- function(x, ...) UseMethod("reloo")

#' @rdname reloo
#' @export
reloo.metabetafit <- function(x, loo = NULL, k_threshold = 0.7, newdata = NULL, resp = NULL,
                              check = TRUE, recompile = NULL, future_args = list(), ...) {
  reloo_observations(x, loo %||% compute_loo(x, deparse1(substitute(x))), k_threshold)
}

#' @rdname reloo
#' @export
reloo.loo <- function(x, fit, ...) reloo.metabetafit(fit, loo = x, ...)

#' @export
psis.metabetafit <- function(log_ratios, newdata = NULL, resp = NULL, model_name = NULL, ...) {
  # in brms, the fitted model is passed as 'log_ratios'
  fit <- log_ratios
  ll <- log_lik(fit, newdata = newdata)
  loo::psis(-ll, r_eff = loo::relative_eff(exp(ll), chain_id = chain_ids(fit)), ...)
}

#' Predictions from K-fold cross-validation
#'
#' @param x A `kfold` object computed with `save_fits = TRUE`.
#' @param method `"posterior_predict"` or `"posterior_epred"`.
#' @param resp Accepted for compatibility.
#' @param ... Passed to the prediction method.
#' @return A list with the responses `y` and a draws x observations matrix
#'   `yrep` of out-of-fold predictions.
#' @export
kfold_predict <- function(x, method = "posterior_predict", resp = NULL, ...) {
  if (is.null(x$fits)) stop2("'x' must be computed by kfold() with 'save_fits = TRUE'.")
  predict_draws <- if (method %in% c("posterior_epred", "fitted")) posterior_epred else posterior_predict
  yrep <- NULL
  for (k in seq_along(x$fits)) {
    held_out <- x$folds == sort(unique(x$folds))[k]
    draws <- predict_draws(x$fits[[k]], newdata = x$data[held_out, , drop = FALSE], allow_new_levels = TRUE, ...)
    yrep <- yrep %||% matrix(NA_real_, nrow(draws), length(x$y))
    yrep[, held_out] <- draws[seq_len(nrow(yrep)), ]
  }
  list(y = x$y, yrep = yrep)
}

#' Combine fits of the same model
#'
#' Combines the draws of several fits of the same model and data (e.g. with
#' different seeds) as additional chains.
#'
#' @param ... Fitted models.
#' @param mlist A list of fitted models.
#' @param check_data Check that all models were fitted to the same data?
#' @return A fitted model.
#' @export
combine_models <- function(..., mlist = NULL, check_data = TRUE) {
  models <- c(list(...), mlist)
  first <- models[[1]]
  for (m in models[-1]) {
    if (!identical(variables(m), variables(first)) || niterations(m) != niterations(first)) {
      stop2("Models must have the same variables and number of iterations.")
    }
    if (check_data && !isTRUE(all.equal(m$data, first$data))) stop2("Models must be fitted to the same data.")
  }
  first$fit <- do.call(posterior::bind_draws, c(lapply(models, `[[`, "fit"), along = "chain"))
  first$criteria <- list()
  first
}

#' Check whether a fitted model needs to be refitted
#'
#' @param fit A fitted model.
#' @param data,algorithm New data or algorithm to compare with those of `fit`.
#' @param sdata,scode Ignored (metabeta has no Stan data or code).
#' @param silent,verbose Report the reason for refitting?
#' @return `TRUE` or `FALSE`.
#' @export
brmsfit_needs_refit <- function(fit, sdata = NULL, scode = NULL, data = NULL, algorithm = NULL,
                                silent = FALSE, verbose = FALSE) {
  data_changed <- !is.null(data) && !isTRUE(all.equal(as.data.frame(data)[names(fit$data)], fit$data, check.attributes = FALSE))
  algorithm_changed <- !is.null(algorithm) && algorithm != fit$algorithm
  if (data_changed && !silent) message("The data have changed.")
  if (algorithm_changed && !silent) message("The algorithm has changed.")
  data_changed || algorithm_changed
}

# Model averaging ---------------------------------------------------------------------

#' Model weighting and averaging
#'
#' Weights of several models (`"stacking"`, `"pseudobma"`, `"loo"`, `"waic"`,
#' `"kfold"` or numeric weights), and posterior or predictive draws averaged
#' across the models with these weights, as in \pkg{brms}.
#'
#' @param x A fitted model.
#' @param ... More fitted models.
#' @param weights Weighting method or numeric weights.
#' @param model_names Names of the models.
#' @param method Prediction method of `pp_average()`.
#' @param ndraws,nsamples Total number of draws.
#' @param summary,probs,robust Summarize the averaged predictions?
#' @param variable,pars Variables to average in `posterior_average()`.
#' @param missing Values of variables missing in some models.
#' @param control Passed to [loo::loo_model_weights()].
#' @param seed Seed for the selection of draws.
#' @return Weights, or averaged draws.
#' @name model_weights
NULL

#' @rdname model_weights
#' @export
model_weights <- function(x, ...) UseMethod("model_weights")

#' @rdname model_weights
#' @export
model_weights.metabetafit <- function(x, ..., weights = "stacking", model_names = NULL) {
  fits <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)$models
  weigh_models(fits, weights)
}

weigh_models <- function(models, weights, control = list()) {
  if (is.numeric(weights)) {
    w <- weights
  } else if (weights %in% c("stacking", "pseudobma")) {
    lls <- lapply(models, log_lik)
    r_effs <- lapply(seq_along(lls), function(i) loo::relative_eff(exp(lls[[i]]), chain_id = chain_ids(models[[i]])))
    w <- as.numeric(do.call(loo::loo_model_weights, c(list(lls, method = weights, r_eff_list = r_effs), control)))
  } else {
    # weights proportional to exp(elpd)
    criterion <- if (weights == "waic") "waic" else "loo"
    elpd <- sapply(names(models), function(name) {
      crit <- models[[name]]$criteria[[weights]] %||% compute_loo(models[[name]], name, criterion)
      crit$estimates[1, "Estimate"]
    })
    w <- exp(elpd - max(elpd))
  }
  stats::setNames(w / sum(w), names(models))
}

# Number of draws taken from each model for a total of 'ndraws'.
draws_per_model <- function(weights, ndraws) {
  n <- floor(weights * ndraws)
  n[which.max(weights)] <- n[which.max(weights)] + ndraws - sum(n)
  n
}

#' @rdname model_weights
#' @export
pp_average <- function(x, ...) UseMethod("pp_average")

#' @rdname model_weights
#' @export
pp_average.metabetafit <- function(x, ..., weights = "stacking", method = "posterior_predict",
                                   ndraws = NULL, nsamples = NULL, summary = TRUE,
                                   probs = c(0.025, 0.975), robust = FALSE, model_names = NULL,
                                   control = list(), seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  split <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)
  w <- weigh_models(split$models, weights, control)
  n <- draws_per_model(w, ndraws %||% nsamples %||% ndraws(x))
  predict_draws <- switch(method,
    posterior_epred = , fitted = posterior_epred,
    posterior_linpred = posterior_linpred,
    posterior_predict
  )
  draws <- do.call(rbind, lapply(which(n > 0), function(i) do.call(predict_draws, c(list(split$models[[i]], ndraws = n[i]), split$args))))
  attr(draws, "weights") <- w
  summarise_predictions(draws, summary, probs, robust)
}

#' @rdname model_weights
#' @export
posterior_average <- function(x, ...) UseMethod("posterior_average")

#' @rdname model_weights
#' @export
posterior_average.metabetafit <- function(x, ..., variable = NULL, pars = NULL, weights = "stacking",
                                          ndraws = NULL, nsamples = NULL, missing = NULL,
                                          model_names = NULL, control = list(), seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  fits <- fitted_models(x, list(...), c(deparse1(substitute(x)), dots_names(...)), model_names)$models
  variable <- variable %||% pars %||% plot_variables(x)
  w <- weigh_models(fits, weights, control)
  n <- draws_per_model(w, ndraws %||% nsamples %||% ndraws(x))
  out <- lapply(which(n > 0), function(i) {
    found <- intersect(variable, variables(fits[[i]]))
    absent <- setdiff(variable, found)
    if (length(absent) && is.null(missing)) {
      stop2("Variables ", collapse_comma(absent), " are missing in model '", names(fits)[i], "'. Use argument 'missing' to fill them in.")
    }
    draws <- as.data.frame(as.matrix(fits[[i]], variable = found, draw = sample.int(ndraws(fits[[i]]), n[i], replace = TRUE)))
    for (v in absent) draws[[v]] <- if (is.list(missing)) missing[[v]] else missing
    draws[variable]
  })
  out <- do.call(rbind, out)
  attr(out, "weights") <- w
  out
}

# Utilities ---------------------------------------------------------------------------

#' Validate new data
#'
#' Checks that `newdata` contains everything needed to predict from `object`, as
#' in \pkg{brms}, and returns it.
#'
#' @param newdata A data frame.
#' @param object A fitted model.
#' @param re_formula,allow_new_levels See [posterior_predict()].
#' @param check_response Must the response be present?
#' @param newdata2,resp,incl_autocor,group_vars,req_vars,... Accepted for compatibility.
#' @return The data frame.
#' @export
validate_newdata <- function(newdata, object, re_formula = NULL, allow_new_levels = FALSE,
                             newdata2 = NULL, resp = NULL, check_response = TRUE,
                             incl_autocor = TRUE, group_vars = NULL, req_vars = NULL, ...) {
  newdata <- as.data.frame(newdata)
  design_newdata(object$metabeta$design, newdata, length(group_coefs(object, re_formula)) > 0, allow_new_levels)
  missing_vars <- setdiff(all.vars(response(object$formula$formula)), names(newdata))
  if (check_response && length(missing_vars)) {
    stop2("The following variables can neither be found in 'data' nor in 'data2':\n", collapse_comma(missing_vars))
  }
  newdata
}

#' Transform into a brmsprior object
#' @param x A data frame with the columns of a `brmsprior` object.
#' @return An object of class `brmsprior`.
#' @export
as.brmsprior <- function(x) new_brmsprior(x)

#' Settings accepted by brm() for compatibility
#'
#' `save_pars()`, `threading()` and `opencl()` create the settings objects of
#' \pkg{brms}; metabeta stores all draws and does not run Stan, so they have no
#' effect. Use `options(metabeta.device = "cuda")` to run the network on a GPU.
#'
#' @param group,latent,all,manual,threads,grainsize,static,force,ids See \pkg{brms}.
#' @return A settings object.
#' @export
save_pars <- function(group = TRUE, latent = FALSE, all = FALSE, manual = NULL) {
  structure(list(group = group, latent = latent, all = all, manual = manual), class = "save_pars")
}

#' @rdname save_pars
#' @export
threading <- function(threads = NULL, grainsize = NULL, static = FALSE, force = FALSE) {
  structure(list(threads = threads, grainsize = grainsize, static = static, force = force), class = "brmsthreads")
}

#' @rdname save_pars
#' @export
opencl <- function(ids = NULL) structure(list(ids = ids), class = "brmsopencl")

#' Shrinkage priors
#'
#' The shrinkage priors of \pkg{brms}, returned as prior strings. metabeta's
#' network was not trained on them, so [brm()] rejects them.
#'
#' @param df,scale_global,df_global,scale_slab,df_slab,par_ratio,autoscale,main,scale,mean_R2,prec_R2,cons_D2
#'   See \pkg{brms}.
#' @return A character string.
#' @export
horseshoe <- function(df = 1, scale_global = 1, df_global = 1, scale_slab = 2, df_slab = 4,
                      par_ratio = NULL, autoscale = TRUE, main = FALSE) {
  sprintf("horseshoe(%s, scale_global = %s)", df, scale_global)
}

#' @rdname horseshoe
#' @export
lasso <- function(df = 1, scale = 1) sprintf("lasso(%s, %s)", df, scale)

#' @rdname horseshoe
#' @export
R2D2 <- function(mean_R2 = 0.5, prec_R2 = 2, cons_D2 = 0.5, autoscale = TRUE, main = FALSE) {
  sprintf("R2D2(%s, %s, %s)", mean_R2, prec_R2, cons_D2)
}

#' Class checks of brms objects
#' @param x An R object.
#' @return `TRUE` or `FALSE`.
#' @name is-brms
NULL

#' @rdname is-brms
#' @export
is.brmsfit_multiple <- function(x) inherits(x, "brmsfit_multiple")

#' @rdname is-brms
#' @export
is.mvbrmsformula <- function(x) inherits(x, "mvbrmsformula")

#' @rdname is-brms
#' @export
is.brmsterms <- function(x) inherits(x, "brmsterms")

#' @rdname is-brms
#' @export
is.mvbrmsterms <- function(x) inherits(x, "mvbrmsterms")

#' @rdname is-brms
#' @export
is.cor_brms <- function(x) inherits(x, "cor_brms")

#' @rdname is-brms
#' @export
is.cor_arma <- function(x) inherits(x, "cor_arma")

#' @rdname is-brms
#' @export
is.cor_cosy <- function(x) inherits(x, "cor_cosy")

#' @rdname is-brms
#' @export
is.cor_sar <- function(x) inherits(x, "cor_sar")

#' @rdname is-brms
#' @export
is.cor_car <- function(x) inherits(x, "cor_car")

#' @rdname is-brms
#' @export
is.cor_fixed <- function(x) inherits(x, "cor_fixed")

#' Convert to coda objects
#'
#' @param x A fitted model.
#' @param pars,fixed Variables to extract.
#' @param combine_chains Return one `mcmc` object instead of an `mcmc.list`?
#' @param inc_warmup Ignored.
#' @param ... Ignored.
#' @return An `mcmc.list` (or `mcmc`) object; needs the package \pkg{coda}.
#' @export
as.mcmc <- function(x, ...) UseMethod("as.mcmc")

#' @rdname as.mcmc
#' @export
as.mcmc.default <- function(x, ...) coda::as.mcmc(x, ...)

#' @rdname as.mcmc
#' @export
as.mcmc.metabetafit <- function(x, pars = NA, fixed = FALSE, combine_chains = FALSE, inc_warmup = FALSE, ...) {
  draws <- as.array(x, pars = pars)
  chain <- function(c) coda::mcmc(matrix(draws[, c, ], ncol = dim(draws)[3], dimnames = list(NULL, dimnames(draws)[[3]])))
  if (combine_chains) coda::mcmc(as.matrix(x, pars = pars)) else coda::mcmc.list(lapply(seq_len(dim(draws)[2]), chain))
}

# Distributions -----------------------------------------------------------------------

#' The location-scale Student-t distribution
#'
#' As in \pkg{brms}; the distribution of metabeta's `student_t(5, ...)` priors.
#'
#' @param x,q Quantiles.
#' @param p Probabilities.
#' @param n Number of draws.
#' @param df Degrees of freedom.
#' @param mu Location.
#' @param sigma Scale.
#' @param log,log.p Return logarithms?
#' @param lower.tail Return `P[X <= x]` (otherwise `P[X > x]`)?
#' @return A numeric vector.
#' @export
dstudent_t <- function(x, df, mu = 0, sigma = 1, log = FALSE) {
  d <- dt((x - mu) / sigma, df, log = TRUE) - log(sigma)
  if (log) d else exp(d)
}

#' @rdname dstudent_t
#' @export
pstudent_t <- function(q, df, mu = 0, sigma = 1, lower.tail = TRUE, log.p = FALSE) {
  pt((q - mu) / sigma, df, lower.tail = lower.tail, log.p = log.p)
}

#' @rdname dstudent_t
#' @export
qstudent_t <- function(p, df, mu = 0, sigma = 1, lower.tail = TRUE, log.p = FALSE) {
  mu + sigma * stats::qt(p, df, lower.tail = lower.tail, log.p = log.p)
}

#' @rdname dstudent_t
#' @export
rstudent_t <- function(n, df, mu = 0, sigma = 1) mu + sigma * stats::rt(n, df)

#' Scaled logit, and logarithm and exponential with offsets
#'
#' `logit_scaled()` computes `logit((x - lb) / (ub - lb))`, `logm1()` computes
#' `log(x - 1)` and `expp1()` computes `exp(x) + 1`, as in \pkg{brms}.
#'
#' @param x A numeric vector.
#' @param lb,ub Bounds.
#' @param base Base of the logarithm.
#' @return A numeric vector.
#' @export
logit_scaled <- function(x, lb = 0, ub = 1) qlogis((x - lb) / (ub - lb))

#' @rdname logit_scaled
#' @export
inv_logit_scaled <- function(x, lb = 0, ub = 1) lb + (ub - lb) * plogis(x)

#' @rdname logit_scaled
#' @export
logm1 <- function(x, base = exp(1)) log(x - 1, base = base)

#' @rdname logit_scaled
#' @export
expp1 <- function(x) exp(x) + 1
