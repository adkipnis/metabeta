# Predictions: posterior_predict(), posterior_epred(), posterior_linpred(), log_lik(),
# predict(), fitted() and residuals(). They are computed in R from the stored draws, so a
# saved fit needs no Python to predict.

# Draws of the linear predictor (S x N), plus sigma (S) for gaussian models.
#
# 're_formula' selects the group-level effects: NULL (all), NA or ~0 (none) or a formula
# such as ~ (1 | g). Rows of 'newdata' with levels of the grouping factor that were not in
# the data get effects for new levels, drawn as specified by 'sample_new_levels'.
linear_predictor <- function(object, newdata = NULL, re_formula = NULL, ndraws = NULL, draw_ids = NULL,
                             allow_new_levels = FALSE, sample_new_levels = "uncertainty", ...) {
  design <- object$metabeta$design
  draws <- select_draws(ndraws(object), ndraws, draw_ids)
  coefs <- group_coefs(object, re_formula)
  if (is.null(newdata)) {
    X <- design$X
    Z <- design$Z
    level <- design$observed[design$groups + 1]
  } else {
    new <- design_newdata(design, newdata, need_groups = length(coefs) > 0, allow_new_levels)
    X <- new$X
    Z <- new$Z
    level <- new$index
  }
  b <- as.matrix(object, variable = paste0("b_", design$b_names), draw = draws)
  eta <- tcrossprod(b, X)
  if (length(coefs)) {
    r <- ranef_draws(object)[draws, , , drop = FALSE]
    r <- add_new_levels(object, r, length(attr(level, "new_levels")), sample_new_levels, draws)
    for (k in coefs) {
      effect <- matrix(r[, , k], nrow = length(draws))[, level, drop = FALSE]
      eta <- eta + sweep(effect, 2, Z[, match(k, design$re_names)], "*")
    }
  }
  sigma <- if (design$likelihood == "gaussian") as.matrix(object, variable = "sigma", draw = draws)[, 1]
  list(eta = eta, sigma = sigma, likelihood = design$likelihood)
}

# Group-level coefficients included by 're_formula'.
group_coefs <- function(object, re_formula) {
  design <- object$metabeta$design
  if (is.null(re_formula)) return(design$re_names)
  if (!inherits(re_formula, "formula") && is.na(re_formula)) return(character(0))
  rhs <- re_formula[[length(re_formula)]]
  labels <- attr(stats::terms(stats::as.formula(call("~", rhs))), "term.labels")
  coefs <- character(0)
  for (label in labels[grepl("|", labels, fixed = TRUE)]) {
    bar <- parse_bar(str2lang(label))
    if (bar$group != design$group) stop2("Group '", bar$group, "' in 're_formula' is not part of the model.")
    coefs <- c(coefs, design_block(stats::as.formula(call("~", bar$lhs)), design$frame)$names)
  }
  unknown <- setdiff(coefs, design$re_names)
  if (length(unknown)) stop2("Group-level effects ", collapse_comma(unknown), " in 're_formula' are not part of the model.")
  intersect(design$re_names, coefs)
}

# Append effects for new levels of the grouping factor to r [draws, levels, coefficients]:
#   "uncertainty": each draw takes the effects of a randomly chosen existing level,
#   "old_levels":  all draws take the effects of one randomly chosen existing level,
#   "gaussian":    each draw is drawn from N(0, Sigma).
add_new_levels <- function(object, r, n_new, how, draws) {
  how <- match.arg(how, c("uncertainty", "gaussian", "old_levels"))
  S <- dim(r)[1]
  L <- dim(r)[2]
  out <- array(0, dim(r) + c(0, n_new, 0), dimnames = list(NULL, NULL, dimnames(r)[[3]]))
  out[, seq_len(L), ] <- r
  for (j in L + seq_len(n_new)) {
    if (how == "uncertainty") {
      chosen <- sample.int(L, S, replace = TRUE)
      for (k in seq_len(dim(r)[3])) out[, j, k] <- r[cbind(seq_len(S), chosen, k)]
    } else if (how == "old_levels") {
      out[, j, ] <- r[, sample.int(L, 1), ]
    } else {
      vc <- VarCorr(object, summary = FALSE)[[1]]
      q <- ncol(vc$sd)
      cor <- vc$cor %||% (array(diag(q), c(q, q, nrow(vc$sd))) |> aperm(c(3, 1, 2)))
      out[, j, ] <- draw_mvn(vc$sd[draws, , drop = FALSE], cor[draws, , , drop = FALSE])
    }
  }
  out
}

# The observed response for the fitted data or 'newdata'.
observed_response <- function(object, newdata = NULL) {
  if (is.null(newdata)) return(get_y(object))
  spec <- parse_model_formula(object$formula)
  model_response(spec, as.data.frame(newdata), object$family$family, environment(object$formula$formula))
}

inverse_link <- function(eta, likelihood) {
  switch(likelihood, gaussian = eta, bernoulli = plogis(eta), poisson = exp(eta))
}

# Draws of new responses given the linear predictor.
draw_response <- function(lp) {
  mu <- inverse_link(lp$eta, lp$likelihood)
  n <- length(mu)
  y <- switch(lp$likelihood,
    gaussian = rnorm(n, mu, lp$sigma),  # sigma recycles along the draws (rows)
    bernoulli = rbinom(n, 1, mu),
    poisson = rpois(n, mu)
  )
  matrix(y, nrow(mu))
}

summarise_predictions <- function(draws, summary, probs, robust) {
  if (!summary) return(draws)
  out <- summarise_matrix(draws, probs, robust)
  rownames(out) <- NULL
  out
}

#' Draws from the posterior predictive and related distributions
#'
#' As in \pkg{brms}: `posterior_predict()` draws new responses,
#' `posterior_epred()` the expected response, `posterior_linpred()` the linear
#' predictor and `log_lik()` the pointwise log-likelihood, each as a draws x
#' observations matrix. `predict()`, `fitted()` and `residuals()` summarize them
#' per observation.
#'
#' @param object A fitted model.
#' @param newdata Data to predict for; by default the data of the fit.
#' @param re_formula Group-level terms to include: `NULL` (all), `NA` or `~0`
#'   (none), or a formula such as `~ (1 | g)`.
#' @param re.form Alias of `re_formula`.
#' @param transform A function applied to the draws.
#' @param ndraws,draw_ids Number of random draws, or specific draws, to use.
#' @param summary Summarize the draws per observation?
#' @param robust Use median and MAD instead of mean and SD?
#' @param probs Quantiles of the summary.
#' @param scale `"response"` or `"linear"`.
#' @param dpar For `posterior_epred()`: `"mu"` or `"sigma"`.
#' @param method For `residuals()`: `"posterior_predict"` or `"posterior_epred"`.
#' @param type For `residuals()`: `"ordinary"` or `"pearson"`.
#' @param prob Probability mass of the interval.
#' @param resp,negative_rt,sort,ntrys,cores,nlpar,incl_thres,pointwise,combine,add_point_estimate
#'   Accepted for compatibility.
#' @param ... `allow_new_levels` (logical) and `sample_new_levels`
#'   (`"uncertainty"`, `"gaussian"` or `"old_levels"`) control predictions for
#'   levels of the grouping factor that were not in the data.
#' @return A draws x observations matrix, or a summary with one row per observation.
#' @name predictions
NULL

#' @rdname predictions
#' @export
posterior_predict.metabetafit <- function(object, newdata = NULL, re_formula = NULL, re.form = NULL,
                                          transform = NULL, resp = NULL, negative_rt = FALSE,
                                          ndraws = NULL, draw_ids = NULL, sort = FALSE, ntrys = 5,
                                          cores = NULL, ...) {
  lp <- linear_predictor(object, newdata, re_formula %||% re.form, ndraws, draw_ids, ...)
  y <- draw_response(lp)
  if (is.null(transform)) y else transform(y)
}

#' @rdname predictions
#' @export
posterior_epred.metabetafit <- function(object, newdata = NULL, re_formula = NULL, re.form = NULL,
                                        resp = NULL, dpar = NULL, nlpar = NULL, ndraws = NULL,
                                        draw_ids = NULL, sort = FALSE, ...) {
  lp <- linear_predictor(object, newdata, re_formula %||% re.form, ndraws, draw_ids, ...)
  if (identical(dpar, "sigma")) {
    if (is.null(lp$sigma)) stop2("This model has no parameter 'sigma'.")
    return(matrix(lp$sigma, nrow(lp$eta), ncol(lp$eta)))
  }
  inverse_link(lp$eta, lp$likelihood)
}

#' @rdname predictions
#' @export
posterior_linpred.metabetafit <- function(object, transform = FALSE, newdata = NULL, re_formula = NULL,
                                          re.form = NULL, resp = NULL, dpar = NULL, nlpar = NULL,
                                          incl_thres = NULL, ndraws = NULL, draw_ids = NULL,
                                          sort = FALSE, ...) {
  lp <- linear_predictor(object, newdata, re_formula %||% re.form, ndraws, draw_ids, ...)
  if (transform) inverse_link(lp$eta, lp$likelihood) else lp$eta
}

#' @rdname predictions
#' @export
log_lik.metabetafit <- function(object, newdata = NULL, re_formula = NULL, resp = NULL,
                                ndraws = NULL, draw_ids = NULL, pointwise = FALSE, combine = TRUE,
                                add_point_estimate = FALSE, cores = NULL, ...) {
  lp <- linear_predictor(object, newdata, re_formula, ndraws, draw_ids, ...)
  mu <- inverse_link(lp$eta, lp$likelihood)
  y <- matrix(observed_response(object, newdata), nrow(mu), ncol(mu), byrow = TRUE)
  switch(lp$likelihood,
    gaussian = dnorm(y, mu, lp$sigma, log = TRUE),
    bernoulli = dbinom(y, 1, mu, log = TRUE),
    poisson = dpois(y, mu, log = TRUE)
  )
}

#' @rdname predictions
#' @export
predict.metabetafit <- function(object, newdata = NULL, re_formula = NULL, transform = NULL,
                                resp = NULL, negative_rt = FALSE, ndraws = NULL, draw_ids = NULL,
                                sort = FALSE, ntrys = 5, cores = NULL, summary = TRUE, robust = FALSE,
                                probs = c(0.025, 0.975), ...) {
  draws <- posterior_predict(object, newdata, re_formula, transform = transform, ndraws = ndraws, draw_ids = draw_ids, ...)
  summarise_predictions(draws, summary, probs, robust)
}

#' @rdname predictions
#' @export
fitted.metabetafit <- function(object, newdata = NULL, re_formula = NULL, scale = c("response", "linear"),
                               resp = NULL, dpar = NULL, nlpar = NULL, ndraws = NULL, draw_ids = NULL,
                               sort = FALSE, summary = TRUE, robust = FALSE, probs = c(0.025, 0.975), ...) {
  draws <- if (match.arg(scale) == "linear") {
    posterior_linpred(object, newdata = newdata, re_formula = re_formula, ndraws = ndraws, draw_ids = draw_ids, ...)
  } else {
    posterior_epred(object, newdata, re_formula, dpar = dpar, ndraws = ndraws, draw_ids = draw_ids, ...)
  }
  summarise_predictions(draws, summary, probs, robust)
}

#' @rdname predictions
#' @export
residuals.metabetafit <- function(object, newdata = NULL, re_formula = NULL, method = "posterior_predict",
                                  type = c("ordinary", "pearson"), resp = NULL, ndraws = NULL,
                                  draw_ids = NULL, sort = FALSE, summary = TRUE, robust = FALSE,
                                  probs = c(0.025, 0.975), ...) {
  method <- match.arg(method, c("posterior_predict", "posterior_epred", "predict", "fitted"))
  lp <- linear_predictor(object, newdata, re_formula, ndraws, draw_ids, ...)
  mu <- inverse_link(lp$eta, lp$likelihood)
  prediction <- if (method %in% c("posterior_predict", "predict")) draw_response(lp) else mu
  y <- observed_response(object, newdata)
  residual <- -sweep(prediction, 2, y)
  if (match.arg(type) == "pearson") {
    sd <- switch(lp$likelihood, gaussian = lp$sigma, bernoulli = sqrt(mu * (1 - mu)), poisson = sqrt(mu))
    residual <- residual / sd
  }
  summarise_predictions(residual, summary, probs, robust)
}

#' @rdname predictions
#' @export
predictive_error.metabetafit <- function(object, newdata = NULL, re_formula = NULL, re.form = NULL,
                                         method = "posterior_predict", resp = NULL, ndraws = NULL,
                                         draw_ids = NULL, sort = FALSE, ...) {
  residuals(object, newdata, re_formula %||% re.form, method = method, ndraws = ndraws,
    draw_ids = draw_ids, summary = FALSE, ...)
}

#' @rdname predictions
#' @export
predictive_interval.metabetafit <- function(object, prob = 0.9, ...) {
  rstantools::predictive_interval(posterior_predict(object, ...), prob = prob)
}
