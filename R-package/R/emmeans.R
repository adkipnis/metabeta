# Support for the emmeans package, as brms provides it: the basis of a reference grid
# is the posterior of the population-level linear predictor at its points, so
# emmeans(), contrast(), joint_tests() etc. summarize posterior draws.

# The data and the population-level terms of a fit (emmeans' recover_data generic).
#' @exportS3Method emmeans::recover_data
recover_data.metabetafit <- function(object, data = NULL, ...) {
  design <- object$metabeta$design
  data <- data %||% object$data
  # `0 + Intercept` refers to a column of ones named Intercept
  if (design$int_special) data$Intercept <- 1
  trms <- stats::delete.response(design$fixed_block$terms)
  emmeans::recover_data(call("brm"), trms, NULL, data = data, ...)
}

# Posterior draws of the linear predictor (or, with `epred = TRUE`, of the expected
# response) on the reference grid (emmeans' emm_basis generic).
#' @exportS3Method emmeans::emm_basis
emm_basis.metabetafit <- function(object, trms, xlev, grid, vcov., re_formula = NA,
                                  epred = FALSE, ...) {
  post <- if (epred) {
    posterior_epred(object, newdata = grid, re_formula = re_formula)
  } else {
    posterior_linpred(object, newdata = grid, re_formula = re_formula)
  }
  attr(post, "n.chains") <- nchains(object)
  misc <- if (epred) list() else emmeans::.std.link.labels(stats::family(object), list())
  list(
    X = diag(ncol(post)), bhat = colMeans(post), nbasis = matrix(NA), V = stats::cov(post),
    dffun = function(k, dfargs) Inf, dfargs = list(), misc = misc, post.beta = post
  )
}
