# Prior draws and densities, and access to the draws of a fit (as_draws_*(), as.matrix(),
# posterior_summary(), rhat(), ...).

# Prior distributions -----------------------------------------------------------------

# Draws from a parsed prior; standard deviations are truncated at zero.
rprior <- function(n, p, truncate = FALSE) {
  x <- switch(p$dist,
    normal = rnorm(n, p$args$mu, p$args$sigma),
    student_t = p$args$mu + p$args$sigma * stats::rt(n, p$args$nu),
    exponential = stats::rexp(n, p$args$lambda)
  )
  if (truncate) abs(x) else x
}

# Log density of a parsed prior; truncation at zero doubles the density.
dprior <- function(x, p, truncate = FALSE) {
  lp <- switch(p$dist,
    normal = dnorm(x, p$args$mu, p$args$sigma, log = TRUE),
    student_t = dt((x - p$args$mu) / p$args$sigma, p$args$nu, log = TRUE) - log(p$args$sigma),
    exponential = stats::dexp(x, p$args$lambda, log = TRUE)
  )
  if (truncate && p$dist != "exponential") lp + log(2) else lp
}

# Correlation matrices from LKJ(eta), by the onion method (Lewandowski et al., 2009).
rlkj <- function(n, K, eta) {
  out <- array(0, dim = c(n, K, K))
  for (s in seq_len(n)) {
    U <- matrix(0, K, K)
    U[1, 1] <- 1
    beta <- eta + (K - 2) / 2
    for (m in seq_len(K - 1)) {
      if (m == 1) {
        r <- 2 * stats::rbeta(1, beta, beta) - 1
        U[1, 2] <- r
        U[2, 2] <- sqrt(1 - r^2)
      } else {
        beta <- beta - 0.5
        y <- stats::rbeta(1, m / 2, beta)
        z <- rnorm(m)
        U[1:m, m + 1] <- sqrt(y) * z / sqrt(sum(z^2))
        U[m + 1, m + 1] <- sqrt(1 - y)
      }
    }
    out[s, , ] <- crossprod(U)
  }
  out
}

# Log density of a correlation matrix under LKJ(eta), with its normalizing constant.
dlkj <- function(R, eta) {
  K <- nrow(R)
  k <- seq_len(K - 1)
  log_constant <- sum((2 * eta - 2 + K - k) * (K - k) * log(2) +
    (K - k) * lbeta(eta + (K - k - 1) / 2, eta + (K - k - 1) / 2))
  (eta - 1) * determinant(R)$modulus[1] - log_constant
}

# The prior of each parameter, parsed.
parsed_priors <- function(design, table) {
  g <- design$group
  list(
    b = lapply(stats::setNames(design$b_names, design$b_names), function(k) {
      if (k == "Intercept" && design$int_type == "centered") NULL else parse_prior(prior_of(table, "b", k))
    }),
    Intercept = if (design$int_type == "centered") parse_prior(prior_of(table, "Intercept")),
    sd = lapply(design$re_names, function(k) parse_prior(prior_of(table, "sd", k, g))),
    eta = if (design$correlated) parse_prior(prior_of(table, "cor", group = g))$args$eta,
    sigma = if (design$likelihood == "gaussian") parse_prior(prior_of(table, "sigma"))
  )
}

# Joint draws of all parameters from the prior (sample_prior = "only"), in the format
# of back_transform().
draws_from_prior <- function(design, table, S) {
  p <- parsed_priors(design, table)
  q <- length(design$re_names)
  b <- sapply(design$b_names, function(k) if (is.null(p$b[[k]])) rep(0, S) else rprior(S, p$b[[k]]))
  b <- matrix(b, S, dimnames = list(NULL, design$b_names))
  intercept <- NULL
  if (!is.null(p$Intercept)) {
    intercept <- rprior(S, p$Intercept)
    slopes <- colnames(b) != "Intercept"
    b[, "Intercept"] <- intercept - b[, slopes, drop = FALSE] %*% design$x_means[slopes]
  }
  sd_c <- matrix(sapply(p$sd, rprior, n = S, truncate = TRUE), S, q)
  cor_c <- if (design$correlated) rlkj(S, q, p$eta) else array(diag(q), dim = c(q, q, S)) |> aperm(c(3, 1, 2))
  uncentered <- uncenter_cov(sd_c, cor_c, design$B)
  r <- array(0, dim = c(length(design$levels), S, q))
  for (l in seq_along(design$levels)) r[l, , ] <- draw_mvn(uncentered$sd, uncentered$cor)
  list(
    b = b, Intercept = intercept, sd = uncentered$sd, cor = uncentered$cor, sd_c = sd_c, cor_c = cor_c,
    sigma = if (!is.null(p$sigma)) rprior(S, p$sigma, truncate = TRUE), r = r
  )
}

# Prior draws stored as variables "prior_*" (sample_prior = "yes"). As in brms there is
# one variable per prior statement: e.g. prior_b for a class-wide prior, prior_b_x for a
# prior on coefficient x.
prior_variables <- function(design, table, S) {
  g <- design$group
  out <- list()
  if (design$int_type == "centered") out$prior_Intercept <- rprior(S, parse_prior(prior_of(table, "Intercept")))
  for (i in which(table$class %in% c("b", "sd") & nzchar(table$coef))) {
    vectorized <- table$source[i] == "(vectorized)"
    name <- if (table$class[i] == "b") {
      if (vectorized) "prior_b" else paste0("prior_b_", table$coef[i])
    } else {
      if (vectorized) paste0("prior_sd_", g) else paste0("prior_sd_", g, "__", table$coef[i])
    }
    if (is.null(out[[name]])) out[[name]] <- rprior(S, parse_prior(table$prior[i]), truncate = table$class[i] == "sd")
  }
  if (design$correlated) {
    # the marginal of one correlation under LKJ(eta) is a scaled Beta(a, a)
    a <- parsed_priors(design, table)$eta - 1 + length(design$re_names) / 2
    out[[paste0("prior_cor_", g)]] <- 2 * stats::rbeta(S, a, a) - 1
  }
  if (design$likelihood == "gaussian") out$prior_sigma <- rprior(S, parse_prior(prior_of(table, "sigma")), truncate = TRUE)
  out
}

# Log prior density of each draw ("lprior" in brms). The priors of sd and cor apply to
# the parametrization the network used (sd_c, cor_c; see design.R).
log_prior <- function(design, table, draws) {
  p <- parsed_priors(design, table)
  lp <- 0
  for (k in design$b_names) {
    lp <- lp + if (is.null(p$b[[k]])) dprior(draws$Intercept, p$Intercept) else dprior(draws$b[, k], p$b[[k]])
  }
  for (k in seq_along(p$sd)) lp <- lp + dprior(draws$sd_c[, k], p$sd[[k]], truncate = TRUE)
  if (design$correlated) lp <- lp + apply(draws$cor_c, 1, dlkj, eta = p$eta)
  if (!is.null(p$sigma)) lp <- lp + dprior(draws$sigma, p$sigma, truncate = TRUE)
  lp
}

# Access to the draws -----------------------------------------------------------------

# The draws of a fit, optionally of some variables. 'pars' is brms' deprecated way of
# selecting variables by regular expression.
get_draws <- function(x, variable = NULL, pars = NULL, regex = FALSE) {
  if (is.null(x$fit)) stop2("The model does not contain posterior draws.")
  if (is.null(variable) && length(pars) && !all(is.na(pars))) {
    variable <- pars
    regex <- TRUE
  }
  if (is.null(variable)) return(x$fit)
  posterior::subset_draws(x$fit, variable = variable, regex = regex)
}

#' @export
variables.metabetafit <- function(x, ...) posterior::variables(get_draws(x))

#' @export
ndraws.metabetafit <- function(x) if (is.null(x$fit)) 0L else posterior::ndraws(x$fit)

#' @export
nchains.metabetafit <- function(x) if (is.null(x$fit)) 0L else posterior::nchains(x$fit)

#' @export
niterations.metabetafit <- function(x) if (is.null(x$fit)) 0L else posterior::niterations(x$fit)

#' @export
nvariables.metabetafit <- function(x, ...) if (is.null(x$fit)) 0L else posterior::nvariables(x$fit)

#' @export
nsamples.metabetafit <- function(object, subset = NULL, incl_warmup = FALSE, ...) ndraws(object)

#' Transform metabeta fits to draws objects
#'
#' Methods for the \pkg{posterior} generics, with the variable names of
#' \pkg{brms} (`b_Intercept`, `sd_Subject__Days`, `r_Subject[308,Days]`, ...).
#'
#' @param x A fitted model.
#' @param variable Names of variables to extract.
#' @param regex Treat `variable` as regular expressions?
#' @param inc_warmup Ignored: metabeta stores no warmup draws.
#' @param ... Ignored.
#' @name draws-brms
NULL

#' @rdname draws-brms
#' @export
as_draws.metabetafit <- function(x, variable = NULL, regex = FALSE, inc_warmup = FALSE, ...) {
  posterior::as_draws_array(get_draws(x, variable, regex = regex))
}

#' @rdname draws-brms
#' @export
as_draws_array.metabetafit <- function(x, variable = NULL, regex = FALSE, inc_warmup = FALSE, ...) {
  posterior::as_draws_array(get_draws(x, variable, regex = regex))
}

#' @rdname draws-brms
#' @export
as_draws_df.metabetafit <- function(x, variable = NULL, regex = FALSE, inc_warmup = FALSE, ...) {
  posterior::as_draws_df(get_draws(x, variable, regex = regex))
}

#' @rdname draws-brms
#' @export
as_draws_matrix.metabetafit <- function(x, variable = NULL, regex = FALSE, inc_warmup = FALSE, ...) {
  posterior::as_draws_matrix(get_draws(x, variable, regex = regex))
}

#' @rdname draws-brms
#' @export
as_draws_list.metabetafit <- function(x, variable = NULL, regex = FALSE, inc_warmup = FALSE, ...) {
  posterior::as_draws_list(get_draws(x, variable, regex = regex))
}

#' @rdname draws-brms
#' @export
as_draws_rvars.metabetafit <- function(x, variable = NULL, regex = FALSE, inc_warmup = FALSE, ...) {
  posterior::as_draws_rvars(get_draws(x, variable, regex = regex))
}

#' @export
as.matrix.metabetafit <- function(x, pars = NA, variable = NULL, draw = NULL, subset = NULL, ...) {
  draws <- posterior::as_draws_matrix(get_draws(x, variable, pars))
  out <- matrix(draws, nrow = nrow(draws), dimnames = list(NULL, colnames(draws)))
  draw <- draw %||% subset
  if (is.null(draw)) out else out[draw, , drop = FALSE]
}

#' @export
as.data.frame.metabetafit <- function(x, row.names = NULL, optional = TRUE, pars = NA,
                                      variable = NULL, draw = NULL, subset = NULL, ...) {
  as.data.frame(as.matrix.metabetafit(x, pars, variable, draw, subset), optional = optional)
}

#' @export
as.array.metabetafit <- function(x, pars = NA, variable = NULL, draw = NULL, subset = NULL, ...) {
  out <- unclass(posterior::as_draws_array(get_draws(x, variable, pars)))
  draw <- draw %||% subset
  if (!is.null(draw)) out <- out[draw, , , drop = FALSE]
  dimnames(out) <- list(iteration = NULL, chain = paste0("chain:", seq_len(dim(out)[2])), variable = dimnames(out)[[3]])
  out
}

#' Extract posterior draws (deprecated)
#'
#' As in \pkg{brms}; use [as_draws_df()] instead.
#'
#' @param x A fitted model.
#' @param pars Regular expressions of variable names.
#' @param fixed Match `pars` exactly?
#' @param add_chain Add columns `chain` and `iter`?
#' @param subset,as.matrix,as.array,... See \pkg{brms}.
#' @return A data frame of draws.
#' @export
posterior_samples <- function(x, pars = NA, ...) UseMethod("posterior_samples")

#' @rdname posterior_samples
#' @export
posterior_samples.metabetafit <- function(x, pars = NA, fixed = FALSE, add_chain = FALSE,
                                          subset = NULL, as.matrix = FALSE, as.array = FALSE, ...) {
  draws <- if (fixed) get_draws(x, variable = pars) else get_draws(x, pars = pars)
  if (as.array) return(unclass(posterior::as_draws_array(draws)))
  out <- matrix(posterior::as_draws_matrix(draws), ncol = posterior::nvariables(draws), dimnames = list(NULL, posterior::variables(draws)))
  if (!is.null(subset)) out <- out[subset, , drop = FALSE]
  if (as.matrix) return(out)
  out <- as.data.frame(out, optional = TRUE)
  if (add_chain) {
    out$chain <- factor(rep(seq_len(nchains(x)), each = niterations(x)))
    out$iter <- rep(seq_len(niterations(x)), nchains(x))
  }
  out
}

#' Summarize posterior draws
#'
#' Columns `Estimate`, `Est.Error` and quantiles, as in \pkg{brms}. Works for
#' fitted models, matrices (draws in rows), 3D arrays (draws in the first
#' dimension) and draws objects.
#'
#' @param x A fitted model or draws.
#' @param probs Quantiles to compute.
#' @param robust Use median and MAD instead of mean and SD?
#' @param pars,variable Variables to summarize.
#' @param ... Further arguments passed to methods.
#' @return A matrix, or a 3D array for 3D input.
#' @export
posterior_summary <- function(x, ...) UseMethod("posterior_summary")

#' @rdname posterior_summary
#' @export
posterior_summary.default <- function(x, probs = c(0.025, 0.975), robust = FALSE, ...) {
  if (posterior::is_draws(x)) x <- matrix(posterior::as_draws_matrix(x), ncol = posterior::nvariables(x), dimnames = list(NULL, posterior::variables(x)))
  if (length(dim(x)) != 3) return(summarise_matrix(x, probs, robust))
  # array [draws, rows, columns] -> array [rows, statistics, columns]
  out <- vapply(seq_len(dim(x)[3]), function(k) summarise_matrix(x[, , k], probs, robust), matrix(0, dim(x)[2], 2 + length(probs)))
  dimnames(out) <- list(dimnames(x)[[2]], c("Estimate", "Est.Error", paste0("Q", probs * 100)), dimnames(x)[[3]])
  out
}

#' @rdname posterior_summary
#' @export
posterior_summary.metabetafit <- function(x, pars = NA, variable = NULL, probs = c(0.025, 0.975),
                                          robust = FALSE, ...) {
  summarise_matrix(as.matrix.metabetafit(x, pars, variable), probs, robust)
}

#' @export
posterior_interval.metabetafit <- function(object, pars = NA, variable = NULL, prob = 0.95, ...) {
  rstantools::posterior_interval(as.matrix.metabetafit(object, pars, variable), prob = prob)
}

#' @export
rhat.metabetafit <- function(x, pars = NULL, ...) rhat_values(get_draws(x, pars = pars))

#' @export
neff_ratio.metabetafit <- function(object, pars = NULL, ...) neff_ratios(get_draws(object, pars = pars))

rhat_values <- function(draws) {
  stats::setNames(posterior::summarise_draws(draws, rhat = posterior::rhat)$rhat, posterior::variables(draws))
}

neff_ratios <- function(draws) {
  ess <- posterior::summarise_draws(draws, ess = posterior::ess_bulk)$ess
  stats::setNames(ess / posterior::ndraws(draws), posterior::variables(draws))
}

#' Extract prior draws
#'
#' Prior draws stored with `sample_prior = "yes"`, named without the `prior_`
#' prefix, as in \pkg{brms}.
#'
#' @param x A fitted model.
#' @param variable,pars Parameters whose prior draws to extract; matched to the
#'   prior draws by the longest prefix.
#' @param ... Ignored.
#' @return A data frame of draws.
#' @export
prior_draws <- function(x, ...) UseMethod("prior_draws")

#' @rdname prior_draws
#' @export
prior_draws.metabetafit <- function(x, variable = NULL, pars = NULL, ...) {
  names <- grep("^prior_", variables(x), value = TRUE)
  if (!length(names)) return(data.frame())
  draws <- as.data.frame(as.matrix.metabetafit(x, variable = names))
  names(draws) <- sub("^prior_", "", names)
  variable <- variable %||% pars
  if (is.null(variable)) return(draws)
  out <- list()
  for (v in variable) {
    matches <- names(draws)[startsWith(v, names(draws))]
    # the prior of b_Intercept is on the centered intercept, so it does not apply
    if (!length(matches) || (v == "b_Intercept" && x$metabeta$design$int_type == "centered")) next
    out[[v]] <- sample(draws[[matches[which.max(nchar(matches))]]])
  }
  data.frame(out, check.names = FALSE)
}

#' @rdname prior_draws
#' @export
prior_samples <- function(x, ...) prior_draws(x, ...)
