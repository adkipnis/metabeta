# summary() and print(), fixef(), ranef(), coef(), VarCorr(), small accessors and update().

#' Checks if argument is a `brmsfit` object
#' @param x An R object.
#' @return `TRUE` or `FALSE`.
#' @export
is.brmsfit <- function(x) inherits(x, "brmsfit")

#' Create a summary of a fitted model
#'
#' Same layout as `brms::summary.brmsfit()`.
#'
#' @param object A fitted model.
#' @param priors Show the priors?
#' @param prob Probability mass of the credible intervals.
#' @param robust Use median and MAD instead of mean and SD?
#' @param mc_se Show Monte Carlo standard errors?
#' @param ... Ignored.
#' @return An object of class `brmssummary`.
#' @export
summary.metabetafit <- function(object, priors = FALSE, prob = 0.95, robust = FALSE, mc_se = FALSE, ...) {
  mb <- object$metabeta
  out <- list(
    formula = object$formula, data_name = object$data.name, nobs = nobs(object),
    ngrps = ngrps(object), prior = if (priors) prior_summary(object, all = FALSE) else empty_prior()
  )
  class(out) <- c("metabetasummary", "brmssummary")
  if (is.null(object$fit)) return(out)

  # one row per parameter: estimate, error, interval, Rhat and effective sample sizes
  probs <- c((1 - prob) / 2, (1 + prob) / 2)
  interval <- function(x) stats::setNames(stats::quantile(x, probs, names = FALSE), paste0(c("l-", "u-"), prob * 100, "% CI"))
  measures <- list(Estimate = if (robust) stats::median else mean)
  if (mc_se) measures$MCSE <- if (robust) posterior::mcse_median else posterior::mcse_mean
  measures <- c(measures, list(
    Est.Error = if (robust) stats::mad else stats::sd, quantiles = interval,
    Rhat = posterior::rhat, Bulk_ESS = posterior::ess_bulk, Tail_ESS = posterior::ess_tail
  ))
  draws <- get_draws(object, "^(b|sd|cor|sigma)(_|$)", regex = TRUE)
  table <- as.data.frame(do.call(posterior::summarise_draws, c(list(draws), measures)))
  rownames(table) <- table$variable
  table$variable <- NULL
  if (any(table$Rhat > 1.05, na.rm = TRUE)) {
    warning2("Parts of the model have not converged (some Rhats are > 1.05). Be careful when analysing the results! We recommend running more iterations and/or setting stronger priors.")
  }

  # split into sections, with brms' row names: sd(Days), cor(Intercept,Days), Days, sigma
  g <- object$metabeta$design$group
  rows <- rownames(table)
  sd_prefix <- paste0("sd_", g, "__")
  cor_prefix <- paste0("cor_", g, "__")
  sd_rows <- rows[startsWith(rows, sd_prefix)]
  cor_rows <- rows[startsWith(rows, cor_prefix)]
  random <- table[c(sd_rows, cor_rows), , drop = FALSE]
  rownames(random) <- c(
    sprintf("sd(%s)", substring(sd_rows, nchar(sd_prefix) + 1)),
    sprintf("cor(%s)", sub("__", ",", substring(cor_rows, nchar(cor_prefix) + 1), fixed = TRUE))
  )
  out$random <- stats::setNames(list(random), g)
  out$fixed <- table[startsWith(rows, "b_"), , drop = FALSE]
  rownames(out$fixed) <- sub("^b_", "", rownames(out$fixed))
  out$spec_pars <- table[rows == "sigma", , drop = FALSE]

  out$sampler <- paste0("metabeta(", mb$method, ")")
  out$method <- mb$method
  out$accept_rate <- mb$safeguards$accept_rate
  out$chains <- nchains(object)
  out$iter <- mb$iter
  out$warmup <- mb$warmup
  out$thin <- mb$thin
  out$total_ndraws <- ndraws(object)
  out
}

#' @export
print.metabetasummary <- function(x, digits = 2, short = getOption("brms.short_summary", FALSE), ...) {
  family <- x$formula$family
  cat(" Family:", family$family, "\n")
  cat("  Links:", paste0("mu = ", family$link, if (family$family == "gaussian") "; sigma = identity"), "\n")
  cat("Formula: ")
  print(x$formula, wsp = 9)
  cat(paste0("   Data: ", x$data_name, " (Number of observations: ", x$nobs, ") \n"))
  if (is.null(x$sampler)) {
    cat("\nThe model does not contain posterior draws.\n")
    return(invisible(x))
  }
  if (!short) {
    cat(paste0("  Draws: ", x$chains, " chains, each with iter = ", x$iter, "; warmup = ", x$warmup,
      "; thin = ", x$thin, ";\n         total post-warmup draws = ", x$total_ndraws, "\n"))
  }
  cat("\n")
  if (nrow(x$prior)) {
    cat("Priors:\n")
    print(x$prior, show_df = FALSE)
    cat("\n")
  }
  cat("Multilevel Hyperparameters:\n")
  for (g in names(x$random)) {
    cat(paste0("~", g, " (Number of levels: ", x$ngrps[[g]], ") \n"))
    print_table(x$random[[g]], digits)
    cat("\n")
  }
  cat("Regression Coefficients:\n")
  print_table(x$fixed, digits)
  cat("\n")
  if (nrow(x$spec_pars)) {
    cat("Further Distributional Parameters:\n")
    print_table(x$spec_pars, digits)
    cat("\n")
  }
  if (!short) {
    cat(paste0("Draws were sampled using ", x$sampler, ". "))
    if (x$method == "imh") {
      cat("For each parameter, Bulk_ESS\nand Tail_ESS are effective sample size measures, and Rhat is the potential\nscale reduction factor on split chains (at convergence, Rhat = 1).\n")
      cat(sprintf("The Metropolis-Hastings acceptance rate of the network proposal was %.2f.\n", x$accept_rate))
    } else if (x$method == "flow") {
      cat("The draws are independent draws from\nmetabeta's network posterior, without Metropolis-Hastings refinement.\n")
    } else {
      cat("The draws are independent draws from the prior.\n")
    }
  }
  invisible(x)
}

# Print numbers with 'digits' decimals, effective sample sizes without (as in brms).
print_table <- function(x, digits) {
  out <- as.matrix(x)
  for (col in colnames(x)) {
    out[, col] <- sprintf(if (col %in% c("Bulk_ESS", "Tail_ESS")) "%.0f" else paste0("%.", digits, "f"), x[[col]])
  }
  print(out, quote = FALSE, right = TRUE)
}

#' @export
print.metabetafit <- function(x, digits = 2, short = getOption("brms.short_summary", FALSE), ...) {
  print(summary(x, ...), digits = digits, short = short)
}

# Population- and group-level effects ------------------------------------------------

#' @export
fixef.metabetafit <- function(object, summary = TRUE, robust = FALSE, probs = c(0.025, 0.975), pars = NULL, ...) {
  b <- as.matrix(object, variable = paste0("b_", object$metabeta$design$b_names))
  colnames(b) <- sub("^b_", "", colnames(b))
  if (!is.null(pars)) b <- b[, pars, drop = FALSE]
  if (summary) summarise_matrix(b, probs, robust) else b
}

# The group-level effects as an array [draws, levels, coefficients].
ranef_draws <- function(object) {
  design <- object$metabeta$design
  levels <- gsub("[ \t\r\n]", ".", design$levels)
  out <- array(NA_real_, c(ndraws(object), length(levels), length(design$re_names)),
    dimnames = list(NULL, design$levels, design$re_names))
  for (k in design$re_names) {
    out[, , k] <- as.matrix(object, variable = paste0("r_", design$group, "[", levels, ",", k, "]"))
  }
  out
}

#' @export
ranef.metabetafit <- function(object, summary = TRUE, robust = FALSE, probs = c(0.025, 0.975),
                              pars = NULL, groups = NULL, ...) {
  r <- ranef_draws(object)
  if (!is.null(pars)) r <- r[, , pars, drop = FALSE]
  if (summary) r <- posterior_summary(r, probs = probs, robust = robust)
  stats::setNames(list(r), object$metabeta$design$group)
}

# coef() = population-level effect + group-level effect, for every coefficient
# (coefficients without group-level effects are the population-level effect).
#' @export
coef.metabetafit <- function(object, summary = TRUE, robust = FALSE, probs = c(0.025, 0.975), ...) {
  b <- fixef(object, summary = FALSE)
  r <- ranef_draws(object)
  names <- union(dimnames(r)[[3]], colnames(b))
  out <- array(0, c(dim(r)[1:2], length(names)), dimnames = list(NULL, dimnames(r)[[2]], names))
  for (k in names) {
    if (k %in% dimnames(r)[[3]]) out[, , k] <- r[, , k]
    if (k %in% colnames(b)) out[, , k] <- out[, , k] + b[, k]
  }
  if (summary) out <- posterior_summary(out, probs = probs, robust = robust)
  stats::setNames(list(out), object$metabeta$design$group)
}

#' @export
VarCorr.metabetafit <- function(x, sigma = 1, summary = TRUE, robust = FALSE, probs = c(0.025, 0.975), ...) {
  design <- x$metabeta$design
  g <- design$group
  names <- design$re_names
  q <- length(names)
  sd <- as.matrix(x, variable = paste0("sd_", g, "__", names))
  colnames(sd) <- names
  summarise <- function(draws) if (summary) posterior_summary(draws, probs = probs, robust = robust) else draws
  group <- list(sd = summarise(sd))
  if (design$correlated) {
    cor <- array(diag(q), c(q, q, nrow(sd))) |> aperm(c(3, 1, 2))
    for (p in cor_pairs(q)) {
      r <- as.matrix(x, variable = paste0("cor_", g, "__", names[p[1]], "__", names[p[2]]))
      cor[, p[1], p[2]] <- cor[, p[2], p[1]] <- r
    }
    cov <- cor
    for (i in seq_len(q)) for (j in seq_len(q)) cov[, i, j] <- cor[, i, j] * sd[, i] * sd[, j]
    dimnames(cor) <- dimnames(cov) <- list(NULL, names, names)
    group$cor <- summarise(cor)
    group$cov <- summarise(cov)
  }
  out <- stats::setNames(list(group), g)
  if (design$likelihood == "gaussian") {
    residual <- as.matrix(x, variable = "sigma")
    colnames(residual) <- x$formula$resp
    out$residual__ <- list(sd = summarise(residual))
  }
  out
}

# Other accessors -------------------------------------------------------------------------

#' @export
nobs.metabetafit <- function(object, resp = NULL, ...) nrow(object$data)

#' Number of levels of grouping factors
#' @param object A fitted model.
#' @param ... Ignored.
#' @return A named list with the number of levels per grouping factor.
#' @export
ngrps <- function(object, ...) UseMethod("ngrps")

#' @rdname ngrps
#' @export
ngrps.metabetafit <- function(object, ...) {
  stats::setNames(list(length(object$metabeta$design$levels)), object$metabeta$design$group)
}

#' @export
formula.metabetafit <- function(x, ...) x$formula

#' @export
family.metabetafit <- function(object, resp = NULL, ...) object$family

#' @export
model.frame.metabetafit <- function(formula, ...) formula$data

#' Extract the response of a fitted model
#' @param x A fitted model.
#' @param resp,sort,warn,... Accepted for compatibility.
#' @return The response vector.
#' @export
get_y <- function(x, resp = NULL, sort = FALSE, warn = FALSE, ...) x$metabeta$design$y

#' Update a fitted model
#'
#' Refit a model with a changed formula, data or other arguments of [brm()];
#' everything else is reused (as in \pkg{brms}).
#'
#' @param object A fitted model.
#' @param formula. Changes to the formula, e.g. `~ . + x`.
#' @param newdata New data to fit the model to.
#' @param recompile Ignored.
#' @param ... Further arguments passed to [brm()].
#' @return A fitted model.
#' @export
update.metabetafit <- function(object, formula., newdata = NULL, recompile = NULL, ...) {
  formula <- object$formula
  if (!missing(formula.)) {
    formula <- brmsformula(stats::update.formula(formula$formula, formula.), center = attr(formula$formula, "center"))
  }
  new_vars <- setdiff(all.vars(formula$formula), c(names(object$data), "Intercept"))
  if (is.null(newdata) && length(new_vars)) {
    stop2("New variables found: ", collapse_comma(new_vars), "\nPlease supply your data again via argument 'newdata'.")
  }
  args <- list(
    formula = formula, data = newdata %||% object$data, family = object$family,
    prior = object$metabeta$user_prior, sample_prior = object$metabeta$sample_prior,
    algorithm = object$algorithm, empty = is.null(object$fit),
    control = list(random_walk = isTRUE(object$metabeta$random_walk))
  )
  sampling <- intersect(c("chains", "iter", "warmup", "thin"), names(object$stan_args))
  args <- c(args, object$stan_args[sampling])
  dots <- list(...)
  if (!is.null(dots$iter) && is.null(dots$warmup)) dots$warmup <- floor(dots$iter / 2)
  args[names(dots)] <- dots
  fit <- do.call(brm, args)
  fit$data.name <- if (is.null(newdata)) object$data.name else deparse1(substitute(newdata))
  fit
}
