# brm(): fit a model, and turn the network's draws into draws with brms' names and meaning.

#' Fit Bayesian generalized linear multilevel models with metabeta
#'
#' Fits a generalized linear mixed model with the pretrained amortized posterior
#' estimator `metabeta`, with the interface of `brms::brm()`. Arguments that only
#' tune Stan (`cores`, `backend`, `init`, ...) are ignored; arguments requesting
#' features metabeta cannot provide raise an error.
#'
#' With `algorithm = "sampling"` (the default), `chains` Metropolis-Hastings
#' chains that use the network's posterior as proposal run `iter - warmup`
#' iterations each after a short burn-in, and every `thin`-th draw is kept. The
#' variational algorithms (`"meanfield"`, `"fullrank"`, `"pathfinder"`) and
#' `"laplace"` return the network's posterior without this correction.
#'
#' Where the network's proposal fits the posterior poorly, the chains repeat few
#' draws and a warning says so. `control = list(random_walk = TRUE)` then
#' continues those chains with random-walk Metropolis-Hastings moves on the same
#' target.
#'
#' @param formula A model formula, `brmsformula` or string; see [brmsformula()].
#' @param data A data frame containing all model variables.
#' @param family `gaussian()`, `bernoulli()`, `binomial()` (one trial) or
#'   `poisson()`, as a family object, function or string.
#' @param prior Priors created by [prior()] or [set_prior()]; `NULL` uses the
#'   defaults listed by [get_prior()].
#' @param sample_prior `"yes"` to store prior draws as variables `prior_*` as
#'   well, `"only"` to draw from the prior only.
#' @param drop_unused_levels Drop unused factor levels? Group-level effects of
#'   unused levels of the grouping factor are drawn from their population
#'   distribution.
#' @param fit A previous fit whose formula, data, family and priors are reused.
#' @param chains,iter,warmup,thin The model yields
#'   `chains * ceiling((iter - warmup) / thin)` draws.
#' @param algorithm `"sampling"`, or `"meanfield"`, `"fullrank"`,
#'   `"pathfinder"`, `"laplace"` for the network's posterior without correction.
#' @param seed Seed; `NA` takes one from R's random numbers, so `set.seed()`
#'   makes fits reproducible.
#' @param file File to save the fit to, or to read it from if it exists.
#' @param file_compress Compression passed to [saveRDS()].
#' @param file_refit `"never"`, `"always"` or `"on_change"` (refit when data,
#'   formula, family or priors changed).
#' @param empty If `TRUE`, return the model without fitting it.
#' @param control A list. `random_walk = TRUE` continues the Metropolis-Hastings
#'   chains with random-walk moves where their effective sample size is low (see
#'   Details); the default is the option `metabeta.random_walk` (`FALSE`). Other
#'   entries, which tune Stan, are ignored.
#' @param save_pars With `save_pars(all = TRUE)` (as brms requires for
#'   [bridge_sampler()]), the marginal likelihood is computed during the fit, so
#'   that [bridge_sampler()] returns at once instead of repeating the fit.
#' @param save_all_pars Deprecated alias of `save_pars = save_pars(all = TRUE)`.
#' @param autocor,data2,cov_ranef,stanvars,stan_funs Not supported; must be `NULL`.
#' @param sparse,knots,save_ranef,save_mevars,init,inits,cores,threads,opencl,normalize,backend,future,silent,save_model,stan_model_args,rename
#'   Ignored.
#' @param ... Ignored.
#' @return An object of class `c("metabetafit", "brmsfit")`.
#' @examples
#' \dontrun{
#' data("sleepstudy", package = "lme4")
#' fit <- brm(Reaction ~ Days + (Days | Subject), data = sleepstudy)
#' summary(fit)
#' }
#' @export
brm <- function(formula, data, family = gaussian(), prior = NULL, autocor = NULL,
                data2 = NULL, cov_ranef = NULL, sample_prior = "no", sparse = NULL,
                knots = NULL, drop_unused_levels = TRUE, stanvars = NULL,
                stan_funs = NULL, fit = NA, save_pars = getOption("brms.save_pars", NULL),
                save_ranef = NULL, save_mevars = NULL, save_all_pars = NULL, init = NULL,
                inits = NULL, chains = 4, iter = getOption("brms.iter", 2000),
                warmup = floor(iter / 2), thin = 1, cores = getOption("mc.cores", 1),
                threads = getOption("brms.threads", NULL),
                opencl = getOption("brms.opencl", NULL),
                normalize = getOption("brms.normalize", TRUE), control = NULL,
                algorithm = getOption("brms.algorithm", "sampling"),
                backend = getOption("brms.backend", "rstan"),
                future = getOption("future", FALSE), silent = 1, seed = NA,
                save_model = NULL, stan_model_args = list(), file = NULL,
                file_compress = TRUE, file_refit = getOption("brms.file_refit", "never"),
                empty = FALSE, rename = TRUE, ...) {
  data_name <- if (missing(data)) "NULL" else deparse1(substitute(data))
  family_given <- !missing(family)

  # a fit saved earlier
  file_refit <- match.arg(file_refit, c("never", "always", "on_change"))
  if (!is.null(file) && !grepl("\\.rds$", file)) file <- paste0(file, ".rds")
  cached <- if (!is.null(file) && file_refit != "always" && file.exists(file)) readRDS(file)
  if (!is.null(cached) && file_refit == "never") return(cached)

  # arguments
  for (arg in c("autocor", "data2", "cov_ranef", "stanvars", "stan_funs")) {
    if (length(get(arg))) stop2("Argument '", arg, "' is not supported by metabeta. Use brms::brm() instead.")
  }
  if (is.logical(sample_prior)) sample_prior <- if (sample_prior) "yes" else "no"
  sample_prior <- match.arg(sample_prior, c("no", "yes", "only"))
  algorithm <- match.arg(algorithm, c("sampling", "meanfield", "fullrank", "pathfinder", "laplace"))
  if (chains < 1 || thin < 1 || warmup < 0 || iter <= warmup) {
    stop2("Arguments 'chains' and 'thin' must be positive and 'iter' larger than 'warmup'.")
  }
  if (is.brmsfit(fit)) {
    formula <- fit$formula
    data <- fit$data
    family <- fit$family
    prior <- fit$metabeta$user_prior
    data_name <- fit$data.name
    family_given <- TRUE
  }

  # the model
  model <- prepare_model(formula, data, family, drop_unused_levels, family_given)
  prior_table <- resolve_priors(model$design, prior)
  hyper <- model_hyper(model$design, prior_table)
  if (!is.null(cached) && !needs_refit(cached, model, prior_table)) return(cached)

  out <- new_metabetafit(model, prior_table, prior, data_name, algorithm, file)
  out$metabeta$hyper <- hyper
  out$metabeta$sample_prior <- sample_prior
  out$metabeta$random_walk <- isTRUE(control$random_walk %||% getOption("metabeta.random_walk", FALSE))
  out$metabeta$compute_evidence <- isTRUE(save_pars$all) || isTRUE(save_all_pars)
  if (empty) return(out)
  out$metabeta$seed <- if (is.na(seed)) sample.int(.Machine$integer.max, 1) else seed
  out <- add_draws(out, chains, iter, warmup, thin)
  if (!is.null(file)) saveRDS(out, file = file, compress = file_compress)
  out
}

# Parse formula and family, and build the design.
prepare_model <- function(formula, data, family, drop_unused_levels = TRUE, family_given = TRUE) {
  bform <- brmsformula(formula)
  # as in brms, a family set in bf() is used unless brm() was given one explicitly
  if (is.null(bform$family) || family_given) bform$family <- validate_family(family)
  spec <- parse_model_formula(bform)
  design <- build_design(spec, data, bform$family, environment(bform$formula), drop_unused_levels)
  list(formula = bform, family = bform$family, design = design)
}

needs_refit <- function(cached, model, prior_table) {
  !identical(deparse1(cached$formula$formula), deparse1(model$formula$formula)) ||
    !identical(cached$family$family, model$family$family) ||
    !identical(as_plain_df(cached$prior), as_plain_df(prior_table)) ||
    !isTRUE(all.equal(cached$data, model$design$frame, check.attributes = FALSE))
}

# A fitted model without draws, with the elements of a brmsfit object.
new_metabetafit <- function(model, prior_table, user_prior, data_name, algorithm, file) {
  design <- model$design
  data <- design$frame
  attr(data, "data_name") <- data_name
  # description of the group-level effects, as in brms
  ranef <- data.frame(
    id = 1, group = design$group, gn = 1L, coef = design$re_names, cn = seq_along(design$re_names),
    resp = "", dpar = "", nlpar = "", cor = design$correlated, type = "", by = "", cov = "",
    dist = "gaussian"
  )
  attr(ranef, "levels") <- stats::setNames(list(design$levels), design$group)
  structure(
    list(
      formula = model$formula, data = data, prior = prior_table, data2 = list(),
      model = "", algorithm = algorithm, backend = "metabeta", stan_args = list(),
      fit = NULL, criteria = list(), file = file, family = model$family, ranef = ranef,
      data.name = data_name,
      version = list(brms = package_version("2.23.0"), metabeta = utils::packageVersion("metabeta")),
      metabeta = list(design = design, user_prior = user_prior)
    ),
    class = c("metabetafit", "brmsfit")
  )
}

# Sample with the network (or from the prior) and store the draws in x$fit.
add_draws <- function(x, chains, iter, warmup, thin) {
  mb <- x$metabeta
  design <- mb$design
  n_iter <- iter - warmup
  n_samples <- chains * n_iter
  if (mb$sample_prior == "only") {
    draws <- draws_from_prior(design, x$prior, n_samples)
    out <- list(method = "prior", burnin = 0, safeguards = list())
  } else {
    out <- tryCatch(
      sample_network(design$model, mb$hyper, design$likelihood, n_samples, chains,
        refine = x$algorithm == "sampling", seed = mb$seed,
        evidence = isTRUE(mb$compute_evidence) && n_samples >= 64, random_walk = isTRUE(mb$random_walk)),
      error = explain_python_error
    )
    report_safeguards(out, n_samples)
    draws <- back_transform(design, out)
  }

  variables <- brms_variables(design, draws)
  if (mb$sample_prior == "yes") variables <- c(variables, prior_variables(design, x$prior, n_samples))
  variables$lprior <- log_prior(design, x$prior, draws)

  # The draws are chain-major: chain c holds draws (c - 1) * n_iter + 1:n_iter.
  M <- do.call(cbind, variables)
  draws_array <- array(M, dim = c(n_iter, chains, ncol(M)), dimnames = list(NULL, NULL, colnames(M)))
  x$fit <- posterior::as_draws_array(draws_array[seq(1, n_iter, by = thin), , , drop = FALSE])

  burnin <- if (out$method == "imh") out$burnin else 0
  x$metabeta[c("method", "safeguards", "iter", "warmup", "thin")] <- list(out$method, out$safeguards, n_iter + burnin, burnin, thin)
  x$metabeta$evidence <- out$evidence
  x$stan_args <- list(chains = chains, iter = iter, warmup = warmup, thin = thin, seed = mb$seed)
  x
}

# From the network's draws to brms' draws -------------------------------------------

# Undo the linear maps of design.R draw by draw. Returns b (S x d), Intercept (the
# intercept at the predictor means, S), sd and cor of the group-level effects (and,
# for the priors, sd_c and cor_c before uncentering), sigma (S) and r (levels x S x q).
back_transform <- function(design, out) {
  y_sd <- design$y_sd
  b <- matrix(NA_real_, nrow(out$ffx), length(design$b_names), dimnames = list(NULL, design$b_names))
  b[, design$perm] <- out$ffx
  b <- sweep(b, 2, y_sd / design$x_scales, "*")
  intercept <- NULL
  if (design$int_type != "none") {
    # intercept = b*_0 * y_sd + y_mean, at the predictor means when the design is centered
    intercept <- b[, "Intercept"] + design$y_mean
    b[, "Intercept"] <- intercept - b[, colnames(b) != "Intercept", drop = FALSE] %*% design$x_means[colnames(b) != "Intercept"]
  }

  sd_c <- sweep(out$sigma_rfx, 2, y_sd / design$z_scales, "*")
  r_c <- out$rfx
  for (k in seq_along(design$re_names)) r_c[, , k] <- r_c[, , k] * y_sd / design$z_scales[[k]]
  # r = B r_c; the covariance of r is B Sigma_c B'
  r <- array(matrix(r_c, ncol = dim(r_c)[3]) %*% t(design$B), dim = dim(r_c))
  uncentered <- uncenter_cov(sd_c, out$corr_rfx, design$B)

  draws <- list(
    b = b, Intercept = if (design$int_type == "centered") intercept,
    sd = uncentered$sd, cor = uncentered$cor, sd_c = sd_c, cor_c = out$corr_rfx,
    sigma = if (!is.null(out$sigma_eps)) out$sigma_eps * y_sd,
    r = r
  )
  # levels without data: their effects come from the population distribution
  unobserved <- setdiff(seq_along(design$levels), design$observed)
  if (length(unobserved)) {
    all_levels <- array(0, dim = c(length(design$levels), dim(r)[2:3]))
    all_levels[design$observed, , ] <- r
    for (l in unobserved) all_levels[l, , ] <- draw_mvn(draws$sd, draws$cor)
    draws$r <- all_levels
  }
  draws
}

# Standard deviations and correlations of Sigma = B Sigma_c B', draw by draw.
uncenter_cov <- function(sd_c, cor_c, B) {
  sd <- sd_c
  cor <- cor_c
  for (s in seq_len(nrow(sd_c))) {
    Sigma <- B %*% (outer(sd_c[s, ], sd_c[s, ]) * cor_c[s, , ]) %*% t(B)
    sd[s, ] <- sqrt(diag(Sigma))
    cor[s, , ] <- stats::cov2cor(Sigma)
  }
  list(sd = sd, cor = cor)
}

# Named list of draws in brms' variable order: b_*, sd_*, cor_*, sigma, Intercept, r_*.
brms_variables <- function(design, draws) {
  g <- design$group
  re <- design$re_names
  out <- list()
  for (k in design$b_names) out[[paste0("b_", k)]] <- draws$b[, k]
  for (k in seq_along(re)) out[[paste0("sd_", g, "__", re[k])]] <- draws$sd[, k]
  if (design$correlated) {
    for (p in cor_pairs(length(re))) {
      out[[paste0("cor_", g, "__", re[p[1]], "__", re[p[2]])]] <- draws$cor[, p[1], p[2]]
    }
  }
  out$sigma <- draws$sigma
  out$Intercept <- draws$Intercept
  levels <- gsub("[ \t\r\n]", ".", design$levels)  # as in brms
  for (k in seq_along(re)) {
    for (l in seq_along(levels)) out[[paste0("r_", g, "[", levels[l], ",", re[k], "]")]] <- draws$r[l, , k]
  }
  out
}
