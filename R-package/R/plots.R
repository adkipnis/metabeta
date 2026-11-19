# Plots: pp_check(), plot(), mcmc_plot(), pairs() and conditional_effects().

# Variables to plot: those selected by 'variable' (exact) or 'pars' (regular expressions),
# by default the population-level effects, standard deviations, correlations and sigma.
plot_variables <- function(x, pars = NA, variable = NULL, regex = FALSE, fixed = FALSE) {
  all <- variables(x)
  if (!is.null(variable) && !regex) return(intersect(variable, all))
  patterns <- variable %||% (if (!all(is.na(pars))) pars) %||% "^(b|sd|cor|sigma)(_|$)"
  if (fixed) return(intersect(patterns, all))
  all[Reduce(`|`, lapply(patterns, grepl, x = all))]
}

#' Posterior predictive checks
#'
#' Posterior predictive checks with \pkg{bayesplot}, as in \pkg{brms}.
#'
#' @param object A fitted model.
#' @param type Name of a `ppc_` (or `ppd_`) function of \pkg{bayesplot} without
#'   the prefix; `"dens_overlay"` by default.
#' @param ndraws Number of draws; 10 for overlay-type plots by default.
#' @param prefix `"ppc"` (compare with the data) or `"ppd"` (predictions only).
#' @param group,x Names of the variables for grouped plots or plots against `x`.
#' @param newdata Data to predict for.
#' @param draw_ids,nsamples Specific draws, or the deprecated alias of `ndraws`.
#' @param resp,subset Accepted for compatibility.
#' @param ... Passed to the \pkg{bayesplot} function.
#' @return A ggplot object.
#' @export
pp_check.metabetafit <- function(object, type, ndraws = NULL, prefix = c("ppc", "ppd"), group = NULL,
                                 x = NULL, newdata = NULL, resp = NULL, draw_ids = NULL,
                                 nsamples = NULL, subset = NULL, ...) {
  if (missing(type)) type <- "dens_overlay"
  prefix <- match.arg(prefix)
  plot_function <- get0(paste0(prefix, "_", type), envir = asNamespace("bayesplot"), mode = "function")
  if (is.null(plot_function)) stop2("Type '", type, "' is not a valid ppc type.")
  arguments <- names(formals(plot_function))

  # as in brms, types that plot single draws use 10 draws by default
  all_draw_types <- c(
    "error_scatter_avg", "error_scatter_avg_vs_x", "intervals", "intervals_grouped", "loo_pit",
    "loo_pit_overlay", "loo_pit_qq", "loo_intervals", "loo_ribbon", "ribbon", "ribbon_grouped",
    "rootogram", "scatter_avg", "scatter_avg_grouped", "stat", "stat_2d", "stat_freqpoly_grouped",
    "stat_grouped", "violin_grouped"
  )
  ndraws <- ndraws %||% nsamples
  if (is.null(ndraws) && is.null(draw_ids) && !type %in% all_draw_types) {
    ndraws <- 10
    message("Using 10 posterior draws for ppc type '", type, "' by default.")
  }
  if (startsWith(type, "loo")) ndraws <- NULL  # LOO plots need all draws
  draws <- select_draws(ndraws(object), ndraws, draw_ids)
  data <- if (is.null(newdata)) object$data else as.data.frame(newdata)

  yrep <- posterior_predict(object, newdata, draw_ids = draws, ...)
  args <- if (prefix == "ppc") list(y = observed_response(object, newdata), yrep = yrep) else list(ypred = yrep)
  if ("group" %in% arguments) args$group <- data[[group %||% stop2("Argument 'group' is required for type '", type, "'.")]]
  if ("x" %in% arguments) args$x <- if (is.null(x)) seq_len(nrow(data)) else data[[x]]
  if (any(c("lw", "psis_object") %in% arguments)) {
    ll <- log_lik(object, newdata, draw_ids = draws)
    psis <- suppressWarnings(loo::psis(-ll, r_eff = loo::relative_eff(exp(ll), chain_id = chain_ids(object)[draws])))
    if ("psis_object" %in% arguments) args$psis_object <- psis else args$lw <- stats::weights(psis)
  }
  dots <- list(...)
  dots[c("allow_new_levels", "sample_new_levels")] <- NULL
  do.call(plot_function, c(args, dots))
}

#' Trace and density plots of MCMC draws
#'
#' @param x A fitted model.
#' @param pars,variable,regex,fixed Variables to plot.
#' @param combo Plot types per variable, see [bayesplot::mcmc_combo()].
#' @param nvariables,N Number of variables per page (`N` is a deprecated alias).
#' @param bins Accepted for compatibility.
#' @param theme A ggplot2 theme.
#' @param plot Print the plots?
#' @param ask Ask before showing the next page?
#' @param newpage Accepted for compatibility.
#' @param ... Passed to [bayesplot::mcmc_combo()].
#' @return A list of plots, invisibly.
#' @export
plot.metabetafit <- function(x, pars = NA, combo = c("hist", "trace"), nvariables = 5, N = NULL,
                             variable = NULL, regex = FALSE, fixed = FALSE, bins = 30, theme = NULL,
                             plot = TRUE, ask = TRUE, newpage = TRUE, ...) {
  variables <- plot_variables(x, pars, variable, regex, fixed)
  pages <- split(variables, ceiling(seq_along(variables) / (N %||% nvariables)))
  plots <- lapply(pages, function(page) {
    bayesplot::mcmc_combo(as.array(x, variable = page), combo = combo, gg_theme = theme, ...)
  })
  if (plot) {
    for (i in seq_along(plots)) {
      if (i > 1) grDevices::devAskNewPage(ask)
      print(plots[[i]])
    }
  }
  invisible(plots)
}

#' MCMC plots implemented in bayesplot
#'
#' @param object A fitted model.
#' @param pars,variable,regex,fixed Variables to plot.
#' @param type Name of a `mcmc_` function of \pkg{bayesplot} without the prefix.
#' @param ... Passed to the \pkg{bayesplot} function.
#' @return A ggplot object.
#' @export
mcmc_plot <- function(object, ...) UseMethod("mcmc_plot")

#' @rdname mcmc_plot
#' @export
mcmc_plot.metabetafit <- function(object, pars = NA, type = "intervals", variable = NULL,
                                  regex = FALSE, fixed = FALSE, ...) {
  if (startsWith(type, "nuts_")) stop2("NUTS diagnostics are not available: metabeta does not use NUTS.")
  plot_function <- get0(paste0("mcmc_", type), envir = asNamespace("bayesplot"), mode = "function")
  if (is.null(plot_function)) stop2("Invalid plot type '", type, "'.")
  draws <- get_draws(object, plot_variables(object, pars, variable, regex, fixed))
  if (type %in% c("rhat", "rhat_hist")) return(plot_function(rhat_values(draws), ...))
  if (type %in% c("neff", "neff_hist")) return(plot_function(neff_ratios(draws), ...))
  plot_function(posterior::as_draws_array(draws), ...)
}

#' @rdname mcmc_plot
#' @export
stanplot <- function(object, ...) mcmc_plot(object, ...)

#' @export
pairs.metabetafit <- function(x, pars = NA, variable = NULL, regex = FALSE, fixed = FALSE, ...) {
  bayesplot::mcmc_pairs(as.array(x, variable = plot_variables(x, pars, variable, regex, fixed)), ...)
}

# Conditional effects ------------------------------------------------------------------

#' Display conditional effects of predictors
#'
#' Expected responses (or predictions) across the range of one predictor, or of
#' two interacting predictors, with the other predictors at their mean or
#' reference level; as in `brms::conditional_effects()`.
#'
#' @param x A fitted model.
#' @param effects Effects to show, e.g. `"Days"` or `"Days:f"`; by default all
#'   main effects and two-way interactions.
#' @param conditions A data frame of conditions, e.g. from [make_conditions()].
#' @param int_conditions Values of the second variable of interactions, as a
#'   named list.
#' @param re_formula Group-level terms to include (default `NA`: none).
#' @param prob Probability mass of the intervals.
#' @param robust Use the median (default) instead of the mean?
#' @param method `"posterior_epred"`, `"posterior_predict"` or `"posterior_linpred"`.
#' @param transform A function applied to the draws.
#' @param resolution Number of points for numeric predictors.
#' @param probs Quantiles of the intervals (instead of `prob`).
#' @param spaghetti,surface,categorical,ordinal,select_points,too_far
#'   Accepted for compatibility.
#' @param ... Passed to the prediction method.
#' @return An object of class `brms_conditional_effects`.
#' @export
conditional_effects <- function(x, ...) UseMethod("conditional_effects")

#' @rdname conditional_effects
#' @export
marginal_effects <- function(x, ...) conditional_effects(x, ...)

#' @rdname conditional_effects
#' @export
conditional_effects.metabetafit <- function(x, effects = NULL, conditions = NULL, int_conditions = NULL,
                                            re_formula = NA, prob = 0.95, robust = TRUE,
                                            method = "posterior_epred", spaghetti = FALSE,
                                            surface = FALSE, categorical = FALSE, ordinal = FALSE,
                                            transform = NULL, resolution = 100, select_points = 0,
                                            too_far = 0, probs = NULL, ...) {
  predict_draws <- switch(method,
    posterior_epred = , fitted = posterior_epred,
    posterior_predict = , predict = posterior_predict,
    posterior_linpred = posterior_linpred
  )
  probs <- probs %||% c((1 - prob) / 2, (1 + prob) / 2)
  data <- x$data
  predictors <- setdiff(all.vars(x$metabeta$design$fixed_block$terms), "Intercept")
  effects <- if (is.null(effects)) default_effects(x) else strsplit(effects, ":", fixed = TRUE)
  for (effect in effects) {
    if (!all(effect %in% predictors)) stop2("'", paste(effect, collapse = ":"), "' is not an effect of the model.")
  }

  # by default one condition: other predictors at their mean or reference level
  if (is.null(conditions)) {
    conditions <- as.data.frame(lapply(data[predictors], typical_value))
    conditions$cond__ <- 1
  }
  conditions$cond__ <- conditions$cond__ %||% seq_len(nrow(conditions))

  response <- deparse1(response(x$formula$formula))
  out <- lapply(effects, function(effect) {
    grid <- effect_grid(data, effect, int_conditions, resolution)
    results <- lapply(seq_len(nrow(conditions)), function(i) {
      fixed <- setdiff(predictors, effect)
      newdata <- cbind(grid, data[rep(1, nrow(grid)), fixed, drop = FALSE])
      for (v in fixed) newdata[[v]] <- conditions[[v]][i] %||% typical_value(data[[v]])
      draws <- predict_draws(x, newdata = newdata, re_formula = re_formula, ...)
      if (!is.null(transform)) draws <- transform(draws)
      qs <- apply(draws, 2, stats::quantile, probs = probs)
      newdata$cond__ <- conditions$cond__[i]
      newdata$effect1__ <- newdata[[effect[1]]]
      if (length(effect) == 2) newdata$effect2__ <- newdata[[effect[2]]]
      newdata$estimate__ <- apply(draws, 2, if (robust) stats::median else mean)
      newdata$se__ <- apply(draws, 2, if (robust) stats::mad else stats::sd)
      newdata$lower__ <- qs[1, ]
      newdata$upper__ <- qs[2, ]
      newdata
    })
    points <- data.frame(effect1__ = data[[effect[1]]], resp__ = get_y(x))
    structure(do.call(rbind, results), effects = effect, response = response, points = points)
  })
  names(out) <- vapply(effects, paste, "", collapse = ":")
  structure(out, class = "brms_conditional_effects")
}

# All main effects and two-way interactions of the population-level formula.
default_effects <- function(x) {
  terms <- x$metabeta$design$fixed_block$terms
  factors <- attr(terms, "factors")
  if (!length(factors)) return(list())
  variables_in <- lapply(rownames(factors), function(r) all.vars(str2lang(r)))
  effects <- lapply(seq_len(ncol(factors)), function(j) setdiff(unique(unlist(variables_in[factors[, j] > 0])), "Intercept"))
  effects <- c(as.list(unique(unlist(effects))), effects[lengths(effects) == 2])
  effects[!duplicated(effects)]
}

typical_value <- function(x) {
  if (is.factor(x)) return(factor(levels(x)[1], levels(x)))
  if (is.character(x)) return(sort(unique(x))[1])
  if (is.logical(x)) return(FALSE)
  mean(x)
}

# Values of the effect's predictors: a grid over a numeric first predictor, all levels of
# factors, and mean -/+ one SD for a numeric second predictor (as in brms).
effect_grid <- function(data, effect, int_conditions, resolution) {
  values <- lapply(seq_along(effect), function(i) {
    v <- data[[effect[i]]]
    if (!is.null(int_conditions[[effect[i]]])) return(int_conditions[[effect[i]]])
    if (is.factor(v)) return(factor(levels(v), levels(v)))
    if (!is.numeric(v)) return(sort(unique(v)))
    if (i == 1) seq(min(v), max(v), length.out = resolution) else signif(mean(v) + c(-1, 0, 1) * stats::sd(v), 3)
  })
  expand.grid(stats::setNames(values, effect), KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
}

#' @export
print.brms_conditional_effects <- function(x, ...) plot(x, ...)

#' @export
plot.brms_conditional_effects <- function(x, ncol = NULL, points = getOption("brms.plot_points", FALSE),
                                          rug = getOption("brms.plot_rug", FALSE), mean = TRUE,
                                          jitter_width = 0, stype = c("contour", "raster"),
                                          line_args = list(), cat_args = list(), errorbar_args = list(),
                                          surface_args = list(), spaghetti_args = list(),
                                          point_args = list(), rug_args = list(), facet_args = list(),
                                          theme = NULL, ask = TRUE, plot = TRUE, ...) {
  plots <- lapply(x, function(effects) {
    effect <- attr(effects, "effects")
    colour <- if (length(effect) == 2) factor(effects$effect2__)
    p <- ggplot2::ggplot(effects, ggplot2::aes(x = .data$effect1__, y = .data$estimate__,
      ymin = .data$lower__, ymax = .data$upper__, colour = colour, fill = colour))
    p <- if (is.numeric(effects$effect1__)) {
      p + ggplot2::geom_ribbon(alpha = 0.2, colour = NA) + ggplot2::geom_line()
    } else {
      p + ggplot2::geom_pointrange(position = ggplot2::position_dodge(width = 0.4))
    }
    if (points) {
      p <- p + ggplot2::geom_point(ggplot2::aes(x = .data$effect1__, y = .data$resp__),
        data = attr(effects, "points"), inherit.aes = FALSE, alpha = 0.4)
    }
    if (length(unique(effects$cond__)) > 1) p <- p + ggplot2::facet_wrap(~cond__, ncol = ncol)
    p + ggplot2::labs(x = effect[1], y = attr(effects, "response"), colour = effect[2], fill = effect[2]) +
      (theme %||% bayesplot::theme_default())
  })
  if (plot) {
    for (i in seq_along(plots)) {
      if (i > 1) grDevices::devAskNewPage(ask)
      print(plots[[i]])
    }
  }
  invisible(plots)
}

#' Prepare fully crossed conditions
#'
#' Conditions for [conditional_effects()]: mean and mean -/+ one SD of numeric
#' variables and all levels of factors, as in \pkg{brms}.
#'
#' @param x A fitted model or a data frame.
#' @param vars Names of the variables to condition on.
#' @param ... Accepted for compatibility.
#' @return A data frame of conditions, with labels in column `cond__`.
#' @export
make_conditions <- function(x, vars, ...) {
  data <- if (is.brmsfit(x)) x$data else as.data.frame(x)
  values <- lapply(data[vars], function(v) {
    if (is.numeric(v)) signif(mean(v) + c(-1, 0, 1) * stats::sd(v), 3) else if (is.factor(v)) levels(v) else sort(unique(v))
  })
  out <- expand.grid(values, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  out$cond__ <- apply(out, 1, function(row) paste(vars, "=", trimws(row), collapse = " & "))
  out
}
