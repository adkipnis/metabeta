# Priors: brmsprior objects, metabeta's default priors, the combination of defaults and
# user priors, and the translation of the result to the network's scale.

PRIOR_COLUMNS <- c("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "tag", "source")

new_brmsprior <- function(df) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  for (col in setdiff(PRIOR_COLUMNS, names(df))) {
    df[[col]] <- rep(if (col %in% c("lb", "ub")) NA_character_ else "", nrow(df))
  }
  df[] <- lapply(df, as.character)
  df <- df[PRIOR_COLUMNS]
  rownames(df) <- NULL
  class(df) <- c("brmsprior", "data.frame")
  df
}

as_plain_df <- function(x) {
  class(x) <- "data.frame"
  x
}

#' Prior definitions for metabeta models
#'
#' Define priors for parameters or classes of parameters, as in \pkg{brms}.
#' `prior()` takes unquoted arguments, `prior_()` one-sided formulas, and
#' `set_prior()` / `prior_string()` character strings.
#'
#' Classes: `b` (population-level effects), `Intercept` (the intercept at the
#' predictor means, as in \pkg{brms}), `sd` (group-level standard deviations),
#' `cor` or `L` (their correlations) and `sigma` (residual standard deviation).
#' The network was trained on these distributions only: `normal(mu, sigma)` or
#' `student_t(5, mu, sigma)` for `b` and `Intercept` (one family for all);
#' `normal(0, sigma)`, `student_t(5, 0, sigma)` or `exponential(rate)` for `sd`
#' (one family for all) and `sigma`, truncated at zero; and `lkj(eta)` for
#' `cor`. Other distributions and flat priors raise an error in [brm()].
#'
#' @param prior A character string defining a distribution (for `prior()` an
#'   unquoted expression, for `prior_()` a one-sided formula).
#' @param class,coef,group Parameter class, coefficient and grouping factor.
#' @param resp,dpar,nlpar,tag,check Accepted for compatibility.
#' @param lb,ub Bounds; only the implicit lower bound 0 of `sd` and `sigma` is
#'   supported.
#' @param ... Arguments passed to `set_prior()`.
#' @return An object of class `brmsprior`.
#' @examples
#' prior(normal(0, 10), class = b) +
#'   prior(normal(0, 20), class = sd, group = Subject)
#' set_prior("lkj(2)", class = "cor")
#' @export
set_prior <- function(prior, class = "b", coef = "", group = "", resp = "", dpar = "",
                      nlpar = "", lb = NA, ub = NA, tag = "", check = TRUE) {
  columns <- list(
    prior = prior, class = class, coef = coef, group = group, resp = resp, dpar = dpar,
    nlpar = nlpar, lb = lb, ub = ub, tag = tag, source = "user"
  )
  n <- max(lengths(columns))
  new_brmsprior(lapply(columns, function(x) rep_len(as.character(x), n)))
}

#' @rdname set_prior
#' @export
prior <- function(prior, ...) {
  args <- as.list(match.call())[-1]
  args <- lapply(args, function(x) if (is.character(x)) x else deparse1(x))
  args$check <- NULL
  do.call(set_prior, args)
}

#' @rdname set_prior
#' @export
prior_ <- function(prior, ...) {
  args <- lapply(list(prior = prior, ...), function(x) {
    if (inherits(x, "formula")) deparse1(x[[length(x)]]) else x
  })
  do.call(set_prior, args)
}

#' @rdname set_prior
#' @export
prior_string <- function(prior, ...) set_prior(prior, ...)

#' @rdname set_prior
#' @export
empty_prior <- function() new_brmsprior(data.frame(prior = character(0)))

#' Checks if argument is a `brmsprior` object
#' @param x An R object.
#' @return `TRUE` or `FALSE`.
#' @export
is.brmsprior <- function(x) inherits(x, "brmsprior")

#' @export
c.brmsprior <- function(x, ..., replace = FALSE) {
  priors <- Filter(Negate(is.null), list(x, ...))
  if (!all(vapply(priors, is.brmsprior, TRUE))) stop2("All objects must be of class 'brmsprior'.")
  out <- do.call(rbind, lapply(priors, as_plain_df))
  key <- paste(out$class, out$coef, out$group, out$resp, out$dpar, out$nlpar)
  if (!replace && anyDuplicated(key)) stop2("Duplicated prior specifications are not allowed.")
  new_brmsprior(out[!duplicated(key, fromLast = TRUE), ])
}

#' @export
"+.brmsprior" <- function(e1, e2) c(e1, e2)

#' @export
print.brmsprior <- function(x, show_df = NULL, ...) {
  if (is.null(show_df)) show_df <- nrow(x) > 1
  if (show_df) {
    print(as_plain_df(x), row.names = FALSE)
  } else {
    bounds <- ifelse(x$lb %in% "0", "<lower=0> ", "")
    cat(paste0(bounds, prior_label(x), " ~ ", ifelse(nzchar(x$prior), x$prior, "(flat)")), sep = "\n")
  }
  invisible(x)
}

# Parameter labels of prior rows as in brms, e.g. "b_Days" or "sd_Subject__Intercept".
prior_label <- function(x) {
  label <- ifelse(nzchar(x$group), paste0(x$class, "_", x$group), x$class)
  ifelse(nzchar(x$coef), paste0(label, ifelse(nzchar(x$group), "__", "_"), x$coef), label)
}

# Default priors ---------------------------------------------------------------------
#
# metabeta's defaults are defined on the network's scale (bambiDefaultPriors() in
# metabeta/utils/priors.py) and shown here on the data scale.

DEFAULT_SCALES <- list(
  gaussian = list(Intercept = 2.5, b = 2.5, sd = 2.5, sigma = 1),
  bernoulli = list(Intercept = 1.5, b = 1, sd = 2.5),
  poisson = list(Intercept = 2.5, b = 2.5, sd = 2.5)
)

default_priors <- function(design) {
  scales <- DEFAULT_SCALES[[design$likelihood]]
  n <- function(x) signif(x, 4)
  intercept <- sprintf("normal(%s, %s)", n(design$y_mean), n(scales$Intercept * design$y_sd))
  row <- function(prior, class, coef = "", group = "", lb = "", source = "default") {
    data.frame(prior = prior, class = class, coef = coef, group = group, lb = lb, ub = "", source = source)
  }
  rows <- list(row("", "b"))
  for (k in design$b_names) {
    if (k == "Intercept" && design$int_type == "centered") {
      rows <- c(rows, list(row(intercept, "Intercept")))
    } else if (k == "Intercept") {
      rows <- c(rows, list(row(intercept, "b", k)))
    } else {
      rows <- c(rows, list(row(sprintf("normal(0, %s)", n(scales$b * design$y_sd / design$x_scales[[k]])), "b", k)))
    }
  }
  if (design$correlated) {
    rows <- c(rows, list(row("lkj(1)", "cor"), row("lkj(1)", "cor", group = design$group, source = "(vectorized)")))
  }
  rows <- c(rows, list(row("", "sd", lb = "0"), row("", "sd", group = design$group, lb = "0")))
  for (k in design$re_names) {
    prior <- sprintf("normal(0, %s)", n(scales$sd * design$y_sd / design$z_scales[[k]]))
    rows <- c(rows, list(row(prior, "sd", k, design$group, lb = "0")))
  }
  if (design$likelihood == "gaussian") {
    rows <- c(rows, list(row(sprintf("student_t(5, 0, %s)", n(scales$sigma * design$y_sd)), "sigma", lb = "0")))
  }
  out <- new_brmsprior(do.call(rbind, rows))
  out[order(tolower(out$class), out$group, out$coef), ]
}

# Combining defaults and user priors -------------------------------------------------

# The full prior table: defaults, overwritten by the user's priors. As in brms, a prior
# on a class (or on a group) applies to all its coefficients that have no prior of their own.
resolve_priors <- function(design, prior = NULL) {
  table <- default_priors(design)
  if (is.null(prior)) return(table)
  if (!is.brmsprior(prior)) stop2("Argument 'prior' must be created by prior() or set_prior().")
  prior$class[prior$class == "L"] <- "cor"
  prior$prior <- sub("^lkj_corr_cholesky\\(", "lkj(", prior$prior)

  for (i in seq_len(nrow(prior))) {
    p <- prior[i, ]
    row <- which(table$class == p$class & table$coef == p$coef & table$group == p$group &
      !nzchar(p$resp) & !nzchar(p$dpar) & !nzchar(p$nlpar))
    if (!length(row)) {
      stop2(
        "The following priors do not correspond to any model parameter: \n",
        prior_label(p), " ~ ", p$prior, "\nFunction 'default_prior' might be helpful to you."
      )
    }
    if (!is.na(p$ub) || (!is.na(p$lb) && !(p$class %in% c("sd", "sigma") && p$lb == "0"))) {
      stop2("Bounded priors are not supported by metabeta (class '", p$class, "').")
    }
    table$prior[row] <- p$prior
    table$source[row] <- "user"
  }

  # Rows are sorted class, group, coef, so a class-wide row comes before group rows.
  for (i in which(table$source == "user" & !nzchar(table$coef))) {
    same_class <- table$class == table$class[i]
    covered <- if (nzchar(table$group[i])) {
      same_class & table$group == table$group[i] & nzchar(table$coef)
    } else {
      same_class & (nzchar(table$coef) | nzchar(table$group))
    }
    covered <- covered & table$source != "user"
    table$prior[covered] <- table$prior[i]
    table$source[covered] <- "(vectorized)"
  }
  table
}

# The prior string of one parameter.
prior_of <- function(table, class, coef = "", group = "") {
  table$prior[table$class == class & table$coef == coef & table$group == group]
}

# Parse a prior string such as "student_t(5, 0, 2.5)" into list(dist, args).
parse_prior <- function(string) {
  if (!nzchar(string)) return(list(dist = "flat", args = list(), string = string))
  e <- str2lang(string)
  dist <- deparse1(e[[1]])
  args <- tryCatch(lapply(as.list(e)[-1], eval, envir = baseenv()), error = function(err) {
    stop2("The arguments of prior '", string, "' must be numbers.")
  })
  known <- list(normal = c("mu", "sigma"), student_t = c("nu", "mu", "sigma"), exponential = "lambda", lkj = "eta")
  if (dist %in% names(known)) {
    if (length(args) != length(known[[dist]])) stop2("Prior '", string, "' has the wrong number of arguments.")
    names(args) <- known[[dist]]
  }
  list(dist = dist, args = args, string = string)
}

# The distribution of a prior, checked against those the network was trained on.
# Location priors (b, Intercept): normal or student_t(5). Scale priors (sd, sigma):
# half-normal, half-student_t(5) or exponential. Returns family code, location and scale.
prior_family <- function(string, parameter, kind = c("location", "scale")) {
  kind <- match.arg(kind)
  p <- parse_prior(string)
  allowed <- if (kind == "location") "normal() and student_t(5, ...)" else "normal(0, ...), student_t(5, 0, ...) and exponential()"
  if (p$dist == "flat") stop2("Flat priors are not supported by metabeta (parameter '", parameter, "'). Specify a proper prior, e.g. normal(0, 10).")
  if (!p$dist %in% c("normal", "student_t", if (kind == "scale") "exponential")) {
    stop2("The prior '", string, "' of parameter '", parameter, "' is not supported by metabeta, which was trained on ", allowed, " priors.")
  }
  if (p$dist == "student_t" && p$args$nu != 5) {
    stop2("metabeta supports student_t priors with 5 degrees of freedom only (parameter '", parameter, "' has '", string, "').")
  }
  if (kind == "scale" && p$dist != "exponential" && p$args$mu != 0) {
    stop2("Priors of standard deviations must be centered at zero (parameter '", parameter, "' has '", string, "').")
  }
  switch(p$dist,
    normal = list(family = 0L, location = p$args$mu, scale = p$args$sigma),
    student_t = list(family = 1L, location = p$args$mu, scale = p$args$sigma),
    exponential = list(family = 2L, location = 0, scale = 1 / p$args$lambda)
  )
}

# Prior hyperparameters on the network's scale, the prior keys of metabeta's coercePriors().
#
# Every prior is a location-scale family, so the linear maps of design.R carry over
# exactly: a slope b_k = b*_k * y_sd / x_scale_k, the intercept = b*_0 * y_sd + y_mean,
# a standard deviation sd_k = sd*_k * y_sd / z_scale_k and sigma = sigma* * y_sd.
model_hyper <- function(design, table) {
  ffx <- lapply(design$b_names, function(k) {
    if (k == "Intercept") {
      class <- if (design$int_type == "centered") "Intercept" else "b"
      p <- prior_family(prior_of(table, class, if (class == "b") k else ""), k, "location")
      p$location <- (p$location - design$y_mean) / design$y_sd
      p$scale <- p$scale / design$y_sd
    } else {
      p <- prior_family(prior_of(table, "b", k), paste0("b_", k), "location")
      p$location <- p$location * design$x_scales[[k]] / design$y_sd
      p$scale <- p$scale * design$x_scales[[k]] / design$y_sd
    }
    p
  })
  sds <- lapply(design$re_names, function(k) {
    p <- prior_family(prior_of(table, "sd", k, design$group), paste0("sd_", design$group, "__", k), "scale")
    p$scale <- p$scale * design$z_scales[[k]] / design$y_sd
    p
  })
  if (length(unique(sapply(ffx, `[[`, "family"))) > 1) {
    stop2("metabeta requires all population-level priors (classes 'b' and 'Intercept') to come from the same family.")
  }
  if (length(unique(sapply(sds, `[[`, "family"))) > 1) {
    stop2("metabeta requires the priors of all group-level standard deviations (class 'sd') to come from the same family.")
  }

  hyper <- list(
    nu_ffx = sapply(ffx, `[[`, "location")[design$perm],
    tau_ffx = sapply(ffx, `[[`, "scale")[design$perm],
    family_ffx = ffx[[1]]$family,
    tau_rfx = sapply(sds, `[[`, "scale"),
    family_sigma_rfx = sds[[1]]$family,
    eta_rfx = 0  # 0 means uncorrelated group-level effects
  )
  if (design$correlated) {
    p <- parse_prior(prior_of(table, "cor", group = design$group))
    if (p$dist != "lkj") stop2("The prior of the correlations must be lkj(eta) (got '", p$string, "').")
    hyper$eta_rfx <- p$args$eta
  }
  if (design$likelihood == "gaussian") {
    p <- prior_family(prior_of(table, "sigma"), "sigma", "scale")
    hyper$tau_eps <- p$scale / design$y_sd
    hyper$family_sigma_eps <- p$family
  }
  hyper
}

# User interface ---------------------------------------------------------------------

#' Default priors of metabeta models
#'
#' The priors metabeta uses by default, in the format of \pkg{brms}: weakly
#' informative normal priors scaled to the data (flat priors are not supported).
#'
#' @param object,formula A model formula (or `brmsformula`), or a fitted model.
#' @param data A data frame containing all model variables.
#' @param family A description of the response distribution.
#' @param autocor,data2,knots,sparse Accepted for compatibility.
#' @param drop_unused_levels Should unused factor levels be dropped?
#' @param ... Further arguments passed to methods.
#' @return An object of class `brmsprior`.
#' @export
default_prior <- function(object, ...) UseMethod("default_prior")

#' @rdname default_prior
#' @export
default_prior.default <- function(object, data, family = gaussian(), autocor = NULL,
                                  data2 = NULL, knots = NULL, drop_unused_levels = TRUE,
                                  sparse = NULL, ...) {
  default_priors(prepare_model(object, data, family, drop_unused_levels)$design)
}

#' @rdname default_prior
#' @export
default_prior.metabetafit <- function(object, ...) default_priors(object$metabeta$design)

#' @rdname default_prior
#' @export
get_prior <- function(formula, ...) default_prior(formula, ...)

#' Validate prior specifications
#'
#' The full prior of a model: the defaults combined with the user's priors.
#'
#' @inheritParams default_prior
#' @param prior Priors specified by the user, e.g. created by [prior()].
#' @param formula A model formula or `brmsformula`.
#' @param sample_prior,data2,knots Accepted for compatibility.
#' @return An object of class `brmsprior`.
#' @export
validate_prior <- function(prior, formula, data, family = gaussian(), sample_prior = "no",
                           data2 = NULL, knots = NULL, drop_unused_levels = TRUE, ...) {
  design <- prepare_model(formula, data, family, drop_unused_levels)$design
  table <- resolve_priors(design, prior)
  model_hyper(design, table)  # checks that metabeta supports the priors
  table
}

#' @export
prior_summary.metabetafit <- function(object, all = TRUE, ...) {
  # like brms, the short form lists the prior statements, not every parameter they cover
  if (all) object$prior else object$prior[nzchar(object$prior$prior) & object$prior$source != "(vectorized)", ]
}
