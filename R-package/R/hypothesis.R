# hypothesis(): tests of hypotheses about (functions of) parameters, as in brms.

#' Non-linear hypothesis testing
#'
#' Evaluate point and one-sided hypotheses about (functions of) parameters, with
#' the interface and output of `brms::hypothesis()`. Evidence ratios of point
#' hypotheses are Savage-Dickey density ratios, which need prior draws, i.e. a
#' model fitted with `sample_prior = "yes"`.
#'
#' @param x A fitted model, or a data frame / matrix / draws object.
#' @param hypothesis Hypotheses such as `"Days > 8"` or `"Intercept - 250 = 0"`.
#' @param class Prefix of the parameter names used in `hypothesis` (default
#'   `"b"`); `""` for full variable names.
#' @param group Grouping factor, for classes `sd` and `cor` or for `scope`.
#' @param scope `"standard"`, or `"ranef"` / `"coef"` to test the hypothesis for
#'   each level of `group`.
#' @param alpha Two-sided intervals have mass `1 - alpha`, one-sided ones
#'   `1 - 2 * alpha`.
#' @param robust Use median and MAD instead of mean and SD?
#' @param seed Seed for the random order of prior draws.
#' @param y Prior draws (for `density_ratio()`).
#' @param point Value at which `density_ratio()` evaluates the densities.
#' @param n Number of points of the density estimates.
#' @param ... Ignored, or passed to [stats::density()] by `density_ratio()`.
#' @return An object of class `brmshypothesis`.
#' @export
hypothesis <- function(x, ...) UseMethod("hypothesis")

#' @rdname hypothesis
#' @export
hypothesis.metabetafit <- function(x, hypothesis, class = "b", group = "",
                                   scope = c("standard", "ranef", "coef"), alpha = 0.05,
                                   robust = FALSE, seed = NULL, ...) {
  if (!is.null(seed)) set.seed(seed)
  scope <- match.arg(scope)
  if (scope == "standard") {
    draws <- as.data.frame(x)
    priors <- function(variables) prior_draws(x, variable = variables)
    return(test_hypotheses(draws, priors, hypothesis, name_prefix(class, group), alpha, robust))
  }
  # one test per level of the grouping factor, on its group-level effects or coefficients
  if (!nzchar(group)) stop2("Argument 'group' is required when 'scope' is '", scope, "'.")
  r <- if (scope == "coef") coef(x, summary = FALSE)[[group]] else ranef(x, summary = FALSE)[[group]]
  tests <- lapply(dimnames(r)[[2]], function(level) {
    draws <- as.data.frame(matrix(r[, level, ], ncol = dim(r)[3], dimnames = list(NULL, dimnames(r)[[3]])))
    test <- test_hypotheses(draws, NULL, hypothesis, "", alpha, robust)
    test$hypothesis <- cbind(Group = level, test$hypothesis)
    test
  })
  out <- tests[[1]]
  out$hypothesis <- do.call(rbind, lapply(tests, `[[`, "hypothesis"))
  out$samples <- do.call(cbind, lapply(tests, `[[`, "samples"))
  out$prior_samples <- do.call(cbind, lapply(tests, `[[`, "prior_samples"))
  out
}

#' @rdname hypothesis
#' @export
hypothesis.default <- function(x, hypothesis, class = "", group = "", alpha = 0.05, robust = FALSE, ...) {
  draws <- as.data.frame(x)
  draws <- draws[setdiff(names(draws), c(".chain", ".iteration", ".draw"))]
  test_hypotheses(draws, NULL, hypothesis, name_prefix(class, group), alpha, robust)
}

# "b" -> "b_", class "sd" with group "g" -> "sd_g__"; class NULL or "" -> no prefix
name_prefix <- function(class, group) {
  class <- class %||% ""
  group <- group %||% ""
  if (nzchar(group)) paste0(class, "_", group, "__") else if (nzchar(class)) paste0(class, "_") else ""
}

# Test each hypothesis on a data frame of draws. 'priors' returns the prior draws of the
# given variables (or is NULL).
test_hypotheses <- function(draws, priors, hypotheses, prefix, alpha, robust) {
  tests <- lapply(seq_along(hypotheses), function(i) {
    test_hypothesis(hypotheses[i], names(hypotheses)[i], draws, priors, prefix, alpha, robust)
  })
  samples <- as.data.frame(lapply(tests, `[[`, "samples"), col.names = paste0("H", seq_along(tests)))
  prior_samples <- as.data.frame(lapply(tests, `[[`, "prior_samples"), col.names = names(samples))
  summary <- do.call(rbind, lapply(tests, `[[`, "summary"))
  rownames(summary) <- NULL
  structure(
    list(hypothesis = summary, samples = samples, prior_samples = prior_samples, class = sub("_+$", "", prefix), alpha = alpha),
    class = "brmshypothesis"
  )
}

# One hypothesis "left (= or < or >) right", tested as "(left)-(right) (= or < or >) 0".
test_hypothesis <- function(h, name, draws, priors, prefix, alpha, robust) {
  h <- gsub("[ \t\r\n]", "", h)
  sign <- regmatches(h, gregexpr("=|<|>", h))[[1]]
  sides <- strsplit(h, "=|<|>")[[1]]
  if (length(sign) != 1 || length(sides) != 2) stop2("Every hypothesis must be of the form 'left (= OR < OR >) right'.")
  expression <- paste0("(", sides[1], ")", if (sides[2] != "0") paste0("-(", sides[2], ")"))

  # Parameter names may contain ':', '[', ']' and ','; brms replaces them to parse.
  readable <- function(x) {
    for (r in list(c(":", "___"), c("[", "."), c("]", "."), c(",", ".."))) x <- gsub(r[1], r[2], x, fixed = TRUE)
    x
  }
  parameters <- names(draws)[startsWith(names(draws), prefix)]
  short_names <- readable(substring(parameters, nchar(prefix) + 1))
  code <- readable(expression)
  used <- unique(regmatches(code, gregexpr("[A-Za-z.][A-Za-z0-9._]*(?![A-Za-z0-9._(])", code, perl = TRUE))[[1]])
  used <- used[!grepl("^\\.?[0-9]", used)]
  unknown <- setdiff(used, short_names)
  if (length(unknown)) stop2("Some parameters cannot be found in the model: \n", collapse_comma(paste0(prefix, unknown)))
  variables <- parameters[match(used, short_names)]
  evaluate <- function(d) as.numeric(eval(str2lang(code), stats::setNames(as.list(d[variables]), used), baseenv()))

  x <- evaluate(draws)
  prior_x <- NULL
  if (!is.null(priors)) {
    prior <- priors(variables)
    if (ncol(prior) == length(variables)) prior_x <- evaluate(prior)
  }
  probs <- if (sign == "=") c(alpha / 2, 1 - alpha / 2) else c(alpha, 1 - alpha)
  interval <- stats::quantile(x, probs, names = FALSE)
  evidence <- switch(sign,
    "=" = if (is.null(prior_x)) NA else density_ratio(x, prior_x, point = 0),
    "<" = sum(x < 0) / sum(x >= 0),
    ">" = sum(x > 0) / sum(x <= 0)
  )
  probability <- if (is.infinite(evidence)) 1 else evidence / (1 + evidence)
  star <- if (sign == "=") interval[1] > 0 || interval[2] < 0 else isTRUE(probability > 1 - alpha)
  summary <- data.frame(
    Hypothesis = if (length(name) && nzchar(name)) name else paste(expression, sign, "0"),
    Estimate = if (robust) stats::median(x) else mean(x),
    Est.Error = if (robust) stats::mad(x) else stats::sd(x),
    CI.Lower = interval[1], CI.Upper = interval[2],
    Evid.Ratio = evidence, Post.Prob = probability,
    Star = if (star) "*" else ""
  )
  list(summary = summary, samples = x, prior_samples = prior_x %||% rep(NA, length(x)))
}

#' @rdname hypothesis
#' @export
density_ratio <- function(x, y = NULL, point = 0, n = 4096, ...) {
  density_at <- function(v, at) {
    # the density grid covers the draws, extended to 'at' if it lies outside them
    from <- min(v)
    to <- max(v)
    if (from > at) from <- at - stats::sd(v) / 4 else if (to < at) to <- at + stats::sd(v) / 4
    d <- stats::density(v, from = from, to = to, n = n, ...)
    stats::spline(d$x, d$y, xout = at)$y
  }
  out <- vapply(point, density_at, 0, v = x)
  if (is.null(y)) out else out / vapply(point, density_at, 0, v = y)
}

#' @export
print.brmshypothesis <- function(x, digits = 2, chars = 20, short = getOption("brms.short_summary", FALSE), ...) {
  table <- x$hypothesis
  long <- nchar(table$Hypothesis) > chars
  table$Hypothesis[long] <- paste0(substr(table$Hypothesis[long], 1, chars - 3), "...")
  numeric <- vapply(table, is.numeric, TRUE)
  table[numeric] <- lapply(table[numeric], round, digits = digits)
  cat(paste0("Hypothesis Tests for class ", x$class, ":\n"))
  print(table, quote = FALSE)
  if (!short) {
    one_sided <- (1 - 2 * x$alpha) * 100
    two_sided <- (1 - x$alpha) * 100
    cat(
      "---\n",
      "'CI': ", one_sided, "%-CI for one-sided and ", two_sided, "%-CI for two-sided hypotheses.\n",
      "'*': For one-sided hypotheses, the posterior probability exceeds ", two_sided, "%;\n",
      "for two-sided hypotheses, the value tested against lies outside the ", two_sided, "%-CI.\n",
      "Posterior probabilities of point hypotheses assume equal prior probabilities.\n",
      sep = ""
    )
  }
  invisible(x)
}

#' @export
plot.brmshypothesis <- function(x, nvariables = 5, N = NULL, ignore_prior = FALSE, chars = 40,
                                colors = NULL, theme = NULL, ask = TRUE, plot = TRUE, ...) {
  densities <- do.call(rbind, lapply(seq_along(x$samples), function(i) {
    label <- x$hypothesis$Hypothesis[i]
    out <- data.frame(value = x$samples[[i]], Type = "Posterior", Hypothesis = label)
    if (!ignore_prior && !all(is.na(x$prior_samples[[i]]))) {
      out <- rbind(out, data.frame(value = x$prior_samples[[i]], Type = "Prior", Hypothesis = label))
    }
    out
  }))
  p <- ggplot2::ggplot(densities, ggplot2::aes(x = .data$value, fill = .data$Type)) +
    ggplot2::geom_density(alpha = 0.7) +
    ggplot2::facet_wrap(~Hypothesis, scales = "free", ncol = 1) +
    ggplot2::labs(x = NULL, y = NULL, fill = NULL) +
    (theme %||% bayesplot::theme_default())
  if (plot) print(p)
  invisible(list(p))
}
