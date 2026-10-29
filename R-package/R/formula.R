# Model formulas: bf() / brmsformula() and the parser that splits a formula into the
# response, the population-level part and the group-level terms.

#' Set up a model formula for use in metabeta
#'
#' Same interface as `brms::brmsformula()`. metabeta fits the main formula of a
#' univariate model; distributional and non-linear formulas are accepted here
#' so that scripts parse, and [brm()] rejects them.
#'
#' @param formula An object of class `formula` (or one that can be coerced to
#'   it) describing the response and the population- and group-level terms.
#' @param ... Additional formulas for distributional or non-linear parameters.
#' @param flist Optional list of such formulas.
#' @param family Same as in [brm()].
#' @param autocor,nl,loop,cmc,sparse,decomp,unused Accepted for compatibility.
#' @param center Center the population-level predictors, so that the prior class
#'   `Intercept` refers to the intercept at the predictor means (default `TRUE`,
#'   as in \pkg{brms}).
#' @return An object of class `c("brmsformula", "bform")`.
#' @examples
#' bf(Reaction ~ Days + (Days | Subject))
#' @export
brmsformula <- function(formula, ..., flist = NULL, family = NULL, autocor = NULL,
                        nl = NULL, loop = NULL, center = NULL, cmc = NULL,
                        sparse = NULL, decomp = NULL, unused = NULL) {
  if (inherits(formula, "brmsformula")) {
    out <- formula
  } else {
    formula <- stats::as.formula(formula)
    if (length(formula) != 3) stop2("Invalid formula: response variable is missing.")
    attr(formula, "nl") <- isTRUE(nl)
    out <- structure(
      list(formula = formula, pforms = list(), pfix = list(), resp = NULL, family = NULL),
      class = c("brmsformula", "bform")
    )
    out$resp <- rename_coef(deparse1(response(formula)))
  }
  if (!is.null(center)) attr(out$formula, "center") <- as.logical(center)
  for (f in c(list(...), flist)) out <- add_formula(out, f)
  if (!is.null(family)) out$family <- validate_family(family)
  if (!is.null(autocor)) out$autocor <- autocor
  out
}

#' @rdname brmsformula
#' @export
bf <- function(formula, ..., flist = NULL, family = NULL, autocor = NULL, nl = NULL, loop = NULL,
               center = NULL, cmc = NULL, sparse = NULL, decomp = NULL) {
  brmsformula(
    formula, ..., flist = flist, family = family, autocor = autocor, nl = nl, loop = loop,
    center = center, cmc = cmc, sparse = sparse, decomp = decomp
  )
}

#' @rdname brmsformula
#' @param dpar,resp Accepted for compatibility.
#' @export
lf <- function(..., flist = NULL, dpar = NULL, resp = NULL, center = NULL, cmc = NULL,
               sparse = NULL, decomp = NULL) {
  structure(list(formulas = c(list(...), flist)), class = "metabeta_lf")
}

#' @rdname brmsformula
#' @export
nlf <- function(formula, ..., flist = NULL, dpar = NULL, resp = NULL, loop = NULL) {
  structure(list(formulas = c(list(formula), list(...), flist)), class = "metabeta_lf")
}

# Store a formula of a distributional or non-linear parameter; brm() will reject it.
add_formula <- function(x, f) {
  if (is.numeric(f)) stop2("Fixing distributional parameters is not supported by metabeta.")
  f <- if (inherits(f, "brmsformula")) f$formula else stats::as.formula(f)
  x$pforms[[deparse1(f[[2]])]] <- f
  x
}

#' @export
"+.bform" <- function(e1, e2) {
  if (inherits(e2, "family") || is.function(e2) || is.character(e2)) {
    e1$family <- validate_family(e2)
  } else if (inherits(e2, "metabeta_lf")) {
    for (f in e2$formulas) e1 <- add_formula(e1, f)
  } else if (!is.null(e2)) {
    e1 <- add_formula(e1, e2)
  }
  e1
}

#' @export
print.brmsformula <- function(x, wsp = 0, digits = 2, ...) {
  cat(deparse1(x$formula), "\n")
  for (f in x$pforms) cat(strrep(" ", wsp), deparse1(f), "\n", sep = "")
  invisible(x)
}

#' Checks if argument is a `brmsformula` object
#' @param x An R object.
#' @return `TRUE` or `FALSE`.
#' @export
is.brmsformula <- function(x) inherits(x, "brmsformula")

# Parsing ------------------------------------------------------------------------------

# Function calls metabeta cannot fit, wherever they appear in a formula.
UNSUPPORTED_TERMS <- c(
  "s", "t2", "gp", "mo", "me", "mi", "cs", "mm", "mmc", "offset", "ar", "arma", "ma", "cosy",
  "car", "sar", "fcor", "unstr", "se", "weights", "cens", "trunc", "rate", "dec", "thres",
  "vreal", "vint", "mvbind", "cbind"
)

is_bar <- function(e) is.call(e) && deparse1(e[[1]]) %in% c("|", "||")

# Names of all functions called in an expression.
called_functions <- function(e) {
  if (!is.call(e)) return(character(0))
  c(deparse1(e[[1]]), unlist(lapply(as.list(e)[-1], called_functions)))
}

# The response expression, without addition terms such as `| trials(1)`.
response <- function(formula) {
  lhs <- formula[[2]]
  if (is_bar(lhs)) lhs[[2]] else lhs
}

# Split a brms formula into its parts, rejecting what metabeta cannot fit.
#
# Returns: resp (response expression), trials (expression or NULL), fixed (formula of
# the population-level part), center, int_special (`0 + Intercept` was used), bars (the
# parsed group-level terms), group (name of the grouping factor) and group_vars.
parse_model_formula <- function(bform) {
  if (length(bform$pforms)) {
    stop2("Distributional and non-linear formulas (", collapse_comma(names(bform$pforms)), ") are not supported by metabeta.")
  }
  if (isTRUE(attr(bform$formula, "nl"))) stop2("Non-linear models are not supported by metabeta.")
  if (!is.null(bform$autocor)) stop2("Autocorrelation structures are not supported by metabeta.")
  formula <- bform$formula

  unsupported <- intersect(called_functions(formula), UNSUPPORTED_TERMS)
  if (length(unsupported)) {
    stop2("Terms of type ", collapse_comma(paste0(unsupported, "()")), " are not supported by metabeta. Use brms::brm() to fit this model.")
  }
  if ("." %in% all.vars(formula)) {
    stop2("The '.' shorthand is not supported by metabeta; list the predictors explicitly.")
  }

  # addition terms: only trials(), for binomial models with one trial
  trials <- NULL
  if (is_bar(formula[[2]])) {
    add <- formula[[2]][[3]]
    if (!is.call(add) || deparse1(add[[1]]) != "trials") {
      stop2("Addition term '", deparse1(add), "' is not supported by metabeta (only trials(1) is).")
    }
    trials <- add[[2]]
  }

  rhs <- formula[[3]]
  if (is_bar(rhs)) stop2("Group-level terms must be enclosed in parentheses, e.g. '(1 | g)'.")
  # terms() keeps each group-level term `(lhs | g)` as one term labelled "lhs | g"
  rhs_terms <- stats::terms(stats::as.formula(call("~", rhs)))
  labels <- attr(rhs_terms, "term.labels")
  is_group_term <- grepl("|", labels, fixed = TRUE)
  has_intercept <- attr(rhs_terms, "intercept") == 1
  fixed <- if (any(!is_group_term)) {
    stats::reformulate(labels[!is_group_term], intercept = has_intercept, env = environment(formula))
  } else {
    stats::as.formula(if (has_intercept) "~ 1" else "~ 0", env = environment(formula))
  }

  bars <- lapply(labels[is_group_term], function(l) parse_bar(str2lang(l)))
  if (!length(bars)) {
    stop2("metabeta fits multilevel models and needs a group-level term such as '(1 | g)'. Use brms::brm() for models without group-level effects.")
  }
  groups <- unique(vapply(bars, `[[`, "", "group"))
  if (length(groups) > 1) {
    stop2("metabeta supports exactly one grouping factor, but the formula uses ", collapse_comma(groups), ". Use brms::brm() for crossed or nested group-level terms.")
  }

  int_special <- "Intercept" %in% labels
  list(
    resp = response(formula),
    trials = trials,
    fixed = fixed,
    center = !isFALSE(attr(formula, "center")) && !int_special,
    int_special = int_special,
    bars = bars,
    group = groups,
    group_vars = bars[[1]]$group_vars
  )
}

# Parse one group-level term: `lhs | g`, `lhs || g` (uncorrelated) or `lhs | ID | g`.
# Terms sharing an ID are modelled as correlated, like in brms.
parse_bar <- function(bar) {
  correlated <- deparse1(bar[[1]]) == "|"
  lhs <- bar[[2]]
  group <- bar[[3]]
  id <- NA_character_
  if (is_bar(lhs)) {
    id <- deparse1(lhs[[3]])
    correlated <- correlated && deparse1(lhs[[1]]) == "|"
    lhs <- lhs[[2]]
  }
  # gr(g, cor = FALSE, id = "p")
  if (is.call(group) && deparse1(group[[1]]) == "gr") {
    args <- as.list(group)[-1]
    named <- names(args) %||% rep("", length(args))
    group <- args[[which(named == "")[1]]]
    options <- args[named != ""]
    unsupported <- setdiff(names(options), c("cor", "id"))
    if (length(unsupported)) stop2("Arguments ", collapse_comma(unsupported), " of gr() are not supported by metabeta.")
    if (!is.null(options$cor)) correlated <- correlated && isTRUE(eval(options$cor, baseenv()))
    if (!is.null(options$id)) id <- as.character(options$id)
  }
  if (is.call(group) && deparse1(group[[1]]) == "/") {
    stop2("Nested grouping factors such as '", deparse1(group), "' expand to several grouping factors; metabeta supports exactly one.")
  }
  vars <- all.vars(group)
  if (!length(vars) || !all(all.names(group) %in% c(":", vars))) {
    stop2("Invalid grouping term '", deparse1(group), "'. Use a variable or an interaction of variables, such as 'g' or 'g1:g2'.")
  }
  list(lhs = lhs, group = paste(vars, collapse = ":"), group_vars = vars, correlated = correlated, id = id)
}
