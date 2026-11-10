# Response families. Every brms family constructor exists (most are generated, see
# R/generated.R), but brm() only accepts the likelihoods metabeta was trained on.

# The likelihoods metabeta can fit, with the one link each network supports.
SUPPORTED_LINKS <- c(gaussian = "identity", bernoulli = "logit", binomial = "logit", poisson = "log")

# Default links of all brms families.
DEFAULT_LINKS <- c(
  gaussian = "identity", student = "identity", skew_normal = "identity", binomial = "logit",
  beta_binomial = "logit", bernoulli = "logit", poisson = "log", negbinomial = "log",
  geometric = "log", Gamma = "inverse", weibull = "log", exponential = "log",
  lognormal = "identity", frechet = "log", inverse.gaussian = "1/mu^2", exgaussian = "identity",
  wiener = "identity", beta = "logit", xbeta = "logit", von_mises = "tan_half",
  asym_laplace = "identity", gen_extreme_value = "identity", categorical = "logit",
  multinomial = "logit", cumulative = "logit", sratio = "logit", cratio = "logit", acat = "logit",
  hurdle_poisson = "log", hurdle_negbinomial = "log", hurdle_gamma = "log",
  hurdle_lognormal = "identity", hurdle_cumulative = "logit", zero_inflated_poisson = "log",
  zero_inflated_negbinomial = "log", zero_inflated_binomial = "logit",
  zero_inflated_beta_binomial = "logit", zero_inflated_beta = "logit",
  zero_one_inflated_beta = "logit", shifted_lognormal = "identity", cox = "log",
  dirichlet = "logit", dirichlet_multinomial = "logit", logistic_normal = "identity"
)

#' Special family functions for brms models
#'
#' Family objects as in \pkg{brms}. Only `gaussian()` (identity link),
#' `bernoulli()` and `binomial()` with a single trial (logit link), and
#' `poisson()` (log link) can be fitted by metabeta; the other constructors
#' exist so that existing scripts run, and [brm()] rejects their families.
#'
#' @param family A character string naming the distribution of the response.
#' @param link A character string naming the link function of the mean.
#' @param link_sigma,link_shape,link_nu,link_phi,link_kappa,link_beta,link_zi,link_hu,link_zoi,link_coi,link_disc,link_bs,link_ndt,link_bias,link_xi,link_alpha,link_quantile
#'   Links of distributional parameters; accepted for compatibility.
#' @param threshold,refcat Accepted for compatibility.
#' @param ... For `mixture()`: mixture components.
#' @param flist,nmix,order Arguments of `mixture()`, which metabeta does not support.
#' @param name,dpars,links,type,lb,ub,vars,loop,specials,log_lik,posterior_predict,posterior_epred,predict,fitted,env
#'   Arguments of `custom_family()`, which metabeta does not support.
#' @return An object of class `c("brmsfamily", "family")`.
#' @examples
#' brmsfamily("gaussian")
#' bernoulli()
#' @export
brmsfamily <- function(family, link = NULL, link_sigma = "log", link_shape = "log",
                       link_nu = "logm1", link_phi = "log", link_kappa = "log",
                       link_beta = "log", link_zi = "logit", link_hu = "logit",
                       link_zoi = "logit", link_coi = "logit", link_disc = "log",
                       link_bs = "log", link_ndt = "log", link_bias = "logit",
                       link_xi = "log1p", link_alpha = "identity",
                       link_quantile = "logit", threshold = "flexible", refcat = NULL) {
  .brmsfamily(family, link = link, slink = substitute(link))
}

# Link names brms accepts unquoted, as in bernoulli(link = logit).
LINK_NAMES <- unique(c(
  DEFAULT_LINKS, "probit", "probit_approx", "cloglog", "cauchit", "softplus", "squareplus",
  "sqrt", "softit", "log1p", "logm1"
))

# The family object. `slink` is the unevaluated link argument: a bare link name is
# taken as that name, as in brms; anything else is evaluated.
.brmsfamily <- function(family, link = NULL, slink = NULL) {
  if (is.symbol(slink) && as.character(slink) %in% LINK_NAMES) link <- as.character(slink)
  if (!family %in% names(DEFAULT_LINKS)) family <- tolower(family)
  if (!family %in% names(DEFAULT_LINKS)) stop2("Family '", family, "' is not a valid brms family.")
  link <- link %||% DEFAULT_LINKS[[family]]
  # linkfun/linkinv from stats where the link exists there (e.g. not "tan_half")
  link_functions <- tryCatch(stats::make.link(link), error = function(e) list())
  structure(
    list(
      family = family,
      link = link,
      linkfun = link_functions$linkfun,
      linkinv = link_functions$linkinv,
      dpars = if (family == "gaussian") c("mu", "sigma") else "mu"
    ),
    class = c("brmsfamily", "family")
  )
}

#' @rdname brmsfamily
#' @export
bernoulli <- function(link = "logit") .brmsfamily("bernoulli", link = link, slink = substitute(link))

#' @rdname brmsfamily
#' @export
mixture <- function(..., flist = NULL, nmix = 1, order = NULL) {
  stop2("Mixture models are not supported by metabeta. Use brms::brm() instead.")
}

#' @rdname brmsfamily
#' @export
custom_family <- function(name, dpars = "mu", links = "identity", type = c("real", "int"),
                          lb = NA, ub = NA, vars = NULL, loop = TRUE, specials = NULL,
                          threshold = "flexible", log_lik = NULL, posterior_predict = NULL,
                          posterior_epred = NULL, predict = NULL, fitted = NULL,
                          env = parent.frame()) {
  stop2("Custom families are not supported by metabeta. Use brms::brm() instead.")
}

#' @export
print.brmsfamily <- function(x, links = FALSE, newline = TRUE, ...) {
  cat("\nFamily:", x$family, "\nLink function:", x$link, "\n")
  if (newline) cat("\n")
  invisible(x)
}

# Turn any accepted 'family' argument (object, function or string) into a brmsfamily.
validate_family <- function(family) {
  if (is.function(family)) family <- family()
  if (is.character(family)) family <- brmsfamily(family)
  if (!inherits(family, "family")) stop2("Argument 'family' is invalid.")
  if (!inherits(family, "brmsfamily")) family <- brmsfamily(family$family, link = family$link)
  family
}

# The likelihood metabeta uses for a family: "gaussian", "bernoulli" or "poisson".
family_likelihood <- function(family) {
  name <- family$family
  if (!name %in% names(SUPPORTED_LINKS)) {
    stop2(
      "Family '", name, "' is not supported by metabeta, which was trained on gaussian, ",
      "bernoulli (or binomial with one trial) and poisson models. Use brms::brm() to fit this model."
    )
  }
  if (family$link != SUPPORTED_LINKS[[name]]) {
    stop2("metabeta supports only the '", SUPPORTED_LINKS[[name]], "' link for family '", name, "' (got '", family$link, "').")
  }
  if (name == "binomial") "bernoulli" else name
}
