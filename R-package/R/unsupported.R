# What metabeta cannot do: functions that need a Stan model raise an error, and so do the
# generated functions in R/generated.R. standata() is the exception: it returns the data.

no_stan <- function(what) {
  stop2(what, " is not available for metabeta models: they are fitted by a pretrained network instead of a Stan program. Use brms::brm() if you need it.")
}

# Called by the functions in R/generated.R.
unsupported <- function(name, reason) {
  message <- switch(reason,
    term = paste0("Terms of type '", name, "()' are not supported by metabeta"),
    gr = "gr() is only meaningful inside a model formula, e.g. '(1 | gr(g, cor = FALSE))'",
    autocor = "Autocorrelation structures are not supported by metabeta",
    formula = "Multivariate, non-linear and measurement-error formulas are not supported by metabeta",
    distribution = paste0("The distribution functions of families metabeta cannot fit, such as '", name, "()', are not included in metabeta"),
    internal = paste0("'", name, "()' is part of the internals of brms and not available in metabeta")
  )
  stop2(message, ". Use brms for it.")
}

#' Parts of brms that metabeta does not provide
#'
#' These functions exist with the arguments of \pkg{brms} so that scripts written
#' for \pkg{brms} fail with an explanation: formula terms such as `s()`, `gp()`
#' or `mo()`, autocorrelation structures, multivariate and non-linear formulas,
#' distribution functions of families metabeta cannot fit, and internal helpers
#' of \pkg{brms}.
#'
#' @return None; these functions raise an error.
#' @name unsupported-brms
#' @keywords internal
NULL

#' Stan code and data of a model
#'
#' metabeta models have no Stan program, so `stancode()` raises an error.
#' `standata()` returns the data in the format of \pkg{brms} (`N`, `Y`, `K`,
#' `X`, `Z_1_1`, `J_1`, ...).
#'
#' @param object,formula A model formula or a fitted model.
#' @param data,family,prior,... Same as in [brm()].
#' @param newdata,re_formula,newdata2,new_objects,incl_autocor Accepted for
#'   compatibility; `newdata` is not supported.
#' @return For `standata()`, a named list.
#' @export
stancode <- function(object, ...) UseMethod("stancode")

#' @rdname stancode
#' @export
stancode.default <- function(object, ...) no_stan("Stan code")

#' @rdname stancode
#' @export
make_stancode <- function(formula, ...) no_stan("Stan code")

#' @rdname stancode
#' @export
standata <- function(object, ...) UseMethod("standata")

#' @rdname stancode
#' @export
standata.default <- function(object, data, family = gaussian(), prior = NULL, ...) {
  stan_data(prepare_model(object, data, family)$design)
}

#' @rdname stancode
#' @export
standata.metabetafit <- function(object, newdata = NULL, re_formula = NULL, newdata2 = NULL,
                                 new_objects = NULL, incl_autocor = TRUE, ...) {
  if (!is.null(newdata)) stop2("standata() with 'newdata' is not supported by metabeta.")
  stan_data(object$metabeta$design)
}

#' @rdname stancode
#' @export
make_standata <- function(formula, ...) standata(formula, ...)

stan_data <- function(design) {
  out <- list(N = length(design$y), Y = design$y, K = ncol(design$X), X = design$X)
  for (k in seq_along(design$re_names)) out[[paste0("Z_1_", k)]] <- design$Z[, k]
  q <- length(design$re_names)
  c(out, list(
    J_1 = design$observed[design$groups + 1], N_1 = length(design$levels), M_1 = q,
    NC_1 = if (design$correlated) choose(q, 2) else 0, prior_only = 0L
  ))
}

#' Functions that need a Stan model
#'
#' Not available for metabeta models; each raises an error that says so.
#'
#' @param x A fitted model.
#' @param ... Ignored.
#' @param recompile,formula,data,family,prior,data2,autocor,cov_ranef,sample_prior,sparse,knots,stanvars,stan_funs,silent,combine,fit,algorithm,seed,file,file_compress,file_refit
#'   Arguments of the \pkg{brms} functions, accepted for compatibility.
#' @return None; these functions raise an error.
#' @name stan-only
NULL

#' @rdname stan-only
#' @export
expose_functions <- function(x, ...) no_stan("Exposing Stan functions")

#' @rdname stan-only
#' @export
control_params <- function(x, ...) no_stan("Sampler control parameters")

#' @rdname stan-only
#' @export
recompile_model <- function(x, recompile = NULL) no_stan("Recompiling the Stan model")

#' @rdname stan-only
#' @export
loo_moment_match <- function(x, ...) no_stan("Moment matching")

#' @rdname stan-only
#' @export
loo_subsample <- function(x, ...) stop2("loo_subsample() is not supported by metabeta; loo() is fast enough for its data sizes.")

#' @rdname stan-only
#' @export
brm_multiple <- function(formula, data, family = gaussian(), prior = NULL, data2 = NULL,
                         autocor = NULL, cov_ranef = NULL, sample_prior = c("no", "yes", "only"),
                         sparse = NULL, knots = NULL, stanvars = NULL, stan_funs = NULL, silent = 1,
                         recompile = FALSE, combine = TRUE, fit = NA,
                         algorithm = getOption("brms.algorithm", "sampling"), seed = NA,
                         file = NULL, file_compress = TRUE,
                         file_refit = getOption("brms.file_refit", "never"), ...) {
  stop2("brm_multiple() is not supported by metabeta. Fit the models with brm() and combine them with combine_models().")
}

#' @export
nuts_params.metabetafit <- function(object, pars = NULL, ...) no_stan("NUTS diagnostics")

#' @export
log_posterior.metabetafit <- function(object, ...) no_stan("The Stan log posterior")
