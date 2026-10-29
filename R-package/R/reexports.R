# Generics owned by other packages, re-exported so that (as with brms) they are
# available after library(metabeta) without loading their home packages.

#' @importFrom nlme fixef
#' @export
nlme::fixef

#' @importFrom nlme ranef
#' @export
nlme::ranef

#' @importFrom nlme VarCorr
#' @export
nlme::VarCorr

#' @importFrom rstantools posterior_predict
#' @export
rstantools::posterior_predict

#' @importFrom rstantools posterior_epred
#' @export
rstantools::posterior_epred

#' @importFrom rstantools posterior_linpred
#' @export
rstantools::posterior_linpred

#' @importFrom rstantools log_lik
#' @export
rstantools::log_lik

#' @importFrom rstantools bayes_R2
#' @export
rstantools::bayes_R2

#' @importFrom rstantools loo_R2
#' @export
rstantools::loo_R2

#' @importFrom rstantools prior_summary
#' @export
rstantools::prior_summary

#' @importFrom rstantools posterior_interval
#' @export
rstantools::posterior_interval

#' @importFrom rstantools predictive_interval
#' @export
rstantools::predictive_interval

#' @importFrom rstantools predictive_error
#' @export
rstantools::predictive_error

#' @importFrom rstantools nsamples
#' @export
rstantools::nsamples

#' @importFrom loo loo
#' @export
loo::loo

#' @importFrom loo waic
#' @export
loo::waic

#' @importFrom loo loo_compare
#' @export
loo::loo_compare

#' @importFrom loo kfold
#' @export
loo::kfold

#' @importFrom loo loo_model_weights
#' @export
loo::loo_model_weights

#' @importFrom loo psis
#' @export
loo::psis

#' @importFrom bayesplot theme_default
#' @export
bayesplot::theme_default

#' @importFrom posterior as_draws
#' @export
posterior::as_draws

#' @importFrom posterior as_draws_df
#' @export
posterior::as_draws_df

#' @importFrom posterior as_draws_array
#' @export
posterior::as_draws_array

#' @importFrom posterior as_draws_matrix
#' @export
posterior::as_draws_matrix

#' @importFrom posterior as_draws_list
#' @export
posterior::as_draws_list

#' @importFrom posterior as_draws_rvars
#' @export
posterior::as_draws_rvars

#' @importFrom posterior ndraws
#' @export
posterior::ndraws

#' @importFrom posterior nchains
#' @export
posterior::nchains

#' @importFrom posterior niterations
#' @export
posterior::niterations

#' @importFrom posterior nvariables
#' @export
posterior::nvariables

#' @importFrom posterior variables
#' @export
posterior::variables

#' @importFrom bayesplot pp_check
#' @export
bayesplot::pp_check

#' @importFrom posterior rhat
#' @export
posterior::rhat

#' @importFrom bayesplot neff_ratio
#' @export
bayesplot::neff_ratio

#' @importFrom bayesplot nuts_params
#' @export
bayesplot::nuts_params

#' @importFrom bayesplot log_posterior
#' @export
bayesplot::log_posterior
