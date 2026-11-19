# Bridge to the metabeta Python package.
#
# R does all brms-specific work. Python only receives a design that is already on the
# network's standardized scale, as a "model dataset" of the public `Api.sample()`, and
# returns draws on that scale, so a fitted model holds plain R arrays and can be saved
# with saveRDS().

.metabeta <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Declare the Python dependency, so that reticulate installs Python and metabeta by
  # itself the first time they are needed. This has no effect when the user chose an
  # interpreter (RETICULATE_PYTHON, use_python(), ...).
  requirement <- getOption("metabeta.python_requirement", "metabeta>=0.6.0")
  try(reticulate::py_require(requirement, python_version = ">=3.12"), silent = TRUE)
}

#' Set up metabeta's Python environment and networks
#'
#' Not required: [brm()] does all of this the first time it is needed. Call
#' `install_metabeta()` to do it upfront, e.g. before working offline. It
#' installs Python and the `metabeta` Python package (through \pkg{reticulate},
#' into a cached environment), downloads the pretrained networks (about 135 MB
#' per family) and checks that they load.
#'
#' @param families The families whose networks to download.
#' @return `TRUE`, invisibly.
#' @examples
#' \dontrun{
#' install_metabeta()
#' }
#' @export
install_metabeta <- function(families = c("gaussian", "bernoulli", "poisson")) {
  for (family in match.arg(families, several.ok = TRUE)) network(family)
  message("metabeta is ready (Python ", reticulate::py_config()$version, ").")
  invisible(TRUE)
}

# Run `code` with Python warnings switched off (R reports problems itself).
without_python_warnings <- function(code) {
  warnings <- reticulate::import("warnings")
  context <- warnings$catch_warnings()
  context$`__enter__`()
  on.exit(context$`__exit__`(NULL, NULL, NULL))
  warnings$simplefilter("ignore")
  code
}

# The pretrained network for a likelihood, loaded once per session.
network <- function(likelihood) {
  if (is.null(.metabeta[[likelihood]])) {
    message(
      "Loading metabeta's ", likelihood, " network. The first time, this installs Python ",
      "packages and downloads the network (about 135 MB), which takes a few minutes."
    )
    api <- import_api()
    .metabeta[[likelihood]] <- without_python_warnings(api$Api$from_pretrained(
      likelihood,
      device = getOption("metabeta.device", "cpu"),
      warmup = FALSE,
      verbose = isTRUE(getOption("metabeta.verbose", FALSE))
    ))
  }
  .metabeta[[likelihood]]
}

import_api <- function() {
  imported <- tryCatch(import_installed("metabeta.models.api"), error = function(e) {
    if (grepl("No module named 'metabeta'|PackageNotFoundError", conditionMessage(e))) {
      stop2(
        "The 'metabeta' Python package is not available to reticulate. Install it into the ",
        "Python environment reticulate uses (see reticulate::py_config()): pip install metabeta"
      )
    }
    if (grepl("WinError 127|DLL load failed", conditionMessage(e))) {
      stop2(
        "torch could not be loaded because this R session already holds a different version ",
        "of a library torch needs. This happens on Windows when R and Python share a conda ",
        "environment. Restart R and call brm() before other numerical code (e.g. solve()), ",
        "or use a Python environment outside of R's conda environment.\n\n", conditionMessage(e)
      )
    }
    stop(e)
  })
  api <- imported$module
  version <- imported$version
  if (utils::compareVersion(version, "0.6.0") < 0) {
    stop2(
      "The 'metabeta' Python package in ", reticulate::py_config()$python, " is version ",
      version, "; the R interface needs version 0.6.0 or later: pip install -U metabeta"
    )
  }
  api
}

# A module of the installed metabeta package, with the package's version. Python looks
# for modules in the working directory first, where a folder named 'metabeta' (e.g. a
# clone of its repository) would hide the installed package, so the import leaves the
# working directory out. Later imports of its submodules follow the package's location.
import_installed <- function(module) {
  helper <- reticulate::py_run_string(local = TRUE, convert = FALSE, "
def import_installed(name):
    import importlib, importlib.metadata, os, sys
    here = os.path.normcase(os.path.abspath(os.getcwd()))
    saved = list(sys.path)
    sys.path[:] = [p for p in saved if os.path.normcase(os.path.abspath(p or '.')) != here]
    try:
        importlib.import_module(name)
        return importlib.metadata.version('metabeta')
    finally:
        sys.path[:] = saved
")
  version <- reticulate::py_to_r(helper$import_installed(module))
  # now in sys.modules, so this returns the installed module
  list(module = reticulate::import(module), version = version)
}

# metabeta runs a fixed number of independent Metropolis-Hastings chains, the module
# constant IMH_N_CHAINS; brm()'s `chains` sets it for the duration of one call.
with_imh_chains <- function(chains, code) {
  module <- reticulate::import("metabeta.models.api")
  old <- module$IMH_N_CHAINS
  reticulate::py_set_attr(module, "IMH_N_CHAINS", as.integer(chains))
  on.exit(reticulate::py_set_attr(module, "IMH_N_CHAINS", old))
  code
}

# The design as a model dataset of metabeta's Api, padded to the dimensions of the
# submodel it is routed to. Rows are ordered by group, as the collator expects.
model_dataset <- function(api, design, hyper, likelihood) {
  np <- reticulate::import("numpy", convert = FALSE)
  utils <- reticulate::import("metabeta.utils.api", convert = FALSE)
  rows <- order(design$groups)
  d <- ncol(design$X)
  q <- ncol(design$Z)
  m <- max(design$groups) + 1L
  # reticulate turns vectors of length one into scalars; these must stay 1D arrays
  for (key in c("nu_ffx", "tau_ffx", "tau_rfx")) hyper[[key]] <- as.array(unname(hyper[[key]]))
  priors <- utils$coercePriors(hyper, d = d, q = q, likelihood_family = LIKELIHOOD_CODES[[likelihood]])
  dataset <- reticulate::dict(
    X = np$asarray(design$X[rows, , drop = FALSE], dtype = "float64"),
    Z = np$asarray(design$Z[rows, , drop = FALSE], dtype = "float64"),
    y = np$asarray(as.array(design$y[rows]), dtype = "float64"),
    groups = np$asarray(as.array(as.integer(design$groups[rows])), dtype = "int64"),
    d = np$array(d), q = np$array(q), n = np$array(length(rows)), m = np$array(m),
    ns = np$asarray(as.array(tabulate(design$groups + 1L, m)), dtype = "int64"),
    sd_y = np$array(1),
    # placeholders for the generating parameters, which a real dataset does not have
    ffx = np$zeros(d), sigma_rfx = np$ones(q), corr_rfx = np$eye(q), rfx = np$zeros(c(m, q))
  )
  if (likelihood == "gaussian") dataset["sigma_eps"] <- np$array(1)
  dataset$update(priors)
  model <- api$model(api$route(dataset)[[1]])
  utils$padModelDataset(dataset, max_d = model$d_ffx, max_q = model$d_rfx)
}

LIKELIHOOD_CODES <- c(gaussian = 0L, bernoulli = 1L, poisson = 2L)

# Sample the posterior of a design on the network's scale.
#
# `design` holds X (n x d, intercept first), Z (n x q), y and 0-based group codes;
# `hyper` holds the prior hyperparameters (see model_hyper()). Returns a list with
# ffx (S x d), sigma_rfx (S x q), corr_rfx (S x q x q), rfx (m x S x q), sigma_eps (S,
# gaussian only), method ("imh" or "flow"), burnin, safeguards and, with
# `evidence = TRUE`, evidence (log marginal likelihood of the standardized outcomes).
sample_network <- function(design, hyper, likelihood, n_samples, chains, refine, seed,
                           evidence = FALSE, random_walk = FALSE) {
  # Tests can replace the network with an R function of the same arguments.
  sampler <- getOption("metabeta.sampler")
  if (is.function(sampler)) {
    return(sampler(design, hyper, likelihood, n_samples, chains, refine, seed, evidence, random_walk))
  }
  api <- network(likelihood)
  result <- without_python_warnings(with_imh_chains(chains, api$sample(
    model_dataset(api, design, hyper, likelihood),
    n_samples = as.integer(n_samples),
    refine = refine,
    random_walk = random_walk && refine,
    evidence = evidence,
    seed = as.integer(seed)
  )))
  d <- ncol(design$X)
  q <- ncol(design$Z)
  m <- max(design$groups) + 1L
  draws <- function(tensor) tensor$detach()$double()$cpu()$numpy()
  p <- result$proposal
  sg <- result$safeguards
  out <- list(
    ffx = matrix(draws(p$ffx)[1, , seq_len(d)], nrow = n_samples),
    sigma_rfx = matrix(draws(p$sigma_rfx)[1, , seq_len(q)], nrow = n_samples),
    rfx = array(draws(p$rfx)[1, seq_len(m), , seq_len(q)], dim = c(m, n_samples, q)),
    method = if (!is.null(sg$accept_rate)) "imh" else "flow",
    burnin = reticulate::import("metabeta.models.api")$IMH_BURNIN,
    safeguards = first_values(sg[setdiff(names(sg), "ess")])
  )
  # eta_rfx = 0 encodes uncorrelated effects, whose correlation dimensions carry no posterior
  out$corr_rfx <- if (q > 1 && hyper$eta_rfx > 0) {
    array(draws(p$corr_rfx)[1, , seq_len(q), seq_len(q)], dim = c(n_samples, q, q))
  } else {
    array(rep(diag(q), each = n_samples), dim = c(n_samples, q, q))
  }
  if (likelihood == "gaussian") out$sigma_eps <- as.vector(draws(p$sigma_eps)[1, ])
  if (evidence) out$evidence <- first_values(result$evidence)
  out
}

# The single dataset's value of each per-dataset entry (scalars stay as they are).
first_values <- function(x) {
  lapply(Filter(Negate(is.null), x), function(v) if (is.character(v)) v else as.vector(v)[1])
}

# Turn the safeguards Python computed (and whose warnings were silenced) into R warnings.
report_safeguards <- function(out, n_samples) {
  sg <- out$safeguards
  moved <- isTRUE(sg$random_walk_ran)
  if (out$method == "imh" && isTRUE(sg$accept_flag) && !moved) {
    warning2(
      sprintf("The Metropolis-Hastings acceptance rate is low (%.2f): ", sg$accept_rate),
      "the network's proposal fits this posterior poorly and the draws may be under-dispersed. ",
      sprintf("Consider at least %d post-warmup draws ", as.integer(sg$suggested_n_samples)),
      "(increase 'iter' or 'chains'), or fit the model with brms instead."
    )
  }
  if (out$method == "imh" && isTRUE(sg$ess_flag) && !moved) {
    warning2(
      sprintf("The Metropolis-Hastings chains repeat few draws (lowest ESS %.0f), ", sg$ess_min),
      "so the intervals may be too narrow. Consider more draws (increase 'iter'), ",
      "control = list(random_walk = TRUE), or fitting the model with brms instead."
    )
  }
  if (moved && isTRUE(sg$ess_min < 10)) {
    warning2(
      sprintf("The Metropolis-Hastings chains had very few distinct draws (lowest ESS %.0f) ", sg$ess_min),
      "before the random-walk moves, which may not have closed the gap; the intervals may still ",
      "be too narrow. Check the results with brms."
    )
  }
  if (isTRUE(sg$map_z_flag)) {
    warning2(
      sprintf("The posterior mean is far from the analytical point estimates (z = %.1f). ", sg$map_z),
      "The data may lie outside the range metabeta was trained on; treat the results with ",
      "caution or fit the model with brms instead."
    )
  }
  if (out$method == "flow" && n_samples < 64) {
    message(
      "Fewer than 64 draws were requested, so the network posterior was returned without ",
      "Metropolis-Hastings refinement."
    )
  }
}

# Translate the router's "outside the checkpoint's limits" error into a readable message.
explain_python_error <- function(e) {
  msg <- conditionMessage(e)
  marker <- "outside the checkpoint's limits: "
  if (!grepl(marker, msg, fixed = TRUE)) stop2("metabeta failed to sample: ", msg)
  misfit <- sub("[\r\n].*$", "", sub(paste0(".*", marker), "", msg))
  stop2(
    "The data lie outside the range metabeta's networks were trained on: ", misfit,
    "\n(d: population-level coefficients, q: group-level coefficients, m: groups, ",
    "n_i: observations per group). Use brms::brm() for this model."
  )
}
