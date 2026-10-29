# Shared fixtures. The mock sampler stands in for the Python network so that the R side
# (design, priors, transformations, methods) is tested without Python; it records its
# inputs and outputs so tests can check the transformations exactly.

mock_env <- new.env()

mock_sampler <- function(design, hyper, likelihood, n_samples, chains, refine, seed,
                         evidence = FALSE, random_walk = FALSE) {
  set.seed(seed)
  S <- n_samples
  d <- ncol(design$X)
  q <- ncol(design$Z)
  m <- length(unique(design$groups))
  cor <- array(0, dim = c(S, q, q))
  for (s in seq_len(S)) {
    if (q > 1 && hyper$eta_rfx > 0) {
      A <- matrix(rnorm(q * q), q)
      R <- stats::cov2cor(crossprod(A) + diag(q))
    } else {
      R <- diag(q)
    }
    cor[s, , ] <- R
  }
  out <- list(
    ffx = matrix(rnorm(S * d, 0, 0.5), S, d),
    sigma_rfx = matrix(abs(rnorm(S * q, 0.5, 0.1)), S, q),
    corr_rfx = cor,
    rfx = array(rnorm(m * S * q, 0, 0.5), dim = c(m, S, q)),
    method = if (refine && n_samples >= 64) "imh" else "flow",
    burnin = 25L,
    safeguards = list(accept_rate = 0.5, accept_threshold = 0.1, accept_flag = FALSE,
      suggested_n_samples = 1000, ess_min = 500, ess_threshold = 150, ess_flag = FALSE,
      random_walk_ran = random_walk, psis_k = 0.2, map_z = 1, map_z_threshold = 5,
      map_z_flag = FALSE)
  )
  if (likelihood == "gaussian") out$sigma_eps <- abs(rnorm(S, 0.5, 0.1))
  if (evidence) out$evidence <- list(log_evidence = -100 - d - q, psis_k = 0.3, pool_psis_k = 0.2)
  mock_env$calls <- c(mock_env$calls, list(list(seed = seed, chains = chains, evidence = evidence, random_walk = random_walk)))
  mock_env$design <- design
  mock_env$hyper <- hyper
  mock_env$out <- out
  out
}

with_sampler <- function(sampler, code) {
  old <- options(metabeta.sampler = sampler)
  on.exit(options(old))
  force(code)
}

with_mock <- function(code) with_sampler(mock_sampler, code)

# a sampler that fails the test if the network is consulted
no_network <- function(...) stop("the network must not be called")

# grouped data with continuous, binary and factor predictors and every response type
sim_data <- function(m = 12, n_i = 8, seed = 1) {
  set.seed(seed)
  n <- m * n_i
  g <- factor(rep(sprintf("s%02d", seq_len(m)), each = n_i))
  x <- rnorm(n, 50, 10)
  z <- rbinom(n, 1, 0.4)
  f <- factor(sample(c("a", "b c", "d-e"), n, replace = TRUE))
  u0 <- rnorm(m, 0, 2)[as.integer(g)]
  u1 <- rnorm(m, 0, 0.1)[as.integer(g)]
  eta <- 1 + 0.05 * (x - 50) + 0.5 * z + u0 / 4 + u1 * (x - 50) / 10
  data.frame(
    y = 100 + 10 * eta + rnorm(n, 0, 3),
    yb = rbinom(n, 1, plogis(eta - 1)),
    yc = rpois(n, exp(eta / 2)),
    x = x, z = z, f = f, g = g
  )
}

python_available <- local({
  # Only when the Python tests may run (not on CRAN-like checks, where probing would make
  # reticulate install Python packages). The import happens before any other test touches
  # LAPACK: with R and torch in one conda environment on Windows, whichever loads its
  # OpenMP runtime first wins.
  available <- identical(Sys.getenv("NOT_CRAN"), "true") && tryCatch(
    !is.null(import_api()),
    error = function(e) FALSE
  )
  function() is.null(getOption("metabeta.sampler")) && isTRUE(available)
})

skip_if_no_python <- function() {
  testthat::skip_on_cran()
  if (!isTRUE(tryCatch(python_available(), error = function(e) FALSE))) {
    testthat::skip("metabeta Python package (>= 0.6.0) is not available")
  }
}
