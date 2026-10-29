# The reparametrization between the brms scale and the network's model scale must be
# exact: the linear predictor implied by the back-transformed draws has to equal the
# model-scale linear predictor mapped back to the data scale, draw by draw.

model_scale_eta <- function() {
  d <- mock_env$design
  out <- mock_env$out
  S <- nrow(out$ffx)
  eta <- tcrossprod(out$ffx, d$X)
  for (k in seq_len(ncol(d$Z))) {
    rk <- t(out$rfx[d$groups + 1L, , k])
    eta <- eta + sweep(rk, 2, d$Z[, k], "*")
  }
  eta
}

expect_exact_transform <- function(fit) {
  design <- fit$metabeta$design
  eta_raw <- posterior_linpred(fit)
  eta_model <- model_scale_eta() * design$y_sd + design$y_mean
  expect_equal(eta_raw, eta_model, tolerance = 1e-8, ignore_attr = TRUE)
}

fit_mock <- function(formula, family = gaussian(), data = sim_data(), ...) {
  with_mock(brm(formula, data = data, family = family, chains = 2, iter = 60, warmup = 20, seed = 3, ...))
}

test_that("draws map back exactly for correlated slopes on uncentered predictors", {
  fit <- fit_mock(y ~ x + z + f + x:z + (1 + x | g))
  expect_true(fit$metabeta$design$z_centered)
  expect_exact_transform(fit)
})

test_that("draws map back exactly for uncorrelated slopes", {
  fit <- fit_mock(y ~ x + (1 + x || g))
  expect_false(fit$metabeta$design$z_centered)
  expect_exact_transform(fit)
})

test_that("draws map back exactly without intercepts", {
  fit <- fit_mock(y ~ 0 + x + (0 + x | g))
  expect_identical(fit$metabeta$design$int_type, "none")
  expect_exact_transform(fit)
})

test_that("draws map back exactly with center = FALSE and with 0 + Intercept", {
  fit <- fit_mock(bf(y ~ x + (1 | g), center = FALSE))
  expect_identical(fit$metabeta$design$int_type, "plain")
  expect_exact_transform(fit)
  fit <- fit_mock(y ~ 0 + Intercept + x + (1 + x | g))
  expect_identical(fit$metabeta$design$int_type, "plain")
  expect_exact_transform(fit)
})

test_that("draws map back exactly for bernoulli and poisson models", {
  fit <- fit_mock(yb ~ x + f + (1 + z | g), family = bernoulli())
  expect_exact_transform(fit)
  fit <- fit_mock(yc ~ x + (1 | g), family = poisson())
  expect_exact_transform(fit)
})

test_that("standard deviations and correlations equal those of B Sigma_c B'", {
  fit <- fit_mock(y ~ x + (1 + x | g))
  design <- fit$metabeta$design
  out <- mock_env$out
  s <- 7
  sd_c <- out$sigma_rfx[s, ] * design$y_sd / design$z_scales
  Sigma <- design$B %*% (outer(sd_c, sd_c) * out$corr_rfx[s, , ]) %*% t(design$B)
  draws <- as.matrix(fit)
  expect_equal(unname(draws[s, c("sd_g__Intercept", "sd_g__x")]), unname(sqrt(diag(Sigma))))
  expect_equal(unname(draws[s, "cor_g__Intercept__x"]), Sigma[1, 2] / sqrt(prod(diag(Sigma))))
})

test_that("the centered intercept is the intercept at the predictor means", {
  fit <- fit_mock(y ~ x + z + (1 | g))
  draws <- as.matrix(fit)
  design <- fit$metabeta$design
  means <- design$x_means[c("x", "z")]
  expect_equal(
    unname(draws[, "Intercept"]),
    drop(draws[, "b_Intercept"] + draws[, c("b_x", "b_z")] %*% means)
  )
})

test_that("priors map exactly to the model scale", {
  data <- sim_data()
  pr <- prior(normal(0, 10), class = b) +
    prior(normal(150, 20), class = Intercept) +
    prior(normal(0, 5), class = sd, coef = x, group = g) +
    prior(exponential(0.5), class = sigma)
  fit <- with_mock(brm(y ~ x + (1 + x | g), data = data, prior = pr, chains = 1, iter = 80, warmup = 10, seed = 1))
  design <- fit$metabeta$design
  hyper <- fit$metabeta$hyper
  sx <- sqrt(mean((data$x - mean(data$x))^2))
  sy <- sqrt(mean((data$y - mean(data$y))^2))
  ix <- match("x", design$b_names[design$perm])
  i0 <- match("Intercept", design$b_names[design$perm])
  expect_equal(hyper$tau_ffx[ix], 10 * sx / sy)
  expect_equal(hyper$nu_ffx[ix], 0)
  expect_equal(hyper$tau_ffx[i0], 20 / sy)
  expect_equal(hyper$nu_ffx[i0], (150 - mean(data$y)) / sy)
  expect_equal(hyper$tau_rfx[match("x", design$re_names)], 5 * sx / sy)
  expect_equal(hyper$tau_eps, 2 / sy)
  expect_identical(hyper$family_sigma_eps, 2L)
})

test_that("network columns put group-level predictors first", {
  fit <- fit_mock(y ~ z + f + x + (1 + x | g))
  design <- fit$metabeta$design
  expect_identical(design$b_names[design$perm][1:2], c("Intercept", "x"))
  # the network was trained with Z = X[, 1:q]
  expect_equal(mock_env$design$Z, mock_env$design$X[, 1:2], ignore_attr = TRUE)
})
