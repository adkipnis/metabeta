fit_once <- local({
  fit <- NULL
  function() {
    if (is.null(fit)) {
      fit <<- with_mock(brm(
        y ~ x + z + f + (1 + x | g), data = sim_data(), chains = 2, iter = 150, warmup = 50,
        seed = 1, sample_prior = "yes"
      ))
    }
    fit
  }
})

test_that("fits have the structure of brmsfit objects", {
  fit <- fit_once()
  expect_s3_class(fit, c("metabetafit", "brmsfit"))
  expect_true(is.brmsfit(fit))
  expect_true(all(c("formula", "data", "prior", "fit", "family", "ranef", "criteria") %in% names(fit)))
  expect_identical(fit$data.name, "sim_data()")
  expect_identical(ndraws(fit), 200L)
  expect_identical(nchains(fit), 2L)
  expect_identical(niterations(fit), 100L)
  expect_identical(nobs(fit), 96L)
  expect_identical(ngrps(fit), list(g = 12L))
  expect_identical(fit$ranef$coef, c("Intercept", "x"))
})

test_that("draws carry brms variable names", {
  vars <- variables(fit_once())
  expect_identical(
    vars[1:10],
    c("b_Intercept", "b_x", "b_z", "b_fbc", "b_fdMe", "sd_g__Intercept", "sd_g__x",
      "cor_g__Intercept__x", "sigma", "Intercept")
  )
  expect_true(all(c("r_g[s01,Intercept]", "r_g[s12,x]", "lprior", "prior_Intercept", "prior_sigma") %in% vars))
  expect_identical(which(vars == "r_g[s01,Intercept]") + 12L, which(vars == "r_g[s01,x]"))
  df <- as_draws_df(fit_once(), variable = "b_x")
  expect_s3_class(df, "draws_df")
  expect_identical(dim(as.matrix(fit_once(), variable = c("b_x", "sigma"))), c(200L, 2L))
  expect_identical(dim(as.array(fit_once(), variable = "b_x")), c(100L, 2L, 1L))
  expect_identical(names(posterior_samples(fit_once(), pars = "^b_")), vars[1:5])
})

test_that("summary() reproduces the brms layout", {
  out <- capture.output(print(fit_once()))
  expect_identical(out[1], " Family: gaussian ")
  expect_identical(out[2], "  Links: mu = identity; sigma = identity ")
  expect_identical(out[3], "Formula: y ~ x + z + f + (1 + x | g) ")
  expect_identical(out[4], "   Data: sim_data() (Number of observations: 96) ")
  expect_identical(out[5], "  Draws: 2 chains, each with iter = 125; warmup = 25; thin = 1;")
  expect_true(all(c("Multilevel Hyperparameters:", "~g (Number of levels: 12) ",
    "Regression Coefficients:", "Further Distributional Parameters:") %in% out))
  expect_match(out[grep("^sd\\(Intercept\\)", out)], "^sd\\(Intercept\\) +-?[0-9]")
  expect_true(any(grepl("^cor\\(Intercept,x\\)", out)))
  s <- summary(fit_once(), prob = 0.9, robust = TRUE)
  expect_identical(colnames(s$fixed), c("Estimate", "Est.Error", "l-90% CI", "u-90% CI", "Rhat", "Bulk_ESS", "Tail_ESS"))
  expect_output(print(summary(fit_once(), priors = TRUE)), "Priors:")
})

test_that("fixef, ranef, coef and VarCorr return brms shapes", {
  fit <- fit_once()
  fe <- fixef(fit)
  expect_identical(dimnames(fe), list(c("Intercept", "x", "z", "fbc", "fdMe"), c("Estimate", "Est.Error", "Q2.5", "Q97.5")))
  expect_identical(dim(fixef(fit, summary = FALSE)), c(200L, 5L))
  re <- ranef(fit)
  expect_identical(names(re), "g")
  expect_identical(dim(re$g), c(12L, 4L, 2L))
  expect_identical(dimnames(re$g)[[3]], c("Intercept", "x"))
  expect_identical(dim(ranef(fit, summary = FALSE)$g), c(200L, 12L, 2L))
  co <- coef(fit, summary = FALSE)$g
  expect_identical(dimnames(co)[[3]], c("Intercept", "x", "z", "fbc", "fdMe"))
  expect_equal(co[, 3, "x"], fixef(fit, summary = FALSE)[, "x"] + ranef(fit, summary = FALSE)$g[, 3, "x"])
  expect_equal(co[, 3, "z"], fixef(fit, summary = FALSE)[, "z"])
  vc <- VarCorr(fit)
  expect_identical(names(vc), c("g", "residual__"))
  expect_identical(names(vc$g), c("sd", "cor", "cov"))
  expect_identical(dim(vc$g$cor), c(2L, 4L, 2L))
  expect_identical(rownames(vc$residual__$sd), "y")
})

test_that("predictions have brms shapes and honour re_formula and new levels", {
  fit <- fit_once()
  expect_identical(dim(posterior_predict(fit)), c(200L, 96L))
  expect_identical(dim(posterior_epred(fit, ndraws = 20)), c(20L, 96L))
  expect_identical(colnames(predict(fit)), c("Estimate", "Est.Error", "Q2.5", "Q97.5"))
  expect_identical(dim(fitted(fit)), c(96L, 4L))
  expect_identical(dim(residuals(fit)), c(96L, 4L))
  expect_identical(dim(log_lik(fit)), c(200L, 96L))
  nd <- data.frame(x = c(40, 60), z = 0, f = "a", g = c("s01", "new"))
  expect_error(posterior_epred(fit, newdata = nd), "allow_new_levels")
  ep <- posterior_epred(fit, newdata = nd, allow_new_levels = TRUE)
  expect_identical(dim(ep), c(200L, 2L))
  ep0 <- posterior_epred(fit, newdata = nd, re_formula = NA)
  lin <- posterior_linpred(fit, newdata = nd, re_formula = NA)
  b <- fixef(fit, summary = FALSE)
  expect_equal(lin[, 1], b[, "Intercept"] + 40 * b[, "x"])
  expect_equal(ep0, lin)
  expect_identical(dim(posterior_epred(fit, newdata = nd[1, ], re_formula = ~ (1 | g))), c(200L, 1L))
  for (how in c("uncertainty", "gaussian", "old_levels")) {
    expect_true(all(is.finite(posterior_epred(fit, newdata = nd, allow_new_levels = TRUE, sample_new_levels = how))))
  }
})

test_that("model comparison criteria work like in brms", {
  fit <- fit_once()
  l <- suppressWarnings(loo(fit))
  expect_s3_class(l, "psis_loo")
  expect_identical(attr(l, "model_name"), "fit")
  fit2 <- suppressWarnings(add_criterion(fit, c("loo", "waic")))
  expect_s3_class(fit2$criteria$loo, "psis_loo")
  expect_error(loo_compare(fit2, fit), "precomputed 'loo'")
  cmp <- loo_compare(fit2, fit2, model_names = c("a", "b"))
  expect_s3_class(cmp, "compare.loo")
  r2 <- bayes_R2(fit)
  expect_identical(dimnames(r2), list("R2", c("Estimate", "Est.Error", "Q2.5", "Q97.5")))
})

test_that("marginal likelihoods, Bayes factors and model probabilities work like in brms", {
  data <- sim_data()
  fit_with <- function(formula, data, sampler = mock_sampler, ...) {
    with_sampler(sampler, brm(formula, data = data, chains = 2, iter = 150, warmup = 50, seed = 1, ...))
  }
  all_pars <- save_pars(all = TRUE)
  fits <- list(a = fit_with(y ~ x + (1 | g), data, save_pars = all_pars), b = fit_with(y ~ x + z + (1 | g), data, save_pars = all_pars))
  # the mock's log evidence is -100 - d - q for the standardized outcomes; the outcomes'
  # standardization adds its Jacobian
  n <- nrow(data)
  y_sd <- sqrt(mean((data$y - mean(data$y))^2))
  ml <- with_sampler(no_network, bridge_sampler(fits$a))  # computed during the fit
  expect_s3_class(ml, "bridge")
  expect_equal(ml$logml, -103 - n * log(y_sd))
  expect_identical(ml$niter, 200)
  expect_output(print(ml), "log marginal likelihood")

  # without save_pars(all = TRUE), the fit is repeated with its seed and chains
  plain <- fit_with(y ~ x + (1 | g), data)
  mock_env$calls <- NULL
  expect_equal(with_mock(bridge_sampler(plain))$logml, ml$logml)
  expect_identical(mock_env$calls, list(list(seed = plain$metabeta$seed, chains = 2, evidence = TRUE, random_walk = FALSE)))
  stored <- with_mock(add_criterion(plain, "marglik"))
  expect_equal(with_sampler(no_network, bridge_sampler(stored))$logml, ml$logml)

  bf <- bayes_factor(fits$a, fits$b, log = TRUE)
  expect_equal(bf$bf, 1)
  expect_output(print(bf), "log Bayes factor in favor of fits$a over fits$b: 1.00000", fixed = TRUE)
  expect_equal(bayes_factor(ml, bridge_sampler(fits$b))$bf, exp(1))
  pp <- post_prob(fits$a, fits$b)
  expect_equal(pp, c("fits$a" = exp(1), "fits$b" = 1) / (1 + exp(1)))
  expect_equal(unname(post_prob(fits$a, fits$b, prior_prob = c(0.2, 0.8))), c(0.2 * exp(1), 0.8) / (0.2 * exp(1) + 0.8))
  expect_error(post_prob(fits$a, fits$b, prior_prob = c(0.5, 0.6)), "do not sum to one")

  # rescaling the outcome changes the marginal likelihood by exactly the Jacobian
  scaled <- fit_with(y ~ x + (1 | g), transform(data, y = y / 1000), save_pars = all_pars)
  expect_equal(bridge_sampler(scaled)$logml, ml$logml + n * log(1000))
  expect_error(bayes_factor(fits$a, scaled), "Responses do not match")

  unreliable <- function(...) {
    out <- mock_sampler(...)
    if (!is.null(out$evidence)) out$evidence$pool_psis_k <- 0.9
    out
  }
  expect_warning(bridge_sampler(fit_with(y ~ x + (1 | g), data, unreliable, save_pars = all_pars)), "Pareto k")
  prior_only <- fit_with(y ~ x + (1 | g), data, sample_prior = "only")
  expect_error(bridge_sampler(prior_only), "fitted to data")
})

test_that("control = list(random_walk = TRUE) reaches the sampler and survives update()", {
  data <- sim_data()
  mock_env$calls <- NULL
  fit <- with_mock(brm(y ~ x + (1 | g), data = data, chains = 2, iter = 150, warmup = 50, seed = 1,
    control = list(random_walk = TRUE, adapt_delta = 0.99)))
  expect_true(mock_env$calls[[1]]$random_walk)
  expect_true(with_mock(update(fit, iter = 200))$metabeta$random_walk)
  expect_true(mock_env$calls[[2]]$random_walk)
})

test_that("hypothesis() reproduces brms' output", {
  fit <- fit_once()
  h <- hypothesis(fit, c("x > 0", "x - z = 0"))
  expect_s3_class(h, "brmshypothesis")
  expect_identical(
    names(h$hypothesis),
    c("Hypothesis", "Estimate", "Est.Error", "CI.Lower", "CI.Upper", "Evid.Ratio", "Post.Prob", "Star")
  )
  expect_identical(h$hypothesis$Hypothesis, c("(x) > 0", "(x-z) = 0"))
  draws <- fixef(fit, summary = FALSE)[, "x"]
  expect_equal(h$hypothesis$CI.Lower[1], unname(quantile(draws, 0.05)))
  expect_false(is.na(h$hypothesis$Evid.Ratio[2]))  # Savage-Dickey from prior draws
  expect_output(print(h), "Hypothesis Tests for class b:")
  hs <- hypothesis(fit, "x > 0", scope = "coef", group = "g")
  expect_identical(nrow(hs$hypothesis), 12L)
  expect_error(hypothesis(fit, "nope > 0"), "cannot be found")
  # class = NULL, as class = "": full variable names
  icc <- hypothesis(fit, "sd_g__Intercept^2 / (sd_g__Intercept^2 + sigma^2) = 0", class = NULL)
  d <- as_draws_df(fit)
  expect_equal(icc$hypothesis$Estimate, mean(d$sd_g__Intercept^2 / (d$sd_g__Intercept^2 + d$sigma^2)))
  expect_output(print(icc), "Hypothesis Tests for class :")
})

test_that("emmeans summarizes the posterior of the linear predictor, as with brms", {
  skip_if_not_installed("emmeans")
  fit <- fit_once()
  em <- summary(emmeans::emmeans(fit, ~ f))
  expect_identical(as.character(em$f), levels(sim_data()$f))
  # emmeans averages over the two values of the 0/1 covariate z
  d <- sim_data()
  grid <- expand.grid(x = mean(d$x), z = 0:1, f = factor(levels(d$f), levels(d$f)))
  eta <- posterior_linpred(fit, newdata = grid, re_formula = NA)
  expect_equal(em$emmean, unname(apply((eta[, c(1, 3, 5)] + eta[, c(2, 4, 6)]) / 2, 2, median)))
  pairs <- summary(emmeans::contrast(emmeans::emmeans(fit, ~ f), "pairwise"))
  expect_identical(nrow(pairs), 3L)
})

test_that("sample_prior = 'only' and empty fits need no network", {
  data <- sim_data()
  fit <- with_sampler(no_network, brm(
    y ~ x + (1 + x | g), data = data, sample_prior = "only", chains = 1, iter = 400, warmup = 0, seed = 1
  ))
  expect_identical(fit$metabeta$method, "prior")
  sd_x <- as.matrix(fit, variable = "b_x")[, 1]
  scale <- as.numeric(sub("normal\\(0, (.*)\\)", "\\1", fit$prior$prior[fit$prior$coef == "x" & fit$prior$class == "b"]))
  expect_equal(sd(sd_x), scale, tolerance = 0.15)
  empty <- with_sampler(no_network, brm(y ~ x + (1 | g), data = data, empty = TRUE))
  expect_null(empty$fit)
  expect_output(print(empty), "does not contain posterior draws")
})

test_that("unused levels get group-level effects from the population distribution", {
  data <- sim_data()
  data$g <- factor(data$g, levels = c(levels(data$g), "unused"))
  fit <- with_mock(brm(y ~ x + (1 | g), data = data, drop_unused_levels = FALSE, chains = 1, iter = 80, warmup = 10, seed = 2))
  expect_identical(ngrps(fit)$g, 13L)
  expect_true("r_g[unused,Intercept]" %in% variables(fit))
})

test_that("file caching, update() and fit reuse behave like brms", {
  data <- sim_data()
  file <- tempfile(fileext = ".rds")
  fit <- with_mock(brm(y ~ x + (1 | g), data = data, chains = 1, iter = 80, warmup = 10, seed = 2, file = file))
  expect_true(file.exists(file))
  again <- with_sampler(no_network, brm(y ~ x + (1 | g), data = data, file = file))
  expect_identical(as.matrix(again), as.matrix(fit))
  expect_error(update(fit, formula. = ~ . + z), "New variables found: 'z'")
  up <- with_mock(update(fit, formula. = ~ . + z, newdata = data, iter = 100))
  expect_true("b_z" %in% variables(up))
  expect_identical(ndraws(up), 50L)
  refit <- with_mock(brm(fit = fit, chains = 2, iter = 40, warmup = 20, seed = 5))
  expect_identical(ndraws(refit), 40L)
})

test_that("plots are produced through bayesplot and ggplot2", {
  fit <- fit_once()
  expect_s3_class(suppressMessages(pp_check(fit)), "ggplot")
  expect_s3_class(pp_check(fit, type = "stat", stat = "mean"), "ggplot")
  expect_s3_class(mcmc_plot(fit, variable = c("b_x", "b_z")), "ggplot")
  ce <- conditional_effects(fit)
  expect_s3_class(ce, "brms_conditional_effects")
  expect_identical(names(ce), c("x", "z", "f"))
  expect_true(all(c("effect1__", "estimate__", "lower__", "upper__") %in% names(ce$x)))
  expect_identical(nrow(ce$x), 100L)
  expect_identical(nrow(ce$f), 3L)
  expect_identical(nrow(make_conditions(fit, "x")), 3L)
})

test_that("Stan-only functions explain that they are unavailable", {
  expect_error(stancode(fit_once()), "not available for metabeta models")
  expect_identical(standata(fit_once())$N, 96L)
})
