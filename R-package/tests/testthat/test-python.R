# End-to-end tests with the pretrained networks (downloaded from Hugging Face on first use).

test_that("sleepstudy matches lme4 and yields valid MCMC diagnostics", {
  skip_if_no_python()
  skip_if_not_installed("lme4")
  data("sleepstudy", package = "lme4")
  fit <- brm(Reaction ~ Days + (Days | Subject), data = sleepstudy, seed = 1)
  expect_identical(fit$metabeta$method, "imh")
  ml <- lme4::fixef(lme4::lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy))
  fe <- fixef(fit)
  expect_lt(abs(fe["Intercept", "Estimate"] - ml[["(Intercept)"]]), fe["Intercept", "Est.Error"])
  expect_lt(abs(fe["Days", "Estimate"] - ml[["Days"]]), fe["Days", "Est.Error"])
  s <- summary(fit)
  expect_true(all(s$fixed$Rhat < 1.05))
  expect_gt(fit$metabeta$safeguards$accept_rate, 0.2)
})

test_that("fits are reproducible under seed and set.seed()", {
  skip_if_no_python()
  data("sleepstudy", package = "lme4")
  a <- brm(Reaction ~ Days + (1 | Subject), data = sleepstudy, seed = 7, chains = 2, iter = 600)
  b <- brm(Reaction ~ Days + (1 | Subject), data = sleepstudy, seed = 7, chains = 2, iter = 600)
  expect_identical(as.matrix(a), as.matrix(b))
  set.seed(3)
  c1 <- brm(Reaction ~ Days + (1 | Subject), data = sleepstudy, chains = 2, iter = 600)
  set.seed(3)
  c2 <- brm(Reaction ~ Days + (1 | Subject), data = sleepstudy, chains = 2, iter = 600)
  expect_identical(as.matrix(c1), as.matrix(c2))
})

test_that("bernoulli and poisson models agree with glmer", {
  skip_if_no_python()
  skip_if_not_installed("lme4")
  data <- sim_data(m = 30, n_i = 15, seed = 4)
  for (case in list(list(f = yb ~ x + z + (1 | g), fam = "binomial", fam_b = bernoulli()),
                    list(f = yc ~ x + z + (1 | g), fam = "poisson", fam_b = poisson()))) {
    fit <- brm(case$f, data = data, family = case$fam_b, seed = 2)
    ml <- lme4::fixef(suppressWarnings(lme4::glmer(case$f, data = data, family = case$fam)))
    fe <- fixef(fit)
    for (k in c("x", "z")) {
      expect_lt(abs(fe[k, "Estimate"] - ml[[k]]), 2 * fe[k, "Est.Error"])
    }
    expect_true(all(summary(fit)$fixed$Rhat < 1.05))
  }
})

test_that("uncorrelated and intercept-free group-level terms run end to end", {
  skip_if_no_python()
  data <- sim_data(m = 20, n_i = 10, seed = 5)
  data$xc <- data$x - mean(data$x)
  fit <- brm(y ~ xc + (1 + xc || g), data = data, seed = 1, chains = 2, iter = 1000)
  expect_false("cor_g__Intercept__xc" %in% variables(fit))
  expect_true(all(is.finite(fixef(fit))))
})

test_that("data outside the trained range fail with an explanation", {
  skip_if_no_python()
  data <- sim_data(m = 3, n_i = 8)
  expect_error(brm(y ~ x + (1 | g), data = data, seed = 1), "outside the range")
})

test_that("too many group-level coefficients for the design report the binding limit", {
  skip_if_no_python()
  data <- sim_data(m = 20, n_i = 10)
  expect_error(
    brm(y ~ x + z + I(x^2) + (1 + x + z | g), data = data, seed = 1),
    "q=3 > max_q=2"
  )
})

test_that("the marginal likelihood is reproducible and penalizes a conflicting prior", {
  skip_if_no_python()
  data("sleepstudy", package = "lme4")
  f <- Reaction ~ Days + (Days | Subject)
  fit <- brm(f, data = sleepstudy, seed = 1, chains = 2, save_pars = save_pars(all = TRUE))
  refit <- brm(f, data = sleepstudy, seed = 1, chains = 2)
  expect_identical(nchains(refit), 2L)
  expect_identical(as.matrix(fit), as.matrix(refit))
  expect_equal(bridge_sampler(refit)$logml, bridge_sampler(fit)$logml)
  # Days changes reaction times by about 10 ms per day; a prior at 0 with SD 0.5 contradicts it
  tight <- suppressWarnings(brm(f, data = sleepstudy, prior = prior(normal(0, 0.5), class = b),
    seed = 1, chains = 2, save_pars = save_pars(all = TRUE)))
  expect_lt(suppressWarnings(bayes_factor(tight, fit, log = TRUE))$bf, -3)
})

test_that("random-walk moves repair chains that repeat few draws", {
  skip_if_no_python()
  data("sleepstudy", package = "lme4")
  data <- transform(sleepstudy, y = as.vector(scale(Reaction)), days = as.vector(scale(Days)))
  tight <- c(prior(normal(0, 0.05), class = b), prior(normal(0, 2.5), class = sd))
  plain <- suppressWarnings(brm(y ~ days + (days | Subject), data = data, prior = tight, seed = 1))
  expect_true(plain$metabeta$safeguards$ess_flag)
  moved <- suppressWarnings(brm(y ~ days + (days | Subject), data = data, prior = tight, seed = 1,
    control = list(random_walk = TRUE)))
  expect_true(moved$metabeta$safeguards$random_walk_ran)
  expect_true(all(summary(moved)$fixed$Rhat < 1.05))
})
