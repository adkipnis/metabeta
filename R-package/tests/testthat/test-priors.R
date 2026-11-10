test_that("prior() and friends build brmsprior objects like brms", {
  p <- prior(normal(0, 10), class = b, coef = x) + prior(lkj(2), class = cor)
  expect_s3_class(p, "brmsprior")
  expect_identical(
    names(p),
    c("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "tag", "source")
  )
  expect_identical(p$prior, c("normal(0, 10)", "lkj(2)"))
  expect_identical(p$coef, c("x", ""))
  expect_identical(p$source, c("user", "user"))
  expect_identical(prior_(~ normal(0, 1), class = ~sd)$class, "sd")
  expect_identical(set_prior(c("normal(0, 1)", "normal(0, 2)"), coef = c("a", "b"))$coef, c("a", "b"))
  expect_error(prior(normal(0, 1)) + prior(normal(0, 2)), "Duplicated")
  expect_output(print(prior(normal(0, 10), class = b)), "b ~ normal(0, 10)", fixed = TRUE)
})

test_that("default priors list every parameter in brms order", {
  data <- sim_data()
  gp <- get_prior(y ~ x + z + (1 + x | g), data = data)
  expect_identical(gp$class, c("b", "b", "b", "cor", "cor", "Intercept", "sd", "sd", "sd", "sd", "sigma"))
  expect_identical(gp$coef, c("", "x", "z", "", "", "", "", "", "Intercept", "x", ""))
  expect_identical(gp$lb[gp$class %in% c("sd", "sigma")], rep("0", 5))
  expect_match(gp$prior[gp$class == "Intercept"], "^normal\\(")
  expect_match(gp$prior[gp$class == "sigma"], "^student_t\\(5, 0, ")
  gpb <- get_prior(yb ~ x + (1 | g), data = data, family = bernoulli())
  expect_identical(gpb$prior[gpb$class == "Intercept"], "normal(0, 1.5)")
  expect_false("sigma" %in% gpb$class)
})

test_that("user priors propagate from class to group to coefficient", {
  data <- sim_data()
  p <- prior(normal(0, 5), class = b) +
    prior(normal(0, 3), class = sd) +
    prior(normal(0, 1), class = sd, group = g, coef = x)
  vp <- validate_prior(p, y ~ x + z + (1 + x | g), data = data)
  b <- vp[vp$class == "b", ]
  expect_identical(b$prior, rep("normal(0, 5)", 3))
  expect_identical(b$source, c("user", "(vectorized)", "(vectorized)"))
  sd <- vp[vp$class == "sd", ]
  expect_identical(sd$prior, c("normal(0, 3)", "normal(0, 3)", "normal(0, 3)", "normal(0, 1)"))
  expect_identical(sd$source, c("user", "(vectorized)", "(vectorized)", "user"))
})

test_that("priors outside the trained families are rejected", {
  data <- sim_data()
  check <- function(p) validate_prior(p, y ~ x + (1 + x | g), data = data)
  expect_error(check(prior(cauchy(0, 2), class = b)), "not supported by metabeta")
  expect_error(check(prior(student_t(3, 0, 2), class = b)), "5 degrees of freedom")
  expect_error(check(set_prior("", class = "b")), "Flat priors")
  expect_error(check(prior(normal(1, 2), class = sd)), "centered at zero")
  expect_error(check(prior(normal(0, 1), class = b, lb = 0)), "Bounded priors")
  expect_error(check(prior(normal(0, 1), class = b, coef = nope)), "do not correspond to any model parameter")
  expect_error(
    check(prior(student_t(5, 0, 2), class = Intercept)),
    "same family"
  )
  expect_silent(check(prior(student_t(5, 0, 2), class = Intercept) + prior(student_t(5, 0, 1), class = b)))
  expect_silent(check(prior(exponential(1), class = sd) + prior(lkj_corr_cholesky(2), class = L)))
})
