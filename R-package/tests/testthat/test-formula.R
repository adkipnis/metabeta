parse <- function(f) parse_model_formula(brmsformula(f))

test_that("group-level terms are parsed like brms", {
  p <- parse(y ~ x + (1 + x | g))
  expect_identical(p$group, "g")
  expect_true(p$bars[[1]]$correlated)
  expect_false(parse(y ~ x + (1 + x || g))$bars[[1]]$correlated)
  expect_identical(parse(y ~ x + (1 | p | g))$bars[[1]]$id, "p")
  expect_false(parse(y ~ x + (x | gr(g, cor = FALSE)))$bars[[1]]$correlated)
  expect_identical(parse(y ~ x + (1 | g1:g2))$group, "g1:g2")
  expect_identical(deparse1(parse(y ~ x + (1 | g) - 1)$fixed), "~x - 1")
  expect_identical(deparse1(parse(y ~ (1 | g))$fixed), "~1")
})

test_that("unsupported model features fail with an explanation", {
  expect_error(parse(y ~ s(x) + (1 | g)), "s\\(\\)")
  expect_error(parse(y ~ x + offset(z) + (1 | g)), "offset")
  expect_error(parse(y ~ x + (1 | a) + (1 | b)), "exactly one grouping factor")
  expect_error(parse(y ~ x + (1 | a/b)), "Nested grouping")
  expect_error(parse(y ~ x), "needs a group-level term")
  expect_error(parse(y | weights(w) ~ x + (1 | g)), "weights()", fixed = TRUE)
  expect_error(parse(bf(y ~ x + (1 | g), sigma ~ x)), "Distributional")
  expect_error(parse(y ~ x + (1 | mm(a, b))), "mm()", fixed = TRUE)
  expect_error(parse(y ~ x + (1 | gr(g, by = z))), "'by'")
})

test_that("families are validated like brms and limited to the trained likelihoods", {
  expect_s3_class(validate_family(gaussian()), "brmsfamily")
  expect_identical(validate_family("poisson")$link, "log")
  expect_identical(validate_family(bernoulli)$family, "bernoulli")
  expect_identical(family_likelihood(validate_family(binomial())), "bernoulli")
  expect_error(family_likelihood(student()), "not supported by metabeta")
  expect_error(family_likelihood(brmsfamily("bernoulli", "probit")), "only the 'logit' link")
  expect_error(brmsfamily("nonsense"), "not a valid brms family")
  expect_output(print(bernoulli()), "Family: bernoulli \nLink function: logit")
  # as in brms, a link may be named without quotes
  expect_identical(bernoulli(link = logit)$link, "logit")
  expect_identical(lognormal(link = identity)$link, "identity")
  expect_identical(brmsfamily("bernoulli", probit)$link, "probit")
  my_link <- "probit"
  expect_identical(bernoulli(link = my_link)$link, "probit")
})

test_that("bf() keeps the brms structure", {
  f <- bf(y ~ x + (x | g), center = FALSE)
  expect_s3_class(f, c("brmsformula", "bform"))
  expect_identical(f$resp, "y")
  expect_false(attr(f$formula, "center"))
  expect_identical((bf(y ~ x + (1 | g)) + poisson())$family$family, "poisson")
  expect_output(print(bf(y ~ x + (1 | g))), "y ~ x + (1 | g)", fixed = TRUE)
})

test_that("designs use brms coefficient names and drop incomplete rows", {
  data <- sim_data()
  data$x[3] <- NA
  expect_warning(
    m <- prepare_model(y ~ I(x^2) + poly(x, 2) + f + x:z + (1 + f | g), data, gaussian()),
    "Rows containing NAs were excluded from the model."
  )
  expect_identical(
    m$design$b_names,
    c("Intercept", "IxE2", "polyx21", "polyx22", "fbc", "fdMe", "x:z")
  )
  expect_identical(m$design$re_names, c("Intercept", "fbc", "fdMe"))
  expect_identical(nrow(m$design$frame), nrow(data) - 1L)
})

test_that("responses are validated per family", {
  data <- sim_data()
  data$yf <- factor(ifelse(data$yb == 1, "b", "a"))
  m <- prepare_model(yf ~ x + (1 | g), data, bernoulli())
  expect_identical(m$design$y, as.numeric(data$yf) - 1)
  expect_error(prepare_model(y ~ x + (1 | g), data, poisson()), "requires integer responses")
  expect_error(prepare_model(y ~ x + (1 | g), data, bernoulli()), "two different values")
  expect_error(prepare_model(y ~ nope + (1 | g), data, gaussian()), "'nope'")
})

test_that("mixed correlation structures are rejected", {
  data <- sim_data()
  expect_error(
    prepare_model(y ~ x + (1 + x | g) + (0 + z | g), data, gaussian()),
    "either all correlations"
  )
  m <- prepare_model(y ~ x + (1 | g) + (0 + x | g), data, gaussian())
  expect_false(m$design$correlated)
  m <- prepare_model(y ~ x + (1 | p | g) + (0 + x | p | g), data, gaussian())
  expect_true(m$design$correlated)
})
