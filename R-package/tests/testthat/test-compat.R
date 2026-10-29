compat_fits <- local({
  fits <- NULL
  function() {
    if (is.null(fits)) {
      data <- sim_data()
      fits <<- with_mock(list(
        a = brm(y ~ x + (1 | g), data = data, chains = 2, iter = 100, warmup = 50, seed = 1),
        b = brm(y ~ x + z + (1 | g), data = data, chains = 2, iter = 100, warmup = 50, seed = 2)
      ))
    }
    fits
  }
})

test_that("every export of brms exists with brms' arguments", {
  skip_if_not_installed("brms")
  b <- getNamespaceExports("brms")
  m <- getNamespaceExports("metabeta")
  expect_identical(setdiff(b, m), character(0))
  for (f in intersect(b, m)) {
    fb <- get(f, asNamespace("brms"))
    fm <- get(f, asNamespace("metabeta"))
    if (is.function(fb) && is.function(fm)) {
      expect_identical(names(formals(fm)), names(formals(fb)), label = f)
    }
  }
})

test_that("deprecated aliases forward to the current functions", {
  a <- compat_fits()$a
  expect_identical(attr(suppressWarnings(LOO(a)), "model_name"), "a")
  expect_s3_class(suppressWarnings(add_loo(a))$criteria$loo, "psis_loo")
  expect_identical(parnames(a), variables(a))
  expect_identical(dim(pp_expect(a, ndraws = 5)), c(5L, 96L))
  expect_identical(restructure(a), a)
})

test_that("model weights and averaging combine several models", {
  fits <- compat_fits()
  w <- suppressWarnings(model_weights(fits$a, fits$b, weights = "loo"))
  expect_equal(sum(w), 1)
  expect_identical(names(w), c("fits$a", "fits$b"))
  pa <- suppressWarnings(pp_average(fits$a, fits$b, weights = c(0.3, 0.7), ndraws = 50, seed = 1))
  expect_identical(dim(pa), c(96L, 4L))
  post <- posterior_average(fits$a, fits$b, weights = c(1, 1), variable = "b_x", ndraws = 40)
  expect_identical(dim(post), c(40L, 1L))
  expect_error(posterior_average(fits$a, fits$b, variable = "b_z", weights = c(1, 1)), "missing")
})

test_that("combine_models adds chains and stubs explain themselves", {
  a <- compat_fits()$a
  ab <- combine_models(a, a)
  expect_identical(nchains(ab), 4L)
  expect_identical(ndraws(ab), 2L * ndraws(a))
  expect_error(s(x), "Terms of type 's\\(\\)' are not supported")
  expect_error(cor_ar(), "Autocorrelation")
  expect_error(dskew_normal(1), "not included in metabeta")
  expect_error(gr(g), "inside a model formula")
  expect_identical(lasso(), "lasso(1, 1)")
})

test_that("Student-t helpers match the location-scale definition", {
  expect_equal(dstudent_t(1.3, 5, 1, 2), dt(0.15, 5) / 2)
  expect_equal(pstudent_t(qstudent_t(0.3, 5, 1, 2), 5, 1, 2), 0.3)
  expect_equal(logm1(expp1(0.7)), 0.7)
})

test_that("validate_newdata and brmsfit_needs_refit check data like brms", {
  a <- compat_fits()$a
  nd <- data.frame(x = 1, g = "s01")
  expect_error(validate_newdata(nd, a), "'y'")
  expect_identical(validate_newdata(nd, a, check_response = FALSE), nd)
  expect_false(brmsfit_needs_refit(a, data = a$data, silent = TRUE))
  changed <- a$data
  changed$y <- changed$y + 1
  expect_true(brmsfit_needs_refit(a, data = changed, silent = TRUE))
})
