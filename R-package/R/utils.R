# Small internal helpers used throughout the package.

stop2 <- function(...) stop(..., call. = FALSE)

warning2 <- function(...) warning(..., call. = FALSE)

`%||%` <- function(x, y) if (is.null(x)) y else x

collapse_comma <- function(x) paste0("'", x, "'", collapse = ", ")

# Coefficient names spelled like brms, e.g. "(Intercept)" -> "Intercept",
# "I(Days^2)" -> "IDaysE2" and factor level "c-d" -> "cMd".
rename_coef <- function(x) {
  x[x == "(Intercept)"] <- "Intercept"
  symbols <- c(" ", "(", ")", "[", "]", ",", "\"", "'", "?", "+", "-", "*", "/", "^", "=", "$")
  replace <- c(rep("", 9), "P", "M", "MU", "D", "E", "EQ", "USD")
  for (i in seq_along(symbols)) {
    x <- gsub(symbols[i], replace[i], x, fixed = TRUE)
  }
  x
}

# Estimate, Est.Error and quantiles of each column of a draws matrix.
summarise_matrix <- function(x, probs = c(0.025, 0.975), robust = FALSE) {
  x <- as.matrix(x)
  estimate <- if (robust) apply(x, 2, stats::median) else colMeans(x)
  error <- if (robust) apply(x, 2, stats::mad) else apply(x, 2, stats::sd)
  quantiles <- matrix(apply(x, 2, stats::quantile, probs = probs, names = FALSE), ncol = length(probs), byrow = TRUE)
  out <- cbind(estimate, error, quantiles)
  dimnames(out) <- list(colnames(x), c("Estimate", "Est.Error", paste0("Q", probs * 100)))
  out
}

# Indices of the draws to use: all, a random subset of 'ndraws', or 'draw_ids'.
select_draws <- function(n_total, ndraws = NULL, draw_ids = NULL) {
  if (!is.null(draw_ids)) {
    if (any(draw_ids < 1 | draw_ids > n_total)) stop2("Some 'draw_ids' indices are out of range.")
    return(as.integer(draw_ids))
  }
  if (is.null(ndraws)) {
    return(seq_len(n_total))
  }
  if (ndraws < 1 || ndraws > n_total) {
    stop2("Argument 'ndraws' should be between 1 and the number of draws (", n_total, ").")
  }
  sort(sample.int(n_total, ndraws))
}

# Index pairs (j, i) with j < i in the order brms names correlations.
cor_pairs <- function(q) {
  pairs <- list()
  for (i in seq_len(q)[-1]) {
    for (j in seq_len(i - 1)) pairs[[length(pairs) + 1]] <- c(j, i)
  }
  pairs
}

# One draw from N(0, diag(sd) R diag(sd)) per row of 'sd' (S x q), with R = cor[s, , ].
draw_mvn <- function(sd, cor) {
  out <- matrix(0, nrow(sd), ncol(sd))
  for (s in seq_len(nrow(sd))) {
    L <- t(chol(matrix(cor[s, , ], ncol(sd))))
    out[s, ] <- sd[s, ] * drop(L %*% rnorm(ncol(sd)))
  }
  out
}

log_mean_exp <- function(x) {
  m <- max(x)
  m + log(mean(exp(x - m)))
}
