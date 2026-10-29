# Compare a project's metabeta fits (saved by harness.R) with the numbers it reported.
#
# Usage: Rscript compare.R <work dir> <name>
# Reads reported/<name>.csv: model, parameter, estimate, error, lower, upper, level.
#   parameter: a draws variable (b_x, sd_g__x, cor_g__x__z, sigma, r_g[level,x]),
#     coef_<g>[level,x] for coef(fit)$g[level, , x], or expr:<R expression of draws>
#   error: the reported posterior SD, or NA if none was reported
#   level: 0.95 for a central interval, hdi0.95 for a highest-density interval
# Writes results/<name>.csv. A number matches when it lies within a tolerance of the
# reported one, in reported posterior SDs (ours where none was reported), beyond half a
# unit of the reported number's last digit (its rounding): 0.25 for the mean and the SD,
# 0.35 for the interval bounds. `status` says whether metabeta fitted the model and
# whether it warned that its draws may be under-dispersed.
args <- commandArgs(TRUE)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
name <- args[2]
root <- file.path(args[1], name)
suppressPackageStartupMessages(library(metabeta))
fits <- new.env()
load(file.path(root, "run.RData"), envir = fits)
reported <- read.csv(file.path(here, "reported", paste0(name, ".csv")), colClasses = "character")

half_unit <- function(s) {
  if (is.na(s) || s == "NA") return(0)
  s <- sub("^-", "", s)
  decimals <- if (grepl(".", s, fixed = TRUE)) nchar(sub(".*\\.", "", s)) else 0
  0.5 * 10^-decimals
}

shortest_interval <- function(x, level) {
  x <- sort(x)
  k <- ceiling(level * length(x))
  i <- which.min(x[k:length(x)] - x[1:(length(x) - k + 1)])
  c(x[i], x[i + k - 1])
}

draws_of <- function(fit, parameter) {
  if (startsWith(parameter, "expr:")) {
    return(eval(str2lang(sub("^expr:", "", parameter)), as.data.frame(as_draws_df(fit))))
  }
  if (startsWith(parameter, "coef_")) {
    m <- regmatches(parameter, regexec("^coef_(.+)\\[(.+),(.+)\\]$", parameter))[[1]]
    return(coef(fit, summary = FALSE)[[m[2]]][, m[3], m[4]])
  }
  d <- as_draws_df(fit)
  if (!parameter %in% names(d)) stop("no variable ", parameter)
  as.numeric(d[[parameter]])
}

# The models whose brm() call warned about the Metropolis-Hastings draws, from the run log.
# (A call can follow a message without a final newline on the same line.)
warned_models <- function(log) {
  lines <- readLines(log, warn = FALSE, encoding = "UTF-8")
  warned <- character()
  current <- NULL
  for (line in lines) {
    call <- regmatches(line, regexec("> ([[:alnum:]_.]+) (<-|=) (.*%>% )?brm\\(", line))[[1]]
    if (length(call)) {
      current <- call[2]
    } else if (!is.null(current) && startsWith(line, "WARNING: The Metropolis")) {
      warned <- union(warned, current)
    }
  }
  warned
}
warned <- warned_models(file.path(root, "run.log"))

rows <- lapply(seq_len(nrow(reported)), function(i) {
  r <- reported[i, ]
  fit <- get0(r$model, envir = fits, inherits = FALSE)
  x <- if (inherits(fit, "metabetafit")) tryCatch(draws_of(fit, r$parameter), error = function(e) NULL)
  if (is.null(x)) return(data.frame(model = r$model, parameter = r$parameter, status = "not fitted"))
  hdi <- startsWith(r$level, "hdi")
  level <- as.numeric(sub("^hdi", "", r$level))
  q <- if (hdi) shortest_interval(x, level) else quantile(x, c(1 - level, 1 + level) / 2, names = FALSE)
  ref <- suppressWarnings(as.numeric(r[c("estimate", "error", "lower", "upper")]))
  ours <- c(mean(x), sd(x), q)
  scale <- if (is.na(ref[2])) ours[2] else ref[2]
  rounding <- vapply(r[c("estimate", "error", "lower", "upper")], half_unit, 0)
  excess <- pmax(abs(ours - ref) - rounding, 0) / scale
  excess[is.na(excess)] <- 0
  data.frame(
    model = r$model, parameter = r$parameter,
    status = if (r$model %in% warned) "warned" else "no warning",
    reported = sprintf("%s (%s) [%s, %s]", r$estimate, r$error, r$lower, r$upper),
    metabeta = sprintf("%.4g (%.3g) [%.4g, %.4g]", ours[1], ours[2], ours[3], ours[4]),
    z_mean = (ours[1] - ref[1]) / scale,
    sd_ratio = ours[2] / ref[2],
    match = excess[1] <= 0.25 && excess[2] <= 0.25 && max(excess[3:4]) <= 0.35
  )
})
columns <- c("model", "parameter", "status", "reported", "metabeta", "z_mean", "sd_ratio", "match")
res <- do.call(rbind, lapply(rows, function(r) {
  r[setdiff(columns, names(r))] <- NA
  r[columns]
}))
res$match[is.na(res$match)] <- FALSE
write.csv(res, file.path(here, "results", paste0(name, ".csv")), row.names = FALSE)

cat(sprintf("%s: %d of %d reported numbers match (%d from models with a warning, %d from models metabeta did not fit)\n",
  name, sum(res$match), nrow(res), sum(res$status == "warned"), sum(res$status == "not fitted")))
