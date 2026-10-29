# Summarize results/*.csv per project and status, and in total; writes results/summary.csv.
#
# Usage: Rscript summarize.R
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "projects.R"))
files <- file.path(here, "results", paste0(names(projects), ".csv"))
results <- do.call(rbind, lapply(files[file.exists(files)], function(f) {
  cbind(project = sub("\\.csv$", "", basename(f)), read.csv(f))
}))

summarize <- function(d) {
  z <- abs(d$z_mean[d$status != "not fitted"])
  data.frame(
    numbers = nrow(d), match = sum(d$match),
    median_abs_z = if (length(z)) round(stats::median(z), 2) else NA,
    max_abs_z = if (length(z)) round(max(z), 2) else NA
  )
}
groups <- split(results, list(results$project, results$status), drop = TRUE)
per_project <- do.call(rbind, lapply(groups, function(d) {
  cbind(project = d$project[1], status = d$status[1], summarize(d))
}))
per_project <- per_project[order(match(per_project$project, names(projects)), per_project$status), ]
total <- do.call(rbind, lapply(split(results, results$status), function(d) {
  cbind(project = "all", status = d$status[1], summarize(d))
}))
summary <- rbind(per_project, total)
rownames(summary) <- NULL
write.csv(summary, file.path(here, "results", "summary.csv"), row.names = FALSE)
print(summary, row.names = FALSE)
