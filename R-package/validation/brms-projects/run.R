# Run the whole check: download each project, run its code with metabeta instead of brms,
# compare the fits with the reported numbers, and summarize.
#
# Usage: Rscript run.R [--work=<dir>] [name ...]
#   name: projects to run (default: all in projects.R)
#   --work: where projects are downloaded and run (default: work/ next to this script)
# Each step runs in a new R process. Run logs go to <work dir>/<name>/run.log.
args <- commandArgs(TRUE)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "projects.R"))
work_arg <- grep("^--work=", args, value = TRUE)
work <- if (length(work_arg)) sub("^--work=", "", work_arg) else file.path(here, "work")
dir.create(work, recursive = TRUE, showWarnings = FALSE)
work <- normalizePath(work)
selected <- setdiff(args, work_arg)
if (!length(selected)) selected <- names(projects)

rscript <- file.path(R.home("bin"), "Rscript")
step <- function(script, ..., stdout = "") {
  status <- system2(rscript, shQuote(c(file.path(here, script), ...)), stdout = stdout, stderr = stdout)
  if (status != 0) cat(script, "failed with status", status, "\n")
  status == 0
}

for (name in selected) {
  t0 <- Sys.time()
  cat(sprintf("== %s (%s)\n", name, format(t0, "%H:%M")))
  if (!step("fetch.R", work, name)) next
  step("harness.R", work, name, stdout = file.path(work, name, "run.log"))
  step("compare.R", work, name)
  cat(sprintf("   %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}
invisible(step("summarize.R"))
