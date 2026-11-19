# Run a project's document with one change: library(brms) becomes library(metabeta).
#
# Usage: Rscript harness.R <work dir> <name>
# The R chunks of the document run one by one in the global environment, in the
# document's directory, as when it is knitted; errors, warnings and messages are logged
# and the run continues. Chunks with eval = FALSE do not run. Writes swapped.R (the code
# that ran) and run.RData (the workspace) to <work dir>/<name>/.
#
# The harness itself lives in its own environment, so the global environment holds only
# what the project's code creates.
local({
  args <- commandArgs(TRUE)
  here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
  source(file.path(here, "projects.R"), local = TRUE)

  # The R chunks of an R Markdown or Quarto document, without those it does not evaluate.
  document_chunks <- function(path) {
    lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
    starts <- grep("^\\s*```+\\s*\\{r[ ,}]", lines)
    chunks <- list()
    for (s in starts) {
      e <- s + which(grepl("^\\s*```+\\s*$", lines[(s + 1):length(lines)]))[1]
      body <- lines[(s + 1):(e - 1)]
      options <- c(lines[s], grep("^#\\|", body, value = TRUE))
      if (any(grepl("eval\\s*=\\s*F(ALSE)?\\b|eval:\\s*false", options))) next
      chunks[[length(chunks) + 1]] <- c(paste("## ----", lines[s]), body)
    }
    chunks
  }

  # The one change: library(brms), library("brms"), or brms among the packages of p_load().
  swap_brms <- function(chunks) {
    n <- 0
    chunks <- lapply(chunks, function(x) {
      library_call <- grepl("^\\s*library\\((['\"]?)brms\\1\\)", x)
      p_load_call <- grepl("p_load\\(", x) & grepl("\\bbrms\\b", x)
      x[library_call] <- sub("library\\((['\"]?)brms\\1\\)", "library(\\1metabeta\\1)", x[library_call])
      x[p_load_call] <- sub("\\bbrms\\b", "metabeta", x[p_load_call])
      for (line in x[library_call | p_load_call]) cat("swapped line: ", trimws(line), "\n", sep = "")
      n <<- n + sum(library_call | p_load_call)
      x
    })
    cat(sprintf("brms -> metabeta on %d line(s)\n", n))
    chunks
  }

  # Replacements that have nothing to do with brms (projects.R, `changes`).
  apply_changes <- function(chunks, changes) {
    for (old in names(changes)) {
      n <- sum(vapply(chunks, function(x) sum(grepl(old, x, fixed = TRUE)), 0))
      cat(sprintf("other change on %d line(s): %s -> %s\n", n, old, changes[[old]]))
      chunks <- lapply(chunks, function(x) gsub(old, changes[[old]], x, fixed = TRUE))
    }
    chunks
  }

  # The packages the code loads with library() or pacman::p_load().
  loaded_packages <- function(code) {
    libs <- regmatches(code, gregexpr("(?<=library\\()['\"]?[A-Za-z0-9.]+", code, perl = TRUE))
    p_load <- regmatches(code, gregexpr("(?<=p_load\\()[^)]*", code, perl = TRUE))
    packages <- trimws(unlist(c(libs, strsplit(unlist(p_load), ","))))
    setdiff(unique(gsub("['\"]", "", packages)), "")
  }

  run_chunks <- function(chunks, until, skip) {
    for (i in seq_along(chunks)) {
      code <- paste(chunks[[i]], collapse = "\n")
      if (!nzchar(trimws(gsub("(?m)^#.*$", "", code, perl = TRUE)))) next
      if (!is.null(skip) && grepl(skip, code, perl = TRUE)) {
        cat(sprintf("\n#### chunk %d skipped\n", i))
        next
      }
      cat(sprintf("\n#### chunk %d\n", i))
      exprs <- tryCatch(parse(text = code, keep.source = TRUE), error = function(e) {
        cat("PARSE ERROR:", conditionMessage(e), "\n")
        NULL
      })
      for (e in exprs) {
        cat("> ", deparse(e, width.cutoff = 100)[1], "\n", sep = "")
        res <- tryCatch(
          withCallingHandlers(withVisible(eval(e, globalenv())),
            warning = function(w) {
              cat("WARNING:", conditionMessage(w), "\n")
              invokeRestart("muffleWarning")
            },
            message = function(m) {
              cat("MESSAGE:", sub("\n?$", "\n", conditionMessage(m)))
              invokeRestart("muffleMessage")
            }
          ),
          error = function(err) {
            cat("ERROR:", conditionMessage(err), "\n")
            NULL
          }
        )
        if (!is.null(res) && res$visible && !inherits(res$value, c("ggplot", "patchwork", "gg"))) {
          try(print(res$value))
        }
      }
      if (!is.null(until) && grepl(until, code, perl = TRUE)) break
    }
  }

  p <- projects[[args[2]]]
  if (is.null(p)) stop("unknown project: ", args[2])
  root <- normalizePath(file.path(args[1], args[2]))
  document <- file.path(root, p$document)
  chunks <- document_chunks(document)
  if (!is.null(p$prelude)) chunks <- c(list(c("## ---- prelude (projects.R)", p$prelude)), chunks)
  chunks <- apply_changes(swap_brms(chunks), p$changes)
  writeLines(unlist(chunks), file.path(root, "swapped.R"))

  packages <- setdiff(loaded_packages(unlist(chunks)), "metabeta")
  missing <- packages[!vapply(packages, requireNamespace, FALSE, quietly = TRUE)]
  if (length(missing)) cat("packages not installed:", missing, "\n")
  # The project's packages are loaded (not attached) first, so that those declaring Python
  # requirements through reticulate (e.g. marginaleffects) do so before Python starts. Then
  # metabeta's Python side is imported, before the project's code runs: on Windows with a
  # conda R, torch must load before other numerical libraries (see the package README).
  for (pkg in setdiff(packages, missing)) {
    try(suppressPackageStartupMessages(requireNamespace(pkg, quietly = TRUE)), silent = TRUE)
  }
  invisible(metabeta:::import_api())

  # every brm(file = ...) fits the model instead of loading a fit an earlier run saved
  options(brms.file_refit = "always")
  setwd(dirname(document))
  t0 <- Sys.time()
  run_chunks(chunks, p$until, p$skip)
  cat(sprintf("\nElapsed: %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  save(list = ls(globalenv(), all.names = TRUE), envir = globalenv(), file = file.path(root, "run.RData"))
})
