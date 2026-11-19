# Download projects' files into <work dir>/<name>/ (see projects.R). Files already there
# are kept.
#
# Usage: Rscript fetch.R <work dir> <name>...
args <- commandArgs(TRUE)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "projects.R"))
work <- args[1]

github_url <- function(gh, path) {
  sprintf("https://raw.githubusercontent.com/%s/%s/%s", gh[["repo"]], gh[["commit"]], utils::URLencode(path))
}

# The files below a directory of the repository, at the commit.
github_dir <- function(gh, dir) {
  url <- sprintf("https://api.github.com/repos/%s/git/trees/%s?recursive=1", gh[["repo"]], gh[["commit"]])
  tree <- jsonlite::fromJSON(url)$tree
  tree$path[tree$type == "blob" & startsWith(tree$path, paste0(dir, "/"))]
}

for (name in args[-1]) {
  p <- projects[[name]]
  if (is.null(p)) stop("unknown project: ", name)
  dir <- file.path(work, name)
  urls <- character()
  if (!is.null(p$github)) {
    paths <- c(p$files, unlist(lapply(p$dirs, github_dir, gh = p$github)))
    urls <- stats::setNames(vapply(paths, github_url, "", gh = p$github), paths)
  }
  if (!is.null(p$osf)) {
    urls <- c(urls, stats::setNames(sprintf("https://osf.io/download/%s/", p$osf), names(p$osf)))
  }
  for (path in names(urls)) {
    dest <- file.path(dir, path)
    if (file.exists(dest)) next
    dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
    # download next to the destination first, so that a failed download leaves no file
    tmp <- paste0(dest, ".part")
    utils::download.file(urls[[path]], tmp, mode = "wb", quiet = TRUE)
    file.rename(tmp, dest)
  }
  for (d in p$mkdir) dir.create(file.path(dir, d), recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("%s: %d files in %s\n", name, length(urls), normalizePath(dir)))
}
