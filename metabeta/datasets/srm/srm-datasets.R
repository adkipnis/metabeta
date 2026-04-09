# install missing packages
options(repos = c(CRAN = "https://cloud.r-project.org"))
packages <- c("rstudioapi", "fs", "arrow")
missing <- packages[!packages %in% installed.packages()[,"Package"]]
if (length(missing) > 0) {
  install.packages(missing, dependencies = T)
}
lapply(packages, library, character.only = T)

# helpers
load.df <- function(fname){
  tmp <- new.env()
  load(fname, envir = tmp)
  df <- tmp[[ls(tmp)]]
  return(df)
}

rename <- function(df, old,  new){
  colnames(df)[colnames(df) == old] <- new
  return(df)
}

# ------------------------------------------------------------------------------
# get cwd
script_path <- dirname(getActiveDocumentContext()$path)
setwd(script_path)
dir.create("parquet", showWarnings = F)

# list all .rdata files
fnames <- list.files(
  path = path(script_path, 'raw'),
  pattern = "\\.rdata$",
  full.names = T,
  ignore.case = T
)
if (length(fnames) == 0){
  print('Warning: No rdata files found!')
}

# Fix column dtypes before writing to parquet.
#
# Two cases need attention:
#
# 1. Binary factors coded as "0"/"1" (e.g. news data_channel_is_*, laborsupply
#    disab).  These come through as categorical dtype in Python and get
#    one-hot encoded instead of being kept as binary numerics.  Casting to
#    integer lets the Python preprocessor's checkBinary() recognise them and
#    leave them untransformed (exclude_binary=True).
#
# 2. schooling$mar76 has mixed NLS numeric codes and the string "yes" as factor
#    levels.  Simplify to a clean binary married/not_married factor so the
#    one-hot encoder produces a meaningful dummy.
fix_dtypes <- function(df, name) {
  # 0/1 binary factors -> integer
  for (col in names(df)) {
    x <- df[[col]]
    if (is.factor(x) && identical(sort(levels(x)), c("0", "1"))) {
      df[[col]] <- as.integer(as.character(x))
    }
  }

  # dataset-specific fixes
  if (name == "schooling" && "mar76" %in% names(df)) {
    df$mar76 <- factor(ifelse(as.character(df$mar76) == "yes", "married", "not_married"))
  }

  return(df)
}

# load each, rename target and save as parquet
for (fname in fnames){
  df <- load.df(fname)
  if(nrow(df) < 50) next
  df <- rename(df, colnames(df)[1], 'y')
  name <- sub("\\.rdata$", "", basename(fname), ignore.case = T)
  df <- fix_dtypes(df, name)
  outname <- sub("\\.rdata$", ".parquet", fname, ignore.case = T)
  outname <- sub("raw", "parquet", outname)
  write_parquet(df, outname)
  print(paste('Saved to', outname))
}

