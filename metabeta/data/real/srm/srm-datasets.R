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

# load each, rename target and save as csv
for (fname in fnames){
  df <- load.df(fname)
  if(nrow(df) < 50) next
  df <- rename(df, colnames(df)[1], 'y')
  outname <- sub("\\.rdata$", ".parquet", fname, ignore.case = T)
  outname <- sub("raw", "parquet", outname)
  write_parquet(df, outname)
  print(paste('Saved to', outname))
}

