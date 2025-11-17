library(rstudioapi)

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
dir.create("csv", showWarnings = F)

# list all .rdata files
fnames <- list.files(
  path = paste0(script_dir, '/raw'),
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
  outname <- sub("\\.rdata$", ".csv", fname, ignore.case = T)
  outname <- sub("raw", "csv", outname)
  write.csv(df, outname, row.names = F)
  print(paste('Saved to', outname))
}

