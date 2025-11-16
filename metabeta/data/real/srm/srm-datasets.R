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
script_path <- normalizePath(sys.frame(1)$ofile)
script_dir <- dirname(script_path)

# list all .rdata files
fnames <- list.files(
  path = script_dir,
  pattern = "\\.rdata$",
  full.names = T,
  ignore.case = T
)

# load each, rename target and save as csv
for (fname in fnames){
  df <- load.df(fname)
  if(nrow(df) < 50) next
  df <- rename(df, colnames(df)[1], 'y')
  outname <- sub("\\.rdata$", ".csv", fname, ignore.case = T)
  write.csv(df, outname, row.names = F)
  print(paste('Saved df to', outname))
}

