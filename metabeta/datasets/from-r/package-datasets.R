# install missing packages
options(repos = c(CRAN = "https://cloud.r-project.org"))
packages <- c(
  "rstudioapi", "fs", "arrow", "lme4", "nlme",
  "mlmRev", "MEMSS", "dplyr"
)
missing <- packages[!packages %in% installed.packages()[,"Package"]]
if (length(missing) > 0) {
  install.packages(missing, dependencies = T)
}
lapply(packages, library, character.only = T)

# set cwd and create subdir
scriptPath <- tryCatch(
  dirname(getActiveDocumentContext()$path),
  error = function(e) getwd()
)
setwd(scriptPath)
dir.create('parquet', showWarnings = F)

# helpers
renameCol <- function(df, old,  new){
  colnames(df)[colnames(df) == old] <- new
  return(df)
}

# ---------------------------------------------------------------------------
# Linear mixed-effects (Gaussian outcomes)

# MathAchieve (d=5)
data('MathAchieve', package = 'MEMSS')
df <- MathAchieve
# model <- lmer(MathAch ~ SES + MEANSES + Minority + Sex  + (1 | School), data = df)
df <- renameCol(df, 'MathAch', 'y')
df <- renameCol(df, 'School', 'group')
write_parquet(df, path('parquet', 'math.parquet'))

# London Exam (d=8)
data('Exam', package = 'mlmRev')
df <- Exam |> select(-c('student'))
# model <- lmer(normexam ~ schavg + standLRT + sex + (1 | school), data = df)
df <- renameCol(df, 'normexam', 'y')
df <- renameCol(df, 'school', 'group')
write_parquet(df, path('parquet', 'london.parquet'))

# GCS (d=3)
data('Gcsemv', package = 'mlmRev')
df <- Gcsemv |> select(-c('student'))
# model <- lmer(written ~ gender + course + (1 | school), data = df)
df <- renameCol(df, 'written', 'y')
df <- renameCol(df, 'school', 'group')
write_parquet(df, path('parquet', 'gcse.parquet'))

# Sleep study (d=2)
data('sleepstudy', package = 'lme4')
df <- sleepstudy #|> select(all_of(selection))
# model <- lmer(Reaction ~ Days + (Days | Subject), data = df)
df <- renameCol(df, 'Reaction', 'y')
df <- renameCol(df, 'Subject', 'group')
write_parquet(df, path('parquet', 'sleep.parquet'))

# Orthodont (d=3)
data('Orthodont', package = 'nlme')
df <- Orthodont
# model <- lme(distance ~ age + Sex, random = ~ 1 | Subject, data = df)
df <- renameCol(df, 'distance', 'y')
df <- renameCol(df, 'Subject', 'group')
write_parquet(df, path('parquet', 'orthodont.parquet'))

# Oxboys (d=3)
data('Oxboys', package = 'nlme')
df <- Oxboys
# model <- lme(height ~ age + Occasion, random = ~ age | Subject, data = df)
df <- renameCol(df, 'height', 'y')
df <- renameCol(df, 'Subject', 'group')
write_parquet(df, path('parquet', 'oxboys.parquet'))

# Penicillin (d=1)
data('Penicillin', package = 'lme4')
df <- Penicillin
# model <- lmer(diameter ~ (1 | plate) + (1 | sample), data = df)
df <- renameCol(df, 'diameter', 'y')
df <- renameCol(df, 'plate', 'group')
write_parquet(df, path('parquet', 'penicillin.parquet'))

# Pastes (d=1)
data('Pastes', package = 'lme4')
df <- Pastes
# model <- lmer(strength ~ 1 + (1 | batch) + (1 | cask), data = df)
df <- renameCol(df, 'strength', 'y')
df <- renameCol(df, 'batch', 'group')
write_parquet(df, path('parquet', 'pastes.parquet'))

# ErgoStool (d=2)
data('ergoStool', package = 'nlme')
df <- ergoStool
# model <- lme(effort ~ Type, random = ~ 1 | Subject, data = df)
df <- renameCol(df, 'effort', 'y')
df <- renameCol(df, 'Subject', 'group')
write_parquet(df, path('parquet', 'ergostool.parquet'))

# Machines (d=2)
data('Machines', package = 'nlme')
df <- Machines
# model <- lme(score ~ Machine, random = ~ 1 | Worker, data = df)
df <- renameCol(df, 'score', 'y')
df <- renameCol(df, 'Worker', 'group')
write_parquet(df, path('parquet', 'machines.parquet'))

# Oats (d=2)
data('Oats', package = 'nlme')
df <- Oats
# model <- lme(yield ~ nitro, random = ~ 1 | Block, data = df)
df <- renameCol(df, 'yield', 'y')
df <- renameCol(df, 'Block', 'group')
write_parquet(df, path('parquet', 'oats.parquet'))

# Pixel (d=2)
data('Pixel', package = 'nlme')
df <- Pixel
# model <- lme(pixel ~ day, random = ~ 1 | Dog, data = df)
df <- renameCol(df, 'pixel', 'y')
df <- renameCol(df, 'Dog', 'group')
write_parquet(df, path('parquet', 'pixel.parquet'))

# Hsb82 (d=4)
data('Hsb82', package = 'mlmRev')
df <- Hsb82
# model <- lmer(mAch ~ ses + meanses + minrty + sx + (1 | school), data = df)
# 'cses' is centered ses (corr=0.85, not caught by correlation filter); 'sector' not in model
df <- df |> select(-c('cses', 'sector'))
df <- renameCol(df, 'mAch', 'y')
df <- renameCol(df, 'school', 'group')
write_parquet(df, path('parquet', 'hsb82.parquet'))

# Chem97 (d=4)
data('Chem97', package = 'mlmRev')
df <- Chem97
# model <- lmer(score ~ age + gcsescore + gender + (1 | school), data = df)
# 'lea' (region) and 'student' (row ID) are not in the reference model; 'gcsecnt' is
# centered gcsescore (r=1)
df <- df |> select(-c('lea', 'student', 'gcsecnt'))
df <- renameCol(df, 'score', 'y')
df <- renameCol(df, 'school', 'group')
write_parquet(df, path('parquet', 'chem97.parquet'))

# ---------------------------------------------------------------------------
# Generalized mixed-effects (non-Gaussian outcomes)

# ---------------------------------------------------------------------------
# Generalized mixed-effects (non-Gaussian outcomes)

# CBPP (d=2, binomial)
data('cbpp', package = 'lme4')
df <- cbpp
# model <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), family = binomial, data = df)
# 'size' is the binomial denominator (total animals at risk), not a fixed-effect predictor
df <- df |> select(-c('size'))
df <- renameCol(df, 'incidence', 'y')
df <- renameCol(df, 'herd', 'group')
write_parquet(df, path('parquet', 'cbpp.parquet'))

# VerbAgg (d=4, binomial)
data('VerbAgg', package = 'lme4')
df <- VerbAgg
# model <- glmer(r2 ~ anger + gender + btype + situ + (1 | id), family = binomial, data = df)
# 'resp' is a 3-level version of the binary target (perfect data leakage); 'item' is a
# random effect (24-level stimulus ID), not a fixed predictor
df <- df |> select(-c('item', 'resp'))
df <- renameCol(df, 'r2', 'y')
df$y <- as.integer(as.character(df$y) == 'Y')
df <- renameCol(df, 'id', 'group')
write_parquet(df, path('parquet', 'verbagg.parquet'))

# Grouseticks (d=3, count)
data('grouseticks', package = 'lme4')
df <- grouseticks
# model <- glmer(TICKS ~ YEAR + HEIGHT + (1 | BROOD), family = poisson, data = df)
# 'INDEX' is a row ID; 'cHEIGHT' is centered HEIGHT (r=1); 'LOCATION' is not in reference model
df <- df |> select(-c('INDEX', 'cHEIGHT', 'LOCATION'))
df <- renameCol(df, 'TICKS', 'y')
df <- renameCol(df, 'BROOD', 'group')
write_parquet(df, path('parquet', 'grouseticks.parquet'))

# Contraception (d=4, binomial)
data('Contraception', package = 'mlmRev')
df <- Contraception
# model <- glmer(use ~ age + urban + livch + (1 | district), family = binomial, data = df)
# 'woman' is a person-level row ID, not a predictor
df <- df |> select(-c('woman'))
df <- renameCol(df, 'use', 'y')
df$y <- as.integer(as.character(df$y) == 'Y')
df <- renameCol(df, 'district', 'group')
write_parquet(df, path('parquet', 'contraception.parquet'))

# InstEval (d=3, grouped ratings)
data('InstEval', package = 'lme4')
df <- InstEval
# model <- lmer(y ~ service + dept + (1 | s), data = df)
# 'd' is the instructor ID (1128 levels, random effect); 'studage' and 'lectage' are not
# in the reference model
df <- df |> select(-c('d', 'studage', 'lectage'))
df <- renameCol(df, 's', 'group')
write_parquet(df, path('parquet', 'insteval.parquet'))

# ---------------------------------------------------------------------------
# Nonlinear mixed-effects (NLME)

# Theoph (PK)
data('Theoph', package = 'datasets')
df <- Theoph
# model <- nlme(conc ~ SSfol(Dose, Time, lKe, lKa, lCl), data = df, fixed = lKe + lKa + lCl ~ 1, random = lKe + lKa + lCl ~ 1 | Subject)
df <- renameCol(df, 'conc', 'y')
df <- renameCol(df, 'Subject', 'group')
write_parquet(df, path('parquet', 'theoph.parquet'))

# Orange (growth)
data('Orange', package = 'datasets')
df <- Orange
# model <- nlme(circumference ~ SSlogis(age, Asym, xmid, scal), data = df, fixed = Asym + xmid + scal ~ 1, random = Asym + xmid + scal ~ 1 | Tree)
df <- renameCol(df, 'circumference', 'y')
df <- renameCol(df, 'Tree', 'group')
write_parquet(df, path('parquet', 'orange.parquet'))

# Indometh (PK)
data('Indometh', package = 'datasets')
df <- Indometh
# model <- nlme(conc ~ SSbiexp(time, A1, lrc1, A2, lrc2), data = df, fixed = A1 + lrc1 + A2 + lrc2 ~ 1, random = A1 + lrc1 + A2 + lrc2 ~ 1 | Subject)
df <- renameCol(df, 'conc', 'y')
df <- renameCol(df, 'Subject', 'group')
write_parquet(df, path('parquet', 'indometh.parquet'))
