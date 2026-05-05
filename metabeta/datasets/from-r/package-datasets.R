# install missing packages
options(repos = c(CRAN = "https://cloud.r-project.org"))
packages <- c(
  "rstudioapi", "fs", "arrow", "lme4", "nlme",
  "mlmRev", "MEMSS", "MASS", "boot", "gamlss.data",
  "glmmTMB", "geepack", "dplyr", "tidyr"
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

expandBinomial <- function(df, success, total){
  df$successes <- df[[success]]
  df$trials <- df[[total]]
  df <- df |> uncount(trials, .id = 'trial')
  df$y <- as.integer(df$trial <= df$successes)
  df <- df |> select(-c(all_of(success), all_of(total), 'successes', 'trial'))
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
# 'sample' is a nested row-level ID (30 unique / 60 rows); not a predictor
df <- df |> select(-c('sample'))
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
# Generalized mixed-effects (Bernoulli outcomes)

# CBPP (d=2, Bernoulli from aggregate binomial)
data('cbpp', package = 'lme4')
df <- cbpp
# model <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), family = binomial, data = df)
# expand aggregate binomial counts to Bernoulli rows; 'size' is the denominator, not a predictor
df <- expandBinomial(df, 'incidence', 'size')
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

# Bacteria (d=5, Bernoulli)
data('bacteria', package = 'MASS')
df <- bacteria
# model <- glmer(y ~ ap + hilo + week + trt + (1 | ID), family = binomial, data = df)
df$y <- as.integer(as.character(df$y) == 'y')
df <- renameCol(df, 'ID', 'group')
write_parquet(df, path('parquet', 'bacteria.parquet'))

# Guatemala immunization (d=16, Bernoulli)
data('guImmun', package = 'mlmRev')
df <- guImmun
# model <- glmer(immun ~ kid2p + mom25p + ord + ethn + momEd + husEd + momWork + rural + pcInd81 + (1 | comm), family = binomial, data = df)
# 'kid' and 'mom' are nested person/family IDs, not fixed-effect predictors
df <- df |> select(-c('kid', 'mom'))
df <- renameCol(df, 'immun', 'y')
df$y <- as.integer(as.character(df$y) == 'Y')
df <- renameCol(df, 'comm', 'group')
write_parquet(df, path('parquet', 'guimmun.parquet'))

# Guatemala prenatal care (d=22, Bernoulli)
data('guPrenat', package = 'mlmRev')
df <- guPrenat
# model <- glmer(prenat ~ childAge + motherAge + birthOrd + indig + momEd + husEd + husEmpl + toilet + TV + pcInd81 + ssDist + (1 | cluster), family = binomial, data = df)
# 'kid' and 'mom' are nested person/family IDs, not fixed-effect predictors
df <- df |> select(-c('kid', 'mom'))
df <- renameCol(df, 'prenat', 'y')
df$y <- as.integer(as.character(df$y) == 'Modern')
df <- renameCol(df, 'cluster', 'group')
write_parquet(df, path('parquet', 'guprenat.parquet'))

# OME (d=6, Bernoulli from aggregate binomial)
data('OME', package = 'MASS')
df <- OME
# model <- glmer(cbind(Correct, Trials - Correct) ~ Age + OME + Loud + Noise + (1 | ID), family = binomial, data = df)
# expand aggregate binomial counts to Bernoulli rows; 'Trials' is the denominator
df <- expandBinomial(df, 'Correct', 'Trials')
df <- renameCol(df, 'ID', 'group')
write_parquet(df, path('parquet', 'ome.parquet'))

# Respiratory infection (d=10, Bernoulli)
data('respInf', package = 'gamlss.data')
df <- respInf
# model <- glmer(time ~ time.1 + age + xero + cosine + sine + female + height + stunted + (1 | id), family = binomial, data = df)
# 'resp' is a vector of ones; 'age1', 'season', and 'time2' are derived/redundant
df <- df |> select(-c('resp', 'age1', 'season', 'time2'))
df <- renameCol(df, 'time', 'y')
df <- renameCol(df, 'id', 'group')
write_parquet(df, path('parquet', 'respinf.parquet'))

# Sugar-cane disease (d=3, Bernoulli from aggregate binomial)
data('cane', package = 'boot')
df <- cane
# model <- glmer(cbind(r, n - r) ~ x + block + (1 | var), family = binomial, data = df)
# expand aggregate binomial counts to Bernoulli rows; 'n' is the denominator
df <- expandBinomial(df, 'r', 'n')
df <- renameCol(df, 'var', 'group')
write_parquet(df, path('parquet', 'cane.parquet'))

# Ohio children wheeze status (d=3, Bernoulli)
data('ohio', package = 'geepack')
df <- ohio
# model <- glmer(resp ~ age + smoke + (1 | id), family = binomial, data = df)
df <- renameCol(df, 'resp', 'y')
df <- renameCol(df, 'id', 'group')
write_parquet(df, path('parquet', 'ohio.parquet'))

# Respiratory illness trial (d=7, Bernoulli)
data('respiratory', package = 'geepack')
df <- respiratory
# model <- glmer(outcome ~ center + treat + sex + age + baseline + visit + (1 | id), family = binomial, data = df)
df <- renameCol(df, 'outcome', 'y')
df <- renameCol(df, 'id', 'group')
write_parquet(df, path('parquet', 'respiratory.parquet'))

# Muscatine obesity (d=5, Bernoulli)
data('muscatine', package = 'geepack')
df <- muscatine
# model <- glmer(obese ~ gender + base_age + age + occasion + (1 | id), family = binomial, data = df)
# 'numobese' is a subject-level summary of the repeated binary outcome
df <- df |> select(-c('numobese'))
df <- renameCol(df, 'obese', 'y')
df$y <- as.integer(as.character(df$y) == 'yes')
df <- renameCol(df, 'id', 'group')
write_parquet(df, path('parquet', 'muscatine.parquet'))

# ---------------------------------------------------------------------------
# Generalized mixed-effects (Poisson outcomes)

# Grouseticks (d=3, Poisson)
data('grouseticks', package = 'lme4')
df <- grouseticks
# model <- glmer(TICKS ~ YEAR + HEIGHT + (1 | BROOD), family = poisson, data = df)
# 'INDEX' is a row ID; 'cHEIGHT' is centered HEIGHT (r=1); 'LOCATION' is not in reference model
df <- df |> select(-c('INDEX', 'cHEIGHT', 'LOCATION'))
df <- renameCol(df, 'TICKS', 'y')
df <- renameCol(df, 'BROOD', 'group')
write_parquet(df, path('parquet', 'grouseticks.parquet'))

# Epilepsy seizures (d=5, Poisson)
data('epil', package = 'MASS')
df <- epil
# model <- glmer(y ~ trt + V4 + lbase + lage + (1 | subject), family = poisson, data = df)
# 'base', 'age', and 'period' are redundant with the transformed/model covariates
df <- df |> select(-c('base', 'age', 'period'))
df <- renameCol(df, 'subject', 'group')
write_parquet(df, path('parquet', 'epil.parquet'))

# Arabidopsis fruits (d=5, Poisson)
data('Arabidopsis', package = 'lme4')
df <- Arabidopsis
# model <- glmer(total.fruits ~ nutrient + amd + status + (1 | popu), family = poisson, data = df)
# 'reg', 'gen', and 'rack' are nesting/blocking IDs, not fixed-effect predictors
df <- df |> select(-c('reg', 'gen', 'rack'))
df <- renameCol(df, 'total.fruits', 'y')
df <- renameCol(df, 'popu', 'group')
write_parquet(df, path('parquet', 'arabidopsis.parquet'))

# Malignant melanoma deaths in Europe (d=3, Poisson)
data('Mmmec', package = 'mlmRev')
df <- Mmmec
# model <- glmer(deaths ~ uvb + offset(log(expected)) + (1 | nation), family = poisson, data = df)
# offset terms are not represented in the current parquet schema, so 'expected' is kept
# as an ordinary predictor for now; 'region' and 'county' are nested location IDs
df <- df |> select(-c('region', 'county'))
df <- renameCol(df, 'deaths', 'y')
df <- renameCol(df, 'nation', 'group')
write_parquet(df, path('parquet', 'mmmec.parquet'))

# Salamanders (d=8, Poisson)
data('Salamanders', package = 'glmmTMB')
df <- Salamanders
# model <- glmer(count ~ spp * mined + cover + sample + DOP + Wtemp + DOY + (1 | site), family = poisson, data = df)
df <- renameCol(df, 'count', 'y')
df <- renameCol(df, 'site', 'group')
write_parquet(df, path('parquet', 'salamanders.parquet'))

# Owl sibling negotiation (d=5, Poisson)
data('Owls', package = 'glmmTMB')
df <- Owls
# model <- glmer(SiblingNegotiation ~ FoodTreatment * SexParent + ArrivalTime + logBroodSize + (1 | Nest), family = poisson, data = df)
# 'NegPerChick' is derived from the outcome and brood size; 'BroodSize' is redundant with
# 'logBroodSize', which approximates the common offset term as a predictor for now
df <- df |> select(-c('NegPerChick', 'BroodSize'))
df <- renameCol(df, 'SiblingNegotiation', 'y')
df <- renameCol(df, 'Nest', 'group')
write_parquet(df, path('parquet', 'owls.parquet'))

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
