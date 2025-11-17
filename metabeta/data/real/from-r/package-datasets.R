library(rstudioapi)
library(lme4)
library(nlme)
library(mlmRev)
library(MEMSS)
library(nycflights13)
library(dplyr)

rename <- function(df, old,  new){
  colnames(df)[colnames(df) == old] <- new
  return(df)
}

# set cwd and create subdir
setwd(dirname(getActiveDocumentContext()$path))
dir.create("csv", showWarnings = F)

# MathAchieve (d=5, q=1)
data("MathAchieve")
selection <- c("MathAch", "SES", "MEANSES", "Minority", "Sex", "School")
df <- MathAchieve |> select(all_of(selection))
model <- lmer(MathAch ~ SES + MEANSES + Minority + Sex  + (1 | School), data = df)
df <- rename(df, 'MathAch', 'y')
df <- rename(df, 'School', 'group')
write.csv(df, "csv/math.csv", row.names = FALSE)

# Exam (d=4, q=1)
data("Exam")
summary(Exam)
selection <- c("normexam", "schavg" , "standLRT", "sex", "school")
df <- Exam |> select(all_of(selection))
model <- lmer(normexam ~ schavg + standLRT + sex + (1 | school), data = df)
df <- rename(df, 'normexam', 'y')
df <- rename(df, 'school', 'group')
write.csv(df, "csv/london.csv", row.names = FALSE)

# GCS (d=3, q=1)
data("Gcsemv")
selection <- c("written", "gender", "course", "school")
df <- Gcsemv |> select(all_of(selection))
model <- lmer(written ~ gender + course + (1 | school), data = df)
df <- rename(df, 'written', 'y')
df <- rename(df, 'school', 'group')
write.csv(df, "csv/gcse.csv", row.names = FALSE)

# Sleep study (d=2, q=2)
data("sleepstudy")
selection <- c("Reaction", "Days", "Subject")
df <- sleepstudy |> select(all_of(selection))
model <- lmer(Reaction ~ Days + (Days | Subject), data = df)
df <- rename(df, 'Reaction', 'y')
df <- rename(df, 'Subject', 'group')
write.csv(df, "csv/sleep.csv", row.names = FALSE)

