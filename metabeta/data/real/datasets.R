library(lme4)
library(nlme)
library(mlmRev)
library(MEMSS)
library(nycflights13)
library(dplyr)

center <- function(df, cols){
  df[cols] <- lapply(df[cols], function(x) scale(x, center = TRUE, scale = FALSE))
  return(df)
}


# MathAchieve (d=5, q=1)
data("MathAchieve")
selection <- c("MathAch", "SES", "MEANSES", "Minority", "Sex", "School")
df <- MathAchieve |> select(all_of(selection))
df <- center(df, c('SES', 'MEANSES'))
model <- lmer(MathAch ~ SES + MEANSES + Minority + Sex  + (1 | School), data = df)
write.csv(df, "math.csv", row.names = FALSE)

# Exam (d=4, q=1)
data("Exam")
summary(Exam)
selection <- c("normexam", "schavg" , "standLRT", "sex", "school")
df <- Exam |> select(all_of(selection))
df <- center(df, c('standLRT'))
model <- lmer(normexam ~ schavg + standLRT + sex + (1 | school), data = df)
write.csv(df, "exam.csv", row.names = FALSE)

# GCS (d=3, q=1)
data("Gcsemv")
selection <- c("written", "gender", "course", "school")
df <- Gcsemv |> select(all_of(selection))
df <- center(df, c('course'))
model <- lmer(written ~ gender + course + (1 | school), data = df)
write.csv(df, "gcsemv.csv", row.names = FALSE)

# Sleep study (d=2, q=2)
data("sleepstudy")
selection <- c("Reaction", "Days", "Subject")
df <- sleepstudy |> select(all_of(selection))
model <- lmer(Reaction ~ Days + (Days | Subject), data = df)
write.csv(df, "sleepstudy.csv", row.names = FALSE)

# # Phenobarb 
# data("Phenobarb")
# selection <- c("conc", "time", "Subject")
# df <- Phenobarb |> select(all_of(selection))
# model <- lmer(conc ~ time + (time | Subject), data = df)
# write.csv(df, "phenobarb.csv", row.names = FALSE)


# NYC Flights
# data("flights")
# selection <- c("dep_delay", "carrier", "origin", "sched_dep_time", "distance", "day", "origin")
# df <- flights |> select(all_of(selection))
# model <- lmer(
#   dep_delay ~ carrier + origin + sched_dep_time + distance + day + (1 | origin), data = df
# )
# write.csv(df, "flights.csv", row.names = FALSE)

