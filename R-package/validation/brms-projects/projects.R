# The projects: where their code and data come from, which document runs and how far.
#
# Nothing of the projects is stored here. fetch.R downloads their files into
# work/<name>/, under the paths they have in the project, from a fixed commit of its
# GitHub repository or from its OSF project.
#
# Each entry has
#   title, url  the project and where it lives
#   reported    where the numbers in reported/<name>.csv come from
#   github      repository and commit that `files` and `dirs` come from
#   files       files to download (GitHub paths), or, with `osf`, local path = OSF file id
#   dirs        directories of the repository to download in full
#   document    the R Markdown or Quarto document that runs, in its own directory as when
#               it is knitted
#   until       stop after the first chunk whose code matches this regular expression
#   skip        chunks whose code matches this regular expression do not run
#   mkdir       directories the document saves fits or results into
#   prelude     code that runs before the document (a book's setup from earlier chapters)
#   changes     replacements in the document that have nothing to do with brms (old = new)

projects <- list(
  kurz_sr2 = list(
    title = "Kurz, Statistical Rethinking with brms, ggplot2, and the tidyverse (2nd ed.), ch. 14",
    url = "https://github.com/ASKurz/Statistical_Rethinking_with_brms_ggplot2_and_the_tidyverse_2_ed",
    reported = "the rendered chapter (_book/14.html)",
    github = c(repo = "ASKurz/Statistical_Rethinking_with_brms_ggplot2_and_the_tidyverse_2_ed",
               commit = "df64d9cb8d50ac850ce24cb277e95ccb88fad0a8"),
    files = "14.qmd",
    document = "14.qmd",
    until = "(?m)^coef\\(b14\\.1\\)$",
    mkdir = "fits"
  ),
  kurz_dbda = list(
    title = "Kurz, Doing Bayesian Data Analysis in brms and the tidyverse, ch. 9",
    url = "https://github.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse",
    reported = "the rendered chapter (_book/09.html)",
    github = c(repo = "ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse",
               commit = "f90517976b2f52cc9b63c7c5e151e85b1d677e63"),
    files = c("09.qmd", "data.R/TherapeuticTouchData.csv"),
    document = "09.qmd",
    until = "(?m)^print\\(fit9\\.1\\)",
    mkdir = "fits"
  ),
  clark = list(
    title = "Clark, Mixed Models with R, ch. Bayesian approaches",
    url = "https://github.com/m-clark/mixed-models-with-R",
    reported = "the rendered chapter (docs/bayesian.html)",
    github = c(repo = "m-clark/mixed-models-with-R", commit = "aea3c381604e4c2c9f3f162a2a8c447c607039da"),
    files = c("bayesian.Rmd", "data/gpa.RData"),
    document = "bayesian.Rmd",
    until = "(?m)^pp_check\\(bayesian_mixed\\)",
    # what the chapter takes from the book's earlier chapters
    prelude = c(
      "library(tidyverse)",
      "library(lme4)",
      "load('data/gpa.RData')",
      "gpa_mixed =  lmer(gpa ~ occasion + (1 + occasion | student), data = gpa)"
    )
  ),
  vandeschoot = list(
    title = "van de Schoot et al., BRMS tutorial: building a multilevel model",
    url = "https://github.com/Rensvandeschoot/Tutorials",
    reported = "the rendered tutorial (multilevel/Tutorial-BRMS.md)",
    github = c(repo = "Rensvandeschoot/Tutorials", commit = "bbebd147b0a63ff984586d400ad59be70c59889f"),
    files = "multilevel/Tutorial BRMS.Rmd",
    document = "multilevel/Tutorial BRMS.Rmd",
    # the repository the tutorial loads its data from no longer exists; a fork has the file
    changes = c(
      "https://github.com/MultiLevelAnalysis/Datasets-third-edition-Multilevel-book/blob/master/" =
        "https://github.com/ElianGriffioen/Datasets-third-edition-Multilevel-book/blob/master/"
    )
  ),
  heiss = list(
    title = "Heiss, Bayes Rules! notebook, ch. 18 (logistic regression of climbing success)",
    url = "https://github.com/andrewheiss/bayesf22-notebook",
    reported = "the chapter's Quarto freeze (_freeze/bayes-rules/18-chapter/execute-results/html.json)",
    github = c(repo = "andrewheiss/bayesf22-notebook", commit = "fbffb638fe3fa4e36171f69d392e987a038db588"),
    files = "bayes-rules/18-chapter.qmd",
    document = "bayes-rules/18-chapter.qmd",
    mkdir = "bayes-rules/18-manual_cache"
  ),
  tal = list(
    title = "Tal et al. (2026), Open Mind, experiment 2",
    url = "https://osf.io/y695j",
    reported = "the paper's Tables 7-10",
    osf = c("Exp2.Rmd" = "t5h7w", "data/exp2_kids.csv" = "69417e43578b1ae48c02d0b3",
            "data/exp2_adults.csv" = "r6vcu"),
    document = "Exp2.Rmd"
  ),
  zhu = list(
    title = "Zhu et al. (2024), Open Mind",
    url = "https://osf.io/cpk92",
    reported = "the rendered analysis on OSF (Data & Analyses/Final Analyses/Final Analyses.pdf)",
    osf = c("Final Analyses.Rmd" = "bcp8v", "RawDataOSF_NoToss.xlsx" = "qjeu4"),
    document = "Final Analyses.Rmd",
    mkdir = "cache"
  ),
  tango = list(
    title = "Prein et al. (2023), Behavior Research Methods (TANGO)",
    url = "https://github.com/ccp-eva/gazecues-methods",
    reported = "the manuscript (paper/manuscript.tex)",
    github = c(repo = "ccp-eva/gazecues-methods", commit = "6f048dbdbf863a350ddd9a47b28c9c82452a2c41"),
    files = c("stats/analysis.Rmd", "data/tango-testtrials.rds"),
    document = "stats/analysis.Rmd",
    until = "(?m)saveRDS\\(mRetestBox",
    # split-half reliabilities: thousands of refits, none of them reported as a brms summary
    skip = "by_split\\(|split_coefs\\(|split_ci\\(|mclapply\\(",
    mkdir = "saves"
  ),
  ma = list(
    title = "Ma et al. (2026), PsyArXiv: what makes a decision good",
    url = "https://github.com/cicl-stanford/good_decisions",
    reported = "the paper's Table 4",
    github = c(repo = "cicl-stanford/good_decisions", commit = "e12f73bdb2aa8ed4c4e1dc49abd3251e404e722b"),
    files = "code/R/good_decisions_analysis.Rmd",
    dirs = "data",
    document = "code/R/good_decisions_analysis.Rmd",
    mkdir = "code/R/models"
  ),
  rose = list(
    title = "Rose et al. (2023), CogSci: teleological properties",
    url = "https://github.com/cicl-stanford/teleological_properties",
    reported = "the rendered analysis (code/R/teleological_properties.html)",
    github = c(repo = "cicl-stanford/teleological_properties", commit = "6d9353957f262bb9fe50f5c33b92e50d03f6c7ab"),
    files = c(
      "code/R/teleological_properties.Rmd",
      file.path("data/experiment2", c("experiment2.csv", "experiment2_demographics.csv",
        "bert_base_property_probabilities.csv", "bert_large_property_probabilities.csv",
        "roberta_large_property_probabilities.csv")),
      file.path("data/experiment3", c("experiment3.csv", "experiment3_demographics.csv")),
      file.path("data/experiment4", c("experiment4.csv", "experiment4_demographics.csv"))
    ),
    document = "code/R/teleological_properties.Rmd",
    mkdir = "code/R/cache"
  ),
  haux = list(
    title = "Haux et al. (2025), Nature Communications: adaptive exploration",
    url = "https://github.com/louhaux/adaptive_exploration",
    reported = "the supplementary information, Table S1",
    github = c(repo = "louhaux/adaptive_exploration", commit = "bfc1c5075cb2be5fd21805d7b5ea220d91f03c47"),
    files = file.path("analysis_experiment", c(
      "experiment_analysis.Rmd", "SourceData_S01_adaptive_exploration_chimpanzees.xlsx",
      "SourceData_S02_adaptive_exploration_data_experiment.txt", "data_risksearch_choice.txt",
      "data_risksearch_exploration_strategy.txt", "data_risksearch_grouped_correlations.txt"
    )),
    document = "analysis_experiment/experiment_analysis.Rmd",
    until = "(?m)conditional_effects\\(m3\\.2\\)"
  )
)
