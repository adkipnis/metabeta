# Rebuttals Table 5 - Real-world datasets from R packages

Datasets used as the held-out real-world test suite. All 23 datasets were sourced from standard R mixed-effects packages ([`lme4`](https://cran.r-project.org/package=lme4), [`nlme`](https://cran.r-project.org/package=nlme), [`mlmRev`](https://cran.r-project.org/package=mlmRev), [`MEMSS`](https://cran.r-project.org/package=MEMSS), [`datasets`](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html)), loaded via [`metabeta/datasets/from-r/package-datasets.R`](../../metabeta/datasets/from-r/package-datasets.R) and preprocessed via [`metabeta/datasets/preprocess.py`](../../metabeta/datasets/preprocess.py).

*d* = number of fixed effects including intercept (after dummy-coding categoricals). *m* = groups. *n* = total observations. *n/group* = min–max observations per group. *H* = group-size entropy ratio $^1$.



## Datasets

| Dataset | Package | Outcome | Likelihood | *d* | *m* | *n* | *n*/group | *H* |
|:---|:---|:---|:---|---:|---:|---:|:---|---:|
| [cbpp](https://rdrr.io/cran/lme4/man/cbpp.html) | [lme4](https://cran.r-project.org/package=lme4) | CBPP incidence | Binomial | 5 | 15 | 56 | 1–4 | 0.989 |
| [Chem97](https://rdrr.io/cran/mlmRev/man/Chem97.html) | [mlmRev](https://cran.r-project.org/package=mlmRev) | A-level chemistry score | Gaussian | 5 | 2409 | 30999 | 1–188 | 0.946 |
| [Contraception](https://rdrr.io/cran/mlmRev/man/Contraception.html) | [mlmRev](https://cran.r-project.org/package=mlmRev) | contraceptive use | Binomial | 6 | 60 | 1934 | 2–118 | 0.948 |
| [Dyestuff](https://rdrr.io/cran/lme4/man/Dyestuff.html) | [lme4](https://cran.r-project.org/package=lme4) | dye yield (g) | Gaussian | 1 | 6 | 30 | 5–5 | 1.000 |
| [ergoStool](https://rdrr.io/cran/nlme/man/ergoStool.html) | [nlme](https://cran.r-project.org/package=nlme) | effort rating | Gaussian | 4 | 9 | 36 | 4–4 | 1.000 |
| [Exam (London)](https://rdrr.io/cran/mlmRev/man/Exam.html) | [mlmRev](https://cran.r-project.org/package=mlmRev) | normalised exam score | Gaussian | 11 | 65 | 4059 | 2–198 | 0.974 |
| [Gcsemv (GCSE)](https://rdrr.io/cran/mlmRev/man/Gcsemv.html) | [mlmRev](https://cran.r-project.org/package=mlmRev) | written GCSE score | Gaussian | 3 | 73 | 1523 | 1–83 | 0.933 |
| [grouseticks](https://rdrr.io/cran/lme4/man/grouseticks.html) | [lme4](https://cran.r-project.org/package=lme4) | tick count on grouse chicks | Poisson | 5 | 118 | 403 | 1–10 | 0.969 |
| [Hsb82](https://rdrr.io/cran/mlmRev/man/Hsb82.html) | [mlmRev](https://cran.r-project.org/package=mlmRev) | math achievement (HSB) | Gaussian | 7 | 160 | 7182 | 14–67 | 0.993 |
| [Indometh](https://rdrr.io/r/datasets/Indometh.html) | [datasets](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html) | indomethacin concentration | Gaussian | 2 | 6 | 66 | 11–11 | 1.000 |
| [InstEval](https://rdrr.io/cran/lme4/man/InstEval.html) | [lme4](https://cran.r-project.org/package=lme4) | instructor rating | Gaussian | 10 | 2972 | 73421 | 1–92 | 0.978 |
| [Machines](https://rdrr.io/cran/nlme/man/Machines.html) | [nlme](https://cran.r-project.org/package=nlme) | productivity score | Gaussian | 3 | 6 | 54 | 9–9 | 1.000 |
| [MathAchieve](https://rdrr.io/cran/MEMSS/man/MathAchieve.html) | [MEMSS](https://cran.r-project.org/package=MEMSS) | math achievement score | Gaussian | 5 | 160 | 7184 | 14–67 | 0.993 |
| [Oats](https://rdrr.io/cran/nlme/man/Oats.html) | [nlme](https://cran.r-project.org/package=nlme) | oat yield | Gaussian | 4 | 6 | 72 | 12–12 | 1.000 |
| [Orange](https://rdrr.io/r/datasets/Orange.html) | [datasets](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html) | orange tree circumference | Gaussian | 2 | 5 | 35 | 7–7 | 1.000 |
| [Orthodont](https://rdrr.io/cran/nlme/man/Orthodont.html) | [nlme](https://cran.r-project.org/package=nlme) | orthodontic distance (mm) | Gaussian | 3 | 27 | 108 | 4–4 | 1.000 |
| [Oxboys](https://rdrr.io/cran/nlme/man/Oxboys.html) | [nlme](https://cran.r-project.org/package=nlme) | height (cm) | Gaussian | 10 | 26 | 234 | 9–9 | 1.000 |
| [Pastes](https://rdrr.io/cran/lme4/man/Pastes.html) | [lme4](https://cran.r-project.org/package=lme4) | paste strength | Gaussian | 3 | 10 | 60 | 6–6 | 1.000 |
| [Penicillin](https://rdrr.io/cran/lme4/man/Penicillin.html) | [lme4](https://cran.r-project.org/package=lme4) | plate diameter (mm) | Gaussian | 6 | 24 | 144 | 6–6 | 1.000 |
| [Pixel](https://rdrr.io/cran/nlme/man/Pixel.html) | [nlme](https://cran.r-project.org/package=nlme) | pixel intensity | Gaussian | 3 | 10 | 102 | 4–14 | 0.972 |
| [sleepstudy](https://rdrr.io/cran/lme4/man/sleepstudy.html) | [lme4](https://cran.r-project.org/package=lme4) | reaction time (ms) | Gaussian | 2 | 18 | 180 | 10–10 | 1.000 |
| [Theoph](https://rdrr.io/r/datasets/Theoph.html) | [datasets](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html) | theophylline concentration | Gaussian | 4 | 12 | 132 | 11–11 | 1.000 |
| [VerbAgg](https://rdrr.io/cran/lme4/man/VerbAgg.html) | [lme4](https://cran.r-project.org/package=lme4) | verbal aggression response | Binomial | 9 | 316 | 7584 | 24–24 | 1.000 |

---
[1]:   $H = \frac{-\sum_{i=1}^{m} p_i \log p_i}{\log m}, \quad p_i = \frac{n_i}{n}$ where $H = 1$ indicates perfectly balanced groups and $H \to 0$ indicates extreme imbalance.
