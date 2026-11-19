# metabeta in place of brms in published analyses

Eleven published analyses with open code and data fit their models with brms and report
the results. Here each one runs its own code with a single change, `library(brms)` →
`library(metabeta)` (or `brms` → `metabeta` in `pacman::p_load()`), and the fits are
compared with the numbers the project reported. brms itself is not rerun.

## Projects

| Name | Project | Models compared | Reported numbers from |
|---|---|---|---|
| `kurz_sr2` | [Kurz, *Statistical Rethinking* with brms](https://github.com/ASKurz/Statistical_Rethinking_with_brms_ggplot2_and_the_tidyverse_2_ed), ch. 14 | 1 gaussian, correlated slopes | rendered chapter |
| `kurz_dbda` | [Kurz, *Doing Bayesian Data Analysis* in brms](https://github.com/ASKurz/Doing-Bayesian-Data-Analysis-in-brms-and-the-tidyverse), ch. 9 | 1 bernoulli | rendered chapter |
| `clark` | [Clark, *Mixed Models with R*](https://github.com/m-clark/mixed-models-with-R), Bayesian chapter | 1 gaussian | rendered chapter |
| `vandeschoot` | [van de Schoot et al., BRMS tutorial](https://github.com/Rensvandeschoot/Tutorials) | 7 gaussian | rendered tutorial |
| `heiss` | [Heiss, *Bayes Rules!* notebook](https://github.com/andrewheiss/bayesf22-notebook), ch. 18 | 1 bernoulli | Quarto freeze of the chapter |
| `tal` | [Tal et al. (2026)](https://osf.io/y695j), *Open Mind*, exp. 2 | 4 bernoulli | paper, Tables 7–10 |
| `zhu` | [Zhu et al. (2024)](https://osf.io/cpk92), *Open Mind* | 14 bernoulli | rendered analysis on OSF |
| `tango` | [Prein et al. (2023)](https://github.com/ccp-eva/gazecues-methods), *Behavior Research Methods* | 2 bernoulli | manuscript |
| `ma` | [Ma et al. (2026)](https://github.com/cicl-stanford/good_decisions), PsyArXiv | 8 gaussian | paper, Table 4 |
| `rose` | [Rose et al. (2023)](https://github.com/cicl-stanford/teleological_properties), CogSci | 3 gaussian | rendered analysis |
| `haux` | [Haux et al. (2025)](https://github.com/louhaux/adaptive_exploration), *Nature Communications* | 3 gaussian | supplementary Table S1 |

`projects.R` pins each GitHub project to a commit and lists every file used.

## How a project runs

* The R chunks of the project's document run in order, in the document's folder, as
  when it is knitted. Chunks with `eval = FALSE` don't run.
* The only change to the code is the swap of brms for metabeta, with two exceptions
  that have nothing to do with brms:
  * The data that van de Schoot et al. load come from a GitHub repository that no longer
    exists, so the URL points to a fork with the same file.
  * Clark's chapter builds on objects from the book's earlier chapters, so their setup
    runs first.
* Some runs stop after the last reported model. TANGO skips its split-half reliability
  analysis, which refits models thousands of times and reports no brms summaries.
* Projects that cache fits with `brm(file = ...)` refit instead of loading a saved fit.

A reported number **matches** when metabeta's posterior mean and SD are within 0.25
reported posterior SDs of the reported values, and the interval bounds within 0.35 SDs,
after allowing for the report's rounding. Where a project reports no SD, metabeta's SD
is the scale.

## Results

Matching reported numbers by whether metabeta warned that its draws may be under-dispersed
(`results/summary.csv`; one row of `results/<name>.csv` per number):

| Name | Numbers | Matched, no warning | Matched, warned | Not fitted |
|---|---:|---:|---:|---:|
| `kurz_sr2` | 40 | 40 / 40 | | |
| `kurz_dbda` | 2 | 2 / 2 | | |
| `clark` | 16 | 16 / 16 | | |
| `vandeschoot` | 44 | 24 / 26 | 3 / 8 | 10 |
| `heiss` | 4 | | 0 / 4 | |
| `tal` | 22 | 4 / 8 | 1 / 14 | |
| `zhu` | 35 | 13 / 18 | 5 / 10 | 7 |
| `tango` | 5 | | 2 / 5 | |
| `ma` | 40 | 39 / 40 | | |
| `rose` | 30 | | 11 / 30 | |
| `haux` | 14 | 10 / 10 | | 4 |
| **all** | **252** | **148 / 160** | **22 / 71** | **21** |

* **Without a warning, 93% of the numbers match**, with a median distance of 0.04
  posterior SDs. Of the 12 that don't, 9 are Bernoulli effects whose posterior is
  narrower than brms's (SD ratio 0.73–0.92), and effects far from zero on the logit scale
  are also pulled toward it (e.g. `tal`, adults: 1.82 against 2.04). This is the prior,
  not the fit: brms puts flat priors on regression coefficients, metabeta weakly
  informative ones. The other 3 lie within 0.4 SDs.
* **With a warning, 31% match.** The warning means what it says: the draws are too
  narrow, and in `heiss` (a nearly separated outcome; acceptance 0) the means are off by 2
  to 3.4 SDs. With `control = list(random_walk = TRUE)`, which the warning suggests,
  `vandeschoot`'s model5 matches brms.
* **Not fitted:** 21 numbers come from models outside the range metabeta was trained on,
  e.g. three random slopes with few population-level effects, or groups with fewer than 5
  observations. `brm()` stops with an error that says so.

The numbers come from runs on 25 October 2026 with metabeta 0.6.0. The first four
projects ran with `run.R`. For the others, `compare.R` evaluated the fits of an earlier
run with the same harness, before it moved into this folder; a rerun of all projects was
stopped when the machine ran short of memory.

## Files

| File | Contents |
|---|---|
| `projects.R` | the projects: sources, document, how far it runs, other changes |
| `fetch.R` | downloads a project's files into `work/<name>/` |
| `harness.R` | runs a project's document with metabeta in place of brms |
| `compare.R` | compares the fits with `reported/<name>.csv`, writes `results/<name>.csv` |
| `summarize.R` | sums up all results in `results/summary.csv` |
| `run.R` | all of the above, for all projects or the ones named |
| `reported/` | the numbers each project reported, one row per parameter |
| `results/` | the comparison of the last run, one row per reported number |

`reported/<name>.csv` has the columns `model` (the fit's name in the project's code),
`parameter` (a draws variable such as `b_x` or `sd_g__Intercept`, `coef_g[level,x]` for
`coef()`, or `expr:` followed by an R expression of draws), `estimate`, `error` (the
posterior SD, `NA` if not reported), `lower`, `upper` and `level` (`0.95` for a central
interval, `hdi0.95` for a highest-density interval).

## Running it

From the `R-package` folder:

```sh
Rscript validation/brms-projects/run.R                 # all projects
Rscript validation/brms-projects/run.R tal zhu         # some of them
Rscript validation/brms-projects/run.R --work=/tmp/w   # download and run elsewhere
```

Downloads, fits and the run logs (`work/<name>/run.log`) go to `work/`, which git
ignores. The projects need the R packages they load (a log line lists those that are
missing). Their code runs for about 11 minutes in total on a laptop CPU, plus about a
half minute per project to start R and load the network.
