# Reproducing the paper's metabeta results

`reproduce-paper.Rmd` reruns the metabeta results of the paper with the R package, through
`brm()` and the package's other brms functions:

1. **Oracle benchmark** (Sec. 3.1): freshly simulated datasets that follow the paper's
   protocol; Table 1, Tables B1–B3, B7 and B8, Figure 2, the IMH head against the raw flow
   (App. C.1) and data-poor groups (App. D.5).
2. **Stress tests** (App. D.1–D.3): misspecified likelihoods and priors, out-of-distribution
   predictors.
3. **Prior sensitivity and Bayes factors** on the paper's real datasets (App. B.7, Sec. 3.4).
4. **Deployment safeguards** (App. D.6).

The notebook's last section lists what it does not reproduce, and its text compares each
result with the paper's.

## Running it

From the `R-package` folder:

```r
rmarkdown::render("validation/reproduce-paper/reproduce-paper.Rmd")
```

It needs the packages lme4, glmmTMB, MASS and rmarkdown besides metabeta. With the default
sizes (32 oracle datasets per family and regime, 16 per stress condition) it fits about
1,450 models and takes about two hours on a laptop CPU. Fits are cached in `cache/` next to
the notebook; delete it to start over. The paper's sizes take much longer:

```r
rmarkdown::render("validation/reproduce-paper/reproduce-paper.Rmd",
                  params = list(n_datasets = 512, n_stress = 128))
```

The rendered `reproduce-paper.html` and `cache/` are not tracked by git.
