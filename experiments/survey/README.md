# GLMM literature survey

`model_equations.csv` holds the 500 GLMM model equations extracted from 1,133
randomly sampled papers across ten interdisciplinary journals (LLM pipeline:
screening + equation extraction; a manually audited subsample showed no errors).
Columns: dataset_id, citation, doi, equation, n_fixed_effects, n_random_effects,
family, family_detail, n_groups (number of grouping factors), source (pdf/code),
source_ref (verbatim quote anchoring the extraction), confidence.

Referenced in the paper as Appendix "A Survey of Published GLMM Analyses".
Coverage claim: family in {Gaussian, Bernoulli, Poisson} and d <= 16 and q <= 5
holds for 82.8% (~83%) of rows.
