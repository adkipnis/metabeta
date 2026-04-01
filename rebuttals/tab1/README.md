# Rebuttals Table 1 - Prior Misspecification

Robustness of the NPE posterior under three types of prior misspecification, evaluated on test set. Metrics are averaged across datasets; **bold** marks the best value per metric within each config.

## Procedure

Results were produced using [`experiments/prior_misspecification.py`](../../experiments/prior_misspecification.py). For each model config:

The model is evaluated on the test set under a baseline condition (correct prior) and all combinations of the following perturbations:
   - **Wrong variance** (τ×{0.33, 3.0}): prior scale hyperparameters scaled by a constant factor
   - **Wrong mean** (μ+{1, 2}σ): prior location for fixed effects shifted by a multiple of the prior width
   - **Wrong family** (fam): prior family indices rotated by +1 (e.g. Normal → Student-t, Half-Normal → Half-Student-t)

## Results
|         Config |           Condition |          R |      NRMSE |         ECE |      ppNLL |
|---------------:|--------------------:|-----------:|-----------:|------------:|-----------:|
|  small-n-mixed |            Baseline | **0.9688** | **0.2183** |      0.0242 | **1.3070** |
|  small-n-mixed |                 fam |     0.9681 |     0.2184 |      0.0274 |     1.3105 |
|  small-n-mixed |                μ+1σ |     0.9664 |     0.2291 |  **0.0136** |     1.3097 |
|  small-n-mixed |          μ+1σ + fam |     0.9674 |     0.2273 |      0.0197 |     1.3120 |
|  small-n-mixed |                μ+2σ |     0.9616 |     0.2568 |     -0.0334 |     1.3204 |
|  small-n-mixed |          μ+2σ + fam |     0.9630 |     0.2524 |     -0.0187 |     1.3204 |
|  small-n-mixed |              τ×0.33 |     0.9618 |     0.2465 |     -0.0716 |     1.3171 |
|  small-n-mixed |        τ×0.33 + fam |     0.9640 |     0.2372 |     -0.0545 |     1.3194 |
|  small-n-mixed |       τ×0.33 + μ+1σ |     0.9543 |     0.2876 |     -0.1132 |     1.3295 |
|  small-n-mixed | τ×0.33 + μ+1σ + fam |     0.9568 |     0.2758 |     -0.0919 |     1.3263 |
|  small-n-mixed |       τ×0.33 + μ+2σ |     0.9321 |     0.3787 |     -0.1918 |     1.4256 |
|  small-n-mixed | τ×0.33 + μ+2σ + fam |     0.9358 |     0.3641 |     -0.1722 |     1.4140 |
|  small-n-mixed |                 τ×3 |     0.9662 |     0.2422 |      0.0510 |     1.3230 |
|  small-n-mixed |           τ×3 + fam |     0.9663 |     0.2436 |      0.0503 |     1.3217 |
|  small-n-mixed |          τ×3 + μ+1σ |     0.9660 |     0.2434 |      0.0505 |     1.3213 |
|  small-n-mixed |    τ×3 + μ+1σ + fam |     0.9661 |     0.2447 |      0.0500 |     1.3202 |
|  small-n-mixed |          τ×3 + μ+2σ |     0.9656 |     0.2468 |      0.0465 |     1.3188 |
|  small-n-mixed |    τ×3 + μ+2σ + fam |     0.9655 |     0.2484 |      0.0469 |     1.3187 |
|    mid-n-mixed |            Baseline | **0.9572** | **0.2583** |      0.0701 |     1.2432 |
|    mid-n-mixed |                 fam |     0.9560 |     0.2627 |      0.0726 |     1.2441 |
|    mid-n-mixed |                μ+1σ |     0.9533 |     0.2742 |      0.0520 |     1.2288 |
|    mid-n-mixed |          μ+1σ + fam |     0.9524 |     0.2766 |      0.0594 |     1.2289 |
|    mid-n-mixed |                μ+2σ |     0.9452 |     0.3114 |      0.0177 |     1.2549 |
|    mid-n-mixed |          μ+2σ + fam |     0.9457 |     0.3082 |      0.0286 |     1.2544 |
|    mid-n-mixed |              τ×0.33 |     0.9522 |     0.2852 |  **0.0024** | **1.2157** |
|    mid-n-mixed |        τ×0.33 + fam |     0.9524 |     0.2784 |      0.0154 |     1.2166 |
|    mid-n-mixed |       τ×0.33 + μ+1σ |     0.9440 |     0.3242 |     -0.0367 |     1.2346 |
|    mid-n-mixed | τ×0.33 + μ+1σ + fam |     0.9450 |     0.3143 |     -0.0228 |     1.2290 |
|    mid-n-mixed |       τ×0.33 + μ+2σ |     0.9226 |     0.3993 |     -0.0858 |     1.3500 |
|    mid-n-mixed | τ×0.33 + μ+2σ + fam |     0.9251 |     0.3867 |     -0.0721 |     1.3121 |
|    mid-n-mixed |                 τ×3 |     0.9458 |     0.3184 |      0.0778 |     1.2474 |
|    mid-n-mixed |           τ×3 + fam |     0.9456 |     0.3199 |      0.0793 |     1.2453 |
|    mid-n-mixed |          τ×3 + μ+1σ |     0.9435 |     0.3251 |      0.0761 |     1.2486 |
|    mid-n-mixed |    τ×3 + μ+1σ + fam |     0.9434 |     0.3266 |      0.0783 |     1.2436 |
|    mid-n-mixed |          τ×3 + μ+2σ |     0.9403 |     0.3359 |      0.0735 |     1.2741 |
|    mid-n-mixed |    τ×3 + μ+2σ + fam |     0.9404 |     0.3373 |      0.0737 |     1.2738 |
| medium-n-mixed |            Baseline | **0.9431** | **0.2993** |      0.0859 | **1.3690** |
| medium-n-mixed |                 fam |     0.9428 |     0.3018 |      0.0870 |     1.3693 |
| medium-n-mixed |                μ+1σ |     0.9390 |     0.3127 |      0.0888 |     1.3732 |
| medium-n-mixed |          μ+1σ + fam |     0.9391 |     0.3139 |      0.0902 |     1.3746 |
| medium-n-mixed |                μ+2σ |     0.9265 |     0.3560 |      0.0496 |     1.4117 |
| medium-n-mixed |          μ+2σ + fam |     0.9276 |     0.3537 |      0.0573 |     1.4119 |
| medium-n-mixed |              τ×0.33 |     0.9320 |     0.3434 |     -0.0166 |     1.3968 |
| medium-n-mixed |        τ×0.33 + fam |     0.9361 |     0.3274 |  **0.0057** |     1.3845 |
| medium-n-mixed |       τ×0.33 + μ+1σ |     0.9202 |     0.3796 |     -0.0544 |     1.4105 |
| medium-n-mixed | τ×0.33 + μ+1σ + fam |     0.9239 |     0.3655 |     -0.0325 |     1.4088 |
| medium-n-mixed |       τ×0.33 + μ+2σ |     0.8920 |     0.4609 |     -0.1239 |     1.6590 |
| medium-n-mixed | τ×0.33 + μ+2σ + fam |     0.8962 |     0.4468 |     -0.1060 |     1.6325 |
| medium-n-mixed |                 τ×3 |     0.9376 |     0.3373 |      0.1116 |     1.3844 |
| medium-n-mixed |           τ×3 + fam |     0.9368 |     0.3410 |      0.1128 |     1.3821 |
| medium-n-mixed |          τ×3 + μ+1σ |     0.9369 |     0.3390 |      0.1207 |     1.3890 |
| medium-n-mixed |    τ×3 + μ+1σ + fam |     0.9361 |     0.3429 |      0.1218 |     1.3883 |
| medium-n-mixed |          τ×3 + μ+2σ |     0.9343 |     0.3473 |      0.1287 |     1.4104 |
| medium-n-mixed |    τ×3 + μ+2σ + fam |     0.9335 |     0.3514 |      0.1292 |     1.4120 |

## Summary
- Wrong family alone barely registers: ΔR < 0.002, ΔNRMSE < 0.005, ΔECE < 0.004, ΔppNLL < 0.004 across all configs.
- Overestimating the prior scale (τ×3) causes mild undercoverage and modest NRMSE degradation: NRMSE increases 11–23% over baseline and ECE shifts +0.008 to +0.027 (toward undercoverage).
- Underestimating it (τ×0.33) causes moderate overcoverage on its own (ECE −0.02 to −0.07), but in combination with mean shifts this worsens to ECE as low as −0.19, and NRMSE increases of 54–73% over baseline.
- Mean shifts compound with τ×0.33 but not with τ×3: adding μ+2σ on top of τ×0.33 raises NRMSE by 0.114–0.132, versus only 0.005–0.018 on top of τ×3 (a 6–29× difference), as to be expected.
