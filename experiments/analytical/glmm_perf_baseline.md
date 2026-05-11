GLMM Analytical Estimator Summary
=================================

Last updated: 2026-05-11.

This file records the current retained Gaussian GLMM analytical benchmark and the
gated REML diagnostic. Historical MAP sweep notes were removed; the reusable
scripts are documented in `README.md`.

Current Production MAP Benchmark
--------------------------------

Command:

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
```

Required suite: mixed train epochs 1-2 and sampled valid/test for
`small|medium|large|huge`. Current production MAP replaces only
`sigma_rfx_est` and the `Psi` diagonal; FFX, sEps, and BLUP remain MoM/EM-derived.

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.5650 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.5819 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6242 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.3700 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.4671 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.5129 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.3849 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.4679 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.4677 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.3932 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.5911 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5050 | 0.1828 | 0.5167 |

Gated REML Diagnostic
---------------------

Command:

```bash
uv run python experiments/analytical/glmm_reml_diagnostic.py --breakdown
```

Retained policy in `glmm_reml_diagnostic.py`:

- initialize REML from MoM/EM;
- keep beta, correlation, sigma(Eps), FFX, BLUP, and BLUP variance fixed;
- optimize only `log_sigma_rfx`;
- use REML only for valid, unclamped rows with `q >= 2` and `n < 2000`;
- otherwise keep current production MAP output.

Full-suite row-weighted sRFX NRMSE:

| Method | sRFX |
| --- | ---: |
| mom_em | 0.5410 |
| current MAP | 0.4898 |
| raw REML | 0.4748 |
| gated REML | 0.4745 |

Required-suite cells:

| Dataset | Partition | current MAP | gated REML |
| --- | --- | ---: | ---: |
| small-n-mixed | train | 0.5650 | 0.5565 |
| small-n-sampled | valid | 0.5819 | 0.5803 |
| small-n-sampled | test | 0.6242 | 0.6183 |
| medium-n-mixed | train | 0.3700 | 0.3419 |
| medium-n-sampled | valid | 0.4671 | 0.4489 |
| medium-n-sampled | test | 0.5129 | 0.4786 |
| large-n-mixed | train | 0.3849 | 0.3644 |
| large-n-sampled | valid | 0.4679 | 0.4399 |
| large-n-sampled | test | 0.4677 | 0.4321 |
| huge-n-mixed | train | 0.3932 | 0.3865 |
| huge-n-sampled | valid | 0.5911 | 0.5531 |
| huge-n-sampled | test | 0.5050 | 0.4839 |

Best Setup by Case
------------------

| Case | Best setup | Evidence |
| --- | --- | --- |
| Overall | gated REML | 0.4745 vs raw REML 0.4748 and current MAP 0.4898 |
| q = 1 | current MAP / gated | 0.4675 vs raw REML 0.4712 |
| q = 2 | raw REML | 0.4679 vs gated 0.4692 and current MAP 0.4789 |
| q >= 3 | raw REML | 0.5166 vs gated 0.5187 and current MAP 0.5585 |
| n >= 2000 | current MAP / gated | 0.4262 vs raw REML 0.4358 |
| n < 2000 | raw REML | beats current MAP and MoM/EM in both retained n bins |
| d <= 4 | gated REML | 0.5877 vs raw REML 0.5902 and current MAP 0.5928 |
| d >= 5 | raw REML | about 6% relative gain over current MAP |
| MAP expands MoM/EM | gated REML | 0.4462 vs raw REML 0.4482 and current MAP 0.4495 |
| MAP shrinks MoM/EM | raw REML | 0.5519 vs current MAP 0.6001 |
| MAP changes MoM/EM by <5% | MoM/EM | both refinements can be worse |
| true sigma <0.75 | raw REML | strongest gains: 4.88-8.03% over current MAP |

Interpretation
--------------

- REML helps most when variance decomposition is hard: `q >= 2`, moderate/low
  `n`, and small true random-effect scales.
- Current MAP is safest for `q = 1` and high-n rows. In these cases the existing
  output calibration is already strong, and the fixed-beta/fixed-correlation REML
  profile can move sigma(RFX) away from the scoring target.
- Row-level cases exist where both MAP and REML are worse than MoM/EM:
  `both_worse_than_mom_rate = 0.2374`. This is concentrated where MAP barely
  changes MoM/EM. MoM/EM is not a good global fallback, but a future three-way
  gate may need a no-refinement branch.

Next Checks
-----------

- Test whether recomputing final GLS/BLUP after gated REML preserves FFX and BLUP.
- If production integration is attempted, keep row-level fallback and expose gate,
  fallback, and clamp rates in diagnostics.
