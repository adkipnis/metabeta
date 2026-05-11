Analytical GLMM Plan
====================

Last updated: 2026-05-11.

Current Decision
----------------

The production GLMM analytical baseline remains the current MAP sRFX path. The
retained experiment candidate is gated REML/profile-MAP over sigma(RFX), kept
output-local until recomputing GLS/BLUP is proven stable.

Production Baseline
-------------------

- Runtime Psi, FFX, sEps, BLUP, and BLUP variance come from the MoM/EM path.
- Current production MAP refines only reported `sigma_rfx_est` and the `Psi`
  diagonal.
- Existing BLUP beta blending and output-only sRFX calibration remain the stable
  production choices.

Retained REML Candidate
-----------------------

`experiments/analytical/glmm_reml_diagnostic.py` now uses one retained policy:

- initialize REML from `glmm(..., map_refine=False)`;
- optimize only `log_sigma_rfx` for 20 Adam steps at learning rate 0.03;
- keep beta, correlation, sigma(Eps), FFX, and BLUP fixed;
- use REML only for valid, unclamped rows with `q >= 2` and `n < 2000`;
- otherwise keep current production MAP output.

Full-suite row-weighted sRFX:

| Method | sRFX |
| --- | ---: |
| mom_em | 0.5410 |
| current MAP | 0.4898 |
| raw REML | 0.4748 |
| gated REML | 0.4745 |

Why the Gate Exists
-------------------

- q = 1: current MAP/gated is better than raw REML.
- q >= 2: raw REML usually helps because variance decomposition is harder.
- n >= 2000: current MAP/gated is safer; the fixed-profile REML objective can be
  too sharp under beta/correlation misspecification.
- MAP changes MoM/EM by <5%: MoM/EM can be best, so a future three-way gate may
  need a no-refinement branch.

Remaining Work
--------------

- Test recomputing GLS/BLUP after gated REML and reject it if FFX/BLUP regress.
- If integrating into production, expose gate/fallback/clamp rates.
- Laplace curvature around the MAP/REML point remains the only other retained
  non-MCMC direction, intended for uncertainty/context features rather than
  direct point-estimate gains.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_reml_diagnostic.py --breakdown
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
