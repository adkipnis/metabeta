# Reference methods rework (`ref-methods`)

Working document for the branch; folded into the PR at retirement and deleted.

## Design

1. **Stack**: PyMC ≥ 6.3, PyTensor 3, ArviZ ≥ 1.3, pymc-extras ≥ 0.15 (Pathfinder).
2. **Storage**: `{partition}.{tag}.npz` per method, arrays only, key prefix equals tag,
   plus `source_sha` of `{partition}.npz`. Tags: `nuts0 nuts1 nuts2 nuts advi0 advi1
   pathfinder0 pathfinder1 laplace`. `Collection(path, fits=(...))` merges sibling files
   and raises on checksum mismatch; `exclude_prefixes` retired. `test.fit.npz` stays
   untouched until merge (no de-partitioning); Laplace's merge step goes. `e2-*`
   directories are out of scope.
3. **NUTS ladder**, every test dataset: L0 = PyMC defaults (tune 1000, draws 1000,
   4 chains, target accept 0.8, tree depth 10); L1 = tune 2000, target accept 0.9;
   L2 = tune 4000, draws 2000, target accept 0.99, tree depth 12. Valid partition: L2
   only. Wall time around `pm.sample` including compilation and spawn; compile time also
   recorded separately.
4. **Filter**: one criterion. R-hat ≤ 1.01, bulk and tail ESS ≥ 400 over all sampled
   variables (Vehtari et al. 2021), divergences zero at L0/L1 and ≤ 0.1 % at L2.
   Tree-depth saturation reported, not filtered. `liberal`/`strict` removed; results on
   converged datasets only, plus one convergence-validation check (converged vs all).
5. **Composite `nuts`**: written by reintegration from the level files; per dataset the
   arrays of the first level that passes the filter (L2 if none), with `nuts_level`,
   `nuts_converged`, cumulative `nuts_duration`. Role split: `nuts2` is the *reference*
   for agreement metrics (8000 draws, no runtime attached); `nuts` is the *competitor* in
   runtime and accuracy-vs-truth tables.
6. **ADVI**: one run, adagrad_window lr 1e-3 (PyMC default optimiser, Kucukelbir et al.
   2017), 100k iterations, no early stopping, ELBO recorded; snapshot schedule
   1k/2k/5k/10k/20k/50k/100k stored; draws written at 10k (`advi0`, PyMC default budget)
   and 100k (`advi1`), each with its own wall time.
7. **Pathfinder**: level 0 = 4 paths, level 1 = 20 paths, 4000 draws both, pymc-extras
   defaults otherwise.
8. **Diagnostics**: NUTS: ESS bulk/tail, R-hat, divergences, tree-depth saturation,
   leapfrog steps, step size, acceptance, E-BFMI, chains, tune, target accept,
   durations, PyMC version. ADVI: ELBO curve, iterations, final ELBO (100 MC samples),
   durations. Pathfinder: paths, LBFGS iterations, Pareto k, compile/compute time.
   Laplace: optimizer success, MAP objective, durations.
9. **Budget rule**: every evaluated posterior uses 4000 draws (L2 thinned when it is
   evaluated against truth); the `nuts2` reference keeps 8000.
10. **Curve**: all ids; points MB⁰, MB, NUTS L0, NUTS composite, ADVI0/1, Pathfinder0/1,
    Laplace vs true parameters with the existing metrics; x = wall time (NUTS 4 cores;
    MB one GPU and 4 CPU cores as separate panels). Plus every NUTS/ADVI/Pathfinder
    level scored against `nuts2` with the paper's agreement metric (false-convergence
    check).
11. **Main tables**: reference `nuts2` with the L2 converged mask; competitors `nuts`,
    `advi1`, `pathfinder1`, `laplace`; `--models all` = MB + these. Runtime tables show
    NUTS L0 and composite.
13. **Laplace** (decided 2026-09-29): fitted with `pymc_extras.fit_laplace` on the same
    `buildPymc` model, pymc-extras defaults, 4000 draws; same priors, extractor, timing and
    writer as the other references. The torch reference fitter
    `metabeta/simulation/laplace.py`, its tests and the PyMC parity reference retire in
    step 3 (the package's `posthoc/laplace_glmm.py` is a different component and stays).
    Pathfinder is kept alongside as the modern cheap reference; LA is the classical one.
12. **Cluster**: one `scripts/fit-ref.sh --method --level --data_id --partition`;
    NUTS 4 CPUs, 16 GB, 6 h (L0/L1) / 12 h (L2); ADVI and Pathfinder 1 CPU. `check.py`
    verifies per-level files and reintegrates.

Order of work: stack upgrade → storage and loader with tests → fitter rework → filter
cleanup → SLURM script → curve experiment.

## Step 1: stack upgrade (done locally)

- `pymc>=6.3.2`, `pymc-extras>=0.15.1`, `arviz>=1.3.0`; bambi moved to 0.21 with it.
- Smoke on `small-n-sampled` dataset 0: NUTS (2 chains, forkserver), ADVI (`pm.fit`)
  and `pymc_extras.fit_pathfinder` all run; `buildPymc`/`extractAll` unchanged. Traces
  are `xarray.DataTree` now; `.posterior[name]`, `az.summary(kind='diagnostics')` and
  `az.bfmi` work as before. `posterior.attrs['sampling_time']` gives pure sampling time.
  Pathfinder returns groups `pathfinder` (compile/compute/total time, pareto_k, path
  status) and `lbfgs` (niter).
- ArviZ 1 dropped `az.psislw`. Replacement `arviz_stats.base.array_stats.psislw` takes
  the *log-likelihood* convention (negates internally). `metabeta/utils/psis.py` wraps
  it with the old conventions (log ratios in, raw normalised weights + k = inf for
  < 25 draws). Verified against ArviZ 0.23.4 in a scratch venv: identical to 1e-15 at
  256 draws; ≤ 1e-2 log-weight differences in the extreme tail at ≥ 512 draws because the
  tail-length rule now follows Vehtari et al. 2024 (`n·r_eff > 225 → 3√(n/r_eff)` else
  `n/5`). LOO-NLL and Pareto k therefore shift by numeric noise relative to the paper's
  current numbers.
- 631 tests pass; the opt-in PyMC Laplace parity check passes
  (`METABETA_RUN_PYMC_LAPLACE=1`).
- macOS only: Xcode 27 rejects the `-ld64` linker flag PyTensor adds
  (pymc-devs/pytensor#2268; predates this upgrade). Local workaround outside the repo:
  an empty `~/.pytensor/shim/libd64.dylib` and `~/.pytensorrc` with
  `[gcc] cxxflags = -L~/.pytensor/shim`.

### Cluster: upgrade the container venv

Checked over SSH (2026-09-28): the image `python312.sif` is Debian 12 with g++ 12.2
(PyTensor 3 needs C++17, so no rebuild), `uv` 0.10.11 lives in `~/.local/bin` and is
visible inside the container through the `$HOME` bind, and `.venv-apptainer` was created
by that uv. The repo on the cluster is on `is2-evidence`. Run on the submit node:

```bash
cd ~/metabeta && git fetch && git checkout ref-methods && git pull
apptainer exec --bind "$HOME:$HOME" ~/containers/python312.sif bash -lc '
  cd ~/metabeta
  UV_PROJECT_ENVIRONMENT=.venv-apptainer uv sync
  source .venv-apptainer/bin/activate
  python -c "import pymc, pytensor, arviz, pymc_extras; print(pymc.__version__, pytensor.__version__, arviz.__version__, pymc_extras.__version__)"
  METABETA_RUN_PYMC_LAPLACE=1 python -m pytest -q tests/simulation/test_laplace_pymc_parity.py
'
```

Expected: `6.3.2 3.3.2 1.3.x 0.15.1`, then `2 passed`. `uv sync` resolves from the
committed lock; never `uv lock` on the cluster. Then one smoke fit:

```bash
sbatch scripts/fit-nuts.sh --data_id small-n-sampled   # after step 3 replaces this with fit-ref.sh
```

## Step 2: per-method storage and loader (done locally)

- `metabeta/utils/fits.py` owns the file format: `FIT_TAGS`, `fitPath`, `availableFits`,
  `sourceSha`, `saveFits` (prefix + leading-axis validation, `source_sha`, refuses to
  overwrite without `force`), `loadFits` (checksum check, optional key subset).
  The SHA-256 of a 100 MB `test.npz` costs about 0.3 s per load.
- `Collection(path, fits=(...))` merges the requested tags; `exclude_prefixes`,
  `has_nuts`, `has_advi` are gone; `Dataloader(..., fits=...)` forwards. The permutation
  and collate loops run over `FIT_TAGS` instead of the hard-coded three methods.
- Writers: `Fitter.reintegrate` and `LaplaceFitter.go` write through `saveFits`; the
  Laplace merge step (`--reintegrate`) is removed. Until step 3 replaces the fitter,
  reintegration defaults to NUTS only (there is no `advi` tag; the old ADVI per-index
  files carry `advi_*` keys).
- Readers: evaluate.py maps models to tags (`_FIT_TAGS`: NUTS → `nuts`, ADVI → `advi1`,
  Laplace → `laplace`); the light path streams `*_rfx` from `fitPath(...)` after
  `loadFits` verified the checksum; `_getPartitionData(partition, fits=...)` replaces the
  `need_fits`/`prefer_fit` flags. cache.py loads one method per Dataloader; train.py,
  check.py, plotting/runtimes.py and `dataFilePath` (no `fit=`) follow.
- Tests: `tests/utils/test_fits.py`; fixtures write fits with `saveFits`. 638 pass.
- Experiments (`experiments/evaluation`, `posthoc`, `simulation`): paths point at
  `test.npz`, fits come from `Collection(..., fits=)` or `loadFits`; ADVI rows and their
  summary caches are keyed by the tag `advi1`. The derived-data generators (`ood_design`,
  `prior_misspec`, `likelihood_misspec`) slice `test.nuts.npz` and re-stamp the checksum
  with `saveFits`. Six scripts had imports broken since earlier refactors (`Proposal`,
  `buildPymc`, `Fitter`, `gaussian_local` moved); repaired in passing.
- `test.fit.npz` / `valid.fit.npz` stay on disk untouched; nothing reads them any more.

## Step 3: fitter rework (done locally)

- `metabeta/simulation/fit.py` is the one fitter: `Fitter(cfg)` with `--method
  nuts|advi|pathfinder|laplace --level --idx`, budgets as module constants (`NUTS_LEVELS`,
  `PATHFINDER_PATHS`, `ADVI_ITER`, snapshot schedules, `N_DRAWS = 4000`). Per-dataset files
  `fits/{stem}_{tag}_{idx:03d}.npz`; `--reintegrate` aggregates a tag with `aggregateFits`
  (zero padding to the largest shape, NaN/'' rows for failed fits) and writes through
  `saveFits`. `nutsadvi.py`, the torch `laplace.py`, their tests and the PyMC parity
  reference are deleted; `bambi_equivalence.py` builds the model directly.
- Every tag carries `_duration` (model build + compile + fit), `_failed`, `_error`.
  NUTS adds names, ESS bulk/tail, R-hat, divergences, tree-depth saturation, leapfrog steps,
  step size, acceptance, E-BFMI, sampling time, draws/tune/target accept/chains, PyMC
  version. ADVI: one 100k run, PyMC default optimiser, ELBO logged at
  1k/2k/5k/10k/20k/50k/100k (mean of the last 100 one-sample estimates), draws at 10k
  (`advi0`) and 100k (`advi1`) with the snapshot's own wall time (sampling time of earlier
  snapshots excluded). Pathfinder: paths, Pareto k, compile/compute time, path status
  counts, L-BFGS iterations; runs its paths sequentially (one core on the cluster).
  Laplace: `fit_laplace` defaults (BFGS, inverse Hessian from the optimiser), optimizer
  success/status/iterations/objective/gradient norm. A fit that raises is recorded as
  failed, NUTS is not guarded (a NUTS crash is a job failure).
- `check.py --partition test|valid` checks every started tag, prints the `fit-ref.sh`
  refit command for gaps and reintegrates complete tags. `scripts/fit-ref.sh --method
  --level --data_id [--partition] [--idx ...] [--n_datasets]` replaces `fit-nuts.sh`,
  `fit-advi.sh` and `fit-selected.sh` (NUTS 4 cores, 6 h at L0/L1 and 12 h at L2; the
  other methods one core, 6 h). Campaign hints of the derived-data generators print the
  three NUTS levels plus the check command.
- Smoke (2-dataset slice of `small-n-sampled`, Mac): all methods fit, reintegrate, load
  through `Collection(fits=...)`; wall times 4–10 s NUTS L0, 5–12 s ADVI 100k, 1–2 s
  Pathfinder, 0.4–6 s Laplace. Pathfinder's Pareto k was 3.8 on the correlated dataset
  (PSIS unreliable), a diagnostic worth reporting later.
- Found while migrating: `fit_laplace` rejects `chains` (deprecated) and BFGS reports
  `success=False` for "precision loss" even at gradient norm 1e-8, so the status and the
  gradient norm are stored rather than the flag alone. The local PyTensor shim needed an
  absolute `-install_name` (noted in memory).
- `scripts/fit-nuts-prior-grid.sh` (E2 prior grid, out of scope) still calls `fit.py --config --idx --method`, which the new CLI accepts; it now fits NUTS level 0.
- 627 tests pass (11 Laplace tests removed, 4 fitter tests added).

## Step 4: convergence filter and composite `nuts` (done locally)

- `metabeta/utils/evaluation.py`: `nutsConverged(diag, prefix)` (numpy, dataset axis
  first) and `nutsConvergeMask(batch, prefix)` (collated batch) implement the single
  criterion: R-hat ≤ 1.01, bulk and tail ESS ≥ 400 over all sampled variables, no
  divergences at levels 0/1 and ≤ 0.1 % at level 2 (`DIVERGENCE_RATE_MAX`); the level is
  read from `{prefix}_level`, which every NUTS fit now stores. `liberal`/`strict` and the
  tree-depth criterion are gone.
- `Fitter.composeNuts` (called by `check.py` once the level files are complete) walks the
  per-dataset level files, keeps the first converged level (highest fitted level otherwise),
  thins to 4000 draws and writes `nuts` with `nuts_level`, `nuts_converged` and the
  cumulative `nuts_duration`. On the valid partition only level 2 exists, so `valid.nuts.npz`
  is the thinned level 2.
- evaluate.py: `--models all` = MB, NUTS (composite), ADVI (`advi1`), PATHFINDER
  (`pathfinder1`, label `PF`), LAPLACE. Every table row is computed on the datasets where
  `nuts2` converged (`_referenceMask`, folded into the common mask of the cached, light and
  full paths; summary caches are keyed by that mask). `--all_datasets` disables the filter:
  running with and without it is the convergence-validation check. The old
  `--converged_subset`, `--convergence_mode`, `--pareto_k_thr`, the conv/loo subset rows and
  the NUTS failure analysis are removed. cache.py defaults to the four competitors.
- The collate step now carries `_draws`, `_level`, `_converged`, `_bfmi`, `_n_steps`,
  `_sampling_time` per fit tag.
- Experiments: every `--convergence_mode` flag, mode loop and strict/liberal caption is gone.
  Agreement and oracle scripts use `nuts2` as the reference with its converged mask and the
  composite `nuts` as the NUTS competitor row; the misspecification and data-poverty
  scripts score their NUTS oracle row on `nuts2` draws and cache under `summary_test_nuts2*`
  (old NUTS LOO caches are not reused). runtimes.py lists `nuts0`, `nuts` (composite),
  `advi1`, `pathfinder1`, `laplace`; its runtime figure still maps only the composite.
  real_posterior's Δtime is relative to `nuts2`. nuts_divergences.py audits each level file
  under the single criterion with the share failing each check. loo_bias.py reports all
  datasets, converged, and converged with Pareto k < 0.7. The E2 prior-sensitivity script
  reads `nuts0`/`advi1` per-point files and needs a refit of its grid before it runs.
- Open after the refit: agreement_marginals' default example datasets were picked under the
  old criterion and may need re-picking; nuts_divergences' LOO column reads
  `summary_test_nuts.pt`, which evaluate.py now writes under a mask tag.

## Step 6: time-accuracy curve (done locally)

- `experiments/evaluation/ref_curve.py` (`CurveExperiment`): per test set it scores MB⁰, MB,
  `nuts0`, `nuts1`, `nuts` (composite), `advi0`, `advi1`, `pathfinder0`, `pathfinder1` and
  `laplace` against the true parameters on the `nuts2`-converged datasets (cached summaries,
  one fit file resident at a time), pairs every method with its median wall time (metabeta:
  runtimes.py latency, CPU timed here, GPU read from a pulled `*_cuda.json` cache), and
  writes `ref_curve_{tag}.csv` (long format), `ref_curve_{tag}_agreement.md` and
  `ref_curve_{tag}_latest.pdf` (panels NRMSE / EACE / LOO-NLL over log wall time; faint
  per-set points, one large marker per method, lines along the NUTS, ADVI and Pathfinder
  ladders). The agreement table is the false-convergence check: r, σ-ratio and rank-MAD of
  each level against `nuts2`, NUTS levels split by their own convergence.
- Smoke on the two-dataset slice: all methods scored, MB⁰/MB at 0.1 s versus NUTS 7–13 s;
  Laplace far off in NRMSE and LOO-NLL (the scale overestimate), the ADVI/Pathfinder ladders
  visible. Reference `nuts2` itself is not a point on the curve.
- The SLURM script (step 5) was folded into step 3 (`scripts/fit-ref.sh`).

## Cluster campaign (next)

1. `git pull` on the cluster, `uv sync` in the container venv (see above).
2. Per test set: `scripts/fit-ref.sh --method nuts --level {0,1,2}`, `--method advi`,
   `--method pathfinder --level {0,1}`, `--method laplace`; valid partition: nuts level 2 only.
3. `metabeta/simulation/check.py --partition test --data_id ...` reintegrates and composes.
4. `metabeta/evaluation/cache.py`, then evaluate.py / oracle_posterior.py / runtimes.py /
   ref_curve.py; `evaluate.py --all_datasets` once for the convergence-validation check.

### Campaign log

- 2026-09-29: cluster repo on `ref-methods`, venv at PyMC 6.3.2 / PyTensor 3.3.2 / ArviZ 1.3.0 /
  pymc-extras 0.15.1. The old torch `test.laplace.npz` files collide with the new tag and were
  moved to `{data_id}/old/`; `test.fit.npz` / `valid.fit.npz` stay untouched as the backup of
  the old NUTS and ADVI fits (delete `old/` and `*.fit.npz` once the new fits are validated).
- Smoke (two datasets of `huge-p-sampled`, every method; two of `small-n-sampled/valid` at
  NUTS L2): all completed, ~3 min per array task including container start and compile.
  Pathfinder Pareto k 2–12 on the huge Poisson sets; Laplace BFGS ends with status 2
  (precision loss) and `jac` all zero, to be inspected against the nuts2 draws.
- Full campaign: 96 arrays (84 test = 12 sets × {nuts0,1,2, advi, pathfinder0,1, laplace};
  12 valid = nuts2), 49 152 tasks, user cap 100 concurrent jobs; expected ~1 day.
- First reintegration (`small-n-sampled`, nuts2, 512/512): 0 failed, 485/512 converged, wall
  time median 75 s / p90 174 s / max 397 s; posterior means agree with the old NUTS fits
  (r = 1.0000 ffx, 0.9996 sigma_rfx). `check.py` runs under `--qos=cpu_priority` while the
  campaign saturates the 100-job cap of `cpu_normal`; composition waits until every started
  tag of a set is complete.
- 12:15: single-core methods (advi, pathfinder, laplace) moved to `--qos cpu_preemptible`
  (own cap of 200 jobs, 3-day wall); NUTS stays on `cpu_normal`. Throughput went from
  ~1.5k to ~5–10k tasks/h; no preemptions observed.
- `small-n-sampled`, all levels reintegrated: converged 286 / 395 / 485 of 512 at L0 / L1 / L2,
  wall time median 45 / 52 / 75 s, no failed fits at any level or method; advi1 vs the old
  ADVI posterior means r = 0.999 (ffx) / 0.996 (sigma_rfx).
- 17:00: every test tag has 512/512 per-index files on all 12 sets; check.py runs per set in
  parallel under `cpu_preemptible` (the serial priority job hit its 2 h limit after 8 sets),
  cache.py chained after each check. Composites so far: medium-b 504, medium-p 488,
  large-n 470, large-b 499 of 512 converged.
- `medium-b-sampled` validation: composite `nuts` vs nuts2 posterior means r = 1.000; advi1
  vs old ADVI r = 0.9997 / 0.994 but **45/512 advi1 runs fail** with PyMC's
  `FloatingPointError: NaN occurred in optimization` before 100k iterations (the old run
  stopped early); Pathfinder Pareto k median 2.0, < 0.7 on 0–1 % of datasets, sigma_rfx
  1.6–1.7× nuts2; **Laplace sigma_rfx is 3.9× nuts2 with r ≈ 0** on this set, the same as
  the old torch Laplace (4.5×, r = 0.18), so it is the method, not the new fitter. BFGS
  status 2 (precision loss) on 483/512, `jac` all zero.
- cache.py failed on the Bernoulli sets: ArviZ 1 raises `All tail values are the same` when
  every draw predicts an observation equally well; `metabeta/utils/psis.py` now returns the
  raw normalised weights with k = inf for such rows (the ArviZ < 1 convention), tests in
  `tests/utils/test_psis.py`. Pushed 7d22fc36 to `origin/ref-methods` so the cluster could
  pull it (the only push of this campaign).
- ADVI failures are a Bernoulli phenomenon: old run 18 / 46 / 50 failed on medium-b /
  large-b / huge-b, the new 100k run 45 / 70 / – (advi0 already 39 / 62), zero on all
  Gaussian and Poisson sets.
- Test composites, all 12 sets (converged of 512): small n/b/p 492/508/504, medium 474/504/488,
  large 470/499/476, huge 444/487/473. Valid composites so far: small-n 483, small-b 508.
- cache.py then failed on the Laplace tag of the small sets: `fit_laplace` returns all-NaN
  draws without raising on 16 (small-n) / 28 (small-b) datasets (Hessian not positive
  definite; BFGS status 2). `Fitter._aggregate` now marks fits with non-finite posterior
  draws as failed (`markNonFinite`, error 'non-finite draws'); pushed c55a93ac and
  reintegrated the laplace tag on every set.
- Still failing after that on 5 sets: Laplace rfx / sigma_rfx draws of up to 1e68 (medium-b)
  and 1e154 (large-n), finite in float64 but inf in the float32 evaluation stack.
  `markNonFinite` now also counts draws beyond float32 range as non-finite (7a1acedb);
  Laplace failure counts after both rules: small-n 16, small-b 28, medium-b 37 (+3),
  large-n 24 (+1). Laplace reintegrated and cache.py rerun on all 12 sets.
- Valid composites: small n/b/p 483/508/492, medium-n 468, medium-p 488, large-n 458.

### Campaign result (test partition, 512 datasets per set)

Converged (NUTS levels and composite) or Pareto k < 0.7 (Pathfinder):

| set | nuts0 | nuts1 | nuts2 | nuts | advi0 | advi1 | pathfinder0 | pathfinder1 | laplace |
|---|---|---|---|---|---|---|---|---|---|

**converged**

| set | nuts0 | nuts1 | nuts2 | nuts | advi0 | advi1 | pathfinder0 | pathfinder1 | laplace |
|---|---|---|---|---|---|---|---|---|---|
| small-n-sampled | 286 | 395 | 485 | 492 |  |  | k<0.7: 3 | k<0.7: 0 |  |
| small-b-sampled | 316 | 450 | 503 | 508 |  |  | k<0.7: 30 | k<0.7: 12 |  |
| small-p-sampled | 212 | 394 | 500 | 504 |  |  | k<0.7: 5 | k<0.7: 5 |  |
| medium-n-sampled | 289 | 388 | 471 | 474 |  |  | k<0.7: 0 | k<0.7: 0 |  |
| medium-b-sampled | 344 | 425 | 502 | 504 |  |  | k<0.7: 5 | k<0.7: 1 |  |
| medium-p-sampled | 260 | 388 | 486 | 488 |  |  | k<0.7: 0 | k<0.7: 0 |  |
| large-n-sampled | 305 | 356 | 466 | 470 |  |  | k<0.7: 0 | k<0.7: 0 |  |
**failed**

| set | nuts0 | nuts1 | nuts2 | nuts | advi0 | advi1 | pathfinder0 | pathfinder1 | laplace |
|---|---|---|---|---|---|---|---|---|---|
| small-n-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 |
| small-b-sampled | 0 | 0 | 0 | 0 | 15 | 18 | 0 | 0 | 28 |
| small-p-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 |
| medium-n-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 |
| medium-b-sampled | 0 | 0 | 0 | 0 | 39 | 45 | 0 | 0 | 39 |
| medium-p-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 |
| large-n-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 |
| large-b-sampled | 0 | 0 | 0 | 0 | 62 | 70 | 0 | 0 | 42 |
| large-p-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| huge-n-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23 |
| huge-b-sampled | 0 | 0 | 0 | 0 | 75 | 85 | 0 | 0 | 56 |
| huge-p-sampled | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 52 |

**median s**

| set | nuts0 | nuts1 | nuts2 | nuts | advi0 | advi1 | pathfinder0 | pathfinder1 | laplace |
|---|---|---|---|---|---|---|---|---|---|
| small-n-sampled | 45 | 52 | 75 | 81 | 33 | 49 | 55 | 57 | 30 |
| small-b-sampled | 46 | 54 | 83 | 73 | 31 | 50 | 58 | 60 | 29 |
| small-p-sampled | 45 | 52 | 75 | 90 | 32 | 48 | 57 | 59 | 29 |
| medium-n-sampled | 55 | 64 | 113 | 85 | 37 | 56 | 67 | 67 | 39 |
| medium-b-sampled | 59 | 66 | 124 | 79 | 37 | 59 | 74 | 70 | 37 |
| medium-p-sampled | 56 | 63 | 120 | 91 | 37 | 57 | 70 | 71 | 37 |
| large-n-sampled | 64 | 70 | 131 | 88 | 41 | 62 | 76 | 76 | 46 |
| large-b-sampled | 65 | 73 | 153 | 77 | 41 | 63 | 75 | 81 | 48 |
| large-p-sampled | 70 | 77 | 155 | 93 | 41 | 62 | 90 | 82 | 47 |
| huge-n-sampled | 74 | 81 | 151 | 94 | 44 | 68 | 81 | 92 | 62 |
| huge-b-sampled | 75 | 88 | 189 | 91 | 46 | 69 | 85 | 93 | 59 |
| huge-p-sampled | 79 | 94 | 197 | 98 | 44 | 68 | 85 | 87 | 59 |


### Compile-time profiling (2026-09-29, huge-p-sampled dataset 0, m=27 n=250 d=16 q=1)

Wall time of one fit = build + fit, PyTensor cache cold (empty `base_compiledir`) vs warm
(same dataset fitted before). Cluster = Xeon Gold 6136, 1 BLAS thread, container g++;
MacBook = M3, clang.

| | Pathfinder 4 paths | Laplace | NUTS L0 (sampling only) |
|---|---|---|---|
| cluster, cold, compiledir on Lustre | 149 s | 38 s | 25 s (7.8 s) |
| cluster, cold, compiledir on node /tmp | 80 s | 32 s | 22 s (7.7 s) |
| cluster, warm | 10 s | 7 s | 13 s (7.6 s) |
| MacBook, cold | 24–37 s | 10 s | 10 s (1.2 s) |
| MacBook, warm | 3 s | 2 s | 2.5 s (1.2 s) |

pymc-extras' `compute_time` contains compilation (warm: 0.2–1.2 s). The cache is only
partly shared across datasets: a new dataset of the same family still compiles 3–13 s on
the MacBook (new broadcast patterns; q = 2 adds the LKJ/Cholesky kernels, 36 s), against
1–3 s for a repeat of the same dataset. Every campaign task started cold, so the recorded
durations of every method are dominated by compilation. Cluster NUTS sampling time for
this dataset was 32 s in the campaign smoke and 7.8 s here: node-to-node noise.

Steady state (cache filling over 11 Poisson datasets, node-local compiledir, one fit per
dataset and method): a *new* dataset still compiles on top of the warm cache.

| per new dataset | Pathfinder 4 paths | Laplace | NUTS L0 wall − sampling |
|---|---|---|---|
| cluster (Xeon 6136) | 34–59 s (pymc-extras compute 25–47 s) | 21–80 s (175 s at m=122, q=3) | 10–20 s |
| MacBook (M3) | 8–15 s | 5–25 s (61 s at m=122, q=3) | 3–6 s |

Repeat of the same dataset: cluster 8 / 6 / 4 s, MacBook 3 / 2 / 1.5 s. Cluster sampling
time of NUTS L0 varies 8–74 s across these datasets, MacBook 1–14 s (5–7× slower cluster
core). So the campaign `duration` = cold compile (≈ 40–70 s on the cluster) + residual
compile + compute; a pre-warmed cache removes only the first term.

### Warm-cache rerun of the compiled methods (2026-09-29, evening)

Decision (Alex): keep the cold NUTS fits, rerun ADVI, Pathfinder and Laplace from a
pre-warmed PyTensor cache, the state of a user who has fitted one such GLMM before, and
report the cold compile cost once. Cold wall times are backed up in
`{data_id}/test.cold_durations.npz` (keys `{tag}_duration/_failed`, Pathfinder
`compile_time/compute_time`) and pulled to `~/Downloads/hpc-pull/cold_durations/`; cold
medians: ADVI 100k 49–69 s, Pathfinder 55–93 s, Laplace 29–62 s. `scripts/warm-cache.sh
--family {n,b,p}` builds `~/pytensor_cache_{f}.tar` from brief fits on the first
random-intercept and first random-slope dataset of `small-{f}-sampled`;
`scripts/fit-ref.sh --warm` unpacks it into a node-local compile directory (all tasks now
compile on node /tmp instead of Lustre). Stale `summary_test_{advi1,pathfinder1,laplace}.pt`
caches are deleted before recaching.
- Valid partition complete (nuts2 → `valid.nuts.npz`, converged of 512): small n/b/p
  483/508/492, medium 468/499/488, large 458/493/479, huge 445/486/469.
- Warm rerun: 456 tasks failed on one node (`cpusrv32`, `/tmp` full → `mkdir: No space
  left on device`); node excluded from the pending arrays, the gaps are refitted from the
  `check.py` output afterwards. Warm ADVI 100k on `small-n-sampled`: median 29 s (cold 49 s).
- Warm rerun done 2026-09-30 00:00 (advi, pathfinder0/1, laplace on all 12 sets; the 456
  gaps refitted; every tag 512/512, reintegrated and composed again). Failure counts are
  unchanged up to ±2 Laplace fits per set (borderline Hessians flip with the kernel
  fusion of the cached build). Median wall time cold → warm (s):

| set | ADVI 10k | ADVI 100k | Pathfinder 4 | Pathfinder 20 | Laplace |
|---|---|---|---|---|---|
| small (n/b/p) | 33/31/32 → 11/10/10 | 49/50/48 → 28/29/27 | 55/58/57 → 17/17/18 | 57/60/59 → 18/18/18 | 30/29/29 → 13/13/13 |
| medium | 37/37/37 → 15/14/14 | 56/59/57 → 34/37/34 | 67/74/70 → 28/26/26 | 67/70/71 → 29/27/27 | 39/37/37 → 21/18/18 |
| large | 41/41/41 → 18/18/18 | 62/63/62 → 37/40/39 | 76/75/90 → 36/34/34 | 76/81/82 → 37/36/36 | 46/48/47 → 27/28/28 |
| huge | 44/46/44 → 20/22/21 | 68/69/68 → 42/45/46 | 81/85/85 → 43/41/43 | 92/93/87 → 45/44/46 | 62/59/59 → 34/39/41 |

  NUTS stays cold (L0 45–79 s, L1 52–94 s, L2 75–197 s median); its compile share is
  `duration − sampling_time − build` (≈ 10–20 s on the cluster).
- cache.py done on all 12 sets (nuts, advi1, pathfinder1, laplace; 30–63 min each).
- Old fits deleted per Alex (2026-09-30): `{data_id}/test.fit.npz`, `valid.fit.npz` and
  `old/test.laplace.npz` on all 12 sampled sets. Per-index `fits/test_inla*` and the
  `fits_warm_*` directories are untouched; the July `summary_test_advi.pt` caches are
  orphaned (no consumer) but left in place.

### Real test sets and evaluation stage (2026-09-30)

- Paper moved to `~/LaTeX/metabeta-iclr`. `appendices/protocol.tex` (app:tra) rewritten for
  the ladder, the single criterion (retention 84–98 % at L2, 41–67 % at L0, 66–88 % at L1,
  composite 87–99 % on the oracle sets), PyMC-default ADVI with two snapshots, Pathfinder,
  pymc-extras Laplace, and the compute/wall-time paragraph (heterogeneous Xeon Gold nodes,
  cold NUTS, warm compiled methods); agreement-metric sentence and the runtime Setup
  paragraph of `result_details.tex` adjusted. Left for later (numbers depend on reruns, TODO
  comments in the tex): `tables/nuts_convergence.tex` (nuts_divergences.py, per-level rows),
  the old strict/liberal mentions in `robustness.tex`, the evidence and prior-grid sections
  of `result_details.tex`, the ADVI early-stopping sentences in `results.tex:79` and the
  `tab:real_p` caption, tab:rt_full / fig:rt. Cluster CPU nodes are heterogeneous
  (Xeon 6126–6248R, EPYC), so "four Xeon Gold 6248R cores" is no longer accurate anywhere.
- Real sets: `~/submit-real.sh` submitted the full ladder on the 11 `*-real` test sets
  (nuts 2/0/1 cold on cpu_normal; advi, pathfinder 0/1, laplace warm on cpu_preemptible;
  77 arrays, 39,424 tasks). Old `test.fit.npz` of the real sets stays until validated.
  Afterwards: `check.py --partition test --data_id {id}` per set (cache.py is not used for
  real data; real_posterior.py computes its own summaries).
- Convergence check: `~/eval-conv-check.sh` submitted 24 GPU jobs (gpu_normal, H100):
  evaluate.py on the 12 paper checkpoints (`--prefix latest --models all`), once with
  `--all_datasets` → `metabeta/outputs/results-all_datasets/`, once converged-only →
  `metabeta/outputs/results-converged/` (`--no-plot --save_tables`; the converged run is
  the baseline of the comparison). Alex reviews the two before the remaining evaluation
  items (oracle_posterior.py, runtimes.py, ref_curve.py, real_posterior.py).
- Convergence check done 2026-09-30 (GPU: 80 GB `gpu_priority` runner for small–large, 300 GB
  for huge; 64 GB OOMs on large and huge). Tables in `metabeta/outputs/results-{all_datasets,
  converged}/data=*/evaluate.{md,tex}` on the cluster, pulled to `~/Downloads/hpc-pull/conv-check/`.
  Converged → all: MB and NUTS move by ≤ 0.02 in r and NRMSE and ≤ 0.015 in ECE/EACE on
  every set (largest: huge-b NUTS r 0.726 → 0.706, MB 0.684 → 0.674); ADVI/PF/LA likewise,
  PF r drops up to 0.04 on huge sets. The filter does not flatter any method. Caveats:
  `--all_datasets` still applies the common fit-success mask (small-n: 495 of 512, Laplace
  failures; more on Bernoulli sets via ADVI), and the `tpd` column is identical in both runs
  (time is not masked). LA NRMSE is inf or ~1e5–1e16 on 8 of 12 sets: draws within float32
  but astronomically large pass `markNonFinite`; needs a decision (see below).
- Decisions (Alex, 2026-09-30): NUTS is the main reference; ADVI and PF are the secondary
  comparisons and appear together wherever a competitor is shown; LA is only a failure check
  against PF in its own small appendix section (GLMMs should not use LA). Draws beyond 1e3
  standardized units count as a failed fit (`DRAW_MAX` in fit.py, error 'degenerate draws';
  0–10 extra Laplace failures per sampled set, optimizer `success` is False on > 90 % of fits
  and useless as a filter). Done: real_posterior.py scores ADVI and PF (`SECONDARY`),
  oracle_posterior.py/oracle_corr.py add the PF row, runtimes plotting gets a `pathfinder`
  condition, npe_errors.py and gaussian_local.py moved to the current tags (`nuts`, `advi1`,
  `pathfinder1`; both were still reading `advi_*` keys, untested since no local checkpoint
  run). `padToModel` padded only `nuts_*`/`advi_*`/`laplace_*`; it now pads every `FIT_TAGS`
  prefix, which `Collection(fits=('nuts2', 'advi1', ...))` relied on. Laplace re-aggregated
  and recached on the 12 sampled sets (`~/reagg-laplace.sh`); real sets get the rule through
  check.py. Paper app:tra states the roles and the bound.
- Paper `app:la` rewritten (2026-09-30) as the failure check against PF: why LA is not a GLMM
  method, the medium-b campaign finding (σ-ratio 3.9, r ≈ 0; torch LA 4.5 / 0.18), BFGS
  precision loss > 90 %, 3–13 % failed fits, PF σ-ratio 1.6–1.7; table caption updated to
  the new rows and core counts; `fig:la` and the table numbers marked TODO until
  oracle_posterior.py / evaluate.py rerun. LA clause removed from app:corr. Main text still
  mentions LA (`sections/results.tex:42`) — Alex's call.
- Laplace re-aggregated + recached under `DRAW_MAX` (2026-09-30): failed of 512 (all labelled
  'degenerate draws'): small n/b/p 20/39/36, medium 22/50/28, large 34/53/34, huge 26/63/60
  (4–12 %; before 17–56).

### Real test sets: campaign result (2026-09-30)

All 11 `*-real` sets: every tag 512/512, reintegrated and composed (`check.py`, 7–24 min per
set). 1,630 of the 39,424 tasks first failed with a full `/tmp` on `cpusrv32`/`cpusrv25` and
were refitted with those nodes excluded (`~/refit-real-gaps.sh`; `SBATCH_EXCLUDE` is ignored
by this SLURM, `scontrol update … ExcNodeList` works). Validation (`~/validate_real.py`):

| set | conv nuts0/1/2/comp | r nuts2 vs old NUTS (ffx, σ) | r advi1 vs old ADVI (ffx, σ) | failed advi1/pf1/la | median s nuts0/nuts2/comp/advi1/pf1/la |
|---|---|---|---|---|---|
| small-n | 272/396/505/507 | 0.9999 0.9994 | 0.9999 0.9594 | 0/0/11 | 42 79 76 34 18 15 |
| small-b | 337/438/508/511 | 0.9999 0.9994 | 0.9996 0.9822 | 41/6/9 | 43 82 54 28 19 16 |
| small-p | 119/168/390/395 | 1.0000 0.9992 | 0.4729 0.4213 | 0/1/2 | 51 147 225 26 19 13 |
| medium-n | 297/376/485/494 | 0.9999 0.9995 | 0.9997 0.9833 | 0/0/10 | 70 158 79 33 33 24 |
| medium-b | 320/406/499/508 | 0.9999 0.9991 | 0.9993 0.9737 | 67/0/12 | 68 163 78 47 65 47 |
| medium-p | 153/304/501/503 | 1.0000 0.9996 | 0.7332 0.3918 | 0/1/8 | 72 165 170 47 65 45 |
| large-n | 285/347/479/485 | 0.9982 0.9984 | 0.9993 0.9836 | 0/0/8 | 78 286 98 38 40 32 |
| large-b | 316/386/496/505 | 0.9999 0.9981 | 0.9991 0.9487 | 55/8/12 | 78 255 85 55 73 55 |
| large-p | 213/340/505/505 | 1.0000 0.9997 | 0.7604 0.2636 | 2/5/14 | 77 183 124 51 71 52 |
| huge-n | 315/343/477/486 | 0.9999 0.9963 | 0.9992 0.9150 | 0/1/5 | 81 244 86 42 50 37 |
| huge-b | 319/362/483/494 | 0.9998 0.9983 | 0.9987 0.9589 | 73/1/12 | 82 304 88 60 79 62 |

nuts2 reproduces the old NUTS posterior means (r ≥ 0.998). ADVI agrees with the old ADVI
except on real Poisson (r 0.26–0.76): the old run there was the early-stopping-exhausted one
the paper flagged, the new 100k-iteration run is the trustworthy side. ADVI fails (NaN) on
41–73 Bernoulli datasets, Pathfinder on 0–8 (all paths failed), Laplace on 2–14. Real Poisson
is hard for NUTS at the default budget (119/512 converged on small-p at L0, 390 at L2). Old
`test.fit.npz` of the real sets and the July `summary_test_{nuts,advi}_*` caches are still in
place (deletion pending Alex's confirmation).
