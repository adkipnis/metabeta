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
