# Contributing to MetaBeta

Thank you for your interest. Help is welcome.

## Ways to help

- **A dataset MetaBeta fits badly.** Open an issue with the formula, the family, the numbers of
  rows and groups, and the summary's last line, which holds the diagnostics. Link the data if it
  is public.
- **A bug.** Open an issue with the smallest example that shows it, or a pull request with a
  failing test and the fix.
- **A model MetaBeta cannot fit yet.** A family, a formula term or a grouping structure: open an
  issue describing the model and a dataset that needs it. The [FAQ](README.md#faq) lists what is
  supported.

For anything bigger than a small fix, open an issue before writing code, so we can agree on the
approach first.

## Questions and support

- Check the [FAQ](README.md#faq) and the [tutorials](README.md#tutorials) first.
- Ask anything else in a [GitHub issue](https://github.com/adkipnis/metabeta/issues).
- MetaBeta is maintained by one person alongside research. Issues are answered as time allows,
  bugs first.

## Set up

You need [uv](https://docs.astral.sh/uv/), and R for `rpy2` in the research stack.

```
git clone https://github.com/adkipnis/metabeta
cd metabeta
uv sync                              # the package, the dev tools and the research stack
uv build                             # the release files that tests/packaging checks
uv run pytest                        # the fast tests
uv run blue --check metabeta tests   # the formatting that CI checks
```

- **Without R**: install what CI installs, `uv sync --no-default-groups --group dev`, then run
  `uv run --no-sync pytest tests/models tests/evaluation tests/utils`.
- **Slow tests**: `uv run pytest -m slow` needs local checkpoints and data under
  `metabeta/outputs/`.

## Find your way

```
metabeta/
  models/       the public Api (api.py), the approximator, Set Transformers and flows
  analytical/   closed-form estimates that join the summaries
  posthoc/      corrections of the networks' draws: importance sampling, Metropolis–Hastings
  evaluation/   posterior summaries, intervals, predictive checks and calibration
  plotting/     the figures, plotParameters and plotPPD among them
  utils/        configs, the checkpoint download, priors, families and shared helpers
  configs/      the presets and model configs, as YAML
  simulation/   priors, covariate generators and the simulator for training data
  training/     the training script
  datasets/     the real datasets and their preprocessing
tests/          mirrors metabeta/
demos/          the tutorial notebooks
experiments/    the scripted studies behind the paper; see its README
scripts/        the cluster job scripts
```

The pipeline runs from simulation to training to evaluation. Users only meet `Api`, which routes
each dataset to a pretrained `Approximator`.

## Pull requests

- Branch off `main`, one change per pull request, linked to its issue. Pull requests are
  squash-merged.
- Add a test with each change; mark a test that needs local checkpoints or data `slow`.
- Format with `uv run blue metabeta tests` and run `uv run pytest`. CI checks both on Python 3.12
  and 3.13.

Style:

- `blue` at 100 columns.
- Absolute imports: `from metabeta.models.api import Api`.
- `PascalCase` for classes, `camelCase` for functions and methods, `snake_case` for variables,
  `UPPER_SNAKE_CASE` for constants.
- Modern typing: `X | Y`, built-in generics, `from __future__ import annotations`. Configs are
  Pydantic models.
- `ValueError` for a bad value or shape, `KeyError` for a missing key, `NotImplementedError` for
  an unsupported mode.
- A tensor's shape is annotated inline (`# (B, m, n, d)`). A reused tensor is never changed in
  place. Broadcasting and indexing, not loops.
- American spelling, in names as in prose.

Three rules protect what users install:

- **Released checkpoints keep loading.** `Api.from_pretrained` downloads the checkpoints under
  the Hugging Face tag `v1`. A change to an architecture or a config they read is discussed in an
  issue first; one that breaks them needs a new tag.
- **The training distribution does not move silently.** The priors, the likelihoods and the
  simulator define what the checkpoints learned. A change to them, to a seed or to a metric
  starts as an issue.
- **The wheel ships only what fitting needs.** `simulation/`, `training/` and `outputs/` stay
  out of it, and research-only dependencies such as PyMC and rpy2 stay in the `research` extra.
  `tests/packaging` checks both.

## License

By contributing, you agree that your changes are released under [Apache-2.0](LICENSE). The
pretrained weights are released separately, on Hugging Face, under CC BY-NC-SA 4.0.
