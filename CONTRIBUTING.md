# Contributing to MetaBeta

## Ways to help

- **A bug.** Open an issue with the smallest example that shows it, or a pull request with a
  failing test and the fix.
- **A feature request.** Open an issue describing the feature and the use case behind it. The
  [FAQ](README.md#faq) lists what MetaBeta can fit today.

For anything bigger than a small fix, please open an issue before writing code, so we can agree on the
approach first.

## Questions and support

- Check the [FAQ](README.md#faq) and the [tutorials](README.md#tutorials) first.
- MetaBeta is maintained by one person alongside research. Issues are answered as time permits.

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

- Branch off `main`. One change per pull request, linked to its issue.
- Please add a test with each significant change; mark a test that needs local checkpoints or data `slow`.
- Format with `uv run blue metabeta tests` and run `uv run pytest`.
- No AI slop allowed. If you use AI to assist you in coding (which you are welcome to), always
  make sure that you have read, reviewed, and understood your code.

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
  place. Use broadcasting and indexing instead of loops.
- Use American spelling.

## License

By contributing, you agree that your changes are released under [Apache-2.0](LICENSE). The
pretrained weights are released separately, on Hugging Face, under CC BY-NC-SA 4.0.
