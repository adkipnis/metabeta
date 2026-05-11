"""Shared helpers for experiment scripts."""

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
METABETA_DIR = REPO_ROOT / 'metabeta'
EXPERIMENTS_DIR = REPO_ROOT / 'experiments'
RESULTS_DIR = EXPERIMENTS_DIR / 'results'
OUTPUTS_DIR = METABETA_DIR / 'outputs'
DATA_DIR = OUTPUTS_DIR / 'data'
EVALUATION_CONFIG_DIR = METABETA_DIR / 'evaluation' / 'configs'
MODEL_CONFIG_DIR = METABETA_DIR / 'configs' / 'models'
CHECKPOINT_DIR = OUTPUTS_DIR / 'checkpoints'


def experimentResultsPath(*parts: str) -> Path:
    """Return a path under the shared experiments/results directory."""
    return RESULTS_DIR.joinpath(*parts)


def evaluationConfigPath(name: str | Path) -> Path:
    """Return an evaluation config path from either a name or explicit path."""
    path = Path(name)
    if path.suffix:
        return path
    return EVALUATION_CONFIG_DIR / f'{path.name}.yaml'


def dataDir(data_id: str) -> Path:
    """Return the output directory for a generated dataset id."""
    return DATA_DIR / data_id


def dataFilePath(
    data_id: str,
    partition: str = 'test',
    epoch: int = 0,
    *,
    fit: bool = False,
) -> Path:
    """Return a generated dataset file path."""
    from metabeta.utils.io import datasetFilename

    path = dataDir(data_id) / datasetFilename(partition, epoch)
    return path.with_suffix('.fit.npz') if fit else path


def modelConfigPath(model_id: str) -> Path:
    """Return the model YAML path for a model id."""
    return MODEL_CONFIG_DIR / f'{model_id}.yaml'


def checkpointFilePath(checkpoint_dir: str | Path, prefix: str) -> Path:
    """Return a checkpoint file path."""
    return Path(checkpoint_dir) / f'{prefix}.pt'


def loadModelConfig(cfg: Namespace | dict[str, Any]):
    """Load an ApproximatorConfig from an experiment namespace or dict."""
    from metabeta.utils.config import modelFromYaml

    values = vars(cfg) if isinstance(cfg, Namespace) else cfg
    return modelFromYaml(
        modelConfigPath(values['model_id']),
        d_ffx=values['max_d'],
        d_rfx=values['max_q'],
        likelihood_family=values.get('likelihood_family', 0),
    )


def loadApproximator(
    cfg: Namespace | dict[str, Any],
    device: torch.device,
    checkpoint_dir: str | Path,
    prefix: str,
    *,
    compile_model: bool = False,
):
    """Build an Approximator and restore checkpoint weights."""
    from metabeta.models.approximator import Approximator

    model = Approximator(loadModelConfig(cfg)).to(device)
    model.eval()

    path = checkpointFilePath(checkpoint_dir, prefix)
    assert path.exists(), f'checkpoint not found: {path}'
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'])

    if compile_model and device.type != 'mps':
        model.compile()

    return model
