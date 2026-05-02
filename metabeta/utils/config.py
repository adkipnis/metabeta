"""Config schemas and config loading helpers.

This module defines typed configuration models (Pydantic) and helpers for
loading model/data configs from YAML. It does not generate preset-based
configs; those live in `metabeta/utils/templates.py`.
"""

import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from metabeta.utils.io import datasetFilename
from metabeta.utils.templates import (
    generateSimulationConfig,
    PRESETS,
    FAMILY_NAMES_REVERSE,
)


class SummarizerConfig(BaseModel):
    d_model: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    d_output: int = Field(gt=0)
    n_blocks: int = Field(gt=0)
    n_isab: int = Field(ge=0, default=0)
    n_inducing: int = Field(gt=0, default=32)
    pooling: str = 'cls'
    activation: str = 'GELU'
    dropout: float = Field(ge=0.0, default=0.01)
    type: Literal['set-transformer'] = 'set-transformer'

    model_config = {'extra': 'allow'}

    def to_dict(self) -> dict:
        return self.model_dump()


class PosteriorConfig(BaseModel):
    n_blocks: int = Field(gt=0)
    subnet_kwargs: dict | None = None
    type: Literal['coupling'] = 'coupling'
    transform: Literal['affine', 'spline'] = 'affine'
    base_family: Literal['normal', 'student'] = 'normal'
    base_trainable: bool = True

    model_config = {'extra': 'allow'}

    def to_dict(self) -> dict:
        return self.model_dump()


class ApproximatorConfig(BaseModel):
    d_ffx: int = Field(gt=0)
    d_rfx: int = Field(ge=0)
    summarizer_l: SummarizerConfig
    summarizer_g: SummarizerConfig
    posterior_l: PosteriorConfig
    posterior_g: PosteriorConfig
    likelihood_family: int = Field(ge=0, default=0)
    posterior_correlation: bool = True
    analytical_context: bool = True
    analytical_blup_from_globals: bool = True
    model_config = {'extra': 'allow'}

    @property
    def d_corr(self) -> int:
        """Number of unconstrained partial-correlation scalars (q*(q-1)//2, or 0)."""
        if self.posterior_correlation and self.d_rfx >= 2:
            return self.d_rfx * (self.d_rfx - 1) // 2
        return 0

    def to_dict(self) -> dict:
        return {
            'd_ffx': self.d_ffx,
            'd_rfx': self.d_rfx,
            'likelihood_family': self.likelihood_family,
            'posterior_correlation': self.posterior_correlation,
            'analytical_context': self.analytical_context,
            'analytical_blup_from_globals': self.analytical_blup_from_globals,
            'summarizer_l': self.summarizer_l.model_dump(),
            'summarizer_g': self.summarizer_g.model_dump(),
            'posterior_l': self.posterior_l.model_dump(),
            'posterior_g': self.posterior_g.model_dump(),
        }


def modelFromYaml(
    cfg_path: Path, d_ffx: int, d_rfx: int, likelihood_family: int = 0
) -> ApproximatorConfig:
    with open(cfg_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    # Global config is the full spec; local inherits from global and overrides what differs
    s_g = model_cfg['summarizer_g']
    p_g = model_cfg['posterior_g']
    return ApproximatorConfig(
        d_ffx=d_ffx,
        d_rfx=d_rfx,
        likelihood_family=likelihood_family,
        posterior_correlation=model_cfg['posterior_correlation'],
        analytical_context=model_cfg.get('analytical_context', True),
        analytical_blup_from_globals=model_cfg.get('analytical_blup_from_globals', True),
        summarizer_g=SummarizerConfig(**s_g),
        summarizer_l=SummarizerConfig(**{**s_g, **model_cfg.get('summarizer_l', {})}),
        posterior_g=PosteriorConfig(**p_g),
        posterior_l=PosteriorConfig(**{**p_g, **model_cfg.get('posterior_l', {})}),
    )


def dataFromYaml(cfg_path: Path, partition: str, epoch: int = 0) -> str:
    with open(cfg_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    data_id = data_cfg.get('data_id', cfg_path.stem)
    return str(Path(data_id) / datasetFilename(partition, epoch))


def loadDataConfig(data_id: str) -> dict:
    """
    Load data configuration by data_id.

    First tries template-based config generation (from data_id pattern like 'small-n-mixed'),
    then falls back to looking for config.yaml in outputs/data/{data_id}/.
    Returns data config dict.
    """

    # Try to parse as template-based data_id: size-family-ds_type
    parts = data_id.split('-')
    if len(parts) >= 3:
        size = parts[0]
        family_str = parts[1]
        ds_type = '-'.join(parts[2:])  # Handle types with hyphens

        # Check if it's a valid template combination
        if size in PRESETS['sizes']:
            # Try parsing family as word (n/b/p) or integer
            family = None
            if family_str in FAMILY_NAMES_REVERSE:
                family = FAMILY_NAMES_REVERSE[family_str]
            else:
                try:
                    family = int(family_str)
                except ValueError:
                    pass

            if family is not None and family in PRESETS['families']:
                # Generate from template
                return generateSimulationConfig(size=size, family=family, ds_type=ds_type)

    # Fallback: try loading from dataset directory (new location)
    root = Path(__file__).resolve().parent
    data_dir = Path(root, '..', 'outputs', 'data', data_id)
    config_path = data_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f'Data config not found for data_id: {data_id}\n'
        f'  - Template-based pattern: <size>-<family>-<ds_type> (e.g., small-n-mixed)\n'
        f'  - Dataset config file: {config_path}\n'
    )


def assimilateConfig(big: argparse.Namespace, small: dict) -> None:
    for k, v in small.items():
        setattr(big, k, v)
