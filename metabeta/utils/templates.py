"""Template-based config generation and validation.

This module generates preset-based simulation/training configs using
`metabeta/configs/presets.yaml`, applies CLI overrides, and validates via
Pydantic. It intentionally does not define model network schemas; those live in
`metabeta/utils/config.py`.

Usage:
    # Generate simulation config
    cfg = generateSimulationConfig(size='small', family=0, ds_type='mixed')

    # Generate training config
    cfg = generateTrainingConfig(size='small', family=0, ds_type='mixed')

    # Save config to checkpoint
    saveConfigToCheckpoint(cfg, Path('checkpoints/small-0-mixed'))

    # Load config from checkpoint
    cfg = loadConfigFromCheckpoint(Path('checkpoints/small-0-mixed'))
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Literal, Optional, Callable
import yaml
from pydantic import BaseModel, Field, field_validator


# Load presets once at module import
_REPO_ROOT = Path(__file__).parent.parent
_PRESETS_PATH = _REPO_ROOT / 'configs' / 'presets.yaml'

with open(_PRESETS_PATH) as f:
    PRESETS = yaml.safe_load(f)

CLI_ONLY = set(PRESETS['cli_only'])

# Family integer to short name mapping
FAMILY_NAMES = {0: 'n', 1: 'b', 2: 'p'}  # normal, bernoulli, poisson
FAMILY_NAMES_REVERSE = {'n': 0, 'b': 1, 'p': 2}


class SimulationConfig(BaseModel):
    """Validates simulation/data generation configs."""

    ds_type: str
    source: str = 'all'
    likelihood_family: Literal[0, 1, 2]
    min_d: int = Field(ge=1, default=2)  # lower bound for d; defines non-overlapping test band
    max_d: int = Field(gt=0)
    min_q: int = Field(ge=1, default=1)  # lower bound for q; defines non-overlapping test band
    max_q: int = Field(ge=0)
    min_m: int = Field(gt=0)
    max_m: int = Field(gt=0)
    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    max_n_total: int = Field(gt=0)
    min_bg_df: int = Field(ge=0, default=0)  # minimum between-group df (m − q); 0 = unconstrained
    min_within_df: int = Field(ge=0, default=2)  # minimum within-group df (n_g − q); prevents near-singular ZtZ
    data_id: Optional[str] = None  # Auto-generated if not provided

    @field_validator('max_d')
    @classmethod
    def max_d_ge_min_d(cls, v, info):
        if info.data.get('min_d') is not None and v < info.data['min_d']:
            raise ValueError(f"max_d ({v}) must be >= min_d ({info.data['min_d']})")
        return v

    @field_validator('max_q')
    @classmethod
    def max_q_ge_min_q(cls, v, info):
        if info.data.get('min_q') is not None and v < info.data['min_q']:
            raise ValueError(f"max_q ({v}) must be >= min_q ({info.data['min_q']})")
        return v

    @field_validator('max_m')
    @classmethod
    def max_m_gt_min_m(cls, v, info):
        if info.data.get('min_m') is not None and v <= info.data['min_m']:
            raise ValueError(f"max_m ({v}) must be > min_m ({info.data['min_m']})")
        return v

    @field_validator('max_n')
    @classmethod
    def max_n_gt_min_n(cls, v, info):
        if info.data.get('min_n') is not None and v <= info.data['min_n']:
            raise ValueError(f"max_n ({v}) must be > min_n ({info.data['min_n']})")
        return v

    model_config = {'extra': 'allow'}


CLI_ONLY_PARAMS: set[str] = {
    'plot',
    'wandb',
    'load_best',
    'load_latest',
    'save_best',
    'save_latest',
}

# Fields excluded from config.yaml (session/environment specific, one-time actions,
# or generate.py runtime params that leak into training configs via cli_only_defaults).
CONFIG_YAML_EXCLUDE: set[str] = CLI_ONLY_PARAMS | {
    'device',       # environment-specific
    'verbosity',    # session preference
    'partition',    # generate.py runtime param
    'begin',        # generate.py runtime param
    'loop',         # generate.py runtime param
    'sgld',         # generate.py runtime param
}


class TrainingConfig(BaseModel):
    """Validates training configs."""

    name: str
    data_id: str
    data_id_valid: Optional[str] = None
    model_id: str

    # Training hyperparameters
    max_epochs: int = Field(gt=0, default=1000)
    bs: int = Field(gt=0, default=32)
    accum_steps: int = Field(gt=0, default=1)
    lr: float = Field(gt=0, default=3e-4)
    max_grad_norm: float = Field(gt=0, default=1.0)
    loss_type: str = 'forward'
    ancestral_forward: bool = False
    patience: int = Field(ge=0, default=0)
    sample_interval: int = Field(gt=0, default=20)
    skip_ref: bool = False

    # Runtime settings (will be overridden by CLI-only params)
    cores: int = Field(gt=0, default=8)
    reproducible: bool = True
    compile: bool = False

    # Evaluation settings
    n_samples: int = Field(gt=0, default=512)
    rescale: bool = True
    importance: bool = False
    sir: bool = False
    sir_iter: int = Field(gt=0, default=8)
    sir_n_proposal: int = Field(gt=0, default=2048)
    plot: bool = True

    # Checkpoint settings
    r_tag: str = ''
    save_latest: bool = True
    save_best: bool = True
    load_latest: bool = False
    load_best: bool = False

    model_config = {'extra': 'allow'}


def generateSimulationConfig(size: str, family: int, ds_type: str, **overrides) -> dict[str, Any]:
    """
    Generate simulation config from size/family presets with specified ds_type.

    Args:
        size: One of tiny/small/medium/large/huge
        family: Likelihood family integer (0=normal, 1=bernoulli, 2=poisson)
        ds_type: Dataset type (toy/flat/scm/mixed/sampled/observed)
        **overrides: Additional overrides from CLI (excluding CLI-only params)

    Returns:
        Validated config dict with auto-generated data_id

    Raises:
        ValueError: If size/family is invalid
        pydantic.ValidationError: If generated config is invalid
    """
    if size not in PRESETS['sizes']:
        raise ValueError(f"Invalid size '{size}'. Choose from: {list(PRESETS['sizes'].keys())}")
    if family not in PRESETS['families']:
        raise ValueError(
            f"Invalid family {family}. Choose from: {list(PRESETS['families'].keys())}"
        )

    # Merge presets
    cfg = {}
    cfg.update(PRESETS['sizes'][size])
    cfg.update(PRESETS['families'][family])
    cfg['ds_type'] = ds_type

    # Apply overrides (excluding CLI-only params)
    for k, v in overrides.items():
        if k not in CLI_ONLY and v is not None:
            cfg[k] = v

    # Auto-generate data_id if not provided
    if 'data_id' not in cfg:
        family_name = FAMILY_NAMES[family]
        cfg['data_id'] = f'{size}-{family_name}-{ds_type}'

    # Validate
    validated = SimulationConfig(**cfg)
    return validated.model_dump()


def generateTrainingConfig(
    size: str,
    family: int,
    ds_type: str,
    valid_ds_type: Optional[str] = None,
    **overrides,
) -> dict[str, Any]:
    """
    Generate training config from size/family presets with specified ds_type.

    Args:
        size: One of tiny/small/medium/large/huge
        family: Likelihood family integer (0=normal, 1=bernoulli, 2=poisson)
        ds_type: Dataset type for training (toy/flat/scm/mixed/sampled/observed)
        valid_ds_type: Optional validation dataset type (defaults to 'sampled')
        **overrides: Additional overrides from CLI (excluding CLI-only params)

    Returns:
        Validated config dict with auto-generated data_id fields

    Raises:
        ValueError: If size/family is invalid
        pydantic.ValidationError: If generated config is invalid
    """
    if size not in PRESETS['sizes']:
        raise ValueError(f"Invalid size '{size}'. Choose from: {list(PRESETS['sizes'].keys())}")
    if family not in PRESETS['families']:
        raise ValueError(
            f"Invalid family {family}. Choose from: {list(PRESETS['families'].keys())}"
        )

    # Auto-generate tags based on size/family/ds_type
    family_name = FAMILY_NAMES[family]
    cfg = {
        'name': f'{size}-{family_name}-{ds_type}',
        'data_id': f'{size}-{family_name}-{ds_type}',
        'data_id_valid': f"{size}-{family_name}-{valid_ds_type or 'sampled'}",
        'model_id': size,
    }

    # Apply overrides (excluding CLI-only params)
    for k, v in overrides.items():
        if k not in CLI_ONLY and v is not None:
            cfg[k] = v

    # Validate
    validated = TrainingConfig(**cfg)  # type: ignore
    return validated.model_dump()


def saveConfigToCheckpoint(cfg: dict[str, Any], checkpoint_dir: Path) -> None:
    """
    Save resolved config to checkpoint directory for reproducibility.

    Session-specific and one-time-action fields (defined in CONFIG_YAML_EXCLUDE)
    are stripped so that loading the saved config for resuming does not re-trigger
    load flags or inherit the original device/verbosity/etc.

    Args:
        cfg: Configuration dict to save
        checkpoint_dir: Path to checkpoint directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filtered = {k: v for k, v in cfg.items() if k not in CONFIG_YAML_EXCLUDE}
    config_path = checkpoint_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(filtered, f, sort_keys=False, default_flow_style=False)


def loadConfigFromCheckpoint(checkpoint_dir: Path) -> dict[str, Any]:
    """
    Load config from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Config dict

    Raises:
        FileNotFoundError: If config.yaml not found in checkpoint dir
    """
    config_path = Path(checkpoint_dir) / 'config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(
            f'No config.yaml found in {checkpoint_dir}. '
            'This checkpoint may be from an older version.'
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def getExplicitArgs() -> set[str]:
    """
    Returns set of argument names explicitly provided via CLI.

    This allows us to distinguish between user-provided values and argparse defaults,
    so we only override YAML config values with explicitly-provided CLI args.
    """
    explicit = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Handle --arg or --arg=value
            arg_name = arg[2:].split('=')[0]
            explicit.add(arg_name)
        elif arg.startswith('-') and len(arg) == 2:
            # Handle short flags like -b, -e
            # Map to long form - need to handle different scripts
            flag_map = {
                'b': 'begin',  # generate.py
                'e': 'epochs',  # generate.py (will be mapped to max_epochs in train.py)
            }
            if arg[1] in flag_map:
                long_form = flag_map[arg[1]]
                explicit.add(long_form)
                # Also add the alternative mapping for train.py
                if long_form == 'epochs':
                    explicit.add('max_epochs')
    return explicit


def setupConfigParser(
    parser: argparse.ArgumentParser,
    config_generator: Callable[..., dict[str, Any]],
    description: str = '',
) -> argparse.Namespace:
    """
    Unified configuration setup for scripts using template-based configs.

    Args:
        parser: Pre-configured ArgumentParser with script-specific arguments
        config_generator: Function to generate config (generateSimulationConfig or generateTrainingConfig)
        description: Parser description

    Returns:
        argparse.Namespace with merged configuration
    """
    if description:
        parser.description = description

    # Parse arguments
    args = parser.parse_args()
    explicit_args = getExplicitArgs()

    # Generate config: either from custom YAML or from templates
    if hasattr(args, 'config') and args.config:
        # Path 1: Load custom YAML config
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        # Only explicit CLI args override YAML values
        for k, v in vars(args).items():
            if k in explicit_args and k != 'config':
                cfg_dict[k] = v
    else:
        # Path 2: Template-based generation with defaults
        args_dict = vars(args)

        # Separate template args from overrides
        template_args = {}
        overrides = {}

        for k, v in args_dict.items():
            if k in ['size', 'family', 'ds_type', 'valid_ds_type']:
                template_args[k] = v
            elif k not in ['config'] and k not in CLI_ONLY:
                if v is not None:  # Only include non-None values
                    overrides[k] = v

        # Generate config using the appropriate generator
        cfg_dict = config_generator(**template_args, **overrides)

    # Handle CLI-only parameters with proper defaults
    cli_only_defaults = {
        'device': 'cpu',
        'wandb': False,
        'seed': 42,
        'verbosity': 1,
        # generate.py runtime-only keys stripped from saved configs
        'partition': 'all',
        'begin': 1,
        'loop': False,
        'sgld': False,
        # training flags added after initial release; default keeps old behaviour
        'ancestral_forward': False,
    }

    for key, default_value in cli_only_defaults.items():
        if key in explicit_args:
            cfg_dict[key] = getattr(args, key)
        else:
            cfg_dict.setdefault(key, default_value)

    return argparse.Namespace(**cfg_dict)
