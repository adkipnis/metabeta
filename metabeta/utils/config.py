import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from metabeta.utils.io import datasetFilename

@dataclass(frozen=True)
class SummarizerConfig:
    d_model: int
    d_ff: int
    d_output: int
    n_blocks: int
    n_isab: int = 0
    activation: str = 'GELU'
    dropout: float = 0.01
    type: str = 'set-transformer'
    
    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop('type')
        return out

@dataclass(frozen=True)
class PosteriorConfig:
    n_blocks: int
    subnet_kwargs: dict | None = None
    type: str = 'flow'
    transform: str = 'spline'
    
    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop('type')
        return out

@dataclass(frozen=True)
class ApproximatorConfig:
    d_ffx: int
    d_rfx: int
    summarizer: SummarizerConfig
    posterior: PosteriorConfig

def modelFromYaml(cfg_path: Path, d_ffx: int, d_rfx: int) -> ApproximatorConfig:
    with open(cfg_path, 'r') as f:
        model_cfg = yaml.safe_load(f)
    cfg_s = SummarizerConfig(**model_cfg['summarizer'])
    cfg_p = PosteriorConfig(**model_cfg['posterior'])
    return ApproximatorConfig(d_ffx=d_ffx,
                              d_rfx=d_rfx,
                              summarizer=cfg_s,
                              posterior=cfg_p)

def dataFromYaml(cfg_path: Path, partition: str,  epoch: int = 0) -> str:
    with open(cfg_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    return datasetFilename(data_cfg, partition, epoch)
