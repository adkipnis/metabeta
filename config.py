from pathlib import Path
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

def getConfig() -> dict:
    cfg = {
            "n_predictors": 1, # number of linear model features (without bias)
            "n_draws": 1024, # number of datasets per epoch
            "batch_size": 128, # number of datasets per batch
            "n_epochs": 1000, # number of epochs
            "lr": 1e-3, # learning rate
            "d_model": 64, # size of embedding layer
            "d_ff": 128, # size of feed forward layer
            "d_out": 16, # size of output layer
            "n_heads": 2, # number of heads
            "n_blocks_e": 2, # number of encoder blocks
            "n_blocks_d": 4, # number of decoder blocks
            "dropout": 0.1, # dropout rate
            "seq_len": 2000, # max length of sequence
            "precision": 4, # tokenizer float representation
            "preload": None, # None or epoch number
            "model_folder": "weights",
            "model_basename": "tmodel-numeric",
            }
    cfg["seq_len_out"] = (1 + cfg["n_predictors"]) * 3 + 2
    cfg["experiment_name"] = f"runs/linear-mse-{cfg['n_predictors']}/{timestamp}"
    return cfg

def getWeightsFilePath(cfg: dict, epoch: int):
    model_folder = cfg["model_folder"]
    model_basename = cfg["model_basename"]
    model_filename = f"{model_basename}-{epoch:02d}.pt"
    return str(Path('.') / model_folder / model_filename)

