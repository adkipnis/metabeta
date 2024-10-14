from pathlib import Path

def getConfig() -> dict:
    return {
            "batch_size": 8,
            "num_epochs": 20,
            "lr": 1e-3,
            "seq_len": 350,
            "d_model": 512,
            "model_folder": "weights",
            "model_basename": "tmodel_",
            "preload": None,
            "tokenizer_file": "tokenizer_{0}.json",
            "experiment_name": "runs/tmodel"
            }

def getWeightsFilePath(config: dict, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

