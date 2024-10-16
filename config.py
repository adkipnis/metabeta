from pathlib import Path

def getConfig() -> dict:
    return {
            "n_predictors": 2, # number of linear model features (without bias)
            "n_samples": 50, # number of subjects per draw
            "n_draws": 256, # number of draws
            "batch_size": 4, # number of concurrent samples
            "n_epochs": 10, # number of epochs
            "lr": 1e-3, # learning rate
            "d_model": 128, # size of embedding layer
            "d_ff": 256, # size of feed forward layer
            "n_heads": 4, # number of heads
            "n_blocks_e": 2, # number of encoder blocks
            "n_blocks_d": 6, # number of decoder blocks
            "dropout": 0.1, # dropout rate
            "seq_len": 2000, # max length of sequence
            "seq_len_out": 4*3 + 2, # max length of output sequence
            "precision": 4, # tokenizer float representation
            "preload": None, # None or epoch number
            "model_folder": "weights",
            "model_basename": "tmodel",
            "experiment_name": "runs/linear"
            }

    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

