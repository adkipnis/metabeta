import os
from typing import Tuple, Callable
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import schedulefree
from generator import Task, LinearModel
from dataset import LinearModelDataset, causalMask
from tokenizer import FloatTokenizer
from model import Transformer
from config import getConfig, getWeightsFilePath

def greedyDecode(model: nn.Module,
                 source: torch.Tensor,
                 source_mask: torch.Tensor,
                 tokenizer: FloatTokenizer,
                 max_len: int,
                 device: torch.device) -> torch.Tensor:
        sos_idx = tokenizer.tokenToIdx("[SOS]")
        eos_idx = tokenizer.tokenToIdx("[EOS]")

        # precompute encoder output and reuse it for every token we get from the decoder
        encoder_output = model.encode(source, source_mask) # (1, seq_len, d_model)

        # initialize decoder input with SOS token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

        # greedy decoding
        while True:
            if decoder_input.size(1) >= max_len:
                break
            decoder_mask = causalMask(decoder_input.size(1)).type_as(source).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.projection(out[:, -1]) # give logprob for last token
            prob_mask = tokenizer.conditionalMask(decoder_input[:, -1].item()).to(device)
            prob[0, ~prob_mask] = -float("inf")
            _, next_token = torch.max(prob, dim=-1)
            next_token = torch.empty(1, 1).type_as(source).fill_(next_token.item()).to(device)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            if next_token.item() == eos_idx:
                break
        return decoder_input.squeeze(0)


class Trainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        self.initial_epoch = 0
        self.seed = 0
        self.current_seed = 0
        print(f"Device: {self.device}")
        self.tokenizer = FloatTokenizer(precision=config["precision"])
        self.model = Transformer(d_model=config["d_model"],
                        d_ff=config["d_ff"],
                        n_heads=config["n_heads"],
                        n_blocks_e=config["n_blocks_e"],
                        n_blocks_d=config["n_blocks_d"],
                        vocab_size=self.tokenizer.getVocabSize(),
                        dropout=config["dropout"]).to(self.device)
        # optimizer and loss
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(),
                                                        lr=config["lr"], eps=1e-9)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tokenToIdx("[PAD]"),
                                           label_smoothing=0.1).to(self.device)

        if config["preload"]:
            self.preload()

        # create model folder and tensorboard writer
        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(config["experiment_name"])

    def preload(self) -> None:
        model_filename = getWeightsFilePath(self.config, epoch=self.config["preload"])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.initial_epoch = state["epoch"] + 1
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state["global_step"]
        self.seed = state["seed"]
        self.current_seed = state["current_seed"]

    def getN(self, seed: int) -> int:
        torch.manual_seed(seed)
        low = 10 * (self.config["n_predictors"] + 1)
        n = torch.randint(low, 100, (1,))
        return int(n.item())

    def getDataset(self, config: dict, seed: int) -> Tuple[DataLoader, DataLoader]:
        # create dataset
        dataset = []
        for _ in range(config["n_draws"]):
            task = Task(n_predictors=self.config["n_predictors"], seed=seed)
            lm = LinearModel(task)
            # n_samples = self.config["n_samples"]
            n_samples = self.getN(seed) # TODO: fix collate_fn
            dataset += [lm.sample(n_samples, seed)]
            seed += 1

        # train validation split
        train_ds_size = int(0.9 * len(dataset))
        val_ds_size = len(dataset) - train_ds_size
        print(f"Train size: {train_ds_size}, Validation size: {val_ds_size}")
        train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])
        train_ds = LinearModelDataset(train_ds_raw, self.tokenizer, self.config["seq_len"])
        val_ds = LinearModelDataset(val_ds_raw, self.tokenizer, self.config["seq_len"])

        # dataloaders
        train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

        return train_dl, val_dl
    
    def runBatch(self, batch: dict) -> float:
        self.model.train()
        encoder_input = batch["encoder_input"].to(self.device) # (batch_size, seq_len)
        encoder_mask = batch["encoder_mask"].to(self.device) # (batch_size, 1, 1, seq_len)
        decoder_input = batch["decoder_input"].to(self.device) # (batch_size, seq_len)
        decoder_mask = batch["decoder_mask"].to(self.device) # (batch_size, 1, seq_len, seq_len)

        # forward pass
        proj_output = self.model(encoder_input, decoder_input, encoder_mask, decoder_mask)

        # CE loss
        label = batch["label"].to(self.device) # (batch_size, seq_len)
        loss = self.ce_loss(proj_output.view(-1, self.tokenizer.getVocabSize()), label.view(-1))
        self.writer.add_scalar("train_loss", loss.item(), self.global_step)
        self.writer.flush()
        self.optimizer.train()

        # # MSE loss
        # tgt = batch["tgt"].to(self.device)
        # pred_idx = torch.argmax(proj_output, dim=-1)
        # pred = torch.tensor(self.tokenizer.batchDecode(pred_idx)).to(self.device)
        # mse_loss = self.filteredMSE(pred, tgt)
        # self.writer.add_scalar("train_loss_mse", mse_loss.item(), self.global_step)
        # self.writer.flush()
        # loss += mse_loss

        # backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        return loss.item()

    def runEpoch(self, epoch: int, seed: int, train: bool = True, val: bool = False) -> None:
        train_dl, val_dl = self.getDataset(config=self.config, seed=seed)
        batch_iterator = tqdm(train_dl, desc=f"Epoch {epoch:02d}")

        # train
        if train:
            for batch in batch_iterator:
                loss = self.runBatch(batch)
                batch_iterator.set_postfix({"loss": loss})

        # validation
        if val:
            self.runValidation(val_dl, batch_iterator.write)

    def runValidation(self,
                      validation_ds: DataLoader,
                      print_msg: Callable, # dont't interfere with tqdm
                      num_examples: int = 2) -> None:
        self.model.eval()
        self.optimizer.eval()
        count = 0
        mses = []

        # get the console window width
        try:
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
        except:
                console_width = 80
        # console_width = 80

        with torch.no_grad():
            for batch in validation_ds:
                count += 1
                error = False

                # get data
                encoder_input = batch["encoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

                # save text for printing
                target_text = batch["tgt"][0].detach().cpu()
                target_tokens = ['[SOS]'] + self.tokenizer.encode(target_text) + ['[EOS]']

                # run greedy decoding
                model_out = greedyDecode(self.model, encoder_input, encoder_mask,
                                         self.tokenizer, self.config["seq_len_out"], self.device)
                model_out_ = model_out.detach().cpu().tolist()
                model_out_tokens = [self.tokenizer.idxToToken(idx) for idx in model_out_]
                try:
                    model_out_text = torch.tensor(self.tokenizer.decode(model_out_tokens))
                    mses += [torch.mean((target_text - model_out_text) ** 2)]
                except:
                    # print_msg(f"Error decoding: {model_out_tokens}")
                    error = True
                    model_out_text = "".join(model_out_tokens)

                # # validation loss pass # TODO
                # proj_output = self.model.projection(model_out)
                # label = batch["label"]
                # loss = self.loss_fn(proj_output.view(-1, self.tokenizer.getVocabSize()), label.view(-1))
                # self.writer.add_scalar("val_loss", loss.item(), self.global_step)
                # self.writer.flush()

                # print some examples
                if count <= num_examples:
                    # print some examples
                    print_msg('-' * console_width)
                    if error:
                        print_msg(f"Target (tokens): {target_tokens}")
                        print_msg(f"Predicted (tokens): {model_out_tokens}")
                    else:
                        print_msg(f"Target: {target_text}")
                        print_msg(f"Predicted: {model_out_text}")
                        print_msg(f"Differences: {target_text - model_out_text}")
                        self.writer.add_scalar("val_mse", mses[-1].item(), self.global_step)
        print_msg(f"Validation MSE: {torch.mean(torch.tensor(mses)).item()}")
   
    def saveModel(self, epoch: int) -> None:
        model_filename = getWeightsFilePath(self.config, epoch)
        torch.save({
            'epoch': epoch,
            'seed': self.seed,
            'current_seed': self.current_seed,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_filename)


def main():
    config = getConfig()
    trainer = Trainer(config)
    epochs = range(trainer.initial_epoch, config['n_epochs'])
    for epoch in epochs:
        trainer.runEpoch(epoch, trainer.current_seed)
        trainer.current_seed += trainer.config["n_draws"]
        trainer.saveModel(epoch)


if __name__ == "__main__":
    main()

