from typing import Tuple, Callable
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from generator import Task, LinearModel
from dataset import LinearModelDataset, causalMask
from tokenizer import FloatTokenizer
from model import Transformer
from config import getConfig, getWeightsFilePath


def getDataset(config: dict) -> Tuple[DataLoader, DataLoader, FloatTokenizer]:
    # create dataset
    tokenizer = FloatTokenizer(precision=config["precision"])
    task = Task(n_predictors=config["n_predictors"])
    lm = LinearModel(task)
    dataset = []
    for _ in range(config["n_draws"]):
        task = Task(n_predictors=config["n_predictors"])
        lm = LinearModel(task)
        dataset += [lm.sample(config["n_samples"])]

    # train validation split
    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    print(f"Train size: {train_ds_size}, Validation size: {val_ds_size}")
    train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])
    train_ds = LinearModelDataset(train_ds_raw, tokenizer, config["seq_len"])
    val_ds = LinearModelDataset(val_ds_raw, tokenizer, config["seq_len"])

    # dataloaders
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dl, val_dl, tokenizer


def trainModel(config):
    # definde device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # get dataset
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dl, val_dl, tokenizer = getDataset(config)
    model = Transformer(d_model=config["d_model"],
                        d_ff=config["d_ff"],
                        n_heads=config["n_heads"],
                        n_blocks_e=config["n_blocks_e"],
                        n_blocks_d=config["n_blocks_d"],
                        vocab_size=tokenizer.getVocabSize(),
                        dropout=config["dropout"]).to(device)

    # tensorboard
    writer = SummaryWriter(config["experiment_name"])

    # optimizer and potentially load weights
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = getWeightsFilePath(config, epoch=config["preload"])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # loss function
    padding_idx = tokenizer.tokenToIdx("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx,
                                  label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['n_epochs']):
        batch_iterator = tqdm(train_dl, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)

            # forward pass
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.projection(decoder_output) # (batch_size, seq_len, vocab_tgt_len)
            label = batch["label"].to(device) # (batch_size, seq_len)

            # (batch_size, seq_len, vocab_tgt_len) -> (batch_size * seq_len, vocab_tgt_len)
            loss = loss_fn(proj_output.view(-1, tokenizer.getVocabSize()), label.view(-1))
            batch_iterator.set_postfix({"loss": loss.item()})
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # validation at the end of each epoch
        runValidation(model, val_dl, tokenizer, config["seq_len_out"], device, batch_iterator.write, global_step, writer)
    
        # save model
        model_filename = getWeightsFilePath(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


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
        _, next_token = torch.max(prob, dim=-1)
        next_token = torch.empty(1, 1).type_as(source).fill_(next_token.item()).to(device)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        if next_token.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


def runValidation(model: nn.Module,
                   validation_ds: DataLoader,
                   tokenizer_src: Tokenizer,
                   tokenizer_tgt: Tokenizer,
                   max_len: int,
                   device: torch.device,
                   print_msg: Callable, # dont't interfere with tqdm
                   global_step: int,
                   writer: SummaryWriter,
                   num_examples: int = 2):
     model.eval()
     count = 0

     source_texts = []
     expected = []
     predicted = []

     # get the console window width
     try:
         with os.popen('stty size', 'r') as console:
             _, console_width = console.read().split()
             console_width = int(console_width)
     except:
             console_width = 80

     with torch.no_grad():
         for batch in validation_ds:
             count += 1

             # get data
             encoder_input = batch["encoder_input"].to(device)
             encoder_mask = batch["encoder_mask"].to(device)
             assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

             # run greedy decoding
             model_out = greedyDecode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

             # save text for printing
             source_text = batch["src_text"][0]
             target_text = batch["tgt_text"][0]
             model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
             source_texts += [source_text]
             expected += [target_text]
             predicted += [model_out_text]

             # print some examples
             print_msg('-' * console_width)
             print_msg(f"Source: {source_text}")
             print_msg(f"Target: {target_text}")
             print_msg(f"Predicted: {model_out_text}")

             if count == num_examples:
                 break
     
     if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

if __name__ == "__main__":
    config = getConfig()
    trainModel(config)


