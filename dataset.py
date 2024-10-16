from typing import List, Dict
import torch
from torch.utils.data import Dataset
from tokenizer import FloatTokenizer as Tokenizer


def causalMask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


class LinearModelDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer: Tokenizer,
                 seq_len: int) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get data pair
        src_target_pair = self.dataset[idx]
        src = src_target_pair['data']
        tgt = src_target_pair['params']

        # encode text to tokens
        enc_input_tokens = self.tokenizer.encode(src).ids
        dec_input_tokens = self.tokenizer.encode(tgt).ids

        # prepare padding tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # SOS, EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # SOS
        assert enc_num_padding_tokens >= 0 and dec_num_padding_tokens >= 0,\
                f"Sentence is too long for seq_len={self.seq_len}"

        # get inputs and label
        encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
                ]
        )

        decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
                ]
        )

        label = torch.cat(
                [
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
                ]
        )

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & \
                causalMask(decoder_input.size(0)) # (1, 1, seq_len) & (1, seq_len, seq_len)

        # sanity checks
        assert encoder_input.size(0) == self.seq_len, "Encoder input size is not correct"
        assert decoder_input.size(0) == self.seq_len, "Decoder input size is not correct"
        assert label.size(0) == self.seq_len, "Label size is not correct"

        return {
                "src": src,
                "tgt": tgt,
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
                "label": label,
                }

