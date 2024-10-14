import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class BilingualDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer_src: Tokenizer,
                 tokenizer_tgt: Tokenizer,
                 src_lang: str,
                 tgt_lang: str,
                 seq_len: int) -> None:
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get text pair
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # encode text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

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

        assert encoder_input.size(0) == self.seq_len, "Encoder input size is not correct"
        assert decoder_input.size(0) == self.seq_len, "Decoder input size is not correct"
        assert label.size(0) == self.seq_len, "Label size is not correct"

        return {
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
                "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causalMask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
                "label": label,
                "src_text": src_text,
                "tgt_text": tgt_text
                }


def causalMask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

