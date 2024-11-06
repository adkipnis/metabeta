from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from tokenizer import FloatTokenizer as Tokenizer


def causalMask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


class TransformerDataset(Dataset):
    def __init__(self,
                 raw: List[Dict[str, torch.Tensor]],
                 tokenizer: Tokenizer,
                 seq_len: int) -> None:
        self.raw = raw
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer.tokenToIdx("[SOS]")], dtype=torch.int64)
        self.sot_token = torch.tensor([tokenizer.tokenToIdx("[SOT]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.tokenToIdx("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.tokenToIdx("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # get data pair
        ds = self.raw[idx]
        predictors = ds['predictors']
        y = ds['y']
        params = ds['params']

        # encode text to tokens
        enc_input_tokens_ = self.tokenizer.encode(predictors)
        enc_input_tokens = [ self.tokenizer.tokenToIdx(token) for token in enc_input_tokens_ ]
        y_tokens_ = self.tokenizer.encode(y)
        y_tokens = [ self.tokenizer.tokenToIdx(token) for token in y_tokens_ ]
        dec_input_tokens_ = self.tokenizer.encode(params)
        dec_input_tokens = [ self.tokenizer.tokenToIdx(token) for token in dec_input_tokens_ ]

        # prepare padding tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - len(y_tokens) - 3 # SOS, SOT, EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # SOS
        assert enc_num_padding_tokens >= 0 and dec_num_padding_tokens >= 0,\
                f"Sentence is {len(enc_input_tokens) + len(y_tokens)} long but maximal sequence length is {self.seq_len}"

        # get inputs and label
        encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.sot_token,
                    torch.tensor(y_tokens, dtype=torch.int64),
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
                "predictors": predictors,
                "y": y,
                "params": params,
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
                "label": label,
                }


def padTensor(tensor: torch.Tensor, shape: Tuple[int, int], pad_val: int = 0) -> torch.Tensor:
    ''' Pad a tensor with a constant value. '''
    padded = torch.full(shape, pad_val, dtype=torch.float32)
    padded[:tensor.shape[0], :tensor.shape[1]] = tensor
    return padded


class RnnDataset(Dataset):
    def __init__(self, data: List[dict], max_samples: int, max_predictors: int) -> None:
        self.data = data
        self.max_samples = max_samples
        self.max_predictors = max_predictors
        self.seeds = torch.tensor([item['seed'] for item in data])

    def __len__(self) -> int:
       return len(self.data)

    def preprocess(self, item: dict) -> dict:
        predictors = item['predictors']
        y = item['y']
        params = item['params']
        return {
            'predictors': padTensor(predictors, (self.max_samples, self.max_predictors + 1)),
            'y': padTensor(y, (self.max_samples, 1)),
            'params': padTensor(params, (self.max_predictors + 1, 1)),
            'n': predictors.shape[0],
            'd': predictors.shape[1],
        }

    def __getitem__(self, idx) -> dict:
        data = self.data[idx]
        return self.preprocess(data)

