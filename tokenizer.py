# encode numbers with three tokens: <sign, mantissa, exponent>
# <sign> is either + or -
# <mantissa> is between 0 and 9999
# <exponent> is between E-5 and E+5
from typing import List
import torch

class FloatTokenizer:
    def __init__(self, precision: int = 4):
        self.precision = precision
        self.base = 10
        self.max_exp = 6
        self.symbol_dict = {"special": ["[PAD]", "[SOS]", "[EOS]", "[SOT]", "[UNK]"],
                            "sign": ["-", "+"],
                            "mantissa": [f"N{i:04d}" for i in range(self.base ** self.precision)],
                            "exponent": [f"E{i}" for i in range(-self.max_exp, self.max_exp + 1)]}
        self.symbols = []
        for key in self.symbol_dict:
            self.symbols.extend(self.symbol_dict[key])
        self.n_symbols = len(self.symbols)
        
        # Create a character set and mapping
        self.sym_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self.idx_to_sym = {idx: sym for sym, idx in self.sym_to_idx.items()}
        
        # invert symbol_dict
        self.cat_to_idx = {cat: [self.tokenToIdx(token) for token in self.symbol_dict[cat]]
                           for cat in self.symbol_dict}
        self.idx_to_cat = {idx: cat for cat in self.symbol_dict
                           for idx in self.cat_to_idx[cat]}

    def tokenToIdx(self, token: str) -> int:
        return self.sym_to_idx.get(token, self.sym_to_idx["[UNK]"])

    def idxToToken(self, idx: int) -> str:
        idx = int(idx)
        return self.idx_to_sym.get(idx, "[UNK]")

    def getVocabSize(self) -> int:
        return self.n_symbols

    def encodeNumber(self, number: float) -> List[str]:
        # round to precision
        number = round(number, self.precision + 2)
        sign = "+" if number >= 0 else "-"
        mantissa, exponent = (f"%.{self.precision}e" % number).split("e")
        mantissa = mantissa.lstrip("-")[:self.precision + 1].replace(".", "")
        exponent = int(exponent) - self.precision + 1
        assert exponent <= self.max_exp, f"Exponent {exponent} is too large: {number} -> {mantissa}{exponent}"
        while exponent < -self.max_exp:
            mantissa = "0" + mantissa[:-1]
            exponent += 1
        return [sign, f"N{mantissa}", f"E{exponent}"]

    def encodeList(self, numbers: List[float]) -> List[str]:
        out = []
        for x in numbers:
            out.extend(self.encodeNumber(x))
        return out

    def encode(self, numbers: torch.Tensor) -> List[str]:
        assert numbers.dim() < 3, "Input must be 1D or 2D tensor"
        out = []
        if numbers.dim() == 2: # features
            for i in range(numbers.shape[0]):
                out.extend(self.encodeList(numbers[i].tolist()))
        elif numbers.dim() == 1: # target
            out.extend(self.encodeList(numbers.tolist()))
        return out

    def decodeNumber(self, tokens: List[str]) -> float:
        s, m, e = tokens
        sign = 1 if s == "+" else -1
        mantissa = int(m[1:])
        exponent = int(e[1:])
        out = float(sign * mantissa * self.base ** exponent)
        return round(out, self.precision + 2)
 
    def decode(self, tokens: List[str]) -> List[float]:
        if tokens[0] == '[SOS]':
            tokens = tokens[1:]
        if '[EOS]' in tokens:
            eos_idx = tokens.index('[EOS]')
            tokens = tokens[:eos_idx]
        out = []
        for i in range(0, len(tokens), 3):
            try:
                num = self.decodeNumber(tokens[i:i+3])
            except:
                # print(f"Error decoding {tokens[i:i+3]}")
                num = float("nan")
            out.append(num)
        return out

    def batchDecode(self, indices: torch.Tensor) -> List[List[float]]:
        tokens = [[self.idxToToken(idx) for idx in row] for row in indices]
        numbers = [self.decode(row) for row in tokens]
        min_len = min(len(row) for row in numbers)
        numbers = [row[:min_len] for row in numbers]
        return numbers


