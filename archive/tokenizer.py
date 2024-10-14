# encode numbers with three tokens: <sign, mantissa, exponent>
# <sign> is either + or -
# <mantissa> is between 0 and 9999
# <exponent> is between E-5 and E+5
import torch
from typing import List

class FloatTokenizer:
    def __init__(self, precision=4):
        self.precision = precision
        self.base = 10
        self.max_exp = 6
        self.symbols = ["<sos>", "<eos>", "+", "-"]
        self.symbols += [f"N{i:04d}" for i in range(self.base ** self.precision)]
        self.symbols += [f"E{i}" for i in range(-self.max_exp, self.max_exp + 1)]
        self.n_symbols = len(self.symbols)
        # Create a character set and mapping
        self.sym_to_idx = {symbol: idx + 1 for idx, symbol in enumerate(self.symbols)}
        self.idx_to_sym = {idx: sym for sym, idx in self.sym_to_idx.items()}

    def _encode(self, number: float) -> List[str]:
        # round to precision
        sign = "+" if number >= 0 else "-"
        mantissa, exponent = (f"%.{self.precision}e" % number).split("e")
        mantissa = mantissa.lstrip("-")[:self.precision + 1].replace(".", "")
        exponent = int(exponent) - self.precision + 1
        assert exponent <= self.max_exp, f"Exponent {exponent} is too large: {number} -> {mantissa}{exponent}"
        while exponent < -self.max_exp:
            mantissa = "0" + mantissa[:-1]
            exponent += 1
        return [sign, f"N{mantissa}", f"E{exponent}"]

    def encode(self, numbers: List[float]) -> List[str]:
        out = ['<sos>']
        for number in numbers:
            out += self._encode(float(number))
        return out + ['<eos>']
    
    def encodeTensor(self, numbers: torch.Tensor) -> List[str]:
        out = []
        for row in numbers:
            out += self.encode(row.tolist())
        return out

    def _decode(self, tokens: List[str]) -> float:
        s, m, e = tokens
        sign = 1 if s == "+" else -1
        mantissa = int(m[1:])
        exponent = int(e[1:])
        return float(sign * mantissa * self.base ** exponent)

    def decode(self, tokens: List[str]) -> List[float]:
        assert tokens[0] == '<sos>', "First token must be <sos>"
        tokens = tokens[1:-1]
        out = []
        for i in range(0, len(tokens), 3):
            out.append(self._decode(tokens[i:i+3]))
        return out


def testFT():
    import numpy as np
    tokenizer = FloatTokenizer()
    numbers = [-0.5, 0.0144, 1000, 2.4242, 70911.1]
    expected = [-0.5, 0.0144, 1000.0, 2.424, 70910.0]
    tokens = tokenizer.encode(numbers)
    decoded_numbers = tokenizer.decode(tokens)
    for i, (number, expected_number) in enumerate(zip(decoded_numbers, expected)):
        passed = np.isclose(number, expected_number)
        print(f"Test {i+1} {passed}: {number} == {expected_number}")
        print(f"Tokens: {tokens[1+i*3:4+i*3]}")


def main():
    testFT()

if __name__ == "__main__":
    main()

