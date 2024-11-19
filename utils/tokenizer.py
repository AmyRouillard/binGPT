# conway_lib/tokenizer.py

import torch
import typing


class Tokenizer:
    def __init__(
        self,
        n_pad: int = 132,
        device: torch.device = torch.device("cpu"),
        pad_byte: int = 0,
    ):
        self.n_pad = n_pad
        self.device = device
        self.pad_byte = pad_byte
        self.end_token = ord("$")  # Define the end token

    def tokenize_str(self, sentence: str, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert (
                len(base) == self.n_pad
            ), f"n_pad is too small, use {len(base)} or greater."
        tensor = torch.Tensor(base)
        return tensor.long().to(self.device)

    def texts_to_sequences(
        self, texts: typing.List[str], encoding="utf8", do_padding=True
    ):
        sentences = [
            self.tokenize_str(sentence, do_padding=do_padding).unsqueeze(0)
            for sentence in texts
        ]
        return torch.cat(sentences, dim=0).to(self.device)

    def sequences_to_texts(self, sequences: torch.Tensor, encoding="utf8"):
        out = []
        for seq in sequences:
            out.append("".join([chr(int(s)) for s in seq]))

        return out


class CustomTokenizer:
    def __init__(
        self,
        n_pad: int = 132,
        device: torch.device = torch.device("cpu"),
        pad_byte: int = 0,
        eos_token: str = "$",
        bos_token: str = "@",
        special_token: str = ">",
    ):
        self.n_pad = n_pad
        self.pad_byte = pad_byte

        self.device = device

        self.token_map = {
            "0": 1,
            "1": 2,
            special_token: 3,
            eos_token: 4,
        }

        if bos_token is not None:
            self.token_map[bos_token] = 5

        self.inverse_token_map = {v: k for k, v in self.token_map.items()}

        self.vocab_size = len(self.token_map) + 1

    def tokenize_str(self, sentence: str, do_padding=True):
        base = [self.token_map.get(c, self.pad_byte) for c in sentence]
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert (
                len(base) == self.n_pad
            ), f"n_pad is too small, use {len(base)} or greater."
        tensor = torch.Tensor(base)
        return tensor.long().to(self.device)

    def texts_to_sequences(
        self, texts: typing.List[str], encoding="utf8", do_padding=True
    ):
        sentences = [
            self.tokenize_str(sentence, do_padding=do_padding).unsqueeze(0)
            for sentence in texts
        ]
        return torch.cat(sentences, dim=0).to(self.device)

    def sequences_to_texts(self, sequences: torch.Tensor, encoding="utf8"):
        out = []
        for seq in sequences:
            out.append("".join([self.inverse_token_map.get(int(s), " ") for s in seq]))

        return out
