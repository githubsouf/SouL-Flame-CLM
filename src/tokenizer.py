import re

import torch


def load_text(path: str) -> str:
    """read a utf-8 text file"""
    # open(path, "r", encoding="utf-8") in a with block, .read()
    # always encoding="utf-8" explicitly. on windows the default is NOT utf-8, u'll get weird chars
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def split_into_tokens(text: str) -> list:
    """regex split on punctuation / -- / whitespace, drop empty strings"""
    # re.split with a CAPTURING group so the separators are kept, not thrown away
    # -> r'([,.:;?_!"()\']|--|\s)'   (same one u already used in tokenizer_demo)
    # then .strip() each item and drop the empty ones
    # punctuation is its own token: "hello," must not become one weird word
    pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def build_vocab(tokens: list, special_tokens: list = None) -> dict:
    """sorted unique tokens (+ <|unk|>, <|endoftext|>) -> {token: id}"""
    # 1. sorted(set(tokens)) -> sorted so the ids are STABLE, run it twice u get the same ids
    # 2. append the specials at the END (<|unk|>, <|endoftext|>)
    # 3. {token: i for i, token in enumerate(...)}
    # the specials go last so adding one later doesnt shift every existing id
    vocab_tokens = sorted(set(tokens))
    specials = special_tokens or ["<|unk|>", "<|endoftext|>"]
    for token in specials:
        if token not in vocab_tokens:
            vocab_tokens.append(token)
    return {token: idx for idx, token in enumerate(vocab_tokens)}


class SimpleTokenizerV2:
    """word-level tokenizer with <|unk|> fallback (your V1 + unknown words)"""

    def __init__(self, vocab: dict):
        # 1. self.str_to_int = vocab
        # 2. self.int_to_str = the reversed dict {i: s}
        # build the reverse ONCE here, not on every decode call
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str, allowed_special=None) -> list:
        """text -> list of ids"""
        # 1. split_into_tokens
        # 2. any token not in the vocab -> replace by "<|unk|>"
        # 3. map to ids
        # thats the only real difference with ur V1: V1 crashed on an unknown word, this one survives
        tokens = split_into_tokens(text)
        unk_id = self.str_to_int["<|unk|>"]
        return [self.str_to_int.get(token, unk_id) for token in tokens]

    def decode(self, ids: list) -> str:
        """list of ids -> text, fix spaces before punctuation"""
        # 1. " ".join of the tokens
        # 2. re.sub to kill the space before , . ? ! " ( )
        # round trip is NOT perfect and thats expected: encode->decode loses some spacing
        text = " ".join(self.int_to_str[idx] for idx in ids)
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)


def get_gpt2_tokenizer():
    """tiktoken.get_encoding("gpt2"), vocab 50,257"""
    # import tiktoken, return tiktoken.get_encoding("gpt2")
    # BPE, so no <|unk|> ever: unknown words get chopped into pieces it knows
    # this is just phase 1, u rewrite BPE urself in week 8
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def text_to_ids(text: str, tokenizer) -> torch.Tensor:
    """str -> (1, T), allow <|endoftext|>"""
    # 1. tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # 2. torch.tensor(ids).unsqueeze(0) -> add the batch dim, model wants (B, T)
    # without allowed_special tiktoken raises on the special token instead of encoding it
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def ids_to_text(ids: torch.Tensor, tokenizer) -> str:
    """(1, T) -> str"""
    # 1. ids.squeeze(0) -> drop the batch dim
    # 2. .tolist() -> tiktoken wants a python list, not a tensor
    # 3. tokenizer.decode(...)
    return tokenizer.decode(ids.squeeze(0).tolist())
