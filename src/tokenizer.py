"""
SouL Flame — tokenizer interface
================================
STATUS: skeleton. Your earlier experiments stay in tokenizer_demo.py / tokenizer_BTE.ipynb.

PLAN
    Phase 1 (now):   wrap tiktoken "gpt2" (vocab 50,257) behind text_to_ids / ids_to_text
    Phase 2 (Week 8): your own BPE in the same interface, so nothing else changes

INTERFACE
    text_to_ids(text, tokenizer) -> torch.LongTensor (1, T)     batch dim added
    ids_to_text(ids, tokenizer)  -> str                         batch dim removed

TESTS TO WRITE YOURSELF
    - round trip: ids_to_text(text_to_ids(s)) == s
    - "<|endoftext|>" must be allowed (allowed_special)
"""
import torch


def load_text(path: str) -> str:
    """read a utf-8 text file"""
    # open(path, "r", encoding="utf-8") in a with block, .read()
    # always encoding="utf-8" explicitly. on windows the default is NOT utf-8, u'll get weird chars
    pass


def split_into_tokens(text: str) -> list:
    """regex split on punctuation / -- / whitespace, drop empty strings"""
    # re.split with a CAPTURING group so the separators are kept, not thrown away
    # -> r'([,.:;?_!"()\']|--|\s)'   (same one u already used in tokenizer_demo)
    # then .strip() each item and drop the empty ones
    # punctuation is its own token: "hello," must not become one weird word
    pass


def build_vocab(tokens: list, special_tokens: list = None) -> dict:
    """sorted unique tokens (+ <|unk|>, <|endoftext|>) -> {token: id}"""
    # 1. sorted(set(tokens)) -> sorted so the ids are STABLE, run it twice u get the same ids
    # 2. append the specials at the END (<|unk|>, <|endoftext|>)
    # 3. {token: i for i, token in enumerate(...)}
    # the specials go last so adding one later doesnt shift every existing id
    pass


class SimpleTokenizerV2:
    """word-level tokenizer with <|unk|> fallback (your V1 + unknown words)"""

    def __init__(self, vocab: dict):
        # 1. self.str_to_int = vocab
        # 2. self.int_to_str = the reversed dict {i: s}
        # build the reverse ONCE here, not on every decode call
        pass

    def encode(self, text: str) -> list:
        """text -> list of ids"""
        # 1. split_into_tokens
        # 2. any token not in the vocab -> replace by "<|unk|>"
        # 3. map to ids
        # thats the only real difference with ur V1: V1 crashed on an unknown word, this one survives
        pass

    def decode(self, ids: list) -> str:
        """list of ids -> text, fix spaces before punctuation"""
        # 1. " ".join of the tokens
        # 2. re.sub to kill the space before , . ? ! " ( )
        # round trip is NOT perfect and thats expected: encode->decode loses some spacing
        pass


def get_gpt2_tokenizer():
    """tiktoken.get_encoding("gpt2"), vocab 50,257"""
    # import tiktoken, return tiktoken.get_encoding("gpt2")
    # BPE, so no <|unk|> ever: unknown words get chopped into pieces it knows
    # this is just phase 1, u rewrite BPE urself in week 8
    pass


def text_to_ids(text: str, tokenizer) -> torch.Tensor:
    """str -> (1, T), allow <|endoftext|>"""
    # 1. tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # 2. torch.tensor(ids).unsqueeze(0) -> add the batch dim, model wants (B, T)
    # without allowed_special tiktoken raises on the special token instead of encoding it
    pass


def ids_to_text(ids: torch.Tensor, tokenizer) -> str:
    """(1, T) -> str"""
    # 1. ids.squeeze(0) -> drop the batch dim
    # 2. .tolist() -> tiktoken wants a python list, not a tensor
    # 3. tokenizer.decode(...)
    pass
