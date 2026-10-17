"""
SouL Flame — next-token dataset (sliding window)
================================================
STATUS: skeleton. Data: data/processed/train.txt and data/processed/val.txt

MATH
    token_ids = tokenizer.encode(text)
    for i in range(0, len(token_ids) - max_length, stride):
        input  = token_ids[i     : i + max_length]
        target = token_ids[i + 1 : i + max_length + 1]    <- shifted by ONE

SHAPES
    one sample   : (max_length,), (max_length,)
    one batch    : (batch_size, max_length) x 2

HINTS
    - stride == max_length -> no overlap (less overfitting between batches).
    - drop_last=True avoids a tiny last batch that spikes the loss.
    - shuffle=True for train, False for val.
    - the val text is small (~2,000 chars): check it still gives >= 1 window.
"""
import torch
from torch.utils.data import Dataset, DataLoader


class SoulFlameDataset(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        # 1. self.input_ids = [] and self.target_ids = []
        # 2. token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        # 3. call self.build_windows(token_ids, max_length, stride)
        # tokenize the WHOLE text once here. doing it in __getitem__ would re-tokenize every epoch
        pass

    def build_windows(self, token_ids: list, max_length: int, stride: int):
        """sliding window -> fills self.input_ids and self.target_ids (target = input shifted by 1)"""
        # for i in range(0, len(token_ids) - max_length, stride):
        #   input  = token_ids[i     : i + max_length]
        #   target = token_ids[i + 1 : i + max_length + 1]    <- shifted by ONE, thats the whole task
        #   append both as torch.tensor
        # the "- max_length" in the range is what stops u from going out of bounds on the target
        # stride == max_length -> no overlap. stride < max_length -> windows overlap, more data but redundant
        pass

    def __len__(self) -> int:
        # len(self.input_ids). the DataLoader needs this to know how many batches
        pass

    def __getitem__(self, idx: int):
        """-> (input (max_length,), target (max_length,))"""
        # return self.input_ids[idx], self.target_ids[idx]
        # a tuple, the DataLoader stacks them into (B, max_length) x2 by itself
        pass


def create_dataloader(text: str, batch_size: int = 4, max_length: int = 256,
                      stride: int = 128, shuffle: bool = True,
                      drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """tokenizer -> SoulFlameDataset -> DataLoader"""
    # 1. tokenizer = get_gpt2_tokenizer()
    # 2. dataset = SoulFlameDataset(text, tokenizer, max_length, stride)
    # 3. return DataLoader(dataset, batch_size=..., shuffle=..., drop_last=..., num_workers=...)
    # drop_last=True -> throw away the last incomplete batch, it makes the loss curve jump
    # num_workers=0 on windows, otherwise it can hang. dont waste an evening on this
    pass


def create_train_val_loaders(train_path: str, val_path: str, cfg: dict,
                             batch_size: int = 2):
    """data/processed/train.txt + val.txt -> (train_loader, val_loader), val: shuffle=False, drop_last=False"""
    # 1. load both files (load_text from src/tokenizer.py)
    # 2. train: shuffle=True, drop_last=True
    # 3. val:   shuffle=False, drop_last=False   <- we want the SAME val loss every time to compare runs
    # 4. max_length = stride = cfg["context_length"]
    # careful: val.txt is small (~2000 chars). if max_length is too big u get 0 windows
    # -> check len(val_loader) > 0 before training, or u'll debug a nan for nothing
    pass
