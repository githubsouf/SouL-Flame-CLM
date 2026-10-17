import math
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.config import SOULFLAME_TINY
from src.checkpoint import save_checkpoint
from src.dataset import create_train_val_loaders
from src.generate import generate_text_simple
from src.model import SoulFlameModel
from src.tokenizer import get_gpt2_tokenizer
from src.tokenizer import ids_to_text, text_to_ids


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor,
                    model, device) -> torch.Tensor:
    """cross-entropy for one batch"""
    # 1. move both to device
    # 2. logits = model(input_batch) -> (B, T, V)
    # 3. torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    #    flatten(0,1) -> (B*T, V), flatten() -> (B*T,). cross_entropy wants 2D + 1D
    # return the TENSOR not .item(), we need it for backward()
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches: int = None) -> float:
    """average loss over (the first num_batches of) a loader"""
    # 1. if the loader is empty -> return float("nan"), dont crash
    # 2. num_batches = min(num_batches, len(data_loader)) if given, else all
    # 3. loop, sum loss.item(), divide by the count
    # .item() here on purpose: we just want a number, no graph, no memory leak
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def perplexity(loss: float) -> float:
    """exp(loss)"""
    # math.exp(loss)
    # = "the model hesitates between N words". at init N ~ vocab_size, if it drops ur learning
    return math.exp(loss)


def evaluate_model(model, train_loader, val_loader, device, eval_iter: int):
    """eval() -> no_grad losses -> train() -> (train_loss, val_loss)"""
    # 1. model.eval()  <- kills dropout, otherwise the val loss is noisy
    # 2. with torch.no_grad(): calc_loss_loader on both, limited to eval_iter batches
    # 3. model.train() before returning !! if u forget, dropout stays off for the rest of training
    model.eval()
    try:
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    finally:
        model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context: str):
    """quick text sample during training to watch progress"""
    # 1. model.eval()
    # 2. text_to_ids -> generate_text_simple -> ids_to_text
    # 3. print it (replace \n by " " so it stays on one line in the terminal)
    # 4. model.train()
    # honestly the best debug tool: u SEE the model going from garbage to almost-words
    model.eval()
    try:
        context_size = model.pos_emb.weight.shape[0]
        idx = text_to_ids(start_context, tokenizer).to(device)
        out = generate_text_simple(model, idx, max_new_tokens=50, context_size=context_size)
        text = ids_to_text(out.cpu(), tokenizer)
        print(text.replace("\n", " "))
    finally:
        model.train()


def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs: int, eval_freq: int, eval_iter: int,
                start_context: str, tokenizer):
    """main loop -> (train_losses, val_losses, tokens_seen)"""
    # for each epoch, for each batch:
    #   1. optimizer.zero_grad()   <- FIRST. pytorch accumulates gradients, if u forget it they pile up
    #   2. loss = calc_loss_batch(...)
    #   3. loss.backward()
    #   4. optimizer.step()
    #   5. count the tokens seen, and every eval_freq steps -> evaluate_model + print
    # at the end of each epoch -> generate_and_print_sample
    # if train loss goes down but val loss goes UP -> overfitting, and on 18k chars it WILL happen fast
    train_losses = []
    val_losses = []
    tokens_seen = []
    total_tokens = 0
    global_step = 0

    model.train()
    for _ in range(num_epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens)
                print(
                    f"step {global_step}: train loss {train_loss:.4f}, "
                    f"val loss {val_loss:.4f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, tokens_seen


def set_seed(seed: int):
    """torch.manual_seed (+ cuda) for reproducibility"""
    # torch.manual_seed(seed), and torch.cuda.manual_seed_all(seed) if cuda is there
    # same seed = same init = same curve. without it u cant compare 2 runs, period
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # ur training script: set_seed -> cfg -> loaders -> model.to(device)
    # -> torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1) -> train_model
    # -> save_checkpoint at the end
    # start with SOULFLAME_TINY on cpu. if it cant overfit the tiny file, dont even try the big one
    set_seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_gpt2_tokenizer()
    cfg = {**SOULFLAME_TINY, "vocab_size": tokenizer.n_vocab}

    root = Path(__file__).resolve().parents[1]
    train_path = root / "data" / "processed" / "train.txt"
    val_path = root / "data" / "processed" / "val.txt"

    train_loader, val_loader = create_train_val_loaders(
        str(train_path),
        str(val_path),
        cfg,
        batch_size=2,
    )

    model = SoulFlameModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        eval_freq=10,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    checkpoint_path = root / "checkpoints" / "soulflame_tiny_first_run.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    final_step = len(tokens_seen)
    save_checkpoint(
        str(checkpoint_path),
        model,
        optimizer,
        cfg,
        step=final_step,
        epoch=2,
    )

    if val_losses:
        print(
            f"saved checkpoint to {checkpoint_path} | eval points: {len(train_losses)} | "
            f"last val loss: {val_losses[-1]:.4f}"
        )
    else:
        print(f"saved checkpoint to {checkpoint_path} | no evals run")
