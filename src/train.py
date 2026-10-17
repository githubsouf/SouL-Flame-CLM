"""
SouL Flame — training loop (next delivery)
==========================================
STATUS: skeleton.

FLOW
    input ids -> forward -> logits (B, T, V) -> cross-entropy vs targets (B, T)
    -> loss.backward() -> optimizer.step() -> optimizer.zero_grad()

SHAPES FOR THE LOSS
    logits.flatten(0, 1) : (B*T, V)
    targets.flatten()    : (B*T,)

CHECKS
    - loss at step 0 ~ ln(vocab_size) (10.82 for 50,257)
    - perplexity = exp(loss)
    - val loss going up while train goes down = overfitting
"""
import torch


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor,
                    model, device) -> torch.Tensor:
    """cross-entropy for one batch"""
    # 1. move both to device
    # 2. logits = model(input_batch) -> (B, T, V)
    # 3. torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    #    flatten(0,1) -> (B*T, V), flatten() -> (B*T,). cross_entropy wants 2D + 1D
    # return the TENSOR not .item(), we need it for backward()
    pass


def calc_loss_loader(data_loader, model, device, num_batches: int = None) -> float:
    """average loss over (the first num_batches of) a loader"""
    # 1. if the loader is empty -> return float("nan"), dont crash
    # 2. num_batches = min(num_batches, len(data_loader)) if given, else all
    # 3. loop, sum loss.item(), divide by the count
    # .item() here on purpose: we just want a number, no graph, no memory leak
    pass


def perplexity(loss: float) -> float:
    """exp(loss)"""
    # math.exp(loss)
    # = "the model hesitates between N words". at init N ~ vocab_size, if it drops ur learning
    pass


def evaluate_model(model, train_loader, val_loader, device, eval_iter: int):
    """eval() -> no_grad losses -> train() -> (train_loss, val_loss)"""
    # 1. model.eval()  <- kills dropout, otherwise the val loss is noisy
    # 2. with torch.no_grad(): calc_loss_loader on both, limited to eval_iter batches
    # 3. model.train() before returning !! if u forget, dropout stays off for the rest of training
    pass


def generate_and_print_sample(model, tokenizer, device, start_context: str):
    """quick text sample during training to watch progress"""
    # 1. model.eval()
    # 2. text_to_ids -> generate_text_simple -> ids_to_text
    # 3. print it (replace \n by " " so it stays on one line in the terminal)
    # 4. model.train()
    # honestly the best debug tool: u SEE the model going from garbage to almost-words
    pass


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
    pass


def set_seed(seed: int):
    """torch.manual_seed (+ cuda) for reproducibility"""
    # torch.manual_seed(seed), and torch.cuda.manual_seed_all(seed) if cuda is there
    # same seed = same init = same curve. without it u cant compare 2 runs, period
    pass


if __name__ == "__main__":
    # ur training script: set_seed -> cfg -> loaders -> model.to(device)
    # -> torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1) -> train_model
    # -> save_checkpoint at the end
    # start with SOULFLAME_TINY on cpu. if it cant overfit the tiny file, dont even try the big one
    pass
