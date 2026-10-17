
import torch


def crop_context(idx: torch.Tensor, context_size: int) -> torch.Tensor:
    """keep the last context_size tokens, (B, T) -> (B, <=context_size)"""
    # idx[:, -context_size:]
    # if T < context_size python just gives u everything, no crash, its fine
    # without this the model breaks as soon as u generate past the context window
    return idx[:, -context_size:]


def last_token_logits(model, idx_cond: torch.Tensor) -> torch.Tensor:
    """model(idx_cond)[:, -1, :] under no_grad -> (B, V)"""
    # 1. with torch.no_grad():  <- we're not training, no need to build the graph, saves memory
    # 2. logits = model(idx_cond) -> (B, T, V)
    # 3. keep only [:, -1, :] -> the prediction for the NEXT token
    # the model predicts at every position but we only care about the last one here
    with torch.no_grad():
        logits = model(idx_cond)
    return logits[:, -1, :]


def greedy_pick(logits: torch.Tensor) -> torch.Tensor:
    """argmax over V -> (B, 1)"""
    # logits.argmax(dim=-1, keepdim=True)
    # keepdim=True -> (B, 1) so u can torch.cat it right after
    # no softmax needed: softmax doesnt change the ORDER, so the argmax is the same. free win
    return logits.argmax(dim=-1, keepdim=True)


def generate_text_simple(model, idx: torch.Tensor, max_new_tokens: int,
                         context_size: int) -> torch.Tensor:
    """greedy loop: crop -> last_token_logits -> greedy_pick -> concat, (B, T) -> (B, T + max_new_tokens)"""
    # for _ in range(max_new_tokens):
    #   1. crop_context
    #   2. last_token_logits
    #   3. greedy_pick
    #   4. idx = torch.cat((idx, next), dim=1)   <- dim=1 = the time axis
    # return idx
    # call model.eval() before using this, otherwise dropout makes it random each time
    for _ in range(max_new_tokens):
        idx_cond = crop_context(idx, context_size)
        logits = last_token_logits(model, idx_cond)
        next_token = greedy_pick(logits)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


# ---------- Week 9: decoding strategies ----------

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """logits / temperature"""
    # divide. thats all
    # < 1 -> sharper, boring but safe. > 1 -> flatter, creative but often nonsense
    # guard temperature == 0 before calling this, it would divide by zero
    pass


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """keep the k biggest logits, others -> -inf"""
    # 1. torch.topk(logits, top_k) -> values + indices
    # 2. take the smallest kept value (values[:, -1])
    # 3. everything below it -> -torch.inf (torch.where is clean for this)
    # after softmax those become exactly 0 -> they can never be sampled
    pass


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """keep the smallest set whose cumulative probability >= p, others -> -inf"""
    # 1. sort the logits descending
    # 2. softmax -> cumsum on the sorted probs
    # 3. cut everything after the cumsum passes p (keep the one that crosses it !)
    # 4. scatter back to the original order, rest -> -inf
    # difference with top_k: the number of candidates ADAPTS to how sure the model is
    pass


def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
    """softmax -> torch.multinomial -> (B, 1)"""
    # 1. probs = torch.softmax(logits, dim=-1)
    # 2. torch.multinomial(probs, num_samples=1) -> (B, 1)
    # here we DO need the real softmax, we're drawing from the distribution, not taking the max
    pass


def generate(model, idx: torch.Tensor, max_new_tokens: int, context_size: int,
             temperature: float = 0.0, top_k: int = None, top_p: float = None,
             eos_id: int = None) -> torch.Tensor:
    """temperature 0 -> greedy, else temperature -> top_k -> top_p -> sample; stop at eos_id"""
    # same loop as generate_text_simple but:
    #   if temperature == 0 -> greedy_pick (deterministic, reproducible)
    #   else -> apply_temperature, then top_k if given, then top_p if given, then sample
    # order matters: temperature FIRST, then the filters, then sample
    # and if eos_id is not None and the token == eos_id -> break
    pass
