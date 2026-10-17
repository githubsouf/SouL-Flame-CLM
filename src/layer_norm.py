"""

STATUS: ts: tests/test_layer_norm.py

MATH 
    mu      = mean(x)
    sigma^2 = var(x)                  <- biased variance (divide by d, not d-1)
    x_hat   = (x - mu) / sqrt(sigma^2 + eps)
    y       = gamma * x_hat + beta

SHAPES
    x      : (B, T, d)
    mu     : (B, T, 1)    
    gamma  : (d,)         trainable, init = ones
    beta   : (d,)         trainable, init = zeros
    y      : (B, T, d)    same shape as x

"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        
        # 1. store eps on self. tiny number, we add it under the sqrt so we never divide by 0
        # 2. self.gamma = nn.Parameter, vector of size emb_dim, filled with ONES
        #    ones = "dont change anything yet", the net learns to scale later
        # 3. self.beta = nn.Parameter, same size, filled with ZEROS (shift, starts neutral)
        # careful: if u use plain torch.ones instead of nn.Parameter the optimizer never sees them
        # no gradient no learning  and test_gamma_beta_are_trainable_and_initialized fails
        #
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))



    def compute_mean(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> mu (B, T, 1)"""
        # mean over the LAST dim only (dim=-1). each token normalizes itself, not the batch
        # keepdim=True -> u get (B, T, 1) not (B, T). we need that 1 so broadcasting works after
        return x.mean(dim=-1, keepdim=True)

    def compute_variance(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> sigma^2 (B, T, 1), biased"""
        # same dim=-1, same keepdim=True
        # unbiased=False !! we divide by d, not d-1. thats what the formula says and what pytorch does
        # if u forget it, the test that compares us to F.layer_norm fails on small d
        return x.var(dim=-1, keepdim=True, unbiased=False)

    def normalize(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """x_hat = (x - mu) / sqrt(sigma^2 + eps) -> (B, T, d)"""
        # 1. subtract the mean -> centered
        # 2. divide by sqrt(var + eps) -> unit variance
        # eps goes INSIDE the sqrt. if variance is exactly 0 (constant token) sqrt(0) = 0 -> nan
        # (B, T, d) / (B, T, 1) -> broadcasting handles it, no loop needed
        return (x - mean) / torch.sqrt(var + self.eps)

    def scale_and_shift(self, x_hat: torch.Tensor) -> torch.Tensor:
        """y = gamma * x_hat + beta -> (B, T, d)"""
        # gamma is (d,), x_hat is (B, T, d) -> broadcasts on the last dim, just write it normally
        # at init gamma=1 beta=0 so this line changes nothing. thats on purpose
        # training decides later if some features need to be louder or shifted
        return self.gamma * x_hat + self.beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """mean -> variance -> normalize -> scale_and_shift"""
        # just chain the 4 methods above, nothing new here
        # shape in == shape out, ALWAYS. if it changed, u broke something
        # dont call .detach() or .numpy() anywhere, it kills the gradient graph
        mean = self.compute_mean(x)
        var = self.compute_variance(x)
        x_hat = self.normalize(x, mean, var)
        return self.scale_and_shift(x_hat)
