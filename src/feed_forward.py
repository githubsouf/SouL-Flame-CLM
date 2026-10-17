import torch
import torch.nn as nn

from src.gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # 1. grab d = cfg["emb_dim"]
        # 2. up = nn.Linear(d, 4*d)   <- with bias, default is True, leave it
        # 3. act = ur GELU()          <- ur own one, not nn.GELU
        # 4. down = nn.Linear(4*d, d)
        # u can wrap the 3 in nn.Sequential, or keep them separate, ur call
        # why 4x ? its just the GPT-2 choice. more room to compute, then squeeze back
        d = cfg["emb_dim"]
        self.up = nn.Linear(d, 4 * d)
        self.act = GELU()
        self.down = nn.Linear(4 * d, d)

    def up_projection(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, T, 4d)"""
        # just call the up Linear. nn.Linear only touches the LAST dim
        # so (B, T, d) -> (B, T, 4d), B and T dont move
        return self.up(x)

    def activation(self, h: torch.Tensor) -> torch.Tensor:
        """GELU element-wise, (B, T, 4d) -> (B, T, 4d)"""
        # call ur GELU. shape doesnt change
        # this is THE important line: without it, up then down = one big Linear
        # two linears stacked = still linear. the net would learn nothing extra
        return self.act(h)

    def down_projection(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, 4d) -> (B, T, d)"""
        # the down Linear, back to d
        # we HAVE to come back to d, otherwise the residual x + ffn(x) cant even add up
        return self.down(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """up_projection -> activation -> down_projection"""
        # chain the 3. shape in == shape out
        # note: every token goes through this alone, no token talks to another here
        # mixing between tokens ONLY happens in attention. this part is per-token thinking
        h = self.up_projection(x)
        h = self.activation(h)
        return self.down_projection(h)
