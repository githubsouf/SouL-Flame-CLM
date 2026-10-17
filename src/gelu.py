"""SouL Flame, GELU activation (tanh approximation) """

import math

import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        # nothing to store. GELU has ZERO parameters, its pure math
        # keep the __init__ anyway so nn.Module is properly set up
        pass

    def inner_cubic_term(self, x: torch.Tensor) -> torch.Tensor:
        """x + 0.044715 * x^3"""
        # literally x + 0.044715 * x**3
        # that magic constant comes from the paper, its fitted to match the exact GELU
        # element-wise -> shape in = shape out, no dim argument anywhere
        return x + 0.044715 * (x ** 3)

    def tanh_gate(self, x: torch.Tensor) -> torch.Tensor:
        """1 + tanh( sqrt(2/pi) * inner_cubic_term(x) )"""
        # 1. sqrt(2/pi) -> use math.sqrt(2/math.pi), its a python float, fine
        # 2. multiply it by inner_cubic_term(x)
        # 3. torch.tanh(...) -> gives u something in ]-1, 1[
        # 4. + 1 -> now its in ]0, 2[. thats the "gate": how much of x passes through
        return 1 + torch.tanh(math.sqrt(2 / math.pi) * self.inner_cubic_term(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """0.5 * x * tanh_gate(x)"""
        # 0.5 * x * gate. the 0.5 brings the gate back to ]0, 1[
        # sanity check in ur head: x=0 -> 0. x=10 -> gate~2 -> ~10 (passes). x=-10 -> gate~0 -> ~0 (blocked)
        # thats the whole idea: smooth switch instead of ReLU's hard cut at 0
        return 0.5 * x * self.tanh_gate(x)
