from __future__ import annotations

import torch
from torch import tensor, Tensor
from torch.nn import Module, ModuleList, RMSNorm

import einx
from einops import rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def entropy(prob):
    return -(prob * log(prob)).sum(dim = -1)

# classes

class fwPKM(Module):
    def __init__(
        self,
        dim,
        *,
        dim_qk = 512,
        dim_v = 512,
        learning_rate = 1.,
        topk = 8,
        lookahead_values = True,
        addressing_loss_weight = 10.
    ):
        super().__init__()

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens
    ):
        return tokens
