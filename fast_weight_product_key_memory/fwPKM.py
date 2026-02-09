from __future__ import annotations

import torch
from torch.nn import Module, ModuleList, RMSNorm

import einx
from einops import rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
        lookahead_values = True
    ):
        super().__init__()

    def forward(
        self,
        tokens
    ):
        return tokens
