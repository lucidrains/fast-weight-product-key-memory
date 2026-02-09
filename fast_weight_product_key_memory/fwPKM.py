from __future__ import annotations
from math import sqrt

import torch
from torch import nn
from torch import tensor, Tensor
from torch.nn import Module, Sequential, RMSNorm

import einx
from einops import rearrange
from einops.layers.torch import Rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def LinearNoBias(dim, dim_out):
    return nn.Linear(dim, dim_out, bias = False)

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
        num_memories = 512 * 512,
        dim_queries_keys = 512,
        dim_values = 512,
        learning_rate = 1.,
        topk = 8,
        lookahead_values = True,
        addressing_loss_weight = 10.
    ):
        super().__init__()
        assert sqrt(num_memories).is_integer(), 'num memories must have an integer square root'

        self.memories = nn.Parameter(torch.randn(num_memories, dim_values) * 1e-2)

        num_keys = int(sqrt(num_memories))
        self.keys = nn.Parameter(torch.randn(2, num_keys, dim_queries_keys))

        # projections

        self.to_queries = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_queries_keys * 2),
            Rearrange('... (two d) -> two ... d', two = 2)
        )

        self.to_gates = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, 1)
        )

        self.to_values = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_values)
        )

        self.to_out = Sequential(
            RMSNorm(dim_values),
            LinearNoBias(dim_values, dim)
        )

        # loss related

        self.addressing_loss_weight = addressing_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens
    ):

        return tokens
