from __future__ import annotations
from math import sqrt

import torch
from torch import nn, cdist
from torch import tensor, Tensor
from torch.nn import Module, Sequential, RMSNorm

import einx
from einops import rearrange, einsum, repeat
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

def inverse_distance_weight(q, k, eps = 1e-3):
    dist = cdist(q, k) ** 2
    return -log(dist, eps = eps)

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
        addressing_loss_weight = 10.
    ):
        super().__init__()
        assert sqrt(num_memories).is_integer(), 'num memories must have an integer square root'

        self.memories = nn.Parameter(torch.randn(num_memories, dim_values))

        # pkm related

        self.topk = topk

        num_keys = int(sqrt(num_memories))
        self.keys = nn.Parameter(torch.randn(2, num_keys, dim_queries_keys) * 1e-2)

        self.num_keys = num_keys
        self.num_memories = num_memories

        # projections

        self.to_queries = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_queries_keys * 2),
            Rearrange('... (two d) -> two ... d', two = 2)
        )

        self.to_gates = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, 1),
            nn.Sigmoid()
        )

        self.to_values = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_values)
        )

        self.to_out = Sequential(
            RMSNorm(dim_values),
            LinearNoBias(dim_values, dim)
        )

        # storing related

        self.addressing_loss_weight = addressing_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    def forward(
        self,
        tokens,
        return_store_grads = False,
        return_aux_loss = False
    ):
        k = self.topk

        q1, q2 = self.to_queries(tokens)
        k1, k2 = self.keys

        # product keys

        # they use a special type of distance from a paper i've seen in the past by a lone author - https://arxiv.org/abs/2310.18805

        dist1, dist2 = inverse_distance_weight(q1, k1), inverse_distance_weight(q2, k2)

        # get the topk closest by idw

        top1, indices1 = dist1.topk(k = k)
        top2, indices2 = dist2.topk(k = k)

        # merge

        indices = einx.add('... i, ... j -> ... (i j)', indices1 * self.num_keys, indices2)
        scores = einx.add('... i, ... j -> ... (i j)', top1, top2)

        # topk again

        top_scores, top_sub_indices = scores.topk(k = k)

        final_indices = indices.gather(-1, top_sub_indices)
        final_scores = top_scores.softmax(dim = -1)

        memories = self.memories[final_indices]

        values = einsum(memories, final_scores, '... topk d, ... topk -> ... d')

        # gates and values

        gates = self.to_gates(tokens)

        target_values = self.to_values(tokens)

        output = target_values.lerp(values, gates)

        out = self.to_out(output)

        if not return_store_grads:
            return out

        # calculating fast weights for episodic memory
        # with lookahead

        final_indices = final_indices[..., :-1, :]
        final_scores = final_scores[..., :-1, :]
        gates = gates[..., :-1, :]

        error = gates * (values[:, :-1] - target_values[:, 1:]) # mse loss with lookahead

        # get update for memories

        memories_grad = einx.multiply('... d, ... topk -> (... topk) d', error, final_scores)

        flattened_final_indices = rearrange(final_indices, '... -> (...)')

        final_indices_expanded = repeat(flattened_final_indices, '... -> (...) d', d = memories_grad.shape[-1])

        fast_weight_memories = torch.zeros_like(self.memories).scatter_reduce_(0, final_indices_expanded, memories_grad, reduce = 'mean', include_self = False)

        if not return_aux_loss:
            return out, fast_weight_memories

        addressing_loss = self.zero

        return out, fast_weight_memories, addressing_loss
