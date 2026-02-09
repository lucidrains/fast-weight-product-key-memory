from __future__ import annotations
from collections import namedtuple
from math import sqrt
from functools import partial

import torch
from torch import nn, stack, cdist
from torch import tensor, Tensor
import torch.nn.functional as F
from torch.nn import Module, Sequential, RMSNorm

import einx
from einops import rearrange, einsum, repeat, reduce
from einops.layers.torch import Rearrange

from torch_einops_utils import shape_with_replace

# constants

Memories = namedtuple('Memories', ('memory_values', 'keys'))

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def LinearNoBias(dim, dim_out):
    return nn.Linear(dim, dim_out, bias = False)

# tensor helpers

def log(t, eps = 1e-20):
    return (t + eps).log()

def l1norm(t, dim = -1, eps = 1e-10):
    return F.normalize(t, dim = -1, p = 1, eps = eps)

def entropy(prob):
    return -(prob * log(prob)).sum(dim = -1)

def z_score(t, dim = -1, eps = 1e-10):
    return (t - t.mean(dim = -1, keepdim = True)) / t.std(dim = -1, keepdim = True).clamp_min(eps)

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

        self.learning_rate = learning_rate

        self.addressing_loss_weight = addressing_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    def forward(
        self,
        tokens,
        return_next_memories = False,
        return_addressing_loss = False,
        past_memories: Memories | None = None,
        detach_next_memories = False,
        idw_eps = 1e-3
    ):
        k, num_keys = self.topk, self.num_keys

        q1, q2 = self.to_queries(tokens)

        # keys for pkm, accounting for fast weight memories

        keys = self.keys

        if exists(past_memories):
            keys = keys + past_memories.keys

        k1, k2 = keys

        # they use a special type of distance from a paper i've seen in the past by a lone author - https://arxiv.org/abs/2310.18805

        dist1 = cdist(q1, k1) ** 2
        dist2 = cdist(q2, k2) ** 2

        score1, score2 = -log(dist1, eps = idw_eps), -log(dist2, eps = idw_eps)

        # get the topk closest by idw

        top1, indices1 = score1.topk(k = k)
        top2, indices2 = score2.topk(k = k)

        # product keys

        indices = einx.add('... i, ... j -> ... (i j)', indices1 * num_keys, indices2)
        scores = einx.add('... i, ... j -> ... (i j)', top1, top2)

        # topk again

        top_scores, top_sub_indices = scores.topk(k = k)

        final_indices = indices.gather(-1, top_sub_indices)
        final_scores = top_scores.softmax(dim = -1)

        memories = self.memories[final_indices]

        # add the past memories

        if exists(past_memories):
            gathered_past_memories = past_memories.memory_values[final_indices]
            memories = memories + gathered_past_memories

        values = einsum(memories, final_scores, '... topk d, ... topk -> ... d')

        # gates and values

        gates = self.to_gates(tokens)

        target_values = self.to_values(tokens)

        target_values = z_score(target_values) # they apparently z-scored the target values for stability

        output = target_values.lerp(values, gates)

        out = self.to_out(output)

        # handle addressing loss

        if return_addressing_loss:
            key1_indices = (final_indices // num_keys).flatten()
            key2_indices = (final_indices % num_keys).flatten()

            flattened_final_scores = final_scores.flatten()

            zeros = torch.zeros(num_keys, device = self.device)

            acc_scores_key1 = zeros.scatter_add(0, key1_indices, flattened_final_scores)
            acc_scores_key2 = zeros.scatter_add(0, key2_indices, flattened_final_scores)

            probs1 = l1norm(acc_scores_key1)
            probs2 = l1norm(acc_scores_key2)

            addressing_loss = -(entropy(probs1) + entropy(probs2)).mean()

            loss = addressing_loss * self.addressing_loss_weight

            out = (out, loss)

        # early return if not storing

        if not return_next_memories:
            return out

        # calculating fast weights for episodic memory
        # with lookahead

        # remove last token for a bunch of variables for easy reading

        q1, q2 = q1[:, :-1], q2[:, :-1]
        dist1, dist2 = dist1[:, :-1], dist2[:, :-1]

        indices1, indices2 = indices1[:, :-1], indices2[:, :-1]

        scores = scores[:, :-1]
        top_sub_indices = top_sub_indices[:, :-1]
        final_indices = final_indices[:, :-1]
        final_scores = final_scores[:, :-1]

        gates = gates[:, :-1]
        target_values = target_values[:, 1:]
        values = values[:, :-1]
        memories = memories[:, :-1]

        # mse loss with lookahead

        error = gates * (values - target_values) * self.learning_rate

        # get update for memories

        memories_grad = einx.multiply('... d, ... topk -> (... topk) d', error, final_scores)

        flattened_final_indices = final_indices.flatten()

        final_indices_expanded = repeat(flattened_final_indices, '... -> (...) d', d = memories_grad.shape[-1])

        next_fast_weight_memories = torch.zeros_like(self.memories).scatter_reduce_(0, final_indices_expanded, memories_grad, reduce = 'mean', include_self = False)

        # step through the softmax backwards and torwards the idw

        final_scores_grad = einsum(error, memories, '... d, ... topk d -> ... topk')

        top_scores_grad = final_scores * (final_scores_grad - (final_scores * final_scores_grad).sum(dim = -1, keepdim = True))

        # now propagate top_scores_grad back to the keys

        sub_indices1 = top_sub_indices // k
        sub_indices2 = top_sub_indices % k

        # 2. back to dist1 and dist2

        final_indices1 = indices1.gather(-1, sub_indices1)
        final_indices2 = indices2.gather(-1, sub_indices2)

        grad_shape = shape_with_replace(dist1, {-1: num_keys})

        dist1_grad = torch.zeros(grad_shape, device = self.device).scatter_add_(-1, final_indices1, top_scores_grad)
        dist2_grad = torch.zeros(grad_shape, device = self.device).scatter_add_(-1, final_indices2, top_scores_grad)

        # gemini flash helped me out with the backwards through idw below

        # 3. back through idw: dist = -log(cdist^2) -> d_dist = -1/cdist^2 * d_cdist^2

        def get_keys_grad(q, k, d_sq, dist_grad):
            # dist = cdist(q, k) ** 2
            # return -log(dist, eps = idw_eps)

            cdist_sq_grad = -dist_grad / (d_sq + idw_eps)

            # d_cdist_sq / d_k = -2 * (q - k)

            diff = einx.subtract('... d, m d -> ... m d', q, k)
            grad = -2 * einx.multiply('... m, ... m d', cdist_sq_grad, diff)

            return reduce(grad, '... m d -> m d', 'sum')

        # stack the grads for key1 and key2 as the fast weight memories

        next_fast_weight_keys = stack((
            get_keys_grad(q1, k1, dist1, dist1_grad),
            get_keys_grad(q2, k2, dist2, dist2_grad)
        ))

        # accumulate the new stored memories with the old

        if exists(past_memories):
            next_fast_weight_memories = next_fast_weight_memories + past_memories.memory_values
            next_fast_weight_keys = next_fast_weight_keys + past_memories.keys

        if detach_next_memories:
            next_fast_weight_memories = next_fast_weight_memories.detach()
            next_fast_weight_keys = next_fast_weight_keys.detach()

        return out, Memories(next_fast_weight_memories, next_fast_weight_keys)
