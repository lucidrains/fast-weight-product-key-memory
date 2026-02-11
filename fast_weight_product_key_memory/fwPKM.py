from __future__ import annotations
from typing import NamedTuple
from math import sqrt
from functools import partial

import torch
from torch import nn, stack, cat, cdist
from torch import tensor, Tensor
import torch.nn.functional as F
from torch.nn import Module, Sequential, RMSNorm

import einx
from einops import rearrange, einsum, repeat, reduce
from einops.layers.torch import Rearrange

from torch_einops_utils import shape_with_replace, safe_cat

# constants

class Memories(NamedTuple):
    memory_values: Tensor
    keys: Tensor
    last_token: Tensor | None = None
    cached_tokens: Tensor | None = None
    token_count: int = 0
    num_cached: int = 0

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_greater_than_zero(n):
    return n > 0

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

def remove_last_token(t):
    return t[:, :-1]

def remove_first_token(t):
    return t[:, 1:]

def get_first_token(t):
    return t[:, :1]

def get_last_token(t):
    return t[:, -1:]

# classes

class fwPKM(Module):
    def __init__(
        self,
        dim,
        *,
        chunk_size = 256,
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

        self.chunk_size = chunk_size

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @property
    def init_memories(self):
        return Memories(torch.zeros_like(self.memories), torch.zeros_like(self.keys))

    def calculate_addressing_loss(
        self,
        indices,
        scores
    ):
        num_keys = self.num_keys
        b, n, _ = indices.shape

        # key indices for the two keypads

        key_indices = stack((indices // num_keys, indices % num_keys))
        scores = repeat(scores, 'b n k -> two b n k', two = 2)

        # compute distribution per token and keypad

        shape = (2, b, n, num_keys)
        probs = torch.zeros(shape, device = self.device)
        probs.scatter_add_(-1, key_indices, scores)

        # average of per-token, per-keypad entropies

        addressing_loss = entropy(probs).mean(dim = 0)
        return addressing_loss * self.addressing_loss_weight

    def retrieve(
        self,
        tokens,
        past_memories: Memories | None = None,
        idw_eps = 1e-3
    ):
        k, num_keys = self.topk, self.num_keys

        q1, q2 = self.to_queries(tokens)

        # keys for pkm, accounting for fast weight memories

        keys = self.keys

        if exists(past_memories):
            keys = keys + past_memories.keys

        k1, k2 = keys

        dist1 = cdist(q1, k1) ** 2
        dist2 = cdist(q2, k2) ** 2

        score1, score2 = -log(dist1, eps = idw_eps), -log(dist2, eps = idw_eps)

        # get the topk closest - using negative distance for stable selection

        _, indices1 = (-dist1).topk(k = k)
        _, indices2 = (-dist2).topk(k = k)

        top1 = score1.gather(-1, indices1)
        top2 = score2.gather(-1, indices2)

        scores = einx.add('... i, ... j -> ... (i j)', top1, top2)

        # product keys

        indices = einx.add('... i, ... j -> ... (i j)', indices1 * num_keys, indices2)

        # for stable product selection, we rank by -( (dist1 + eps) * (dist2 + eps) )
        # which is equivalent to ranking by log sums but numerically robust

        s1 = (dist1 + idw_eps).gather(-1, indices1)
        s2 = (dist2 + idw_eps).gather(-1, indices2)
        prod_dist = einx.multiply('... i, ... j -> ... (i j)', s1, s2)

        _, sub_indices = (-prod_dist).topk(k = k)

        final_indices = indices.gather(-1, sub_indices)

        # scores are reconstructed from the log-scores of the winners

        top_scores = scores.gather(-1, sub_indices)

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

        target_values = z_score(target_values)

        output = target_values.lerp(values, gates)

        out = self.to_out(output)

        # return everything needed for store

        intermediates = dict(
            q1 = q1,
            q2 = q2,
            dist1 = dist1,
            dist2 = dist2,
            indices1 = indices1,
            indices2 = indices2,
            scores = scores,
            sub_indices = sub_indices,
            final_indices = final_indices,
            final_scores = final_scores,
            gates = gates,
            target_values = target_values,
            values = values,
            memories = memories,
            k1 = k1,
            k2 = k2
        )

        return out, intermediates

    def store(
        self,
        intermediates: dict,
        past_memories: Memories | None = None,
        detach_next_memories = False,
        idw_eps = 1e-3
    ):
        k, num_keys = self.topk, self.num_keys

        (
            q1, q2,
            final_indices, final_scores,
            gates, values, memories,
            indices1, indices2, sub_indices,
            dist1, dist2
        ) = [remove_last_token(intermediates[key]) for key in (
            'q1', 'q2',
            'final_indices', 'final_scores',
            'gates', 'values', 'memories',
            'indices1', 'indices2', 'sub_indices',
            'dist1', 'dist2'
        )]

        target_values = remove_first_token(intermediates['target_values'])

        # mse loss with lookahead

        error = gates * (target_values - values) * self.learning_rate # swapped for gradient descent

        memories_grad = einx.multiply('... d, ... topk -> (... topk) d', error, final_scores)

        flattened_final_indices = final_indices.flatten()
        final_indices_expanded = repeat(flattened_final_indices, '... -> (...) d', d = memories_grad.shape[-1])

        next_fast_weight_memories = torch.zeros_like(self.memories).scatter_reduce_(0, final_indices_expanded, memories_grad, reduce = 'mean', include_self = False)

        final_scores_grad = einsum(error, memories, '... d, ... topk d -> ... topk')
        top_scores_grad = final_scores * (final_scores_grad - (final_scores * final_scores_grad).sum(dim = -1, keepdim = True))

        # now propagate top_scores_grad back to the keys

        sub_indices1 = sub_indices // k
        sub_indices2 = sub_indices % k

        final_indices1 = indices1.gather(-1, sub_indices1)
        final_indices2 = indices2.gather(-1, sub_indices2)

        grad_shape = shape_with_replace(dist1, {-1: num_keys})

        dist1_grad = torch.zeros(grad_shape, device = self.device).scatter_add_(-1, final_indices1, top_scores_grad)
        dist2_grad = torch.zeros(grad_shape, device = self.device).scatter_add_(-1, final_indices2, top_scores_grad)

        def get_keys_grad(q, k, d_sq, dist_grad):
            cdist_sq_grad = -dist_grad / (d_sq + idw_eps)
            diff = einx.subtract('... d, m d -> ... m d', q, k)
            grad = -2 * einx.multiply('... m, ... m d', cdist_sq_grad, diff)
            return reduce(grad, '... m d -> m d', 'sum')

        next_fast_weight_keys = stack((
            get_keys_grad(q1, intermediates['k1'], dist1, dist1_grad),
            get_keys_grad(q2, intermediates['k2'], dist2, dist2_grad)
        ))

        if exists(past_memories):
            next_fast_weight_memories = next_fast_weight_memories + past_memories.memory_values
            next_fast_weight_keys = next_fast_weight_keys + past_memories.keys

        if detach_next_memories:
            next_fast_weight_memories = next_fast_weight_memories.detach()
            next_fast_weight_keys = next_fast_weight_keys.detach()

        return next_fast_weight_memories, next_fast_weight_keys

    def forward(
        self,
        tokens,
        return_next_memories = False,
        return_addressing_loss = False,
        past_memories: Memories | None = None,
        detach_next_memories = False,
        idw_eps = 1e-3
    ):
        past_mem = default(past_memories, self.init_memories)
        num_tokens, count, chunk_size = tokens.shape[1], past_mem.token_count, self.chunk_size

        # calc segments reaching chunk boundaries

        to_bound = chunk_size - (count % chunk_size)
        rem = max(0, num_tokens - to_bound)
        split_sizes = (min(num_tokens, to_bound), *([chunk_size] * (rem // chunk_size)), rem % chunk_size)
        segments = tokens.split(list(filter(is_greater_than_zero, split_sizes)), dim = 1)

        out_list, loss_list = [], []

        for segment in segments:
            # potential chunked store across boundary

            if past_mem.num_cached == chunk_size:
                _, s_inter = self.retrieve(
                    cat((past_mem.cached_tokens, get_first_token(segment)), dim = 1),
                    past_memories = past_mem,
                    idw_eps = idw_eps
                )

                mv, mk = self.store(s_inter, past_memories = past_mem, idw_eps = idw_eps)
                past_mem = past_mem._replace(memory_values = mv, keys = mk, cached_tokens = None, num_cached = 0)

            # retrieve outputs

            out, inter = self.retrieve(
                safe_cat((past_mem.last_token, segment), dim = 1),
                past_memories = past_mem,
                idw_eps = idw_eps
            )

            # update state

            mv, mk = past_mem.memory_values, past_mem.keys

            # handle causal chaining and slicing

            indices, scores, slen = inter['final_indices'], inter['final_scores'], segment.shape[1]

            if exists(past_mem.last_token):
                out, indices, scores = [remove_first_token(t) for t in (out, indices, scores)]

            cached = safe_cat((past_mem.cached_tokens, segment), dim = 1)
            past_mem = Memories(mv, mk, get_last_token(segment), cached, past_mem.token_count + slen, past_mem.num_cached + slen)

            out_list.append(out)
            loss_list.append(self.calculate_addressing_loss(indices, scores))

        # finalize next memories

        if detach_next_memories:
            past_mem = past_mem._replace(
                memory_values = past_mem.memory_values.detach(),
                keys = past_mem.keys.detach()
            )

        # finalize return

        out = cat(out_list, dim = 1)
        res = (out, cat(loss_list, dim = 1)) if return_addressing_loss else out
        return (res, past_mem) if return_next_memories else res
