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

def divisible_by(num, den):
    return (num % den) == 0

def LinearNoBias(dim, dim_out):
    return nn.Linear(dim, dim_out, bias = False)

# tensor helpers

def log(t, eps = 1e-20):
    return (t + eps).log()

def l1norm(t, dim = -1, eps = 1e-10):
    return F.normalize(t, dim = dim, p = 1, eps = eps)

def entropy(prob):
    return -(prob * log(prob)).sum(dim = -1)

def z_score(t, dim = -1, eps = 1e-10):
    numer = (t - t.mean(dim = dim, keepdim = True))
    denom = t.std(dim = dim, keepdim = True)
    return numer / denom.clamp_min(eps)

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
        heads = 4,
        chunk_size = 256,
        num_memories = 512 * 512,
        dim_queries_keys = 512,
        dim_values = 512,
        learning_rate = 1.,
        topk = 8,
        addressing_loss_weight = 10.,
        mse_loss_weight_to_keys = 1.
    ):
        super().__init__()
        assert sqrt(num_memories).is_integer(), 'num memories must have an integer square root'
        assert addressing_loss_weight > 0., 'addressing loss weight must be greater than 0.'

        self.heads = heads
        self.dim_head_qk = dim_queries_keys // heads
        self.dim_head_v = dim_values // heads

        self.memories = nn.Parameter(torch.randn(num_memories, heads, self.dim_head_v))

        # pkm related

        self.topk = topk

        num_keys = int(sqrt(num_memories))
        self.keys = nn.Parameter(torch.randn(2, heads, num_keys, self.dim_head_qk) * 1e-2)

        self.num_keys = num_keys
        self.num_memories = num_memories

        # projections

        self.to_queries = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_queries_keys * 2),
             Rearrange('... (two h d) -> two ... h d', two = 2, h = heads)
        )

        self.to_gates = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, 1),
            nn.Sigmoid(),
            Rearrange('... 1 -> ... 1 1')
        )

        self.to_values = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_values),
            Rearrange('... (h d) -> ... h d', h = heads)
        )

        self.to_out = Sequential(
            RMSNorm(dim_values),
            LinearNoBias(dim_values, dim)
        )

        # storing related

        self.learning_rate = learning_rate

        self.addressing_loss_weight = addressing_loss_weight

        self.chunk_size = chunk_size

        self.mse_loss_weight_to_keys = mse_loss_weight_to_keys

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @property
    def has_mse_loss_weight_to_keys(self):
        return self.mse_loss_weight_to_keys > 0.

    @property
    def init_memories(self):
        return Memories(torch.zeros_like(self.memories), torch.zeros_like(self.keys))

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

        dist1 = torch.sum(einx.subtract('b n h d, h k d -> b n h k d', q1, k1) **2, dim = -1)
        dist2 = torch.sum(einx.subtract('b n h d, h k d -> b n h k d', q2, k2) **2, dim = -1)

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

        h_idx = torch.arange(self.heads, device=self.device).view(1, 1, self.heads, 1)
        memories = self.memories[final_indices, h_idx, :]

        # add the past memories

        if exists(past_memories):
            gathered_past_memories = past_memories.memory_values[final_indices, h_idx, :]
            memories = memories + gathered_past_memories

        values = einsum(memories, final_scores, '... topk d, ... topk -> ... d')

        # gates and values

        gates = self.to_gates(tokens)

        target_values = self.to_values(tokens)

        target_values = z_score(target_values)

        output = target_values.lerp(values, gates)

        output = rearrange(output, 'b n h d -> b n (h d)')
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

        memories_grad = einx.multiply('... h d, ... h topk -> (... topk) h d', error, final_scores)

        final_indices_expanded = repeat(final_indices, 'b n h k -> (b n k) h d', d = memories_grad.shape[-1])

        next_fast_weight_memories = torch.zeros_like(self.memories).scatter_reduce_(0, final_indices_expanded, memories_grad, reduce = 'mean', include_self = False)

        # unconditionally stack queries and keys for vectorization processing

        k1, k2 = intermediates['k1'], intermediates['k2']
        q1, q2 = intermediates['q1'], intermediates['q2']
        indices1, indices2 = intermediates['indices1'], intermediates['indices2']

        keys = stack((k1, k2))
        queries = stack((q1, q2))
        indices = stack((indices1, indices2))

        b, n, h, _, num_keys = queries.shape[1], queries.shape[2], queries.shape[3], queries.shape[4], keys.shape[2]

        dist_tgt = (einx.subtract('two b n h d, two h k d -> two b n h k d', queries, keys) ** 2).sum(dim = -1)

        # initialize base distances gradient (tracks the NEGATIVE gradient for gradient descent)

        dist_grad = torch.zeros((2, b, n, h, num_keys), device = self.device)

        # analytical addressing loss gradient to the keys

        # scores are negative log distances

        score_tgt = -log(dist_tgt, eps = idw_eps)

        # mask out everything except retrieved top k

        mask = torch.full_like(score_tgt, float('-inf'))
        mask.scatter_(-1, indices, score_tgt.gather(-1, indices))

        # compute distribution per token and keypad

        probs = F.softmax(mask, dim = -1)

        # average of per-token, per-keypad entropies

        p_bar = probs.mean(dim = (1, 2))

        # analytical addressing loss gradient
        # from marginal entropy H(p) = -p * log(p) -> dH/dp = -log(p) - 1

        weights = self.addressing_loss_weight * self.learning_rate
        dp_bar = -log(p_bar, eps = 1e-8) - 1.0
        dp_bar = dp_bar * weights

        dprobs = repeat(dp_bar, 'two h k -> two b n h k', b = b, n = n) / (b * n)

        dmask = probs * (dprobs - (probs * dprobs).sum(dim = -1, keepdim = True))
        dscore = torch.zeros_like(score_tgt).scatter_(-1, indices, dmask.gather(-1, indices))

        ddist = -dscore / (dist_tgt + idw_eps)

        # ddist here is the positive gradient of addressing loss
        # since dist_grad represents the negative gradient for gradient descent, we subtract it

        dist_grad = dist_grad - ddist

        if self.has_mse_loss_weight_to_keys:
            final_scores_grad = einsum(error, memories, '... d, ... topk d -> ... topk')
            top_scores_grad = final_scores * (final_scores_grad - (final_scores * final_scores_grad).sum(dim = -1, keepdim = True))

            # now propagate top_scores_grad back to the keys

            sub_indices1 = sub_indices // k
            sub_indices2 = sub_indices % k

            final_indices1 = indices1.gather(-1, sub_indices1)
            final_indices2 = indices2.gather(-1, sub_indices2)

            top_scores_grad_stack = stack((top_scores_grad, top_scores_grad))
            final_indices_stack = stack((final_indices1, final_indices2))

            top_scores_grad_stack = top_scores_grad_stack * self.mse_loss_weight_to_keys
            dist_grad.scatter_add_(-1, final_indices_stack, top_scores_grad_stack)

        # unify backwards tracking onto the queries and keys -> directly yields negative gradient to ADD to keys

        diff = einx.subtract('two b n h d, two h m d -> two b n h m d', queries, keys)
        grad = -2 * einx.multiply('... h m, ... h m d -> ... h m d', dist_grad, diff)
        next_fast_weight_keys = einx.sum('two b n h m d -> two h m d', grad)

        # accumulate on top of old memories

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
        past_memories: Memories | None = None,
        detach_next_memories_every: int | None = None,
        idw_eps = 1e-3
    ):
        past_mem = default(past_memories, self.init_memories)
        num_tokens, count, chunk_size = tokens.shape[1], past_mem.token_count, self.chunk_size

        # calc segments reaching chunk boundaries

        to_bound = chunk_size - (count % chunk_size)
        remainder = max(0, num_tokens - to_bound)
        num_chunks, chunk_remainder = divmod(remainder, chunk_size)

        split_sizes = (min(num_tokens, to_bound), *([chunk_size] * num_chunks), chunk_remainder)
        segments = tokens.split(list(filter(is_greater_than_zero, split_sizes)), dim = 1)

        out_list = []

        for chunk_index, segment in enumerate(segments):
            # periodic truncated bptt - detach memories every N chunks

            should_detach = exists(detach_next_memories_every) and divisible_by(chunk_index + 1, detach_next_memories_every)

            # potential chunked store across boundary

            if past_mem.num_cached == chunk_size:
                _, store_inter = self.retrieve(
                    cat((past_mem.cached_tokens, get_first_token(segment)), dim = 1),
                    past_memories = past_mem,
                    idw_eps = idw_eps
                )

                mv, mk = self.store(store_inter, past_memories = past_mem, detach_next_memories = should_detach, idw_eps = idw_eps)
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

        # finalize next memories

        if should_detach:
            past_mem = past_mem._replace(
                memory_values = past_mem.memory_values.detach(),
                keys = past_mem.keys.detach()
            )

        # finalize return

        res = cat(out_list, dim = 1)
        return (res, past_mem) if return_next_memories else res
