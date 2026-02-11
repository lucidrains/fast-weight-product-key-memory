# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "tqdm",
#   "einops",
#   "local-attention",
#   "accelerate",
#   "fast-weight-product-key-memory",
#   "fire"
# ]
# ///

from __future__ import annotations

import math
import gzip
import random
import tqdm
import fire
import numpy as np

import torch
from torch.optim import Adam
from torch import nn, Tensor
from torch.nn import Module, ModuleList, RMSNorm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from local_attention import LocalAttention, LocalMHA
from local_attention.transformer import FeedForward
from einops import rearrange, repeat, einsum

from fast_weight_product_key_memory import fwPKM

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def gumbel_noise(logits):
    noise = torch.zeros_like(logits).uniform_(0, 1)
    return -torch.log(-torch.log(noise))

def sample(logits, temperature = 1., filter_thres = 0.9):
    logits = logits / max(temperature, 1e-10)
    if filter_thres < 1.:
        logits = top_k(logits, thres = filter_thres)

    return torch.argmax(logits + gumbel_noise(logits), dim = -1)

# sampling helpers

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None

    for _ in range(sample_num_times):
        logits, cache = net(out, return_cache = True, cache = cache)
        last_logit = logits[:, -1]

        sampled_token = sample(last_logit, temperature = temperature, filter_thres = filter_thres)

        out = torch.cat((out, sampled_token[:, None]), dim = -1)

    return out[..., prompt_seq_len:]

# transformer with potential fwPKM layers

class Transformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        local_attn_window_size = 256,
        neural_memory_layers: tuple[int, ...] | None = None,
        neural_memory_kwargs: dict = dict()
    ):
        super().__init__()
        neural_memory_layers = default(neural_memory_layers, ())

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = ModuleList([])

        for i in range(depth):
            layer_idx = i + 1

            maybe_neural_memory = fwPKM(dim = dim, **neural_memory_kwargs) if layer_idx in neural_memory_layers else None

            self.layers.append(ModuleList([
                maybe_neural_memory,
                LocalMHA(dim = dim, window_size = local_attn_window_size, prenorm = True, causal = True, use_rotary_pos_emb = False),
                FeedForward(dim = dim)
            ]))

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        self.norm = RMSNorm(dim)

    def forward(
        self,
        x,
        return_loss = False,
        return_cache = False,
        cache = None
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        has_cache = exists(cache)

        if has_cache:
            x = x[:, -1:]
            layer_caches, offset = cache
        else:
            layer_caches = []
            offset = 0

        n, device = x.shape[1], x.device

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device) + offset)

        total_addressing_loss = self.zero

        new_layer_caches = []
        layer_caches = iter(default(layer_caches, []))

        for neural_memory, attn, ff in self.layers:
            nm_cache, attn_cache = next(layer_caches, (None, None))

            next_nm_cache = nm_cache

            if exists(neural_memory):
                neural_memory_res, next_nm_cache = neural_memory(
                    x,
                    return_addressing_loss = True,
                    past_memories = nm_cache,
                    return_next_memories = True
                )

                neural_memory_out, addressing_loss = neural_memory_res

                x = x + neural_memory_out
                total_addressing_loss = total_addressing_loss + addressing_loss.mean()

            attn_out, next_attn_cache = attn(
                x,
                cache = attn_cache,
                return_cache = True
            )

            x = attn_out + x
            x = ff(x) + x

            new_layer_caches.append((next_nm_cache, next_attn_cache))

        x = self.norm(x)
        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, (new_layer_caches, offset + n)

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)

        return loss + total_addressing_loss, (loss, total_addressing_loss)

# training function

def train(
    num_batches = 100000,
    batch_size = 4,
    grad_accum_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    generate_every = 500,
    generate_length = 512,
    seq_len = 512,
    prime_length = 128,
    dim = 512,
    depth = 6,
    local_attn_window_size = 256,
    neural_memory_layers = (2, 4)
):
    # accelerator

    accelerator = Accelerator(gradient_accumulation_steps = grad_accum_every)
    device = accelerator.device

    # model setup

    model = Transformer(
        num_tokens = 256,
        dim = dim,
        max_seq_len = seq_len,
        depth = depth,
        local_attn_window_size = local_attn_window_size,
        neural_memory_layers = neural_memory_layers
    )

    # prepare enwik8 data

    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return self.data.size(0) // self.seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
            return full_seq

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # optimizer

    optim = Adam(model.parameters(), lr = learning_rate)

    # prepare everything with accelerator

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval = 10.0, desc = "training", disable = not accelerator.is_main_process):
        model.train()

        with accelerator.accumulate(model):
            data = next(train_loader)

            loss, (ar_loss, addr_loss) = model(data, return_loss = True)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.5)

            optim.step()
            optim.zero_grad()


        if divisible_by(i, 10):
            accelerator.print(f"step {i} training loss: {ar_loss.item():.3f}, addressing loss: {addr_loss.item():.3f}")

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)
                loss, (ar_loss, addr_loss) = model(valid_data, return_loss = True)
                accelerator.print(f"validation loss: {ar_loss.item():.3f}, addressing loss: {addr_loss.item():.3f}")

        if divisible_by(i, generate_every) and accelerator.is_main_process:
            model.eval()

            unwrapped_model = accelerator.unwrap_model(model)

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(device)

            prime = decode_tokens(inp)
            print(f"\nINPUT: {prime}")

            prompt = inp[None, ...]

            sampled = base_decoding(unwrapped_model, prompt, generate_length)

            base_decode_output = decode_tokens(sampled[0])

            print(f"\nOUTPUT: {base_decode_output}\n")

if __name__ == "__main__":
    fire.Fire(train)
