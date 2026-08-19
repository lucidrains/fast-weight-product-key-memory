# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einx",
#     "einops",
#     "fire",
#     "torch",
#     "torch-einops-utils",
#     "tqdm"
# ]
# ///

import fire
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from fast_weight_product_key_memory import fwPKM

# helpers

def exists(val):
    return val is not None

class MemorizingModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        *,
        chunk_size = 4,
        heads = 1,
        num_memories = 256,
        dim_queries_keys = 64,
        learning_rate = 1.,
        topk = 4,
        addressing_loss_weight = 1e-3,
        use_memory = True,
        use_gdn_update = False
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, dim)

        self.memory = None
        if use_memory:
            self.memory = fwPKM(
                dim = dim,
                heads = heads,
                num_memories = num_memories,
                dim_queries_keys = dim_queries_keys,
                dim_values = dim,
                learning_rate = learning_rate,
                topk = topk,
                chunk_size = chunk_size,
                addressing_loss_weight = addressing_loss_weight,
                use_gdn_update = use_gdn_update
            )

        self.head = nn.Linear(dim, num_tokens)

    def forward(self, x):
        h = self.embed(x)

        if exists(self.memory):
            h = h + self.memory(h)

        return self.head(h)

# training

def train(
    seed = 42,
    num_tokens = 32,
    dim = 128,
    half_len = 8,
    chunk_size = 4,
    heads = 1,
    num_memories = 256,
    dim_queries_keys = 64,
    learning_rate = 1.,
    topk = 4,
    addressing_loss_weight = 1e-3,
    num_batches = 2000,
    lr = 1e-3,
    eval_batches = 15,
    eval_every = 500,
    use_gdn_update = False
):
    assert chunk_size <= half_len, "the chunk len should be less than or equal to the half len, or the memory isn't being properly tested"

    results = dict()

    for use_memory in (False, True):
        torch.manual_seed(seed)

        model = MemorizingModel(
            num_tokens,
            dim,
            chunk_size = chunk_size,
            heads = heads,
            num_memories = num_memories,
            dim_queries_keys = dim_queries_keys,
            learning_rate = learning_rate,
            topk = topk,
            addressing_loss_weight = addressing_loss_weight,
            use_memory = use_memory,
            use_gdn_update = use_gdn_update
        )
        optim = Adam(model.parameters(), lr = lr)

        label = 'fwPKM' if use_memory else 'Baseline'

        def eval_acc():
            model.eval()
            with torch.no_grad():
                eval_half = torch.randint(0, num_tokens, (1, half_len))
                eval_seq = torch.cat((eval_half, eval_half), dim = -1)
                eval_x, eval_labels = eval_seq[:, :-1], eval_seq[:, 1:]

                preds = model(eval_x).argmax(dim = -1)
                return (preds[:, (half_len - 1):] == eval_labels[:, (half_len - 1):]).float().mean().item()

        pbar = tqdm(range(num_batches), desc = label)
        last_accs = []

        for i in pbar:
            model.train()

            half = torch.randint(0, num_tokens, (1, half_len))
            seq = torch.cat((half, half), dim = -1)

            x, labels = seq[:, :-1], seq[:, 1:]

            logits = model(x)

            # only calculate cross entropy loss on the second half, the recall portion

            loss = F.cross_entropy(
                logits[:, (half_len - 1):].reshape(-1, num_tokens),
                labels[:, (half_len - 1):].reshape(-1)
            )

            loss.backward()
            optim.step()
            optim.zero_grad()

            # periodic eval to check convergence

            if eval_every > 0 and i > 0 and i % eval_every == 0:
                pbar.write(f'{label} batch {i}: eval acc {eval_acc():.1%}')

            if i >= (num_batches - eval_batches):
                last_accs.append(eval_acc())

        results[label] = sum(last_accs) / len(last_accs)

    # report

    print(f'\n{"-" * 40}')
    for label, acc in results.items():
        print(f'  {label}: {acc:.1%}')
    print(f'{"-" * 40}')

    memory_acc = results['fwPKM']
    print(f'\n  {"✅ Memory works!" if memory_acc > 0.75 else "❌ No clear advantage."}')

if __name__ == '__main__':
    fire.Fire(train)
