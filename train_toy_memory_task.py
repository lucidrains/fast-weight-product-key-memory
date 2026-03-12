# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "torch",
#     "tqdm"
# ]
# ///

import fire
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from fast_weight_product_key_memory import fwPKM

# helpers

def exists(val):
    return val is not None

# model

class MemorizingModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        use_memory = True
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, dim)

        self.memory = fwPKM(
            dim = dim,
            heads = 1,
            num_memories = 256,
            dim_queries_keys = 64,
            dim_values = dim,
            learning_rate = 1.,
            topk = 4,
            chunk_size = 4,
            addressing_loss_weight = 1e-3
        ) if use_memory else None

        self.head = nn.Linear(dim, num_tokens)

    def forward(self, x):
        h = self.embed(x)

        if exists(self.memory):
            h = h + self.memory(h)

        return self.head(h)

# training

def train(
    seed: int = 42,
    num_tokens: int = 32,
    dim: int = 128,
    half_len: int = 8,
    num_batches: int = 2000,
    lr: float = 1e-3
):
    results = dict()

    for use_memory in (False, True):
        torch.manual_seed(seed)

        model = MemorizingModel(num_tokens, dim, use_memory = use_memory)
        optim = Adam(model.parameters(), lr = lr)

        label = 'fwPKM' if use_memory else 'Baseline'
        pbar = tqdm(range(num_batches), desc = label)
        last_accs = []

        for i in pbar:
            model.train()

            half = torch.randint(0, num_tokens, (1, half_len))
            seq = torch.cat((half, half), dim = -1)

            x, labels = seq[:, :-1], seq[:, 1:]

            loss = F.cross_entropy(
                model(x).reshape(-1, num_tokens),
                labels.reshape(-1)
            )

            loss.backward()
            optim.step()
            optim.zero_grad()

            if i >= (num_batches - 15):
                model.eval()
                with torch.no_grad():
                    preds = model(x).argmax(dim = -1)
                    acc = (preds[:, half_len:] == labels[:, half_len:]).float().mean()
                    last_accs.append(acc.item())

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
