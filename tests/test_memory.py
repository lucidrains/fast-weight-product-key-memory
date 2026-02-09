import pytest
param = pytest.mark.parametrize

import torch

def test_memory():
    from fast_weight_product_key_memory.fwPKM import fwPKM

    pkm = fwPKM(512)

    tokens = torch.randn(2, 256, 512)

    (_, _), memories = pkm(tokens, return_addressing_loss = True, return_next_memories = True)
    retrieved, addressing_loss = pkm(tokens, return_addressing_loss = True, past_memories = memories)

    assert tokens.shape == retrieved.shape
    assert addressing_loss.numel() == 1
