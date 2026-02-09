import pytest
param = pytest.mark.parametrize

import torch

def test_memory():
    from fast_weight_product_key_memory.fwPKM import fwPKM

    pkm = fwPKM(512)

    tokens = torch.randn(2, 256, 512)
    out, addressing_loss = pkm(tokens, return_addressing_loss = True)

    assert tokens.shape == out.shape
    assert addressing_loss.numel() == 1
