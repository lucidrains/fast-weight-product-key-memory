import pytest
param = pytest.mark.parametrize

import torch

def test_memory():
    from fast_weight_product_key_memory.fwPKM import fwPKM

    pkm = fwPKM(512)

    tokens = torch.randn(2, 256, 512)
    out = pkm(tokens)

    assert tokens.shape == out.shape
