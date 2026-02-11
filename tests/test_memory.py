import torch
import pytest
from fast_weight_product_key_memory import fwPKM, Memories

# helpers

param = pytest.mark.parametrize

# tests

def test_memory():
    pkm = fwPKM(512)
    tokens = torch.randn(2, 256, 512)

    (out, addressing_loss), memories = pkm(
        tokens,
        return_addressing_loss = True,
        return_next_memories = True
    )

    retrieved, addressing_loss = pkm(
        tokens,
        return_addressing_loss = True,
        past_memories = memories
    )

    assert tokens.shape == retrieved.shape
    assert addressing_loss.numel() == 1

def test_fw_pkm_basic_parity():
    dim = 32
    seq_len = 32
    chunk_size = 16

    model = fwPKM(
        dim = dim,
        num_memories = 16 * 16,
        dim_queries_keys = 16,
        dim_values = 16,
        chunk_size = chunk_size,
        topk = 4,
        learning_rate = 1.
    )

    tokens = torch.randn(1, seq_len, dim)

    # parallel processing

    parallel_out, parallel_state = model(tokens, return_next_memories = True)

    # sequential manual chunked processing

    seq_a = tokens[:, :16]
    seq_b = tokens[:, 16:]
    
    out_a, state_a = model(seq_a, return_next_memories = True)
    out_b, state_b = model(seq_b, return_next_memories = True, past_memories = state_a)
    
    sequential_out = torch.cat((out_a, out_b), dim = 1)

    # verify parity

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-6)
    assert torch.allclose(parallel_state.memory_values, state_b.memory_values, atol = 1e-6)
    assert torch.allclose(parallel_state.keys, state_b.keys, atol = 1e-6)

@param('seq_len, chunk_size', [
    (16, 8),
    (32, 16),
    (48, 16)
])
def test_fw_pkm_parity_multi(seq_len, chunk_size):
    dim = 32
    model = fwPKM(
        dim = dim,
        num_memories = 16 * 16,
        dim_queries_keys = 16,
        dim_values = 16,
        chunk_size = chunk_size,
        topk = 4,
        learning_rate = 1.
    )

    tokens = torch.randn(1, seq_len, dim)

    # parallel

    parallel_out, parallel_state = model(tokens, return_next_memories = True)

    # sequential

    state = None
    sequential_outputs = []

    for chunk in tokens.split(chunk_size, dim = 1):
        out, state = model(
            chunk,
            return_next_memories = True,
            past_memories = state
        )
        sequential_outputs.append(out)
    
    sequential_out = torch.cat(sequential_outputs, dim = 1)
    
    assert torch.allclose(parallel_out, sequential_out, atol = 1e-6)
    assert torch.allclose(parallel_state.memory_values, state.memory_values, atol = 1e-6)

def test_fw_pkm_addressing_loss_unpacked():
    dim = 32
    model = fwPKM(
        dim = dim,
        num_memories = 16 * 16,
        dim_queries_keys = 16,
        dim_values = 16,
        topk = 4,
        learning_rate = 1.
    )

    seq = torch.randn(1, 16, dim)
    
    # check returns

    res = model(seq, return_addressing_loss = True)
    assert isinstance(res, tuple) and len(res) == 2 
    
    res = model(seq, return_next_memories = True)
    assert isinstance(res, tuple) and len(res) == 2
    
    res = model(seq, return_addressing_loss = True, return_next_memories = True)
    assert isinstance(res, tuple) and len(res) == 2 
    assert isinstance(res[0], tuple) and len(res[0]) == 2

def test_fw_pkm_token_by_token():
    dim = 32
    model = fwPKM(
        dim = dim,
        num_memories = 4 * 4,
        dim_queries_keys = 8,
        dim_values = 8,
        chunk_size = 1,
        topk = 2
    )

    seq = torch.randn(1, 5, dim)
    parallel_out, parallel_state = model(seq, return_next_memories = True)

    state = None
    serial_out = []

    for i in range(5):
        token = seq[:, i:i+1]
        out, state = model(
            token,
            return_next_memories = True,
            past_memories = state
        )
        serial_out.append(out)
        
    serial_out = torch.cat(serial_out, dim = 1)
    
    assert torch.allclose(parallel_out, serial_out, atol = 1e-6)
    assert torch.allclose(parallel_state.memory_values, state.memory_values, atol = 1e-6)

def test_fw_pkm_causality():
    dim = 32
    seq_len = 8
    model = fwPKM(
        dim = dim,
        num_memories = 16 * 16,
        dim_queries_keys = 16,
        dim_values = 16,
        topk = 4,
        learning_rate = 1.
    )

    tokens = torch.randn(1, seq_len, dim, requires_grad = True)
    
    output = model(tokens)
    
    for i in range(seq_len):
        loss = output[0, i].sum()
        loss.backward(retain_graph = True)
        
        if i < seq_len - 1:
            after_grads = tokens.grad[0, i+1:]
            assert torch.all(after_grads == 0)
            
        tokens.grad.zero_()

@param('chunk_size', [8, 16])
def test_fw_pkm_unaligned_parity(chunk_size):
    dim = 32
    seq_len = 32
    model = fwPKM(
        dim = dim,
        num_memories = 16 * 16,
        dim_queries_keys = 16,
        dim_values = 16,
        chunk_size = chunk_size,
        topk = 4,
        learning_rate = 1.
    )

    tokens = torch.randn(1, seq_len, dim)

    # parallel

    parallel_out, parallel_state = model(tokens, return_next_memories = True)

    # sequential with unaligned chunks

    curr_state = None
    serial_outputs = []
    chunk_step = 5

    for i in range(0, seq_len, chunk_step):
        chunk = tokens[:, i:i+chunk_step]
        out, curr_state = model(
            chunk,
            past_memories = curr_state,
            return_next_memories = True
        )
        serial_outputs.append(out)
    
    sequential_out = torch.cat(serial_outputs, dim = 1)

    # verify parity

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-6)
    assert torch.allclose(parallel_state.memory_values, curr_state.memory_values, atol = 1e-6)
