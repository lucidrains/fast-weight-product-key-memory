import pytest
param = pytest.mark.parametrize

import torch
from fast_weight_product_key_memory import fwPKM

# tests

def default_kwargs():
    return dict(
        dim = 32,
        num_memories = 16 * 16,
        dim_queries_keys = 16,
        dim_values = 16,
        chunk_size = 8,
        topk = 4,
        learning_rate = 1.
    )

@param('heads', [1, 4])
@param('mse_loss_weight_to_keys', [0., 1.])
def test_gdn_update_forward(heads, mse_loss_weight_to_keys):
    pkm = fwPKM(
        heads = heads,
        mse_loss_weight_to_keys = mse_loss_weight_to_keys,
        use_gdn_update = True,
        **default_kwargs()
    )
    tokens = torch.randn(2, 32, 32)

    out, memories = pkm(tokens, return_next_memories = True)
    retrieved = pkm(tokens, past_memories = memories)

    assert tokens.shape == retrieved.shape

@param('heads', [1, 4])
def test_gdn_update_learned_forget_gate(heads):
    pkm = fwPKM(
        heads = heads,
        use_gdn_update = True,
        **default_kwargs()
    )
    tokens = torch.randn(1, 32, 32)

    out, _ = pkm(tokens, return_next_memories = True)
    out.sum().backward()

    assert pkm.forget_gate_decay.grad is not None
    assert pkm.to_forget_gates.weight.grad is not None

def test_gdn_update_extra_params_only_when_enabled():
    pkm = fwPKM(**default_kwargs())
    assert not hasattr(pkm, 'to_forget_gates')
    assert not hasattr(pkm, 'forget_gate_decay')

    pkm = fwPKM(use_gdn_update = True, **default_kwargs())
    assert hasattr(pkm, 'to_forget_gates')
    assert hasattr(pkm, 'forget_gate_decay')

def test_gdn_update_matches_default_at_zero_decay():
    torch.manual_seed(42)
    pkm_gdn = fwPKM(use_gdn_update = True, **default_kwargs())

    torch.manual_seed(42)
    pkm_base = fwPKM(**default_kwargs())

    tokens = torch.randn(1, 32, 32)

    out_gdn, mem_gdn = pkm_gdn(tokens, return_next_memories = True)
    out_base, mem_base = pkm_base(tokens, return_next_memories = True)

    # zero decay init (forget gate = 1) reduces exactly to the default update rule

    assert torch.allclose(out_gdn, out_base, atol = 1e-6)
    assert torch.allclose(mem_gdn.memory_values, mem_base.memory_values, atol = 1e-6)
    assert torch.allclose(mem_gdn.keys, mem_base.keys, atol = 1e-6)

@param('heads', [1, 4])
@param('mse_loss_weight_to_keys', [0., 1.])
def test_gdn_update_parity(heads, mse_loss_weight_to_keys):
    model = fwPKM(
        heads = heads,
        mse_loss_weight_to_keys = mse_loss_weight_to_keys,
        use_gdn_update = True,
        **default_kwargs()
    )

    tokens = torch.randn(1, 32, 32)

    parallel_out, parallel_state = model(tokens, return_next_memories = True)

    out_a, state_a = model(tokens[:, :16], return_next_memories = True)
    out_b, state_b = model(tokens[:, 16:], return_next_memories = True, past_memories = state_a)

    sequential_out = torch.cat((out_a, out_b), dim = 1)

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-6)
    assert torch.allclose(parallel_state.memory_values, state_b.memory_values, atol = 1e-6)
    assert torch.allclose(parallel_state.keys, state_b.keys, atol = 1e-6)

@param('heads', [1, 4])
def test_gdn_update_unaligned_parity(heads):
    model = fwPKM(
        heads = heads,
        use_gdn_update = True,
        **default_kwargs()
    )

    seq_len = 32
    tokens = torch.randn(1, seq_len, 32)

    parallel_out, parallel_state = model(tokens, return_next_memories = True)

    curr_state = None
    serial_outputs = []

    for i in range(0, seq_len, 5):
        chunk = tokens[:, i:i+5]
        out, curr_state = model(chunk, past_memories = curr_state, return_next_memories = True)
        serial_outputs.append(out)

    sequential_out = torch.cat(serial_outputs, dim = 1)

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-6)
    assert torch.allclose(parallel_state.memory_values, curr_state.memory_values, atol = 1e-6)

@param('heads', [1, 4])
@param('mse_loss_weight_to_keys', [0., 1.])
def test_gdn_update_batch_parity(heads, mse_loss_weight_to_keys):
    model = fwPKM(
        heads = heads,
        mse_loss_weight_to_keys = mse_loss_weight_to_keys,
        use_gdn_update = True,
        **default_kwargs()
    )

    batch, seq_len = 3, 32
    tokens = torch.randn(batch, seq_len, 32)

    parallel_out, parallel_state = model(tokens, return_next_memories = True)

    curr_state = None
    serial_outputs = []

    for i in range(0, seq_len, 5):
        chunk = tokens[:, i:i+5]
        out, curr_state = model(
            chunk,
            past_memories = curr_state,
            return_next_memories = True
        )
        serial_outputs.append(out)

    sequential_out = torch.cat(serial_outputs, dim = 1)

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-6)
    assert torch.allclose(parallel_state.memory_values, curr_state.memory_values, atol = 1e-6)
    assert torch.allclose(parallel_state.keys, curr_state.keys, atol = 1e-6)

@param('heads', [1, 4])
@param('mse_loss_weight_to_keys', [0., 1.])
def test_gdn_update_batch_independence(heads, mse_loss_weight_to_keys):
    model = fwPKM(
        heads = heads,
        mse_loss_weight_to_keys = mse_loss_weight_to_keys,
        use_gdn_update = True,
        **default_kwargs()
    )

    batch, seq_len = 3, 32
    tokens = torch.randn(batch, seq_len, 32)

    # reordering batch elements must not affect per-element results (no cross-contamination)
    # loose tolerance: reordering changes memory layout, and CPU kernels produce ~1e-5
    # alignment-dependent float noise in the projections; genuine cross-batch coupling would be O(1)

    perm = torch.randperm(batch)
    inv_perm = torch.argsort(perm)

    out, state = model(tokens, return_next_memories = True)
    out_perm, state_perm = model(tokens[perm], return_next_memories = True)

    assert torch.allclose(out_perm[inv_perm], out, atol = 1e-4)
    assert torch.allclose(state_perm.memory_values[inv_perm], state.memory_values, atol = 1e-4)
    assert torch.allclose(state_perm.keys[:, inv_perm], state.keys, atol = 1e-4)
