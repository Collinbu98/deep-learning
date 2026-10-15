import importlib.util
from pathlib import Path

import pytest
import torch


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Offline tests: synthetic inputs only, no data download and no training.
attention_exp = load_module("self_attention_exp_for_mha", "05_self_attention.py")
block_exp = load_module("transformer_block_exp_for_mha", "06_transformer_block.py")

DEVICE = torch.device("cpu")


def _mha(d_model=8, n_heads=1, seed=0):
    torch.manual_seed(seed)
    return attention_exp.MultiHeadSelfAttention(d_model, n_heads).to(DEVICE)


def _input(batch_size=2, seq_len=4, d_model=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, d_model, device=DEVICE)


# ----------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------


def test_single_head_output_shape_matches_input():
    model = _mha(d_model=8, n_heads=1)
    x = _input(d_model=8)

    output = model(x)

    assert output.shape == (2, 4, 8)


def test_multi_head_output_shape_matches_input():
    model = _mha(d_model=8, n_heads=4)
    x = _input(d_model=8)

    output = model(x)

    assert output.shape == (2, 4, 8)


def test_attention_weights_shape_has_a_head_axis():
    model = _mha(d_model=8, n_heads=4)
    x = _input(d_model=8)

    _, weights = model(x, return_weights=True)

    assert weights.shape == (2, 4, 4, 4)  # [B, H, T, T]


# ----------------------------------------------------------------------
# Validation of configurations
# ----------------------------------------------------------------------


@pytest.mark.parametrize("d_model, n_heads", [(8, 3), (6, 4), (10, 3)])
def test_incompatible_embedding_dimension_and_head_count_rejected(d_model, n_heads):
    with pytest.raises(ValueError):
        attention_exp.MultiHeadSelfAttention(d_model, n_heads)


@pytest.mark.parametrize("n_heads", [0, -1])
def test_non_positive_head_count_rejected(n_heads):
    with pytest.raises(ValueError):
        attention_exp.MultiHeadSelfAttention(8, n_heads)


# ----------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_outputs_are_finite(n_heads):
    model = _mha(d_model=8, n_heads=n_heads)
    x = _input(d_model=8)

    output, weights = model(x, return_weights=True, causal=True)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_attention_weights_are_nonnegative_and_rows_sum_to_one(n_heads):
    model = _mha(d_model=8, n_heads=n_heads)
    x = _input(d_model=8)

    _, weights = model(x, return_weights=True)

    assert (weights >= 0).all()
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


# ----------------------------------------------------------------------
# Causal masking
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_causal_weights_are_zero_above_the_diagonal(n_heads):
    model = _mha(d_model=8, n_heads=n_heads)
    x = _input(d_model=8)

    _, weights = model(x, return_weights=True, causal=True)

    t = x.size(1)
    strictly_above = torch.triu(
        torch.ones(t, t, dtype=torch.bool), diagonal=1
    )
    assert (weights[..., strictly_above] == 0).all()
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def _forced_identity_mha(d_model=8, n_heads=4):
    """Deterministic multi-head model whose projections are the identity.

    With Wq = Wk = Wv = I the per-head scores are raw input similarities; the
    output projection is left random. No random q/k/v weights are involved, so
    the causality comparison is exact.
    """
    model = attention_exp.MultiHeadSelfAttention(d_model, n_heads).to(DEVICE)
    with torch.no_grad():
        for projection in (model.query, model.key, model.value):
            projection.weight.copy_(torch.eye(d_model))
            projection.bias.zero_()
    return model


def _future_input(d_model=8, seq_len=4, k=2):
    shared = torch.full((d_model,), 2.0, device=DEVICE)
    x = torch.zeros(1, seq_len, d_model, device=DEVICE)
    x[0, 0] = shared
    x[0, k] = shared
    x[0, 1] = torch.arange(d_model, device=DEVICE, dtype=torch.float32) * 0.1
    x[0, 3] = -shared
    return x, k


def test_future_token_change_leaves_earlier_causal_outputs_unchanged():
    model = _forced_identity_mha()
    x, k = _future_input()

    before = model(x, causal=True)

    changed = x.clone()
    changed[0, k] = 3.0
    after = model(changed, causal=True)

    assert torch.allclose(before[0, :k], after[0, :k], atol=1e-6)
    assert not torch.allclose(before[0, k:], after[0, k:])


# ----------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_heads", [1, 4])
def test_backpropagation_gives_finite_gradients(n_heads):
    model = _mha(d_model=8, n_heads=n_heads)
    x = _input(d_model=8)

    output = model(x, causal=True)
    output.pow(2).mean().backward()

    for name, param in model.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

    assert model.query.weight.grad.abs().sum() > 0
    assert model.out_proj.weight.grad.abs().sum() > 0


# ----------------------------------------------------------------------
# Transformer block integration
# ----------------------------------------------------------------------


def test_block_defaults_to_single_head_and_preserves_behavior():
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16).to(DEVICE)
    assert block.n_heads == 1
    assert type(block.attention).__name__ == "SelfAttention"

    x = torch.randn(2, 5, 16, device=DEVICE)
    assert block(x, causal=True).shape == x.shape


@pytest.mark.parametrize("n_heads", [2, 4])
def test_block_supports_multiple_heads(n_heads):
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16, n_heads=n_heads).to(DEVICE)
    assert block.n_heads == n_heads
    assert type(block.attention).__name__ == "MultiHeadSelfAttention"

    x = torch.randn(2, 5, 16, device=DEVICE)
    output = block(x, causal=True)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_block_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        block_exp.TransformerBlock(10, n_heads=3)
