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
attention_exp = load_module("self_attention_exp_for_dropout", "05_self_attention.py")
block_exp = load_module("transformer_block_exp_for_dropout", "06_transformer_block.py")
lm_exp = load_module("char_lm_exp_for_dropout", "07_char_language_model.py")

DEVICE = torch.device("cpu")


def _tiny_text():
    return "Hello world! Hello world again."


# ----------------------------------------------------------------------
# Defaults and backward compatibility
# ----------------------------------------------------------------------


def test_dropout_is_disabled_by_default():
    assert attention_exp.SelfAttention(8).dropout == 0.0
    assert attention_exp.MultiHeadSelfAttention(8, 2).dropout == 0.0
    assert block_exp.TransformerBlock(16).dropout == 0.0

    tokenizer = lm_exp.CharTokenizer(_tiny_text())
    model = lm_exp.CharLanguageModel(tokenizer.vocab_size, d_model=16, n_layers=1)
    assert model.dropout == 0.0
    assert model.embed_dropout.p == 0.0


def test_zero_dropout_is_a_no_op_in_train_mode():
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16, n_heads=4, dropout=0.0).to(DEVICE)
    x = torch.randn(2, 5, 16, device=DEVICE)

    block.train()
    first = block(x)
    second = block(x)
    block.eval()
    evaluated = block(x)

    assert torch.allclose(first, second)
    assert torch.allclose(first, evaluated)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_dropout_rejected_by_attention_modules(bad):
    with pytest.raises(ValueError):
        attention_exp.SelfAttention(8, dropout=bad)
    with pytest.raises(ValueError):
        attention_exp.MultiHeadSelfAttention(8, 2, dropout=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.0])
def test_invalid_dropout_rejected_by_block(bad):
    with pytest.raises(ValueError):
        block_exp.TransformerBlock(16, dropout=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 3.0])
def test_invalid_dropout_rejected_by_language_model(bad):
    tokenizer = lm_exp.CharTokenizer(_tiny_text())
    with pytest.raises(ValueError):
        lm_exp.CharLanguageModel(tokenizer.vocab_size, d_model=16, n_layers=1, dropout=bad)


def test_zero_and_high_but_valid_dropout_accepted():
    attention_exp.SelfAttention(8, dropout=0.0)
    attention_exp.SelfAttention(8, dropout=0.999)


# ----------------------------------------------------------------------
# Shapes, loss and gradients with dropout enabled
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_heads", [1, 4])
def test_block_shapes_and_finite_gradients_with_dropout(n_heads):
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16, n_heads=n_heads, dropout=0.3).to(DEVICE)
    block.train()
    x = torch.randn(2, 6, 16, device=DEVICE)

    output = block(x, causal=True)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    output.pow(2).mean().backward()
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name


def test_language_model_finite_loss_and_gradients_with_dropout():
    torch.manual_seed(0)
    tokenizer = lm_exp.CharTokenizer(_tiny_text())
    model = lm_exp.CharLanguageModel(
        tokenizer.vocab_size, d_model=16, n_layers=1, context_len=8, n_heads=4, dropout=0.2
    ).to(DEVICE)
    model.train()

    x = torch.randint(0, tokenizer.vocab_size, (2, 6), device=DEVICE)
    y = torch.randint(0, tokenizer.vocab_size, (2, 6), device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
    loss.backward()

    assert logits.shape == (2, 6, tokenizer.vocab_size)
    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()


# ----------------------------------------------------------------------
# Train stochasticity vs. eval determinism
# ----------------------------------------------------------------------


def test_training_mode_dropout_is_stochastic():
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16, n_heads=4, dropout=0.5).to(DEVICE)
    block.train()
    x = torch.randn(4, 8, 16, device=DEVICE)

    first = block(x)
    second = block(x)

    assert not torch.allclose(first, second)


def test_evaluation_mode_is_deterministic_with_dropout():
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16, n_heads=4, dropout=0.5).to(DEVICE)
    block.eval()
    x = torch.randn(4, 8, 16, device=DEVICE)

    with torch.no_grad():
        first = block(x)
        second = block(x)

    assert torch.allclose(first, second)


def test_language_model_train_stochastic_eval_deterministic():
    torch.manual_seed(0)
    tokenizer = lm_exp.CharTokenizer(_tiny_text())
    model = lm_exp.CharLanguageModel(
        tokenizer.vocab_size, d_model=16, n_layers=1, context_len=8, dropout=0.5
    ).to(DEVICE)
    x = torch.randint(0, tokenizer.vocab_size, (4, 6), device=DEVICE)

    model.train()
    train_first = model(x)
    train_second = model(x)
    assert not torch.allclose(train_first, train_second)

    model.eval()
    with torch.no_grad():
        eval_first = model(x)
        eval_second = model(x)
    assert torch.allclose(eval_first, eval_second)


# ----------------------------------------------------------------------
# Causal masking still holds with dropout enabled
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_heads", [1, 4])
def test_causal_weights_still_masked_in_train_mode_with_dropout(n_heads):
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16, n_heads=n_heads, dropout=0.5).to(DEVICE)
    block.train()
    x = torch.randn(2, 6, 16, device=DEVICE)

    norm_x = block.layer_norm1(x)
    _, weights = block.attention(norm_x, return_weights=True, causal=True)

    t = x.size(1)
    strictly_above = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    assert (weights[..., strictly_above] == 0).all()
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_future_token_change_leaves_earlier_outputs_unchanged_with_dropout():
    torch.manual_seed(0)
    block = block_exp.TransformerBlock(16, n_heads=4, dropout=0.5).to(DEVICE)
    block.eval()  # dropout disabled, so the comparison is exact

    x = torch.randn(1, 8, 16, device=DEVICE)
    with torch.no_grad():
        before = block(x, causal=True)

    k = 2
    changed = x.clone()
    changed[0, k] = 3.0
    with torch.no_grad():
        after = block(changed, causal=True)

    diff = (after[0] - before[0]).abs().max(dim=-1).values
    assert (diff[:k] < 1e-6).all()


# ----------------------------------------------------------------------
# The language model propagates dropout into its blocks
# ----------------------------------------------------------------------


def test_language_model_passes_dropout_into_blocks():
    tokenizer = lm_exp.CharTokenizer(_tiny_text())
    model = lm_exp.CharLanguageModel(
        tokenizer.vocab_size, d_model=16, n_layers=2, n_heads=4, dropout=0.3
    )

    assert model.dropout == 0.3
    assert model.embed_dropout.p == 0.3
    for block in model.blocks:
        assert block.dropout == 0.3
        assert block.attention.dropout == 0.3
        assert block.attention.attn_dropout.p == 0.3
        assert block.attention.out_dropout.p == 0.3
        assert block.ffn_hidden_dropout.p == 0.3
        assert block.ffn_output_dropout.p == 0.3


def test_single_head_block_uses_weight_dropout():
    block = block_exp.TransformerBlock(16, n_heads=1, dropout=0.3)
    assert block.attention.attn_dropout.p == 0.3
