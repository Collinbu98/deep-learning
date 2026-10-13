import importlib.util
from pathlib import Path

import torch


def load_module(name, filename):
    """Import an experiment file whose name is not a valid module name."""
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# These tests are offline: no data download, no training, just tensor shapes and
# gradient flow on small synthetic inputs.
experiment = load_module("self_attention_experiment", "05_self_attention.py")

DEVICE = torch.device("cpu")


def _model_and_input(batch_size=2):
    torch.manual_seed(experiment.SEED)
    model = experiment.SelfAttention(experiment.EMBED_DIM).to(DEVICE)
    x = torch.randn(batch_size, experiment.SEQ_LEN, experiment.EMBED_DIM, device=DEVICE)

    return model, x


def test_output_shape_matches_the_input_shape():
    model, x = _model_and_input()

    output = model(x)

    assert output.shape == (2, experiment.SEQ_LEN, experiment.EMBED_DIM)


def test_attention_weights_have_shape_batch_by_tokens_by_tokens():
    model, x = _model_and_input()

    _, weights = model(x, return_weights=True)

    assert weights.shape == (2, experiment.SEQ_LEN, experiment.SEQ_LEN)


def test_attention_weights_are_nonnegative_and_rows_sum_to_one():
    model, x = _model_and_input()

    _, weights = model(x, return_weights=True)

    assert (weights >= 0).all()
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_backpropagation_gives_finite_gradients_for_the_projections():
    model, x = _model_and_input()

    output = model(x)
    loss = output.pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

    assert model.query.weight.grad.abs().sum() > 0
    assert model.value.bias.grad.abs().sum() > 0


def test_module_works_with_a_batch_larger_than_one():
    model, x = _model_and_input(batch_size=5)

    output, weights = model(x, return_weights=True)

    assert output.shape == (5, experiment.SEQ_LEN, experiment.EMBED_DIM)
    assert weights.shape == (5, experiment.SEQ_LEN, experiment.SEQ_LEN)
    assert torch.allclose(
        weights.sum(dim=-1), torch.ones(5, experiment.SEQ_LEN), atol=1e-6
    )


# ----------------------------------------------------------------------
# Causal attention
# ----------------------------------------------------------------------


def test_non_causal_attention_remains_the_default():
    model, x = _model_and_input()

    default = model(x)
    explicit = model(x, causal=False)

    assert torch.allclose(default, explicit)


def test_causal_output_and_weight_shapes():
    model, x = _model_and_input()

    output, weights = model(x, return_weights=True, causal=True)

    assert output.shape == (2, experiment.SEQ_LEN, experiment.EMBED_DIM)
    assert weights.shape == (2, experiment.SEQ_LEN, experiment.SEQ_LEN)


def test_causal_weights_are_zero_above_the_diagonal():
    model, x = _model_and_input()

    _, weights = model(x, return_weights=True, causal=True)

    strictly_above = torch.triu(
        torch.ones(experiment.SEQ_LEN, experiment.SEQ_LEN, dtype=torch.bool),
        diagonal=1,
    )
    assert (weights[..., strictly_above] == 0).all()


def test_causal_attention_rows_sum_to_one():
    model, x = _model_and_input()

    _, weights = model(x, return_weights=True, causal=True)

    assert (weights >= 0).all()
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def _forced_attention_model(embed_dim=experiment.EMBED_DIM):
    """A deterministic model whose projections are the identity.

    With Wq = Wk = Wv = I, scores = X X^T / sqrt(D) and the output is a convex
    combination of the raw input rows. No random weights are involved, so tests
    about which positions can influence which are exact rather than fragile.
    """
    model = experiment.SelfAttention(embed_dim).to(DEVICE)
    with torch.no_grad():
        for projection in (model.query, model.key, model.value):
            projection.weight.copy_(torch.eye(embed_dim))
            projection.bias.zero_()

    return model


def _future_attention_input():
    """[1, T, D] where token 0 and token k share a large vector.

    With identity projections token 0 gets a large non-causal attention weight
    on token k, so (non-causally) token 0's output is guaranteed to depend on
    token k. Returns the input and the index k.
    """
    dim = experiment.EMBED_DIM
    seq = experiment.SEQ_LEN
    k = 2

    shared = torch.full((dim,), 2.0, device=DEVICE)
    x = torch.zeros(1, seq, dim, device=DEVICE)
    x[0, 0] = shared
    x[0, k] = shared
    x[0, 1] = torch.arange(dim, device=DEVICE, dtype=torch.float32) * 0.1
    x[0, 3] = -shared

    return x, k


def test_future_token_change_leaves_earlier_causal_outputs_unchanged():
    model = _forced_attention_model()
    x, k = _future_attention_input()

    before = model(x, causal=True)

    changed = x.clone()
    changed[0, k] = 3.0
    after = model(changed, causal=True)

    # Positions strictly before k must be untouched; k and later may move.
    assert torch.allclose(before[0, :k], after[0, :k], atol=1e-6)
    assert not torch.allclose(before[0, k:], after[0, k:])


def test_future_token_change_affects_earlier_outputs_when_non_causal():
    model = _forced_attention_model()
    x, k = _future_attention_input()

    # Precondition: token 0 really does attend to the future token k here.
    _, weights = model(x, return_weights=True)
    assert weights[0, 0, k] > 0.1

    before = model(x)

    changed = x.clone()
    changed[0, k] = 3.0
    after = model(changed)

    assert not torch.allclose(before[0, 0], after[0, 0])


def test_causal_mask_keeps_backpropagation_finite():
    model, x = _model_and_input()

    output = model(x, causal=True)
    output.pow(2).mean().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

