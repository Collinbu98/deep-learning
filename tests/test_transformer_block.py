import importlib.util
from pathlib import Path

import torch


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = load_module("transformer_block_experiment", "06_transformer_block.py")

DEVICE = torch.device("cpu")


def _model_and_input(batch_size=2, seq_len=8, embed_dim=32, seed=experiment.SEED):
    torch.manual_seed(seed)
    model = experiment.TransformerBlock(embed_dim).to(DEVICE)
    x = torch.randn(batch_size, seq_len, embed_dim, device=DEVICE)
    return model, x


def test_output_shape_equals_input_shape():
    model, x = _model_and_input()
    output = model(x)
    assert output.shape == x.shape


def test_module_has_trainable_parameters():
    model, _ = _model_and_input()
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) > 0
    # Sanity: sums of grads would be nonzero after backward on params
    assert sum(p.numel() for p in params) > 0


def test_backpropagation_produces_finite_gradients():
    model, x = _model_and_input()
    output = model(x, causal=True)
    loss = output.pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

    # Ensure attention and FFN params got gradients
    assert model.attention.query.weight.grad is not None
    assert torch.isfinite(model.attention.query.weight.grad).all()
    assert model.ffn[2].weight.grad is not None  # final Linear in FFN
    assert torch.isfinite(model.ffn[2].weight.grad).all()


def test_outputs_are_finite():
    model, x = _model_and_input()
    output = model(x)
    assert torch.isfinite(output).all()


def test_perturbing_future_token_leaves_earlier_positions_unchanged():
    model, x = _model_and_input(seq_len=8)
    model.eval()
    with torch.no_grad():
        out_base = model(x, causal=True)

    k = 2
    pert = x.clone()
    pert[0, k] = 3.0
    with torch.no_grad():
        out_pert = model(pert, causal=True)

    diff = (out_pert[0] - out_base[0]).abs().max(dim=-1).values
    assert (diff[:k] < 1e-6).all()


def test_block_works_for_batch_greater_than_one_and_short_sequence():
    model, x = _model_and_input(batch_size=4, seq_len=5, embed_dim=16)
    model.eval()
    with torch.no_grad():
        out = model(x, causal=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
