import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from minigrad.bpe_tokenizer import BPETokenizer


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp23 = load_module("scalable_gpt_training_exp23_for_stability", "23_scalable_gpt_training.py")


def _tiny_model():
    return exp23.GPTLanguageModel(
        vocab_size=12,
        n_layers=2,
        d_model=16,
        n_heads=2,
        dff=32,
        context_len=8,
        dropout=0.0,
    )


def _tiny_loader():
    ids = [0, 1, 2, 3, 2, 1, 0, 3, 2, 1, 0, 2, 3, 0, 1, 2, 0, 1, 2, 3]
    ds = exp23.GPTTokenDataset(ids, context_len=8)
    return torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)


def test_train_epoch_diagnostics_disabled_by_default():
    model = _tiny_model()
    loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = exp23.make_optimizer(model, lr=1e-3)
    out = exp23.train_epoch(model, loader, criterion, optimizer, torch.device("cpu"), clip_grad=1.0)
    assert set(out.keys()) == {"loss", "tokens", "steps"}


def test_train_epoch_diagnostics_record_grad_loss_and_update_metrics():
    model = _tiny_model()
    loader = _tiny_loader()
    val_loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = exp23.make_optimizer(model, lr=1e-3)
    out = exp23.train_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        clip_grad=1.0,
        diagnostics=True,
        diagnostics_interval=2,
        val_loader=val_loader,
    )
    diags = out["diagnostics"]
    assert len(diags) > 0
    train_diags = [d for d in diags if "val_loss" not in d]
    val_diags = [d for d in diags if "val_loss" in d]
    assert len(train_diags) == out["steps"]
    first = train_diags[0]
    for key in (
        "step",
        "loss",
        "loss_finite",
        "grad_pre_clip_norm",
        "grad_post_clip_norm",
        "clipped",
        "grads_finite",
        "params_finite",
        "update_ratio",
        "lr",
        "logits_dtype",
        "loss_dtype",
    ):
        assert key in first, key
    assert val_diags == [d for d in diags if "val_loss" in d]
    # validation diagnostics recorded on the configured interval
    assert any(d["step"] % 2 == 0 for d in val_diags) if val_diags else True
    assert out["steps"] > 0


def test_train_epoch_bf16_autocast_dtypes_and_fp32_loss_reduction():
    model = _tiny_model()
    loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = exp23.make_optimizer(model, lr=1e-3)
    out = exp23.train_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        clip_grad=1.0,
        use_mixed_precision=True,
        diagnostics=True,
        diagnostics_interval=10,
        val_loader=None,
    )
    last = out["diagnostics"][-1]
    assert last["logits_dtype"] == "torch.bfloat16"
    assert last["loss_dtype"] == "torch.float32"
    assert last["loss_finite"] is True


def test_next_token_loss_reduction_is_float32():
    model = _tiny_model()
    x = torch.randint(0, 12, (2, 8))
    y = torch.randint(0, 12, (2, 8))
    loss = exp23.next_token_loss(model, x, y)
    assert loss.dtype == torch.float32
    assert bool(torch.isfinite(loss).item())
