import argparse
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from minigrad.bpe_tokenizer import BPETokenizer
from minigrad.lora import (
    LoRALinear,
    apply_lora,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
)


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = load_module("lora_adaptation_experiment", "10_lora_adaptation.py")
exp08 = experiment.exp08
exp09 = experiment.exp09

DEVICE = torch.device("cpu")

PRETRAIN_TEXT = "elizabeth and darcy walked the long road home. " * 20
DOMAIN_TEXT = "the candle wax melts into a bright flame of fire! " * 15


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _tokenizer():
    return BPETokenizer(vocab_size=48).fit(PRETRAIN_TEXT)


def _model(tokenizer, **kwargs):
    defaults = dict(d_model=16, n_layers=2, context_len=8, n_heads=1, dropout=0.0)
    defaults.update(kwargs)
    return exp08.BPETokenLanguageModel(tokenizer.vocab_size, **defaults).to(DEVICE)


def _wrapped(base, rank=2, alpha=4):
    return LoRALinear(base, rank=rank, alpha=alpha).to(DEVICE)


def _write_base_checkpoint(tmp_path, tokenizer):
    model = _model(tokenizer)
    path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "tokenizer": tokenizer.to_dict(),
        },
        path,
    )
    return path


# --------------------------------------------------------------------------
# The LoRA layer itself
# --------------------------------------------------------------------------

def test_lora_output_matches_base_at_initialization():
    torch.manual_seed(0)
    base = nn.Linear(7, 5)
    x = torch.randn(3, 4, 7)
    wrapped = _wrapped(base, rank=2, alpha=4)
    with torch.no_grad():
        expected = base(x)
        actual = wrapped(x)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_lora_initialization_and_shapes():
    base = nn.Linear(7, 5)
    wrapped = _wrapped(base, rank=3, alpha=6)
    assert wrapped.lora_A.shape == (3, 7)   # [rank, in_features]
    assert wrapped.lora_B.shape == (5, 3)   # [out_features, rank]
    # B starts at zero (no update at init), A starts nonzero.
    assert torch.count_nonzero(wrapped.lora_B) == 0
    assert torch.count_nonzero(wrapped.lora_A) > 0
    assert wrapped.scaling == pytest.approx(6 / 3)
    # Default alpha is the rank itself (scale 1).
    assert LoRALinear(nn.Linear(4, 4), rank=4).scaling == pytest.approx(1.0)


def test_lora_output_shape_is_preserved():
    base = nn.Linear(6, 9)
    wrapped = _wrapped(base, rank=2, alpha=2)
    x = torch.randn(2, 5, 6)
    out = wrapped(x)
    assert out.shape == (2, 5, 9)


def test_lora_applies_the_expected_scaling():
    base = nn.Linear(4, 4)
    wrapped = _wrapped(base, rank=2, alpha=4)
    with torch.no_grad():
        wrapped.lora_B.copy_(torch.ones(4, 2))
        wrapped.lora_A.copy_(torch.ones(2, 4))
    x = torch.ones(1, 1, 4)
    with torch.no_grad():
        base_out = base(x)
        lora_out = wrapped(x)
    # lora_A(x): sum of 4 ones -> 4 per position.
    # lora_B(.): sum of 2 times 4 -> 8 per output feature.
    # scaled by alpha/rank = 4/2 = 2 -> 16.
    expected_delta = 8 * 2
    assert torch.allclose(lora_out - base_out, torch.full_like(base_out, expected_delta), atol=1e-5)


def test_lora_follows_the_base_layer_device_and_dtype():
    base = nn.Linear(4, 4, dtype=torch.float64)
    wrapped = LoRALinear(base, rank=2, alpha=4)
    assert wrapped.lora_A.dtype == torch.float64
    assert wrapped.lora_B.dtype == torch.float64
    x = torch.randn(2, 4, dtype=torch.float64)
    out = wrapped(x)
    assert out.dtype == torch.float64
    assert out.shape == (2, 4)
    # Moving the whole module keeps base and adapter in sync.
    wrapped = wrapped.to(torch.float32)
    assert wrapped.lora_A.dtype == torch.float32
    out = wrapped(torch.randn(2, 4))
    assert out.dtype == torch.float32


def test_lora_raises_when_adapter_and_base_dtypes_disagree():
    base = nn.Linear(4, 4, dtype=torch.float64)
    wrapped = LoRALinear(base, rank=2, alpha=4)
    # Simulate a partial dtype move that left the adapter behind.
    wrapped.base.to(torch.float32)
    with pytest.raises(RuntimeError, match="dtype"):
        wrapped(torch.randn(2, 4, dtype=torch.float64))


def test_invalid_rank_and_config_raise():
    base = nn.Linear(4, 4)
    for bad_rank in (0, -1, -100):
        with pytest.raises(ValueError):
            LoRALinear(base, rank=bad_rank)
    for bad_rank in (2.0, "2", None, True):
        with pytest.raises(TypeError):
            LoRALinear(base, rank=bad_rank)
    for bad_alpha in (0, -1, -0.5):
        with pytest.raises(ValueError):
            LoRALinear(base, rank=2, alpha=bad_alpha)
    with pytest.raises(TypeError):
        LoRALinear(base, rank=2, alpha="8")
    for bad_dropout in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            LoRALinear(base, rank=2, alpha=4, dropout=bad_dropout)
    with pytest.raises(TypeError):
        LoRALinear(nn.LayerNorm(4), rank=2, alpha=4)


def test_base_weights_and_bias_are_frozen():
    base = nn.Linear(5, 3)
    wrapped = _wrapped(base, rank=2, alpha=4)
    assert wrapped.base.weight.requires_grad is False
    assert wrapped.base.bias is not None
    assert wrapped.base.bias.requires_grad is False
    # The base parameters are not reachable through the wrapper's gradients.
    x = torch.randn(2, 5, requires_grad=True)
    out = wrapped(x)
    out.sum().backward()
    assert wrapped.base.weight.grad is None
    assert wrapped.base.bias.grad is None
    assert x.grad is not None


def test_lora_parameters_receive_gradients_and_update():
    torch.manual_seed(0)
    base = nn.Linear(5, 3)
    wrapped = _wrapped(base, rank=2, alpha=4)
    optimizer = torch.optim.AdamW(list(lora_parameters(wrapped)), lr=1e-2)
    before_A = wrapped.lora_A.detach().clone()
    before_B = wrapped.lora_B.detach().clone()
    x = torch.randn(4, 5)
    loss = wrapped(x).pow(2).mean()
    loss.backward()
    assert wrapped.lora_A.grad is not None and torch.isfinite(wrapped.lora_A.grad).all()
    assert wrapped.lora_B.grad is not None and torch.isfinite(wrapped.lora_B.grad).all()
    optimizer.step()
    assert not torch.equal(before_A, wrapped.lora_A)
    assert not torch.equal(before_B, wrapped.lora_B)


# --------------------------------------------------------------------------
# Integration with the Experiment 08 model
# --------------------------------------------------------------------------

def test_apply_lora_targets_attention_projections():
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    replaced = apply_lora(model, ["attention.query", "attention.key", "attention.value"],
                           rank=2, alpha=4)
    assert len(replaced) == 6  # 2 blocks x 3 projections
    assert "blocks.0.attention.query" in replaced
    assert "blocks.1.attention.value" in replaced
    assert isinstance(model.blocks[0].attention.query, LoRALinear)
    assert isinstance(model.blocks[0].attention.key, LoRALinear)
    assert isinstance(model.blocks[0].attention.value, LoRALinear)
    # Non-targeted modules stay untouched.
    assert isinstance(model.blocks[0].ffn[0], nn.Linear)
    assert not isinstance(model.blocks[0].ffn[0], LoRALinear)


def test_apply_lora_supports_multi_head_output_projection():
    tokenizer = _tokenizer()
    model = _model(tokenizer, n_heads=2)
    replaced = apply_lora(model, ["attention.out_proj"], rank=2, alpha=4)
    assert replaced == ["blocks.0.attention.out_proj", "blocks.1.attention.out_proj"]
    assert isinstance(model.blocks[0].attention.out_proj, LoRALinear)


def test_apply_lora_raises_when_nothing_matches():
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    with pytest.raises(ValueError):
        apply_lora(model, ["does.not.exist"], rank=2, alpha=4)
    with pytest.raises(ValueError):
        apply_lora(model, [], rank=2, alpha=4)


def test_only_lora_parameters_are_trainable_in_the_integrated_model():
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    apply_lora(model, ["attention.query", "attention.key", "attention.value"], rank=2, alpha=4)
    mark_only_lora_trainable(model)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = {
        f"blocks.{b}.attention.{proj}.{ab}"
        for b in range(2)
        for proj in ("query", "key", "value")
        for ab in ("lora_A", "lora_B")
    }
    assert trainable == expected
    # Nothing else (embeddings, layer norms, LM head, base weights) is trained.
    for name, p in model.named_parameters():
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        assert p.requires_grad == is_lora, name
    total = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 3 projections x 2 blocks x (A: 2x16 + B: 16x2)
    assert trainable_params == 3 * 2 * (2 * 16 + 16 * 2)
    assert trainable_params < 0.05 * total


def test_wrapped_model_reproduces_base_model_at_initialization():
    tokenizer = _tokenizer()
    torch.manual_seed(0)
    model = _model(tokenizer)
    torch.manual_seed(0)
    reference = _model(tokenizer)  # same seed -> identical weights
    x = torch.randint(0, tokenizer.vocab_size, (2, 6))
    with torch.no_grad():
        expected = reference(x)
    apply_lora(model, ["attention.query", "attention.key", "attention.value"], rank=2, alpha=4)
    with torch.no_grad():
        actual = model(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_base_model_behavior_is_unchanged_by_lora():
    """The Experiment 08 model itself is not modified by this experiment."""
    tokenizer = _tokenizer()
    torch.manual_seed(0)
    model = _model(tokenizer)
    torch.manual_seed(0)
    reference = _model(tokenizer)
    x = torch.randint(0, tokenizer.vocab_size, (2, 6))
    with torch.no_grad():
        assert torch.allclose(model(x), reference(x), atol=1e-6)
    # Wrapping a copy does not change the original module's outputs.
    apply_lora(model, ["attention.query"], rank=2, alpha=4)
    with torch.no_grad():
        assert torch.allclose(model(x), reference(x), atol=1e-6)


def test_frozen_base_parameters_remain_unchanged_during_training():
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    apply_lora(model, ["attention.query", "attention.key", "attention.value"], rank=2, alpha=4)
    mark_only_lora_trainable(model)

    frozen_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not (name.endswith("lora_A") or name.endswith("lora_B"))
    }
    lora_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    model.train()
    x = torch.randint(0, tokenizer.vocab_size, (4, 8))
    y = torch.randint(0, tokenizer.vocab_size, (4, 8))
    loss = criterion(model(x).view(-1, tokenizer.vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()

    for name, p in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            assert not torch.equal(lora_before[name], p), f"{name} did not update"
        else:
            assert torch.equal(frozen_before[name], p), f"{name} changed"


def test_lora_state_dict_contains_only_the_adapter():
    tokenizer = _tokenizer()
    model = _model(tokenizer)
    apply_lora(model, ["attention.query"], rank=2, alpha=4)
    state = lora_state_dict(model)
    assert len(state) == 4  # 2 blocks x (A, B)
    assert all(k.endswith(("lora_A", "lora_B")) for k in state)
    assert all(isinstance(v, torch.Tensor) for v in state.values())


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def _train_and_save(tmp_path, tokenizer, method="lora"):
    path = _write_base_checkpoint(tmp_path, tokenizer)
    model, tok, _ = exp09.load_checkpoint(path, DEVICE)
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if method == "lora":
        experiment.build_lora_model(model, ["attention.query"], rank=2, alpha=4)
    train_ids = exp09.encode_text(tok, DOMAIN_TEXT)
    val_ids = exp09.encode_text(tok, DOMAIN_TEXT[:20])
    result = experiment.train_adaptation(
        model, tok, train_ids, val_ids,
        epochs=1, batch_size=4, lr=1e-2, device=DEVICE,
    )
    best_state = result["best_state"]
    if method == "lora":
        lora_state = {k: v for k, v in best_state.items()
                      if k.endswith("lora_A") or k.endswith("lora_B")}
        saved_base = base_state
        lora_config = {"rank": 2, "alpha": 4, "target_layers": ["attention.query"], "dropout": 0.0}
    else:
        lora_state = {}
        saved_base = best_state
        lora_config = None
    out = tmp_path / f"adapted_{method}.pt"
    experiment.save_adapted_checkpoint(
        out,
        base_state=saved_base,
        lora_state=lora_state,
        model_config=model.config_dict(),
        tokenizer=tok,
        lora_config=lora_config,
        method=method,
        history=result["history"],
        best_epoch=result["best_epoch"],
        best_val_loss=result["best_domain_val_loss"],
        base_checkpoint=str(path),
        args=argparse.Namespace(seed=0, epochs=1),
    )
    # Put the in-memory model back into its best state so it can be compared
    # with the reloaded checkpoint.
    model.load_state_dict(best_state)
    return out, model


def test_saving_and_reloading_an_adapted_model_preserves_its_output(tmp_path):
    tokenizer = _tokenizer()
    ckpt_path, model = _train_and_save(tmp_path, tokenizer, method="lora")
    reloaded, reloaded_tokenizer, ckpt = experiment.load_adapted_checkpoint(ckpt_path, DEVICE)
    assert ckpt["method"] == "lora"
    assert ckpt["lora_config"]["rank"] == 2
    assert ckpt["lora_config"]["target_layers"] == ["attention.query"]
    assert reloaded_tokenizer.to_dict() == tokenizer.to_dict()

    x = torch.randint(0, tokenizer.vocab_size, (2, 6))
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)


def test_reloaded_lora_model_keeps_the_base_frozen(tmp_path):
    """A reloaded adapter must report the same trainable budget as training."""
    tokenizer = _tokenizer()
    ckpt_path, _ = _train_and_save(tmp_path, tokenizer, method="lora")
    reloaded, _, _ = experiment.load_adapted_checkpoint(ckpt_path, DEVICE)
    total, trainable, pct = experiment.count_parameters(reloaded)
    # 2 wrapped layers (one per block) x (A: 2x16 + B: 16x2)
    assert trainable == 2 * (2 * 16 + 16 * 2)
    frozen = sum(
        p.numel()
        for name, p in reloaded.named_parameters()
        if not (name.endswith("lora_A") or name.endswith("lora_B"))
    )
    assert total == trainable + frozen
    assert frozen > 20 * trainable  # the vast majority stays frozen
    assert 0 < pct < 5
    for name, p in reloaded.named_parameters():
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        assert p.requires_grad == is_lora, name


def test_full_finetune_checkpoint_roundtrip(tmp_path):
    tokenizer = _tokenizer()
    ckpt_path, model = _train_and_save(tmp_path, tokenizer, method="full")
    reloaded, _, ckpt = experiment.load_adapted_checkpoint(ckpt_path, DEVICE)
    assert ckpt["method"] == "full"
    assert ckpt["lora_state_dict"] == {}
    total, trainable, _ = experiment.count_parameters(reloaded)
    assert trainable == total  # nothing is frozen in the full baseline
    x = torch.randint(0, tokenizer.vocab_size, (2, 6))
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)


def test_checkpoint_metadata_records_base_and_lora_config(tmp_path):
    tokenizer = _tokenizer()
    ckpt_path, _ = _train_and_save(tmp_path, tokenizer, method="lora")
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert ckpt["stage"] == "lora_adaptation"
    assert ckpt["base_checkpoint"].endswith("base.pt")
    assert ckpt["model_config"]["vocab_size"] == tokenizer.vocab_size
    assert ckpt["tokenizer"]["type"] == "bpe_char"
    assert ckpt["lora_config"]["rank"] == 2
    assert ckpt["lora_config"]["alpha"] == 4
    # The adapter is small compared to the frozen base weights.
    lora_params = sum(v.numel() for v in ckpt["lora_state_dict"].values())
    base_params = sum(v.numel() for v in ckpt["base_state_dict"].values())
    assert lora_params < base_params


def test_load_adapted_checkpoint_rejects_broken_files(tmp_path):
    tokenizer = _tokenizer()
    with pytest.raises(FileNotFoundError):
        experiment.load_adapted_checkpoint(tmp_path / "missing.pt", DEVICE)
    torch.save({"model_config": {}}, tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="missing required key"):
        experiment.load_adapted_checkpoint(tmp_path / "broken.pt", DEVICE)
