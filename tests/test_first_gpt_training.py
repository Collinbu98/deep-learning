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


e25 = load_module("first_gpt_training_exp25", "25_first_gpt_training.py")
EXP23 = e25.exp23
DEVICE = torch.device("cpu")


def _tiny_generation_model():
    tokenizer = BPETokenizer(vocab_size=32).fit("abcdefghij klmnopqrst")
    vocab_size = int(tokenizer.vocab_size)
    model = EXP23.GPTLanguageModel(
        vocab_size=vocab_size,
        n_layers=2,
        d_model=16,
        n_heads=2,
        dff=32,
        context_len=16,
        dropout=0.0,
    )
    return model, tokenizer


def _tiny_train_loader():
    ids = [0, 1, 2, 3, 2, 1, 0, 3, 2, 1, 0, 2, 3, 0, 1, 2, 0, 1, 2, 3]
    ds = EXP23.GPTTokenDataset(ids, context_len=8)
    return torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)


def test_first_gpt_recommended_config_is_valid_and_scale():
    cfg = e25.first_gpt_training_config()
    assert cfg["vocab_size"] == 256
    assert cfg["n_layers"] == 16
    assert cfg["d_model"] == 512
    assert cfg["n_heads"] == 8
    assert cfg["dff"] == 768
    assert cfg["context_len"] == 256
    assert cfg["batch_size"] == 4
    assert cfg["epochs"] == 3
    assert cfg["lr"] == 1e-3
    assert cfg["val_split"] == 0.1
    assert cfg["seed"] == 0
    assert cfg["use_mixed_precision"] is True
    assert EXP23.gpt_parameter_count(256, 16, 512, 8, 768, 256) == 29_840_640


def test_first_gpt_tiny_model_parameter_count_matches_closed_form():
    model, _ = _tiny_generation_model()
    assert EXP23.count_parameters(model) == EXP23.gpt_parameter_count(
        int(model.vocab_size), int(model.n_layers), int(model.d_model),
        int(model.n_heads), int(model.dff), int(model.context_len),
    )


def test_first_gpt_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        e25.first_gpt_training_config(epochs=0)
    with pytest.raises(ValueError):
        e25.first_gpt_training_config(batch_size=0)
    with pytest.raises(ValueError):
        e25.first_gpt_training_config(lr=0.0)
    with pytest.raises(ValueError):
        e25.first_gpt_training_config(val_split=1.0)
    with pytest.raises(ValueError):
        e25.first_gpt_training_config(clip_grad=0.0)


def test_first_gpt_checkpoint_paths():
    d = "/tmp/opencode/first_gpt_ckpt_test"
    assert e25.best_checkpoint_path(d) == Path(d) / "best.pt"
    assert e25.latest_checkpoint_path(d) == Path(d) / "latest.pt"
    assert e25.epoch_checkpoint_path(d, 1) == Path(d) / "epoch_1.pt"
    assert e25.epoch_checkpoint_path(d, 3) == Path(d) / "epoch_3.pt"
    assert e25.best_checkpoint_path() == e25.DEFAULT_CHECKPOINT_DIR / "best.pt"


def test_first_gpt_generation_settings_resolution():
    gs = e25.resolve_generation_settings()
    assert gs["max_new_tokens"] == 40
    assert gs["temperature"] == 0.8
    assert gs["top_k"] is None
    with pytest.raises(ValueError):
        e25.resolve_generation_settings({"temperature": 0.0})
    with pytest.raises(ValueError):
        e25.resolve_generation_settings({"max_new_tokens": 0})
    with pytest.raises(ValueError):
        e25.resolve_generation_settings({"top_k": 0})


def test_first_gpt_mixed_precision_training_step_is_finite():
    loader = _tiny_train_loader()
    model, _ = _tiny_generation_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = EXP23.make_optimizer(model, lr=1e-3)
    first = EXP23.train_epoch(
        model, loader, criterion, optimizer, DEVICE, clip_grad=1.0, use_mixed_precision=True
    )
    second = EXP23.train_epoch(
        model, loader, criterion, optimizer, DEVICE, clip_grad=1.0, use_mixed_precision=True
    )
    assert torch.isfinite(torch.tensor(first["loss"])).item()
    assert torch.isfinite(torch.tensor(second["loss"])).item()
    val = EXP23.validate_epoch(model, loader, criterion, DEVICE)
    assert torch.isfinite(torch.tensor(val["loss"])).item()
    assert val["perplexity"] > 0.0


def test_first_gpt_generate_samples_deterministic_per_stage_seed():
    model, tokenizer = _tiny_generation_model()
    out = e25.generate_samples(
        model,
        tokenizer,
        ["abc", "def"],
        generation_settings={"max_new_tokens": 3, "temperature": 0.8, "top_k": None},
        seed=0,
        device=DEVICE,
    )
    assert set(out.keys()) == {"abc", "def"}
    assert all(isinstance(v, str) for v in out.values())
    out2 = e25.generate_samples(
        model,
        tokenizer,
        ["abc", "def"],
        generation_settings={"max_new_tokens": 3, "temperature": 0.8, "top_k": None},
        seed=0,
        device=DEVICE,
    )
    assert out == out2
