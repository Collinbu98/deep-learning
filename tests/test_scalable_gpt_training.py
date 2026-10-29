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


exp23 = load_module("scalable_gpt_training_exp23", "23_scalable_gpt_training.py")

DEVICE = torch.device("cpu")


def _tiny_config(**overrides):
    config = {
        "vocab_size": 12,
        "n_layers": 2,
        "d_model": 16,
        "n_heads": 2,
        "dff": 32,
        "context_len": 8,
        "dropout": 0.0,
    }
    config.update(overrides)
    return config


def _make_model(**overrides):
    return exp23.GPTLanguageModel(**exp23.validate_gpt_config(_tiny_config(**overrides)))


def _tiny_token_ids():
    return [0, 1, 2, 3, 2, 1, 0, 3, 2, 1, 0, 2, 3, 0, 1, 2, 0, 1, 2, 3]


def test_gpt_model_output_shape_and_finite_logits():
    model = _make_model()
    x = torch.randint(0, 12, (2, 5))
    logits = model(x)
    assert tuple(logits.shape) == (2, 5, 12)
    assert torch.isfinite(logits).all()
    assert exp23.count_parameters(model) == exp23.gpt_parameter_count(12, 2, 16, 2, 32, 8)
    assert exp23.estimate_parameter_memory_bytes(exp23.count_parameters(model)) == (
        exp23.count_parameters(model) * torch.float32.itemsize
    )


def test_gpt_model_config_validation():
    with pytest.raises(ValueError):
        exp23.GPTLanguageModel(**exp23.validate_gpt_config(_tiny_config(n_heads=3)))
    with pytest.raises(ValueError):
        exp23.GPTLanguageModel(**exp23.validate_gpt_config(_tiny_config(context_len=1)))
    with pytest.raises(ValueError):
        exp23.GPTLanguageModel(**exp23.validate_gpt_config(_tiny_config(dropout=1.0)))
    with pytest.raises(ValueError):
        exp23.GPTLanguageModel(**exp23.validate_gpt_config(_tiny_config(dff=0)))
    with pytest.raises(ValueError):
        exp23.GPTLanguageModel(vocab_size=1, n_layers=1, d_model=4, n_heads=1, context_len=4)
    ok = exp23.GPTLanguageModel(**exp23.validate_gpt_config(_tiny_config()))
    assert ok.context_len == 8


def test_gpt_causal_masking_future_tokens_do_not_change_earlier_position_logits():
    model = _make_model(context_len=8)
    model.eval()
    x = torch.randint(0, 12, (2, 8))
    with torch.no_grad():
        base = model(x)
        x2 = x.clone()
        x2[0, -1] = (int(x2[0, -1]) + 1) % 12
        changed = model(x2)
        assert torch.equal(base[0, :7], changed[0, :7])
        assert not torch.equal(base[0, 7], changed[0, 7])


def test_gpt_dropout_train_vs_eval_determinism():
    model = _make_model(dropout=0.5)
    model.eval()
    x = torch.randint(0, 12, (2, 6))
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)
    torch.manual_seed(0)
    model.train()
    c = model(x)
    torch.manual_seed(0)
    model.train()
    d = model(x)
    assert torch.equal(c, d)


def test_gpt_dataset_input_target_shift():
    ids = _tiny_token_ids()
    ds = exp23.GPTTokenDataset(ids, context_len=4)
    assert len(ds) == len(ids) - 4
    x, y = ds[0]
    assert x.shape == y.shape == (4,)
    assert torch.equal(y[:-1], x[1:])
    assert y[3] == ids[4]


def test_gpt_training_updates_parameters_and_loss_decreases():
    ids = _tiny_token_ids()
    ds = exp23.GPTTokenDataset(ids, context_len=4)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    model = exp23.GPTLanguageModel(
        vocab_size=4, n_layers=1, d_model=8, n_heads=1, dff=16, context_len=4
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = exp23.make_optimizer(model, lr=0.05)
    first = exp23.train_epoch(model, loader, criterion, optimizer, DEVICE)
    second = exp23.train_epoch(model, loader, criterion, optimizer, DEVICE)
    assert torch.isfinite(torch.tensor(first["loss"])).item()
    assert torch.isfinite(torch.tensor(second["loss"])).item()
    assert second["loss"] < first["loss"]


def test_checkpoint_save_load_and_resume():
    model = _make_model()
    optimizer = exp23.make_optimizer(model, lr=1e-3)
    training_state = {"step": 7, "optimizer_state_dict": optimizer.state_dict()}
    path = exp23.save_checkpoint(
        "/tmp/opencode/gpt_ckpt_test.pt",
        model=model,
        config=model.config_dict(),
        tokenizer=BPETokenizer(vocab_size=8).fit("abcdefgh"),
        training_state=training_state,
        args={"smoke": True},
    )
    payload = exp23.load_checkpoint(path)
    restored_model = _make_model()
    restored_optimizer = exp23.make_optimizer(restored_model, lr=1e-3)
    restored = exp23.restore_training_state(payload, restored_model, restored_optimizer, DEVICE)
    assert restored["step"] == 7
    assert "state" in restored["optimizer_state"]
    with pytest.raises(ValueError):
        exp23.restore_training_state(payload, _make_model(context_len=16), None, DEVICE)


def test_generation_output_shape_max_length_context_window():
    tokenizer = BPETokenizer(vocab_size=8).fit("abcdefgh")
    model = _make_model(vocab_size=tokenizer.vocab_size, context_len=8)
    out = exp23.generate(model, tokenizer, prompt="abc", max_new_tokens=5, temperature=0.8, device=DEVICE)
    assert len(out) == 3 + 5
    short_model = _make_model(vocab_size=tokenizer.vocab_size, context_len=4)
    out_short = exp23.generate(short_model, tokenizer, prompt="abcdefgh", max_new_tokens=2, temperature=0.8, device=DEVICE)
    assert len(out_short) == 4 + 2
    with pytest.raises(ValueError):
        exp23.generate(model, tokenizer, prompt="abc", max_new_tokens=5, temperature=0.0, device=DEVICE)


def test_device_selection_and_generation_validation():
    assert exp23.get_device("cpu") == torch.device("cpu")
    assert exp23.get_device("auto") in (torch.device("cpu"), torch.device("cuda:0"))
    with pytest.raises(RuntimeError):
        exp23.get_device("cuda:99")
    tokenizer = BPETokenizer(vocab_size=8).fit("abcdefgh")
    model = _make_model(vocab_size=tokenizer.vocab_size)
    with pytest.raises(ValueError):
        exp23.generate(model, tokenizer, prompt="abc", max_new_tokens=5, temperature=0.0, device=DEVICE)
    with pytest.raises(ValueError):
        exp23.generate(model, tokenizer, prompt="abc", max_new_tokens=5, temperature=0.8, top_k=0, device=DEVICE)


def test_data_split_no_leakage_between_train_and_val_windows():
    text = "abcdefgh" * 40
    train_text, val_text = exp23.split_corpus(text, 0.2)
    assert len(train_text) + len(val_text) == len(text)
    tokenizer, train_ids, val_ids = exp23.fit_tokenizer_and_encode(train_text, val_text, 64)
    train_ds = exp23.GPTTokenDataset(train_ids, 8)
    val_ds = exp23.GPTTokenDataset(val_ids, 8)
    train_tokens = set()
    for i in range(len(train_ds)):
        x, y = train_ds[i]
        train_tokens.update(x.tolist())
        train_tokens.update(y.tolist())
    val_tokens = set()
    for i in range(len(val_ds)):
        x, y = val_ds[i]
        val_tokens.update(x.tolist())
        val_tokens.update(y.tolist())
    assert not (val_tokens & train_tokens)
