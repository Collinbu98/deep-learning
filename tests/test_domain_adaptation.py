import importlib.util
from pathlib import Path

import pytest
import torch

from minigrad.bpe_tokenizer import BPETokenizer


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = load_module("domain_adaptation_experiment", "09_domain_adaptation.py")
exp08 = experiment.exp08
DEVICE = torch.device("cpu")

# "Pretraining" corpus (stands in for Pride and Prejudice) and a distinct
# domain corpus with characters outside the pretraining alphabet ('!', 'z').
PRETRAIN_TEXT = "elizabeth and darcy walked the long road home. " * 20
DOMAIN_FULL_TEXT = ("the candle wax melts into a bright flame of fire! " * 12) + (
    "heat the wax and watch it fizz! " * 5
)


def _tiny_model(tokenizer):
    return exp08.BPETokenLanguageModel(
        tokenizer.vocab_size, d_model=16, n_layers=1, context_len=8, n_heads=1, dropout=0.0
    ).to(DEVICE)


def _write_checkpoint(tmp_path, vocab_size=48, name="exp08_ckpt.pt"):
    tokenizer = BPETokenizer(vocab_size=vocab_size).fit(PRETRAIN_TEXT)
    model = _tiny_model(tokenizer)
    path = tmp_path / name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "tokenizer": tokenizer.to_dict(),
            "best_epoch": 1,
            "best_val_loss": 0.0,
        },
        path,
    )
    return path, tokenizer


def _domain_ids(tokenizer, val_frac=0.2):
    train_text, val_text = experiment.split_sequence(DOMAIN_FULL_TEXT, val_frac)
    return (
        experiment.encode_text(tokenizer, train_text),
        experiment.encode_text(tokenizer, val_text),
        train_text,
        val_text,
    )


# --------------------------------------------------------------------------
# Checkpoint / config / tokenizer loading and compatibility
# --------------------------------------------------------------------------

def test_checkpoint_loading_roundtrip(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    model, tok, ckpt = experiment.load_checkpoint(path, DEVICE)
    assert tok.vocab_size == tokenizer.vocab_size
    assert tok.alphabet == tokenizer.alphabet
    assert tok.merges == tokenizer.merges
    assert tok.unk_id == tokenizer.unk_id
    assert model.config_dict() == ckpt["model_config"]
    assert model.vocab_size == tok.vocab_size
    # Weights really were restored: outputs match a freshly reconstructed net.
    x = torch.randint(0, tok.vocab_size, (1, 5))
    torch.manual_seed(7)
    reference = exp08.BPETokenLanguageModel(**ckpt["model_config"]).to(DEVICE)
    reference.load_state_dict(ckpt["model_state_dict"])
    with torch.no_grad():
        assert torch.allclose(model(x), reference(x), atol=1e-6)
    # Fixed tokenizer round-trips text through checkpoint serialization.
    assert BPETokenizer.from_dict(ckpt["tokenizer"]).encode(PRETRAIN_TEXT) == tok.encode(
        PRETRAIN_TEXT
    )


def test_checkpoint_loading_rejects_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        experiment.load_checkpoint(tmp_path / "missing.pt", DEVICE)
    good = {"model_state_dict": {}, "model_config": {}, "tokenizer": {}}
    # Missing tokenizer key.
    bad = {k: v for k, v in good.items() if k != "tokenizer"}
    torch.save(bad, tmp_path / "no_tok.pt")
    with pytest.raises(ValueError, match="tokenizer"):
        experiment.load_checkpoint(tmp_path / "no_tok.pt", DEVICE)
    # Incomplete model_config: must refuse rather than guess architecture.
    torch.save(
        {
            "model_state_dict": {},
            "model_config": {"vocab_size": 10, "d_model": 16},
            "tokenizer": {"type": "bpe_char", "alphabet": ["a"], "merges": []},
        },
        tmp_path / "no_cfg.pt",
    )
    with pytest.raises(ValueError, match="guess"):
        experiment.load_checkpoint(tmp_path / "no_cfg.pt", DEVICE)


def test_checkpoint_rejects_tokenizer_vocab_mismatch(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    ckpt = torch.load(path, weights_only=False)
    ckpt["model_config"] = dict(ckpt["model_config"], vocab_size=tokenizer.vocab_size + 1)
    bad_path = tmp_path / "mismatch.pt"
    torch.save(ckpt, bad_path)
    with pytest.raises(ValueError, match="incompatible"):
        experiment.load_checkpoint(bad_path, DEVICE)


def test_split_sequence(tmp_path):
    train_text, val_text = experiment.split_sequence(DOMAIN_FULL_TEXT, 0.2)
    assert train_text + val_text == DOMAIN_FULL_TEXT
    assert train_text and val_text
    # Deterministic.
    assert (train_text, val_text) == experiment.split_sequence(DOMAIN_FULL_TEXT, 0.2)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            experiment.split_sequence("abc", bad)
    with pytest.raises(ValueError):
        experiment.split_sequence("", 0.1)


# --------------------------------------------------------------------------
# Domain splitting: validation data never appears in training windows
# --------------------------------------------------------------------------

def test_training_windows_exclude_validation_data(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    train_ids, val_ids, train_text, val_text = _domain_ids(tokenizer, val_frac=0.3)
    assert train_text + val_text == DOMAIN_FULL_TEXT  # disjoint, complete split
    # Text with unseen characters masked to U+FFFD by the tokenizer round-trip.
    masked_train = tokenizer.decode(tokenizer.encode(train_text))
    masked_val = tokenizer.decode(tokenizer.encode(val_text))
    ctx = 4
    ds = exp08.TokenDataset(train_ids, ctx)
    assert len(ds) == len(train_ids) - ctx
    for idx in range(len(ds)):
        x, y = ds[idx]
        # Windows are exact slices of the domain-TRAIN stream only: they never
        # reach past the end of train_ids into validation tokens.
        assert torch.equal(x, train_ids[idx : idx + ctx])
        assert torch.equal(y, train_ids[idx + 1 : idx + 1 + ctx])
        # Text-level check: windows decode to segments of the training text.
        assert tokenizer.decode(x.tolist()) in masked_train
        assert tokenizer.decode(y.tolist()) in masked_train
    # The validation stream is entirely separate and never used for windows.
    assert val_ids.numel() > 0
    assert tokenizer.decode(val_ids.tolist()) == masked_val


def test_encode_text_covers_every_character(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    ids = experiment.encode_text(tokenizer, DOMAIN_FULL_TEXT)
    assert sum(tokenizer.token_char_length(int(t)) for t in ids.tolist()) == len(DOMAIN_FULL_TEXT)
    # Characters outside the pretraining alphabet became UNK (nothing dropped).
    assert (ids == tokenizer.unk_id).sum() > 0


# --------------------------------------------------------------------------
# Training / validation behavior
# --------------------------------------------------------------------------

def _adapt(tmp_path, tokenizer, model, **overrides):
    train_ids, val_ids, _, _ = _domain_ids(tokenizer)
    kwargs = dict(
        epochs=1,
        batch_size=4,
        lr=1e-2,
        device=DEVICE,
        checkpoint_path=tmp_path / "adapted.pt",
        context_len=8,
    )
    kwargs.update(overrides)
    return experiment.train_adaptation(
        model, tokenizer, train_ids, val_ids, **kwargs
    ), train_ids, val_ids


def test_training_updates_parameters(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    model, _, _ = experiment.load_checkpoint(path, DEVICE)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result, _, _ = _adapt(tmp_path, tokenizer, model)
    changed = any(
        not torch.equal(before[k], v) for k, v in model.state_dict().items()
    )
    assert changed, "training must update parameters"
    assert len(result["history"]) == 1
    h = result["history"][0]
    for key in ("train_loss", "domain_val_per_token", "domain_val_nats_per_char", "seconds"):
        assert key in h
    assert torch.isfinite(torch.tensor(h["train_loss"])).item()
    assert Path(result["checkpoint_path"]).exists()


def test_validation_evaluation_does_not_update_parameters(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    model, _, _ = experiment.load_checkpoint(path, DEVICE)
    _, val_ids, _, _ = _domain_ids(tokenizer)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metrics1 = experiment.evaluate(model, val_ids, tokenizer, DEVICE, batch_size=4)
    metrics2 = experiment.evaluate(model, val_ids, tokenizer, DEVICE, batch_size=4)
    for k in before:
        assert torch.equal(before[k], model.state_dict()[k]), f"{k} changed during evaluation"
    assert metrics1["per_token_loss"] == pytest.approx(metrics2["per_token_loss"], rel=1e-6)
    assert metrics1["nats_per_char"] == pytest.approx(metrics2["nats_per_char"], rel=1e-6)
    assert metrics1["n_tokens"] == int(val_ids.numel())
    assert metrics1["n_tokens_scored"] == int(val_ids.numel()) - 1


def test_best_checkpoint_selection_and_reload(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    model, _, _ = experiment.load_checkpoint(path, DEVICE)
    result, _, val_ids = _adapt(tmp_path, tokenizer, model, epochs=3)
    history = result["history"]
    losses = [e["domain_val_per_token"] for e in history]
    expected_best = losses.index(min(losses)) + 1
    assert result["best_epoch"] == expected_best
    assert result["best_domain_val_loss"] == min(losses)
    # Reload from disk: identical behavior to the in-memory best model, and
    # the saved tokenizer is the frozen original one.
    reloaded, tok2, ckpt = experiment.load_checkpoint(result["checkpoint_path"], DEVICE)
    assert ckpt["stage"] == "domain_adaptation"
    assert ckpt["best_epoch"] == expected_best
    assert tok2.to_dict() == tokenizer.to_dict()
    m1 = experiment.evaluate(model, val_ids, tokenizer, DEVICE, batch_size=4)
    m2 = experiment.evaluate(reloaded, val_ids, tok2, DEVICE, batch_size=4)
    assert m1["per_token_loss"] == pytest.approx(m2["per_token_loss"], rel=1e-5)
    assert m1["nats_per_char"] == pytest.approx(m2["nats_per_char"], rel=1e-5)


def test_tokenizer_stays_frozen_during_adaptation(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    model, _, _ = experiment.load_checkpoint(path, DEVICE)
    frozen = tokenizer.to_dict()
    _adapt(tmp_path, tokenizer, model)
    assert tokenizer.to_dict() == frozen
    assert [list(m) for m in tokenizer.merges] == frozen["merges"]


# --------------------------------------------------------------------------
# Generation and reproducibility
# --------------------------------------------------------------------------

def test_generation_from_adapted_checkpoint(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    model, _, _ = experiment.load_checkpoint(path, DEVICE)
    result, _, _ = _adapt(tmp_path, tokenizer, model)
    adapted, tok2, _ = experiment.load_checkpoint(result["checkpoint_path"], DEVICE)
    prompt = "the "  # lowercase: all characters are in the fixture alphabet
    out1 = experiment.seeded_generate(
        adapted, tok2, prompt, seed=42, max_new_tokens=10, temperature=0.8, device=DEVICE
    )
    out2 = experiment.seeded_generate(
        adapted, tok2, prompt, seed=42, max_new_tokens=10, temperature=0.8, device=DEVICE
    )
    assert isinstance(out1, str) and out1.startswith(prompt)
    assert len(out1) > len(prompt)
    assert out1 == out2  # fixed seed -> reproducible sampling
    # Original checkpoint generates too, with the same interface.
    original, tok_orig, _ = experiment.load_checkpoint(path, DEVICE)
    out_orig = experiment.seeded_generate(
        original, tok_orig, prompt, seed=42, max_new_tokens=10, temperature=0.8, device=DEVICE
    )
    assert isinstance(out_orig, str) and out_orig.startswith(prompt)


def test_adaptation_reproducible_with_same_seed(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    losses = []
    for run in range(2):
        torch.manual_seed(0)
        model, _, _ = experiment.load_checkpoint(path, DEVICE)
        result, _, _ = _adapt(tmp_path, tokenizer, model,
                              checkpoint_path=tmp_path / f"adapted{run}.pt")
        losses.append(result["history"][0]["train_loss"])
    assert losses[0] == pytest.approx(losses[1], rel=1e-6)


def test_invalid_context_len_rejected(tmp_path):
    path, tokenizer = _write_checkpoint(tmp_path)
    model, _, _ = experiment.load_checkpoint(path, DEVICE)
    _, val_ids, _, _ = _domain_ids(tokenizer)
    with pytest.raises(ValueError):
        experiment.train_adaptation(
            model, tokenizer, experiment.encode_text(tokenizer, DOMAIN_FULL_TEXT), val_ids,
            epochs=1, batch_size=4, lr=1e-3, device=DEVICE,
            checkpoint_path=tmp_path / "x.pt", context_len=999,
        )
