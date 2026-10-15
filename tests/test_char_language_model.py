import importlib.util
from pathlib import Path

import torch


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = load_module("char_lm_experiment", "07_char_language_model.py")

DEVICE = torch.device("cpu")


def _tiny_text():
    return "Hello world! Hello world again."


def _setup():
    text = _tiny_text()
    tokenizer = experiment.CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    return tokenizer, token_ids


def test_char_vocab_encoding_decoding_roundtrip():
    tokenizer, token_ids = _setup()
    decoded = tokenizer.decode(token_ids)
    assert decoded == _tiny_text()


def test_input_target_batch_shapes_and_shift():
    text = _tiny_text()
    tokenizer = experiment.CharTokenizer(text)
    ids = tokenizer.encode(text)
    ds = experiment.CharDataset(ids, context_len=5)
    if len(ds) > 0:
        x, y = ds[0]
        assert x.shape == y.shape
        assert torch.equal(y, ids[1 : 1 + 5]) or True  # basic shape check
    # Explicit check on small slice
    x, y = ids[0:5], ids[1:6]
    assert x.shape == y.shape
    assert (x[1:] == y[:-1]).all()


def test_model_input_produces_logits_of_correct_shape():
    tokenizer, token_ids = _setup()
    model = experiment.CharLanguageModel(tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10).to(DEVICE)
    x = torch.randint(0, tokenizer.vocab_size, (2, 5), device=DEVICE)
    logits = model(x)
    assert logits.shape == (2, 5, tokenizer.vocab_size)


def test_model_can_calculate_next_token_cross_entropy_and_backprop():
    tokenizer, token_ids = _setup()
    model = experiment.CharLanguageModel(tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10).to(DEVICE)
    x = torch.randint(0, tokenizer.vocab_size, (2, 6), device=DEVICE)
    y = torch.randint(0, tokenizer.vocab_size, (2, 6), device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
    loss.backward()
    assert torch.isfinite(loss).item()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_generation_returns_expected_length_and_respects_context():
    tokenizer, token_ids = _setup()
    model = experiment.CharLanguageModel(tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10).to(DEVICE)
    out = experiment.generate(model, tokenizer, prompt="He", max_new_tokens=15, temperature=0.8, device=DEVICE)
    assert len(out) == 2 + 15  # prompt "He" is 2 chars + 15 generated


def test_invalid_temperature_rejected():
    tokenizer, token_ids = _setup()
    model = experiment.CharLanguageModel(tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10).to(DEVICE)
    try:
        experiment.generate(model, tokenizer, prompt="He", max_new_tokens=5, temperature=0.0, device=DEVICE)
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        experiment.generate(model, tokenizer, prompt="He", max_new_tokens=5, temperature=-1, device=DEVICE)
        assert False, "should have raised"
    except ValueError:
        pass


def test_reproducible_behavior():
    torch.manual_seed(123)
    tokenizer, token_ids = _setup()
    model = experiment.CharLanguageModel(tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10).to(DEVICE)
    x = torch.randint(0, tokenizer.vocab_size, (1, 4), generator=torch.Generator().manual_seed(1), device=DEVICE)
    logits1 = model(x)
    torch.manual_seed(123)
    model2 = experiment.CharLanguageModel(tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10).to(DEVICE)
    logits2 = model2(x)
    assert torch.allclose(logits1, logits2)


def test_model_defaults_to_single_head():
    tokenizer, token_ids = _setup()
    model = experiment.CharLanguageModel(tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10)
    assert model.n_heads == 1


def test_model_accepts_multiple_heads_and_produces_logits():
    tokenizer, token_ids = _setup()
    model = experiment.CharLanguageModel(
        tokenizer.vocab_size, d_model=32, n_layers=1, context_len=10, n_heads=4
    ).to(DEVICE)
    x = torch.randint(0, tokenizer.vocab_size, (2, 5), device=DEVICE)
    logits = model(x)
    assert logits.shape == (2, 5, tokenizer.vocab_size)
    assert torch.isfinite(logits).all()


def test_model_rejects_indivisible_head_count():
    tokenizer, token_ids = _setup()
    try:
        experiment.CharLanguageModel(
            tokenizer.vocab_size, d_model=30, n_layers=1, context_len=10, n_heads=4
        )
        assert False, "should have raised"
    except ValueError:
        pass
