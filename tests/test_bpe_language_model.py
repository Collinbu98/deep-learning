import importlib.util
from pathlib import Path

import torch

from minigrad.bpe_tokenizer import BPETokenizer


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = load_module("bpe_lm_experiment", "08_bpe_language_model.py")

DEVICE = torch.device("cpu")


def _train_text():
    return "abcabc abcabc abcabc " * 5


def _val_text():
    # Contains 'z' and '!', which never appear in the training text.
    return "abc z!z abc"


def _setup():
    tokenizer = BPETokenizer(vocab_size=16).fit(_train_text())
    return tokenizer


def _model(tokenizer, **kwargs):
    defaults = dict(d_model=16, n_layers=1, context_len=8, n_heads=1, dropout=0.0)
    defaults.update(kwargs)
    return experiment.BPETokenLanguageModel(tokenizer.vocab_size, **defaults).to(DEVICE)


def test_model_logits_shape():
    tokenizer = _setup()
    model = _model(tokenizer)
    x = torch.randint(0, tokenizer.vocab_size, (2, 5), device=DEVICE)
    logits = model(x)
    assert logits.shape == (2, 5, tokenizer.vocab_size)
    assert torch.isfinite(logits).all()


def test_model_backprop_with_unk_tokens():
    tokenizer = _setup()
    model = _model(tokenizer)
    x = torch.randint(0, tokenizer.vocab_size, (2, 6), device=DEVICE)
    y = torch.randint(0, tokenizer.vocab_size, (2, 6), device=DEVICE)
    y[0, 0] = tokenizer.unk_id  # ensure UNK appears among targets
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(x).view(-1, tokenizer.vocab_size), y.view(-1))
    loss.backward()
    assert torch.isfinite(loss).item()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_validation_text_with_unseen_chars_encodes_to_unk_not_gaps():
    tokenizer = _setup()
    val_text = _val_text()
    val_ids = tokenizer.encode(val_text)
    unk = tokenizer.unk_id
    assert     unk in val_ids
    # 'z' appears twice and '!' once in the validation text.
    assert val_ids.count(unk) == 3
    # No characters are silently skipped: coverage is exact.
    assert sum(tokenizer.token_char_length(i) for i in val_ids) == len(val_text)
    assert tokenizer.decode(val_ids) == val_text.replace("z", "�").replace("!", "�")


def test_val_dataset_and_eval_epoch_handle_unk():
    tokenizer = _setup()
    val_ids = torch.tensor(tokenizer.encode(_val_text()), dtype=torch.long)
    context_len = 4
    ds = experiment.TokenDataset(val_ids, context_len)
    assert len(ds) == len(val_ids) - context_len
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == y.shape == (context_len,)
        assert (x >= 0).all() and (x < tokenizer.vocab_size).all()
        assert (y >= 0).all() and (y < tokenizer.vocab_size).all()
        # Shifted next-token targets: y is x advanced by one position.
        assert torch.equal(y[:-1], x[1:])
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=2, shuffle=False)
    model = _model(tokenizer, context_len=context_len)
    criterion = torch.nn.CrossEntropyLoss()
    loss = experiment.eval_epoch(model, dl, criterion, DEVICE)
    assert torch.isfinite(torch.tensor(loss)).item()
    assert loss > 0.0


def test_bpe_char_weights_match_tokenizer():
    tokenizer = _setup()
    weights = experiment.bpe_char_weights(tokenizer)
    assert len(weights) == tokenizer.vocab_size
    assert weights == [tokenizer.token_char_length(i) for i in range(tokenizer.vocab_size)]
    assert weights[tokenizer.unk_id] == 1


def test_nats_per_character_on_val_stream():
    from minigrad.lm_eval import nats_per_character

    tokenizer = _setup()
    val_text = _val_text()
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    model = _model(tokenizer, context_len=8)
    metrics = nats_per_character(
        model, val_ids, experiment.bpe_char_weights(tokenizer), device=DEVICE
    )
    # All tokens except the first are scored; char weights cover the rest of
    # the raw validation text exactly.
    first_token_chars = tokenizer.token_char_length(int(val_ids[0]))
    assert metrics["n_tokens_scored"] == len(val_ids) - 1
    assert metrics["sum_chars"] == len(val_text) - first_token_chars
    assert metrics["nats_per_char"] > 0
    assert torch.isfinite(torch.tensor(metrics["nats_per_char"])).item()


def test_generation_with_unknown_prompt_chars_is_not_stripped():
    tokenizer = _setup()
    model = _model(tokenizer)
    prompt = "To ¶"  # '¶' was never seen in training
    prefix = tokenizer.decode(tokenizer.encode(prompt))
    out = experiment.generate(
        model, tokenizer, prompt, max_new_tokens=5, temperature=0.8, device=DEVICE
    )
    # The prompt (with ¶ as UNK/U+FFFD) survives as a prefix: not silently stripped.
    assert out.startswith(prefix)
    assert "¶" not in out  # original unseen char is represented as U+FFFD
    assert "�" in prefix


def test_generation_rejects_bad_temperature():
    tokenizer = _setup()
    model = _model(tokenizer)
    for bad in (0.0, -1.0):
        try:
            experiment.generate(model, tokenizer, "ab", max_new_tokens=2, temperature=bad, device=DEVICE)
            assert False, "should have raised"
        except ValueError:
            pass
