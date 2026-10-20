import math

import pytest
import torch
import torch.nn.functional as F

from minigrad.lm_eval import nats_per_character


class StubLM:
    """A causal 'model' that always returns the same logits."""

    def __init__(self, logits, context_len):
        # logits: [vocab_size]
        self.vocab_size = logits.numel()
        self.context_len = context_len
        self._logits = logits

    def eval(self):
        pass

    def __call__(self, x):
        return self._logits.expand(x.shape[0], x.shape[1], -1)


def test_known_answer_constant_logits():
    torch.manual_seed(0)
    logits = torch.randn(3)
    model = StubLM(logits, context_len=4)
    stream = [0, 1, 2, 0, 1]
    metrics = nats_per_character(model, stream, [1, 1, 1], device=torch.device("cpu"))
    # Single window scores targets at positions 1..4 (position 0 is context).
    expected_nll = 0.0
    for t in stream[1:]:
        expected_nll += -F.log_softmax(logits, dim=-1)[t].item()
    assert metrics["n_tokens_scored"] == len(stream) - 1
    assert metrics["sum_chars"] == len(stream) - 1
    assert metrics["sum_nll"] == pytest.approx(expected_nll, rel=1e-5)
    assert metrics["nats_per_char"] == pytest.approx(expected_nll / (len(stream) - 1), rel=1e-5)


def test_every_position_scored_exactly_once_across_windows():
    torch.manual_seed(1)
    logits = torch.randn(5)
    model = StubLM(logits, context_len=4)
    stream = list(range(5)) * 2  # length 10
    metrics = nats_per_character(model, stream, [1] * 5, device=torch.device("cpu"))
    assert metrics["n_tokens_scored"] == len(stream) - 1
    assert metrics["n_windows"] == 3  # starts 0, 4, 8 (last window shorter)
    # Cross-check against a direct per-position computation using the same
    # truncated-history windows the metric documents.
    expected_nll = 0.0
    T = 4
    for s in range(0, len(stream) - 1, T):
        x_len = min(T, len(stream) - s - 1)
        x = torch.tensor([stream[s : s + x_len]])
        y = torch.tensor([stream[s + 1 : s + 1 + x_len]])
        nll = F.cross_entropy(model(x).reshape(-1, 5), y.reshape(-1), reduction="sum")
        expected_nll += nll.item()
    assert metrics["sum_nll"] == pytest.approx(expected_nll, rel=1e-5)


def test_character_weights_normalize_summed_nll():
    logits = torch.zeros(4)
    model = StubLM(logits, context_len=8)
    # Token 3 stands for two characters; everything else for one.
    stream = [2, 2, 3, 1, 2]
    weights = [1, 1, 1, 2]  # indexed by token id (vocab_size = 4)
    metrics = nats_per_character(model, stream, weights, device=torch.device("cpu"))
    # Scored targets: positions 1..4 -> ids [2, 3, 1, 2], weights [1, 2, 1, 1].
    assert metrics["sum_chars"] == 5.0
    assert metrics["n_tokens_scored"] == 4
    assert metrics["nats_per_char"] == pytest.approx(metrics["sum_nll"] / 5.0)


def test_callable_weights_and_sequence_equivalent():
    logits = torch.randn(3)
    model = StubLM(logits, context_len=16)
    stream = [0, 1, 2, 1]
    seq = nats_per_character(model, stream, [1, 2, 3], device=torch.device("cpu"))
    fn = nats_per_character(model, stream, {0: 1, 1: 2, 2: 3}.__getitem__, device=torch.device("cpu"))
    assert seq["sum_nll"] == pytest.approx(fn["sum_nll"], rel=1e-6)
    assert seq["sum_chars"] == fn["sum_chars"] == 2 + 3 + 2


def test_empty_and_tiny_streams():
    model = StubLM(torch.zeros(3), context_len=4)
    for stream in [[], [0]]:
        metrics = nats_per_character(model, stream, [1, 1, 1], device=torch.device("cpu"))
        assert metrics["n_tokens_scored"] == 0
        assert math.isnan(metrics["nats_per_char"])


def test_invalid_chars_per_token():
    model = StubLM(torch.zeros(3), context_len=4)
    with pytest.raises(ValueError):
        nats_per_character(model, [0, 1], [1, 1], device=torch.device("cpu"))  # too short
    with pytest.raises(ValueError):
        nats_per_character(model, [0, 1], [1, 0, 1], device=torch.device("cpu"))  # zero weight


def test_accepts_list_token_ids_and_eval_mode():
    model = StubLM(torch.randn(3), context_len=4)
    metrics = nats_per_character(model, (0, 1, 2), [1, 1, 1], device=torch.device("cpu"))
    assert metrics["n_tokens_scored"] == 2
    assert metrics["sum_nll"] > 0
