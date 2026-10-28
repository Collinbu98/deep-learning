import pytest

from experiments.verification_utils.eval import make_passage
from experiments.verification_utils.windowed_nli import (
    WindowedNLIClaimVerifier,
    evaluate_windowed,
    truncation_summary,
)
from minigrad.claim_verification import Claim
from minigrad.evidence_windows import (
    SENTENCE,
    SENTENCE_ADJACENT,
    WHOLE_PASSAGE,
    build_evidence_windows,
    build_windows_for_passages,
    split_sentences,
)
from minigrad.nli_verification import (
    FakeNLIBackend,
    NLIModelUnavailable,
)

DEFAULT_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def logits_for(label, high=5.0, low=-2.0):
    return [
        high if DEFAULT_ID2LABEL[label_id] == label else low
        for label_id in sorted(DEFAULT_ID2LABEL)
    ]


def labelled_rows(pairs):
    rows = []
    for premise, _ in pairs:
        text = premise.lower()
        if "contradict" in text:
            rows.append(logits_for("contradiction"))
        elif "entail" in text:
            rows.append(logits_for("entailment"))
        else:
            rows.append(logits_for("neutral"))
    return rows


def test_sentence_splitting_handles_punctuation_empty_and_boundary_free_text():
    text = (
        "Dr. Smith paid 3.14 dollars. The U.S. team met e.g. the chair! "
        "Did they agree? Yes."
    )

    sentences = split_sentences(text)

    assert [sentence.text for sentence in sentences] == [
        "Dr. Smith paid 3.14 dollars.",
        "The U.S. team met e.g. the chair!",
        "Did they agree?",
        "Yes.",
    ]
    assert split_sentences("") == ()
    assert split_sentences("   \n\t ") == ()
    assert [s.text for s in split_sentences("No punctuation here")] == [
        "No punctuation here"
    ]


def test_sentence_splitting_handles_ellipsis_and_closing_quotes():
    sentences = split_sentences('He paused... "Really?" She left.')

    assert [sentence.text for sentence in sentences] == [
        "He paused...",
        '"Really?"',
        "She left.",
    ]


def test_window_traceability_and_determinism():
    passage = make_passage("jupiter#0", "One fact. Two facts. Three facts.")

    windows = build_evidence_windows(passage, SENTENCE)

    assert [window.window_id for window in windows] == [
        "jupiter#0::sentence:0",
        "jupiter#0::sentence:1",
        "jupiter#0::sentence:2",
    ]
    assert all(window.source_id == "jupiter#0" for window in windows)
    assert all(window.doc_id == "jupiter" for window in windows)
    assert all(window.chunk_index == 0 for window in windows)
    assert all(
        window.text == passage.text[window.char_start : window.char_end].strip()
        for window in windows
    )
    assert windows == build_evidence_windows(passage, SENTENCE)


def test_adjacent_windows_preserve_context_and_deduplicate():
    passage = make_passage("mars#0", "One. Two. Three.")

    windows = build_evidence_windows(
        passage, SENTENCE_ADJACENT, adjacent_context=1
    )

    assert [(window.start_sentence, window.end_sentence) for window in windows] == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert [window.text for window in windows] == [
        "One. Two.",
        "One. Two. Three.",
        "Two. Three.",
    ]
    assert len({window.window_id for window in windows}) == len(windows)


def test_whole_passage_and_empty_passage_windows():
    passage = make_passage("venus#0", "First sentence. Second sentence.")

    whole = build_evidence_windows(passage, WHOLE_PASSAGE)
    empty = build_evidence_windows(make_passage("empty#0", "  \n "), SENTENCE)

    assert len(whole) == 1
    assert whole[0].text == passage.text
    assert whole[0].start_sentence == 0
    assert whole[0].end_sentence == 1
    assert empty == ()


def test_build_windows_for_multiple_passages_preserves_mapping_order():
    passages = {
        "b#0": make_passage("b#0", "B one. B two."),
        "a#0": make_passage("a#0", "A one."),
    }

    windows = build_windows_for_passages(passages, SENTENCE)

    assert [window.source_id for window in windows] == ["b#0", "b#0", "a#0"]


def test_contradictory_and_neutral_windows_use_conservative_aggregation():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passages = {
        "a#0": make_passage(
            "a#0", "This sentence is neutral. This sentence contradicts."
        ),
        "b#0": make_passage("b#0", "This sentence is neutral."),
    }
    claim = Claim("A factual claim.", ("a#0", "b#0"))

    result = verifier.verify(claim, passages)

    assert result.status == "contradicted"
    assert result.label == "contradiction"
    assert [window.label for window in result.per_window] == [
        "neutral",
        "contradiction",
        "neutral",
    ]
    assert all(window.raw_scores for window in result.per_window)
    assert all(set(window.scores) == set(DEFAULT_ID2LABEL.values()) for window in result.per_window)


def test_entailment_and_contradiction_windows_are_mixed_not_supported():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passages = {
        "a#0": make_passage("a#0", "This sentence entails."),
        "b#0": make_passage("b#0", "This sentence contradicts."),
    }
    claim = Claim("A factual claim.", ("a#0", "b#0"))

    result = verifier.verify(claim, passages)

    assert result.status == "insufficient_evidence"
    assert result.label == "mixed"
    assert result.decision_confidence is None


def test_neutral_windows_never_combine_into_entailment():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passages = {
        "a#0": make_passage("a#0", "Neutral one. Neutral two."),
    }
    claim = Claim("A claim needing both parts.", ("a#0",))

    result = verifier.verify(claim, passages)

    assert result.status == "insufficient_evidence"
    assert result.label == "neutral"
    assert [window.label for window in result.per_window] == ["neutral", "neutral"]


def test_single_entailment_with_neutral_windows_is_supported_but_not_combined():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passage = make_passage(
        "a#0", "This sentence entails. This sentence is neutral."
    )
    claim = Claim("A factual claim.", ("a#0",))

    result = verifier.verify(claim, {"a#0": passage})

    assert result.status == "supported"
    assert result.label == "entailment"
    assert result.decision_confidence is not None
    assert [window.label for window in result.per_window] == [
        "entailment",
        "neutral",
    ]


def test_invalid_citation_is_rejected_before_window_prediction():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passage = make_passage("a#0", "This sentence entails.")
    claim = Claim("A factual claim.", ("missing#0",))

    result = verifier.verify(claim, {"a#0": passage})

    assert result.status == "invalid_citation"
    assert result.per_window == ()
    assert backend.calls == []


def test_empty_passage_returns_insufficient_without_model_call():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passage = make_passage("a#0", "   ")
    claim = Claim("A factual claim.", ("a#0",))

    result = verifier.verify(claim, {"a#0": passage})

    assert result.status == "insufficient_evidence"
    assert result.label is None
    assert "No non-empty evidence windows" in result.explanation
    assert backend.calls == []


def test_backend_unavailable_and_inference_errors_are_preserved():
    class UnavailableBackend(FakeNLIBackend):
        def predict_batch(self, pairs):
            raise NLIModelUnavailable("model is not cached")

    class BrokenBackend(FakeNLIBackend):
        def predict_batch(self, pairs):
            raise RuntimeError("simulated crash")

    passage = make_passage("a#0", "This sentence entails.")
    claim = Claim("A factual claim.", ("a#0",))

    unavailable = WindowedNLIClaimVerifier(
        backend=UnavailableBackend(), strategy=SENTENCE
    ).verify(claim, {"a#0": passage})
    broken = WindowedNLIClaimVerifier(
        backend=BrokenBackend(), strategy=SENTENCE
    ).verify(claim, {"a#0": passage})

    assert unavailable.status == "backend_unavailable"
    assert unavailable.per_window == ()
    assert "not cached" in unavailable.error
    assert broken.status == "verification_error"
    assert broken.per_window == ()
    assert "simulated crash" in broken.error


def test_windowed_results_are_deterministic_and_expose_raw_predictions():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passage = make_passage("a#0", "This sentence entails. Neutral context.")
    claim = Claim("A factual claim.", ("a#0",))

    first = verifier.verify(claim, {"a#0": passage})
    second = verifier.verify(claim, {"a#0": passage})

    assert first.to_dict() == second.to_dict()
    assert first.per_window[0].window_id == "a#0::sentence:0"
    assert first.per_window[1].window_id == "a#0::sentence:1"
    assert first.per_window[0].scores["entailment"] == pytest.approx(
        first.per_window[0].scores["entailment"]
    )


def test_evaluate_windowed_records_counts_runtime_and_truncation():
    class TokenCountingBackend(FakeNLIBackend):
        def input_token_counts(self, pairs):
            return [1000 for _ in pairs]

    cases = [
        {
            "id": "t1",
            "type": "direct",
            "claim": "A factual claim.",
            "citations": ["a#0"],
            "passages": {"a#0": "This sentence entails. Neutral context."},
            "expected": "supported",
            "rationale": "test",
        }
    ]
    backend = TokenCountingBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(
        backend=backend, strategy=SENTENCE, max_length=512
    )

    results = evaluate_windowed(cases, verifier)

    assert results[0]["correct"] is True
    assert results[0]["windows_scored"] == 2
    assert results[0]["truncation"]["checked"] is True
    assert results[0]["truncation"]["truncated_windows"] == 2
    assert results[0]["runtime_seconds"] >= 0.0


def test_truncation_summary_reports_missing_counter():
    backend = FakeNLIBackend(outputs=labelled_rows)
    verifier = WindowedNLIClaimVerifier(backend=backend, strategy=SENTENCE)
    passage = make_passage("a#0", "One sentence.")
    result = verifier.verify(
        Claim("A claim.", ("a#0",)),
        {"a#0": passage},
    )

    summary = truncation_summary(verifier, "A claim.", result.per_window)

    assert summary["checked"] is False
    assert "input_token_counts" in summary["reason"]
