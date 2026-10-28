import pytest

from experiments.verification_utils.confidence_policy import (
    CONFIDENCE_AWARE_POLICY,
    ConfidenceAwareNLIClaimVerifier,
    apply_confidence_policy,
    select_threshold,
)
from experiments.verification_utils.eval import make_passage
from experiments.verification_utils.windowed_nli import (
    WindowedNLIClaimVerifier,
    evaluate_windowed,
)
from minigrad.claim_verification import Claim
from minigrad.evidence_windows import SENTENCE, WHOLE_PASSAGE
from minigrad.nli_verification import FakeNLIBackend, NLIModelUnavailable

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def logits_for(label, high=5.0, low=-2.0):
    return [
        high if ID2LABEL[label_id] == label else low
        for label_id in sorted(ID2LABEL)
    ]


def rows_for_confidences(pairs):
    rows = []
    for premise, _ in pairs:
        text = premise.lower()
        if "low contradiction" in text:
            rows.append([1.0, 0.0, 0.0])
        elif "contradict" in text:
            rows.append(logits_for("contradiction"))
        elif "entail" in text:
            rows.append(logits_for("entailment"))
        else:
            rows.append(logits_for("neutral"))
    return rows


def verify_with_policy(text, *, threshold=0.70, strategy=SENTENCE):
    backend = FakeNLIBackend(outputs=rows_for_confidences)
    verifier = ConfidenceAwareNLIClaimVerifier(
        backend=backend,
        strategy=strategy,
        contradiction_threshold=threshold,
    )
    passage = make_passage("a#0", text)
    return verifier.verify(Claim("A factual claim.", ("a#0",)), {"a#0": passage})


def test_low_confidence_contradiction_is_abstained_without_losing_raw_scores():
    baseline = WindowedNLIClaimVerifier(
        backend=FakeNLIBackend(outputs=rows_for_confidences),
        strategy=SENTENCE,
    ).verify(
        Claim("A factual claim.", ("a#0",)),
        {"a#0": make_passage("a#0", "Low contradiction.")},
    )
    result = verify_with_policy("Low contradiction.")

    assert baseline.status == "contradicted"
    assert result.status == "insufficient_evidence"
    assert result.label == "contradiction"
    assert result.policy_outcome == "low_confidence_contradiction"
    assert result.decision_policy == CONFIDENCE_AWARE_POLICY
    assert result.abstained is True
    assert result.contradiction_threshold == pytest.approx(0.70)
    assert result.max_contradiction_confidence == pytest.approx(
        baseline.decision_confidence
    )
    assert result.max_contradiction_confidence < 0.70
    assert result.per_window == baseline.per_window
    assert result.aggregate_scores == baseline.aggregate_scores
    assert "below the development-set threshold" in result.explanation


def test_threshold_boundary_preserves_contradiction_at_or_above_threshold():
    baseline = verify_with_policy("Low contradiction.", threshold=0.50)
    demoted = verify_with_policy("Low contradiction.", threshold=0.60)

    assert baseline.status == "contradicted"
    assert baseline.policy_outcome == "unanimous_contradiction"
    assert baseline.abstained is False
    assert demoted.status == "insufficient_evidence"
    assert demoted.policy_outcome == "low_confidence_contradiction"


def test_strong_contradiction_and_neutral_context_are_preserved():
    unanimous = verify_with_policy("Strong contradiction.")
    with_neutral = verify_with_policy(
        "Strong contradiction. Neutral context."
    )

    assert unanimous.status == "contradicted"
    assert unanimous.policy_outcome == "unanimous_contradiction"
    assert with_neutral.status == "contradicted"
    assert with_neutral.policy_outcome == "contradiction_with_neutral_context"
    assert with_neutral.label_counts == {"contradiction": 1, "neutral": 1}


def test_mixed_entailment_and_low_contradiction_remains_insufficient():
    result = verify_with_policy("This sentence entails. Low contradiction.")

    assert result.status == "insufficient_evidence"
    assert result.label == "mixed"
    assert result.policy_outcome == "mixed_entailment_contradiction"
    assert result.abstained is True


def test_supported_and_neutral_only_decisions_are_not_promoted_by_policy():
    supported = verify_with_policy("This sentence entails. Neutral context.")
    neutral = verify_with_policy("Neutral one. Neutral two.")

    assert supported.status == "supported"
    assert supported.policy_outcome == "entailment_with_neutral_context"
    assert supported.abstained is False
    assert neutral.status == "insufficient_evidence"
    assert neutral.policy_outcome == "neutral_only"
    assert neutral.label == "neutral"


def test_whole_passage_policy_uses_one_window_and_keeps_metadata():
    result = verify_with_policy(
        "This sentence entails. Neutral context.",
        strategy=WHOLE_PASSAGE,
    )

    assert result.status == "supported"
    assert result.strategy == WHOLE_PASSAGE
    assert len(result.per_window) == 1
    assert result.policy_outcome == "unanimous_entailment"
    assert result.model_metadata["decision_policy"] == CONFIDENCE_AWARE_POLICY
    assert result.model_metadata["contradiction_threshold"] == pytest.approx(0.70)


def test_invalid_citation_is_not_repaired_or_scored():
    backend = FakeNLIBackend(outputs=rows_for_confidences)
    verifier = ConfidenceAwareNLIClaimVerifier(
        backend=backend,
        strategy=SENTENCE,
        contradiction_threshold=0.70,
    )

    result = verifier.verify(
        Claim("A factual claim.", ("missing#0",)),
        {"a#0": make_passage("a#0", "This sentence entails.")},
    )

    assert result.status == "invalid_citation"
    assert result.policy_outcome == "preserved_non_nli_status"
    assert result.abstained is False
    assert result.per_window == ()
    assert backend.calls == []


def test_backend_errors_remain_distinct_from_insufficient_evidence():
    class UnavailableBackend(FakeNLIBackend):
        def predict_batch(self, pairs):
            raise NLIModelUnavailable("not cached")

    class BrokenBackend(FakeNLIBackend):
        def predict_batch(self, pairs):
            raise RuntimeError("simulated crash")

    passage = make_passage("a#0", "This sentence entails.")
    claim = Claim("A factual claim.", ("a#0",))

    unavailable = ConfidenceAwareNLIClaimVerifier(
        backend=UnavailableBackend(), strategy=SENTENCE
    ).verify(claim, {"a#0": passage})
    broken = ConfidenceAwareNLIClaimVerifier(
        backend=BrokenBackend(), strategy=SENTENCE
    ).verify(claim, {"a#0": passage})

    assert unavailable.status == "backend_unavailable"
    assert unavailable.policy_outcome == "preserved_non_nli_status"
    assert broken.status == "verification_error"
    assert broken.policy_outcome == "preserved_non_nli_status"


def test_empty_evidence_stays_insufficient_without_a_model_call():
    backend = FakeNLIBackend(outputs=rows_for_confidences)
    verifier = ConfidenceAwareNLIClaimVerifier(
        backend=backend,
        strategy=SENTENCE,
        contradiction_threshold=0.70,
    )

    result = verifier.verify(
        Claim("A factual claim.", ("a#0",)),
        {"a#0": make_passage("a#0", "   ")},
    )

    assert result.status == "insufficient_evidence"
    assert result.policy_outcome == "no_evidence_windows"
    assert result.abstained is True
    assert backend.calls == []


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        ConfidenceAwareNLIClaimVerifier(
            backend=FakeNLIBackend(outputs=rows_for_confidences),
            contradiction_threshold=1.01,
        )


def test_evaluate_windowed_records_policy_metadata():
    cases = [
        {
            "id": "t1",
            "type": "irrelevant_citation",
            "claim": "A factual claim.",
            "citations": ["a#0"],
            "passages": {"a#0": "Low contradiction."},
            "expected": "insufficient_evidence",
            "rationale": "test",
        }
    ]
    verifier = ConfidenceAwareNLIClaimVerifier(
        backend=FakeNLIBackend(outputs=rows_for_confidences),
        strategy=SENTENCE,
        contradiction_threshold=0.70,
    )

    rows = evaluate_windowed(cases, verifier)

    assert rows[0]["correct"] is True
    assert rows[0]["decision_policy"] == CONFIDENCE_AWARE_POLICY
    assert rows[0]["policy_outcome"] == "low_confidence_contradiction"
    assert rows[0]["contradiction_threshold"] == pytest.approx(0.70)
    assert rows[0]["abstained"] is True


def test_select_threshold_uses_accuracy_recall_then_coverage():
    candidates = [
        {
            "threshold": 0.50,
            "accuracy": 0.80,
            "contradicted_recall": 1.0,
            "incorrect_contradiction_rate": 0.30,
            "abstention_rate": 0.20,
        },
        {
            "threshold": 0.70,
            "accuracy": 0.90,
            "contradicted_recall": 1.0,
            "incorrect_contradiction_rate": 0.20,
            "abstention_rate": 0.30,
        },
        {
            "threshold": 0.75,
            "accuracy": 0.90,
            "contradicted_recall": 1.0,
            "incorrect_contradiction_rate": 0.20,
            "abstention_rate": 0.30,
        },
        {
            "threshold": 0.90,
            "accuracy": 0.90,
            "contradicted_recall": 0.60,
            "incorrect_contradiction_rate": 0.00,
            "abstention_rate": 0.40,
        },
    ]

    selected = select_threshold(candidates)

    assert selected["threshold"] == 0.70
