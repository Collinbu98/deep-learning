import pytest

from experiments.verification_utils.confidence_policy import (
    CONFIDENCE_AWARE_POLICY,
)
from experiments.verification_utils.eval import make_passage
from experiments.verification_utils.relevance_policy import (
    RELEVANCE_AWARE_POLICY,
    RELEVANCE_METHOD,
    RelevanceAwareNLIClaimVerifier,
    apply_relevance_policy,
    assess_relevance,
    extract_topic_anchors,
)
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


def rows_for_premises(pairs):
    rows = []
    for premise, _ in pairs:
        text = premise.lower()
        if "low contradiction" in text:
            rows.append([1.0, 0.0, 0.0])
        elif "contradict" in text:
            rows.append(logits_for("contradiction"))
        elif "entail" in text:
            rows.append(logits_for("entailment"))
        elif "neutral" in text:
            rows.append(logits_for("neutral"))
        else:
            rows.append(logits_for("contradiction"))
    return rows


def verify_source(claim_text, evidence_text, source_id="jupiter#0", **kwargs):
    threshold = kwargs.pop("threshold", 0.70)
    strategy = kwargs.pop("strategy", WHOLE_PASSAGE)
    backend = FakeNLIBackend(outputs=rows_for_premises)
    verifier = RelevanceAwareNLIClaimVerifier(
        backend=backend,
        strategy=strategy,
        contradiction_threshold=threshold,
        **kwargs,
    )
    passage = make_passage(source_id, evidence_text)
    return verifier.verify(
        Claim(claim_text, (source_id,)), {source_id: passage}
    )


def test_extract_topic_anchors_handles_possessives_and_function_words():
    assert extract_topic_anchors(
        "The Solar System's largest planet is Jupiter."
    ) == ("solar", "system", "jupiter")
    assert extract_topic_anchors(
        "A single rotation on Venus takes 243 Earth days."
    ) == ("venus", "earth")
    assert extract_topic_anchors("Jupiter's moons are cold.") == ("jupiter",)
    assert extract_topic_anchors("it has no capitalised proper noun") == ()


def test_assessment_rejects_unrelated_subject_and_keeps_topical_match():
    backend = FakeNLIBackend(outputs=[logits_for("contradiction")])
    base = WindowedNLIClaimVerifier(
        backend=backend, strategy=SENTENCE
    )
    unrelated = base.verify(
        Claim("Saturn has 95 known moons.", ("jupiter#0",)),
        {"jupiter#0": make_passage("jupiter#0", "Jupiter has 95 known moons.")},
    )
    related = base.verify(
        Claim("Jupiter has 79 known moons.", ("jupiter#0",)),
        {"jupiter#0": make_passage("jupiter#0", "Jupiter has 95 known moons.")},
    )

    unrelated_assessment = assess_relevance(
        unrelated.claim.text, unrelated.per_window
    )
    related_assessment = assess_relevance(related.claim.text, related.per_window)

    assert unrelated_assessment.contradiction_relevant is False
    assert unrelated_assessment.reason == (
        "no_contradiction_window_matches_claim_anchor"
    )
    assert related_assessment.contradiction_relevant is True
    assert related_assessment.matched_anchors == ("jupiter",)


def test_source_identifier_can_supply_relevance_for_pronoun_evidence():
    result = verify_source(
        "Jupiter has 79 known moons.",
        "It has 95 known moons.",
        source_id="jupiter#0",
    )

    assert result.status == "contradicted"
    assert result.policy_outcome == "unanimous_contradiction"
    assert result.policy_details["relevance"]["contradiction_relevant"] is True
    assert "jupiter" in result.policy_details["relevance"]["matched_anchors"]


def test_irrelevant_high_confidence_contradiction_is_demoted_without_score_loss():
    backend = FakeNLIBackend(outputs=[logits_for("contradiction")])
    baseline = WindowedNLIClaimVerifier(
        backend=backend, strategy=WHOLE_PASSAGE
    ).verify(
        Claim("Saturn has 95 known moons.", ("jupiter#0",)),
        {"jupiter#0": make_passage("jupiter#0", "Jupiter has 95 known moons.")},
    )
    result = verify_source(
        "Saturn has 95 known moons.",
        "Jupiter has 95 known moons.",
        source_id="jupiter#0",
    )

    assert baseline.status == "contradicted"
    assert result.status == "insufficient_evidence"
    assert result.label == "contradiction"
    assert result.policy_outcome == "irrelevant_contradiction"
    assert result.decision_policy == RELEVANCE_AWARE_POLICY
    assert result.abstained is True
    assert result.decision_confidence == pytest.approx(
        baseline.decision_confidence
    )
    assert result.aggregate_scores == baseline.aggregate_scores
    assert result.per_window == baseline.per_window
    assert result.policy_details["relevance"]["anchors"] == ["saturn"]
    assert "topically unrelated" in result.explanation
    assert result.model_metadata["decision_policy"] == RELEVANCE_AWARE_POLICY
    assert result.model_metadata["relevance_method"] == RELEVANCE_METHOD


def test_relevant_contradiction_is_preserved_by_the_guard():
    result = verify_source(
        "Jupiter has 79 known moons.",
        "Jupiter has 95 known moons.",
    )

    assert result.status == "contradicted"
    assert result.policy_outcome == "unanimous_contradiction"
    assert result.abstained is False
    assert result.policy_details["relevance"]["contradiction_relevant"] is True


def test_no_claim_anchors_preserves_base_decision_instead_of_guessing():
    result = verify_source(
        "it has 79 known moons",
        "Jupiter has 95 known moons.",
    )

    assert result.status == "contradicted"
    assert result.policy_outcome == "unanimous_contradiction"
    assert result.policy_details["relevance"]["anchors"] == []
    assert result.policy_details["relevance"]["reason"] == "no_claim_anchors"
    assert (
        result.policy_details["relevance"]["contradiction_relevant"] is None
    )


def test_relevance_guard_never_promotes_neutral_or_entailment():
    neutral = verify_source(
        "Jupiter has 79 known moons.",
        "Neutral context.",
    )
    supported = verify_source(
        "Jupiter has 79 known moons.",
        "This sentence entails.",
    )

    assert neutral.status == "insufficient_evidence"
    assert neutral.policy_outcome == "neutral_only"
    assert supported.status == "supported"
    assert supported.policy_outcome == "unanimous_entailment"
    assert supported.policy_details["base_policy"] == CONFIDENCE_AWARE_POLICY


def test_low_confidence_contradiction_keeps_exp19_outcome_even_if_relevant():
    result = verify_source(
        "Jupiter has 79 known moons.",
        "Low contradiction.",
    )

    assert result.status == "insufficient_evidence"
    assert result.policy_outcome == "low_confidence_contradiction"
    assert result.policy_details["confidence_threshold"] == pytest.approx(0.70)
    assert result.policy_details["relevance"]["contradiction_relevant"] is True


def test_incidental_subject_mention_is_documented_as_a_limitation():
    result = verify_source(
        "Earth has a thick atmosphere of carbon dioxide.",
        "Surface pressure is about 92 times the pressure on Earth. "
        "A runaway greenhouse effect makes Venus the hottest planet.",
        source_id="venus#0",
    )

    assert result.status == "contradicted"
    assert result.policy_details["relevance"]["anchors"] == ["earth"]
    assert result.policy_details["relevance"]["contradiction_relevant"] is True


def test_apply_relevance_policy_accepts_precomputed_assessment():
    backend = FakeNLIBackend(outputs=[logits_for("contradiction")])
    base = WindowedNLIClaimVerifier(
        backend=backend, strategy=SENTENCE
    ).verify(
        Claim("Saturn has 95 known moons.", ("jupiter#0",)),
        {"jupiter#0": make_passage("jupiter#0", "Jupiter has 95 known moons.")},
    )
    assessment = assess_relevance(base.claim.text, base.per_window)
    result = apply_relevance_policy(base, assessment)

    assert result.status == "insufficient_evidence"
    assert result.policy_outcome == "irrelevant_contradiction"


def test_invalid_citation_is_not_scored_or_repaired():
    backend = FakeNLIBackend(outputs=rows_for_premises)
    verifier = RelevanceAwareNLIClaimVerifier(
        backend=backend,
        strategy=SENTENCE,
        contradiction_threshold=0.70,
    )

    result = verifier.verify(
        Claim("Jupiter has 95 known moons.", ("missing#0",)),
        {"jupiter#0": make_passage("jupiter#0", "Jupiter has 95 known moons.")},
    )

    assert result.status == "invalid_citation"
    assert result.policy_outcome == "preserved_non_nli_status"
    assert result.per_window == ()
    assert result.decision_policy == RELEVANCE_AWARE_POLICY
    assert backend.calls == []


def test_backend_errors_stay_distinct():
    class UnavailableBackend(FakeNLIBackend):
        def predict_batch(self, pairs):
            raise NLIModelUnavailable("not cached")

    verifier = RelevanceAwareNLIClaimVerifier(
        backend=UnavailableBackend(), strategy=SENTENCE
    )
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    result = verifier.verify(
        Claim("Jupiter has 95 known moons.", ("jupiter#0",)),
        {"jupiter#0": passage},
    )

    assert result.status == "backend_unavailable"
    assert result.policy_outcome == "preserved_non_nli_status"
    assert result.decision_policy == RELEVANCE_AWARE_POLICY


def test_evaluate_windowed_records_relevance_details():
    cases = [
        {
            "id": "t1",
            "type": "irrelevant_citation",
            "claim": "Saturn has 95 known moons.",
            "citations": ["jupiter#0"],
            "passages": {"jupiter#0": "Jupiter has 95 known moons."},
            "expected": "insufficient_evidence",
            "rationale": "test",
        }
    ]
    verifier = RelevanceAwareNLIClaimVerifier(
        backend=FakeNLIBackend(outputs=rows_for_premises),
        strategy=WHOLE_PASSAGE,
        contradiction_threshold=0.70,
    )

    rows = evaluate_windowed(cases, verifier)

    assert rows[0]["correct"] is True
    assert rows[0]["decision_policy"] == RELEVANCE_AWARE_POLICY
    assert rows[0]["policy_outcome"] == "irrelevant_contradiction"
    relevance = rows[0]["policy_details"]["relevance"]
    assert relevance["method"] == RELEVANCE_METHOD
    assert relevance["contradiction_relevant"] is False
