import pytest

from experiments.verification_utils.paraphrase_policy import (
    PARAPHRASE_POLICY,
    ParaphraseAwareNLIClaimVerifier,
    extract_numbers,
    extract_predicate_group,
    paraphrase_supports_claim,
)
from experiments.verification_utils.eval import make_passage
from experiments.verification_utils.windowed_nli import (
    WindowedNLIClaimVerifier,
)
from minigrad.claim_verification import Claim
from minigrad.nli_verification import FakeNLIBackend
from minigrad.nli_verification import NLIModelUnavailable

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def logits_for(label, high=5.0, low=-2.0):
    return [
        high if ID2LABEL[label_id] == label else low
        for label_id in sorted(ID2LABEL)
    ]


def neutral_rows(pairs):
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
            rows.append(logits_for("neutral"))
    return rows


def verify_paraphrase(claim_text, evidence_text, *, backend_outputs=None, strategy="whole_passage"):
    backend = FakeNLIBackend(outputs=backend_outputs or neutral_rows)
    verifier = ParaphraseAwareNLIClaimVerifier(
        backend=backend,
        strategy=strategy,
        adjacent_context=0,
    )
    passage = make_passage("jupiter#0", evidence_text)
    return verifier.verify(Claim(claim_text, ("jupiter#0",)), {"jupiter#0": passage})


def test_paraphrase_support_accepts_duration_paraphrase_with_compatible_number():
    assert paraphrase_supports_claim(
        "Jupiter rotates once every 9.9 hours.",
        "A day on Jupiter lasts only about 9.9 hours, which is the shortest day of any planet.",
    )
    assert extract_predicate_group("Jupiter rotates once every 9.9 hours.") == "duration"
    assert extract_numbers("Jupiter rotates once every 9.9 hours.") == (9.9,)


def test_paraphrase_support_rejects_insufficient_fact_paraphrase():
    assert not paraphrase_supports_claim(
        "Mars has liquid water on its surface today.",
        "Evidence suggests that liquid water once flowed on the surface.",
    )


def test_paraphrase_support_rejects_number_mismatch():
    assert not paraphrase_supports_claim(
        "Jupiter's orbital period is about 12 Earth years.",
        "One year on Jupiter lasts about 11.86 Earth years.",
    )


def test_paraphrase_aware_policy_promotes_neutral_paraphrase_to_supported():
    result = verify_paraphrase(
        "Jupiter rotates once every 9.9 hours.",
        "A day on Jupiter lasts only about 9.9 hours, which is the shortest day of any planet.",
    )

    assert result.status == "supported"
    assert result.label == "entailment"
    assert result.policy_outcome == "paraphrase_supported"
    assert result.decision_policy == "paraphrase_aware"
    assert result.abstained is False
    assert result.decision_confidence > 0.0


def test_paraphrase_aware_policy_does_not_promote_contradicted_claim():
    backend = FakeNLIBackend(outputs=[[5.0, -2.0, -2.0]])
    verifier = ParaphraseAwareNLIClaimVerifier(
        backend=backend,
        strategy="whole_passage",
        adjacent_context=0,
    )
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    result = verifier.verify(
        Claim("Jupiter has 79 known moons.", ("jupiter#0",)),
        {"jupiter#0": passage},
    )

    assert result.status == "contradicted"
    assert result.policy_outcome == "unanimous_contradiction"
    assert result.abstained is False


def test_paraphrase_aware_policy_does_not_promote_mixed_nli_result():
    def mixed_backend(pairs):
        rows = []
        for premise, _ in pairs:
            if "and orbits" in premise:
                rows.append([5.0, -2.0, -2.0])
            else:
                rows.append([-2.0, 5.0, -2.0])
        return rows

    backend = FakeNLIBackend(outputs=mixed_backend)
    verifier = ParaphraseAwareNLIClaimVerifier(
        backend=backend,
        strategy="sentence",
        adjacent_context=0,
    )
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons. Jupiter has 95 known moons and orbits.")
    result = verifier.verify(
        Claim("Jupiter has moons.", ("jupiter#0",)),
        {"jupiter#0": passage},
    )

    assert result.status == "insufficient_evidence"
    assert result.label == "mixed"
    assert result.policy_outcome == "mixed_entailment_contradiction"


def test_paraphrase_aware_policy_does_not_score_invalid_citation():
    backend = FakeNLIBackend(outputs=neutral_rows)
    verifier = ParaphraseAwareNLIClaimVerifier(
        backend=backend,
        strategy="whole_passage",
        adjacent_context=0,
    )
    passage = make_passage("jupiter#0", "A day on Jupiter lasts only about 9.9 hours.")
    result = verifier.verify(
        Claim("Jupiter rotates once every 9.9 hours.", ("missing#0",)),
        {"jupiter#0": passage},
    )

    assert result.status == "invalid_citation"
    assert result.policy_outcome == "preserved_non_nli_status"
    assert result.per_window == ()
    assert backend.calls == []


def test_paraphrase_policy_constant_is_reported():
    assert PARAPHRASE_POLICY == "paraphrase_aware"
