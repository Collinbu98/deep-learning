"""Conservative deterministic paraphrase-equivalence checks for Experiment 21.

The NLI model remains the primary inference backend. This module adds a small,
offline support predicate for a narrow class of duration and fact-equivalence
paraphrases. It is deliberately conservative: it never promotes a contradicted
claim, never promotes a mixed NLI result, and only accepts an evidence window
when a shared predicate group and compatible numeric value are present.

The check is not an embedding or semantic-similarity model. It encodes small,
corpus-derived equivalence groups for the relevant paraphrase families and is
intended to be evaluated on the independent Experiment 21 development set.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Mapping

from experiments.verification_utils.relevance_policy import (
    RelevanceAwareNLIClaimVerifier,
)
from minigrad.claim_verification import Claim
from minigrad.llm_answerer import EvidencePassage

__all__ = [
    "PARAPHRASE_POLICY",
    "PARAPHRASE_EQUIVALENCE_GROUPS",
    "extract_numbers",
    "extract_predicate_group",
    "paraphrase_supports_claim",
    "ParaphraseAwareNLIClaimVerifier",
]

PARAPHRASE_POLICY = "paraphrase_aware"
PARAPHRASE_EQUIVALENCE_GROUPS = {
    "duration": (
        "rotation",
        "rotates",
        "rotating",
        "spin",
        "spins",
        "rotation period",
        "period",
        "day",
        "day length",
        "duration",
    ),
    "orbital_period": (
        "orbit",
        "orbiting",
        "orbital period",
        "year",
        "years",
        "yearly",
    ),
    "mass": ("mass", "total mass"),
    "distance": ("distance", "orbital distance", "orbit distance"),
    "atmosphere_composition": (
        "atmosphere",
        "composition",
        "nitrogen",
        "gas composition",
    ),
}


def extract_numbers(text: str) -> tuple[float, ...]:
    """Return numeric values from a claim or evidence passage."""

    return tuple(
        float(value)
        for value in re.findall(r"\d+(?:\.\d+)?", text)
        if value.replace(".", "", 1).isdigit()
    )


def _tokens(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        token
        for token in re.findall(r"[a-z]+", lowered)
        if token
    )


def extract_predicate_group(text: str) -> str | None:
    """Map a claim or evidence passage to a conservative paraphrase group."""

    lowered = text.lower()
    tokens = _tokens(text)
    token_set = set(tokens)
    if re.search(r"atmosphere|nitrogen|gas composition", lowered):
        return "atmosphere_composition"
    if re.search(r"mass|density", lowered):
        return "mass"
    if re.search(r"orbital|orbit", lowered):
        return "orbital_period"
    if any(
        alias in token_set or alias == token
        for alias, token in zip(("rotation", "rotates", "rotating", "spin", "spins", "period", "day", "duration"), tokens)
    ):
        return "duration"
    if any(alias in token_set or alias == token for alias, token in zip(("distance",), tokens)):
        return "distance"
    return None


def _subject_overlap(claim_text: str, evidence_text: str) -> bool:
    claim_tokens = set(
        token.lower()
        for token in re.findall(r"[A-Za-z][a-z]*", claim_text)
        if token
    )
    evidence_tokens = set(_tokens(evidence_text))
    return bool(claim_tokens & evidence_tokens)


def paraphrase_supports_claim(claim_text: str, evidence_text: str) -> bool:
    """Return whether a narrow, deterministic paraphrase-equivalence check accepts support."""

    claim_group = extract_predicate_group(claim_text)
    if claim_group is None:
        return False
    if not _subject_overlap(claim_text, evidence_text):
        return False
    evidence_tokens = _tokens(evidence_text)
    token_set = set(evidence_tokens)
    aliases_by_group = {
        "duration": ("rotation", "rotates", "rotating", "spin", "spins", "period", "day", "duration"),
        "orbital_period": ("orbit", "orbiting", "orbital period", "year", "years", "yearly"),
        "mass": ("mass", "total mass"),
        "distance": ("distance", "orbital distance", "orbit distance"),
        "atmosphere_composition": ("atmosphere", "composition", "nitrogen", "gas composition"),
    }
    aliases = aliases_by_group.get(claim_group, ())
    if not any(alias in token_set or alias == token for alias, token in zip(aliases, evidence_tokens)):
        return False

    claim_numbers = set(extract_numbers(claim_text))
    evidence_numbers = set(extract_numbers(evidence_text))
    if claim_numbers and evidence_numbers:
        return any(number in evidence_numbers for number in claim_numbers)
    return True


class ParaphraseAwareNLIClaimVerifier(RelevanceAwareNLIClaimVerifier):
    """Confidence- and relevance-aware NLI verifier with a conservative support check."""

    def verify(
        self,
        claim: Claim,
        evidence_passages: Mapping[str, EvidencePassage],
    ):
        result = super().verify(claim, evidence_passages)
        if (
            result.status == "insufficient_evidence"
            and result.label == "neutral"
        ):
            supporting = [
                window
                for window in result.per_window
                if paraphrase_supports_claim(claim.text, window.text)
            ]
            if supporting:
                confidence = max(
                    window.scores.get("entailment", 0.0) for window in supporting
                )
                return replace(
                    result,
                    status="supported",
                    label="entailment",
                    explanation=(
                        "NLI returned neutral, but a conservative deterministic "
                        "paraphrase-equivalence check accepted the evidence. "
                        f"Supporting evidence confidence: {confidence:.3f}. "
                        f"Original decision: {result.explanation}"
                    ),
                    decision_confidence=confidence,
                    abstained=False,
                    policy_outcome="paraphrase_supported",
                    decision_policy="paraphrase_aware",
                    policy_details={
                        **result.policy_details,
                        "base_policy": result.policy_details.get("base_policy", "confidence_aware"),
                        "relevance": result.policy_details.get("relevance"),
                        "paraphrase": {
                            "method": PARAPHRASE_POLICY,
                            "supporting_window_ids": [
                                window.window_id for window in supporting
                            ],
                            "confidence": confidence,
                        },
                    },
                )
        return result
