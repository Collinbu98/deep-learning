"""Lexical evidence-relevance guard for Experiment 20.

The Experiment 19 confidence policy can only reject contradictions whose NLI
score is low. It cannot reject a high-confidence contradiction when the cited
passage is about a different subject. This module adds an explicit, opt-in
relevance check that is deliberately independent of the NLI scores.

The check is intentionally small and lexical. It extracts capitalised topic
anchors from the claim (for example ``Jupiter`` or ``Solar System``) and asks
whether a contradiction window, or the source/document identifier associated
with that window, contains any of those anchors. If there is no overlap, the
contradiction is demoted to ``insufficient_evidence``. If the evidence shares
an anchor, or the claim has no extractable anchors, the base policy is left
unchanged.

This is a topical guard, not a semantic relevance model. It does not repair
paraphrases, does not promote neutral predictions, and can be fooled by an
incidental mention of the claim's subject in an otherwise irrelevant passage.
Those limitations are reported rather than hidden.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from experiments.verification_utils.confidence_policy import (
    CONFIDENCE_AWARE_POLICY,
    ConfidenceAwareNLIClaimVerifier,
)
from experiments.verification_utils.windowed_nli import (
    WindowedNLIResult,
    WindowPrediction,
)
from minigrad.bm25 import tokenize
from minigrad.claim_verification import Claim
from minigrad.evidence_windows import split_sentences
from minigrad.llm_answerer import EvidencePassage

__all__ = [
    "RELEVANCE_AWARE_POLICY",
    "RELEVANCE_METHOD",
    "WindowRelevance",
    "RelevanceAssessment",
    "extract_topic_anchors",
    "assess_relevance",
    "apply_relevance_policy",
    "RelevanceAwareNLIClaimVerifier",
]

RELEVANCE_AWARE_POLICY = "relevance_aware"
RELEVANCE_METHOD = "lexical_topic_anchor_guard"

_ANCHOR_TOKEN = re.compile(r"\b[A-Z][a-z]{1,}(?:['\u2019]s)?\b")
_POSSESSIVE = re.compile(r"['\u2019]s$", re.IGNORECASE)
_NON_ANCHOR_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "their",
        "there",
        "he",
        "she",
        "we",
        "i",
        "you",
        "your",
        "our",
        "my",
        "his",
        "her",
        "some",
        "many",
        "most",
        "all",
        "both",
        "each",
        "every",
        "no",
        "not",
        "if",
        "when",
        "while",
        "because",
        "although",
        "however",
        "in",
        "on",
        "at",
        "for",
        "from",
        "to",
        "with",
        "without",
        "after",
        "before",
        "during",
        "since",
        "as",
        "by",
        "of",
        "and",
        "or",
        "but",
        "then",
        "thus",
        "therefore",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "can",
        "could",
        "might",
        "must",
        "should",
        "would",
        "will",
        "within",
        "between",
        "around",
        "about",
        "only",
        "also",
        "more",
        "than",
        "such",
    }
)


def _normalise_anchor(value: str) -> str:
    value = _POSSESSIVE.sub("", value)
    tokens = tokenize(value)
    return tokens[0] if len(tokens) == 1 else ""


def extract_topic_anchors(text: str) -> tuple[str, ...]:
    """Return ordered lower-case proper-noun anchors from a claim.

    Tokens are only candidates when they begin with a capital letter and are
    followed by lower-case letters. Common sentence-initial words and function
    words are ignored. Possessives are normalised with the BM25 tokenizer so
    ``Jupiter's`` and ``Jupiter`` match the same evidence term.
    """

    anchors: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        for match in _ANCHOR_TOKEN.finditer(sentence.text):
            anchor = _normalise_anchor(match.group(0))
            if len(anchor) < 2 or anchor in _NON_ANCHOR_WORDS:
                continue
            if anchor in seen:
                continue
            seen.add(anchor)
            anchors.append(anchor)
    return tuple(anchors)


@dataclass(frozen=True)
class WindowRelevance:
    """Relevance trace for one scored evidence window."""

    window_id: str
    source_id: str
    doc_id: str
    label: str
    text_excerpt: str
    matched_anchors: tuple[str, ...]
    relevant: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "source_id": self.source_id,
            "doc_id": self.doc_id,
            "label": self.label,
            "text_excerpt": self.text_excerpt,
            "matched_anchors": list(self.matched_anchors),
            "relevant": self.relevant,
        }


@dataclass(frozen=True)
class RelevanceAssessment:
    """Claim-level assessment over all NLI windows."""

    anchors: tuple[str, ...]
    matched_anchors: tuple[str, ...]
    contradiction_window_ids: tuple[str, ...]
    relevant_contradiction_window_ids: tuple[str, ...]
    irrelevant_contradiction_window_ids: tuple[str, ...]
    contradiction_relevant: bool | None
    reason: str
    windows: tuple[WindowRelevance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": RELEVANCE_METHOD,
            "anchors": list(self.anchors),
            "matched_anchors": list(self.matched_anchors),
            "contradiction_window_ids": list(
                self.contradiction_window_ids
            ),
            "relevant_contradiction_window_ids": list(
                self.relevant_contradiction_window_ids
            ),
            "irrelevant_contradiction_window_ids": list(
                self.irrelevant_contradiction_window_ids
            ),
            "contradiction_relevant": self.contradiction_relevant,
            "reason": self.reason,
            "windows": [window.to_dict() for window in self.windows],
        }


def _excerpt(text: str, limit: int = 200) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _matched_anchors(
    anchors: Sequence[str], window: WindowPrediction
) -> tuple[str, ...]:
    evidence_terms = set(
        tokenize(
            " ".join((window.text, window.source_id, window.doc_id))
        )
    )
    return tuple(anchor for anchor in anchors if anchor in evidence_terms)


def assess_relevance(
    claim_text: str, windows: Sequence[WindowPrediction]
) -> RelevanceAssessment:
    """Assess whether contradiction windows share a topic anchor with a claim."""

    anchors = extract_topic_anchors(claim_text)
    rows: list[WindowRelevance] = []
    matched: list[str] = []
    contradiction_ids: list[str] = []
    relevant_contradiction_ids: list[str] = []
    irrelevant_contradiction_ids: list[str] = []

    for window in windows:
        window_matches = _matched_anchors(anchors, window)
        for anchor in window_matches:
            if anchor not in matched:
                matched.append(anchor)
        relevant: bool | None
        if not anchors:
            relevant = None
        else:
            relevant = bool(window_matches)
        rows.append(
            WindowRelevance(
                window_id=window.window_id,
                source_id=window.source_id,
                doc_id=window.doc_id,
                label=window.label,
                text_excerpt=_excerpt(window.text),
                matched_anchors=window_matches,
                relevant=relevant,
            )
        )
        if window.label != "contradiction":
            continue
        contradiction_ids.append(window.window_id)
        if relevant:
            relevant_contradiction_ids.append(window.window_id)
        else:
            irrelevant_contradiction_ids.append(window.window_id)

    if not anchors:
        contradiction_relevant = None
        reason = "no_claim_anchors"
    elif not contradiction_ids:
        contradiction_relevant = None
        reason = "no_contradiction_windows"
    elif relevant_contradiction_ids:
        contradiction_relevant = True
        reason = "contradiction_window_matches_claim_anchor"
    else:
        contradiction_relevant = False
        reason = "no_contradiction_window_matches_claim_anchor"

    return RelevanceAssessment(
        anchors=anchors,
        matched_anchors=tuple(matched),
        contradiction_window_ids=tuple(contradiction_ids),
        relevant_contradiction_window_ids=tuple(
            relevant_contradiction_ids
        ),
        irrelevant_contradiction_window_ids=tuple(
            irrelevant_contradiction_ids
        ),
        contradiction_relevant=contradiction_relevant,
        reason=reason,
        windows=tuple(rows),
    )


def apply_relevance_policy(
    result: WindowedNLIResult,
    assessment: RelevanceAssessment | None = None,
) -> WindowedNLIResult:
    """Apply the relevance guard after the confidence-aware base policy.

    Only a ``contradicted`` result whose contradiction windows have no claim
    anchor is changed. The original label, aggregate scores, and raw window
    predictions are retained so the NLI evidence remains inspectable.
    """

    if assessment is None:
        assessment = assess_relevance(result.claim.text, result.per_window)
    details = {
        **dict(result.policy_details),
        "base_policy": CONFIDENCE_AWARE_POLICY,
        "confidence_threshold": result.contradiction_threshold,
        "relevance": assessment.to_dict(),
    }

    if (
        result.status == "contradicted"
        and assessment.contradiction_relevant is False
    ):
        return replace(
            result,
            status="insufficient_evidence",
            explanation=(
                "The NLI model predicted contradiction, but no contradiction "
                "window shared a topic anchor with the claim. Treating the "
                "evidence as topically unrelated rather than as a "
                "contradiction. Original decision: "
                f"{result.explanation}"
            ),
            decision_policy=RELEVANCE_AWARE_POLICY,
            policy_outcome="irrelevant_contradiction",
            abstained=True,
            policy_details=details,
        )

    return replace(
        result,
        decision_policy=RELEVANCE_AWARE_POLICY,
        policy_details=details,
    )


class RelevanceAwareNLIClaimVerifier(ConfidenceAwareNLIClaimVerifier):
    """Confidence-aware verifier plus the opt-in lexical relevance guard."""

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "decision_policy": RELEVANCE_AWARE_POLICY,
                "relevance_method": RELEVANCE_METHOD,
            }
        )
        return metadata

    def verify(
        self, claim: Claim, evidence_passages: Mapping[str, EvidencePassage]
    ) -> WindowedNLIResult:
        result = super().verify(claim, evidence_passages)
        result = apply_relevance_policy(result)
        metadata = dict(result.model_metadata)
        metadata.update(
            {
                "decision_policy": RELEVANCE_AWARE_POLICY,
                "relevance_method": RELEVANCE_METHOD,
            }
        )
        return replace(result, model_metadata=metadata)
