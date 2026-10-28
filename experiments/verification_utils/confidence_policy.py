"""Confidence-aware NLI decision policy for Experiment 19.

The Experiment 17/18 conservative policy remains the default. This module adds
an opt-in policy that only changes one kind of decision: a contradiction whose
strongest contradiction window falls below a development-set threshold is
demoted to ``insufficient_evidence``. It never promotes neutral predictions to
entailment, never turns an uncertain case into ``supported``, and never repairs
citation-validation or backend/inference errors.

Softmax values are model confidences, not calibrated probabilities. The
threshold is selected on a separate development set and must not be tuned on
the fixed Experiment 15 benchmark.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Mapping, Sequence

from experiments.verification_utils.windowed_nli import (
    WindowedNLIClaimVerifier,
    WindowedNLIResult,
)
from minigrad.claim_verification import Claim
from minigrad.evidence_windows import WHOLE_PASSAGE, WINDOW_STRATEGIES
from minigrad.llm_answerer import EvidencePassage
from minigrad.nli_verification import NLIPredictor

__all__ = [
    "CONFIDENCE_AWARE_POLICY",
    "DEFAULT_CONTRADICTION_THRESHOLD",
    "ConfidenceAwareNLIClaimVerifier",
    "apply_confidence_policy",
    "select_threshold",
]

CONFIDENCE_AWARE_POLICY = "confidence_aware"
DEFAULT_CONTRADICTION_THRESHOLD = 0.70
_NON_NLI_STATUSES = {
    "invalid_citation",
    "backend_unavailable",
    "verification_error",
}


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError("contradiction threshold must be between 0 and 1")
    return value


def _label_counts(result: WindowedNLIResult) -> dict[str, int]:
    return dict(Counter(window.label for window in result.per_window))


def _max_contradiction_confidence(
    result: WindowedNLIResult,
) -> float | None:
    values = [
        float(window.scores.get(window.label, 0.0))
        for window in result.per_window
        if window.label == "contradiction"
    ]
    return max(values) if values else None


def apply_confidence_policy(
    result: WindowedNLIResult,
    threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
) -> WindowedNLIResult:
    """Apply the confidence-aware rule to an existing windowed NLI result.

    The raw per-window predictions and original aggregate scores are retained.
    ``policy_outcome`` records why the final status was kept or changed.
    """

    threshold = _validate_threshold(threshold)
    counts = _label_counts(result)
    max_contradiction = _max_contradiction_confidence(result)
    common = {
        "decision_policy": CONFIDENCE_AWARE_POLICY,
        "contradiction_threshold": threshold,
        "label_counts": counts,
        "max_contradiction_confidence": max_contradiction,
    }

    if result.status in _NON_NLI_STATUSES or not result.per_window:
        outcome = (
            "no_evidence_windows"
            if result.status == "insufficient_evidence"
            else "preserved_non_nli_status"
        )
        return replace(
            result,
            **common,
            policy_outcome=outcome,
            abstained=result.status == "insufficient_evidence",
        )

    if counts.get("entailment", 0) and counts.get("contradiction", 0):
        return replace(
            result,
            **common,
            policy_outcome="mixed_entailment_contradiction",
            abstained=True,
        )

    if counts.get("contradiction", 0):
        assert max_contradiction is not None
        if max_contradiction < threshold:
            return replace(
                result,
                status="insufficient_evidence",
                explanation=(
                    "Strongest contradiction confidence "
                    f"{max_contradiction:.3f} is below the development-set "
                    f"threshold {threshold:.2f}; abstaining rather than "
                    f"asserting contradiction. Original decision: "
                    f"{result.explanation}"
                ),
                decision_confidence=max_contradiction,
                **common,
                policy_outcome="low_confidence_contradiction",
                abstained=True,
            )
        outcome = (
            "contradiction_with_neutral_context"
            if counts.get("neutral", 0)
            else "unanimous_contradiction"
        )
        return replace(
            result,
            **common,
            policy_outcome=outcome,
            abstained=False,
        )

    if counts.get("entailment", 0):
        outcome = (
            "entailment_with_neutral_context"
            if counts.get("neutral", 0)
            else "unanimous_entailment"
        )
        return replace(
            result,
            **common,
            policy_outcome=outcome,
            abstained=False,
        )

    return replace(
        result,
        **common,
        policy_outcome="neutral_only",
        abstained=True,
    )


def select_threshold(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose a development-set threshold without using benchmark results.

    Ranking is deterministic and favours, in order: accuracy, contradiction
    recall, fewer incorrect contradictions among expected-insufficient cases,
    fewer abstentions, and then the lower threshold. The final tie-break avoids
    unnecessary abstention when two thresholds behave equally well.
    """

    if not candidates:
        raise ValueError("at least one threshold candidate is required")
    return min(
        candidates,
        key=lambda candidate: (
            -float(candidate["accuracy"]),
            -float(candidate["contradicted_recall"]),
            float(candidate["incorrect_contradiction_rate"]),
            float(candidate["abstention_rate"]),
            float(candidate["threshold"]),
        ),
    )


class ConfidenceAwareNLIClaimVerifier(WindowedNLIClaimVerifier):
    """Windowed NLI verifier with the opt-in confidence-aware policy."""

    def __init__(
        self,
        *,
        backend: NLIPredictor,
        strategy: str = WHOLE_PASSAGE,
        adjacent_context: int = 1,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        max_length: int = 512,
        model_name: str | None = None,
    ):
        if strategy not in WINDOW_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {WINDOW_STRATEGIES}; got {strategy!r}"
            )
        super().__init__(
            backend=backend,
            strategy=strategy,
            adjacent_context=adjacent_context,
            aggregation="conservative",
            max_length=max_length,
            model_name=model_name,
        )
        self.contradiction_threshold = _validate_threshold(
            contradiction_threshold
        )

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "decision_policy": CONFIDENCE_AWARE_POLICY,
                "contradiction_threshold": self.contradiction_threshold,
            }
        )
        return metadata

    def verify(
        self, claim: Claim, evidence_passages: Mapping[str, EvidencePassage]
    ) -> WindowedNLIResult:
        result = super().verify(claim, evidence_passages)
        result = apply_confidence_policy(result, self.contradiction_threshold)
        metadata = dict(result.model_metadata)
        metadata.update(
            {
                "decision_policy": CONFIDENCE_AWARE_POLICY,
                "contradiction_threshold": self.contradiction_threshold,
            }
        )
        return replace(result, model_metadata=metadata)
