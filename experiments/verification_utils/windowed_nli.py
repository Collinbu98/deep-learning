"""Windowed NLI verification for Experiment 18.

The existing :class:`~minigrad.nli_verification.NLIClaimVerifier` remains the
whole-passage baseline. This module turns cited passages into traceable
evidence windows and delegates prediction, label validation, conservative
aggregation, and error handling to the existing verifier.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from experiments.verification_utils.eval import make_passage
from minigrad.claim_verification import Claim, check_citation_validity
from minigrad.evidence_windows import (
    SENTENCE,
    WINDOW_STRATEGIES,
    EvidenceWindow,
    build_windows_for_passages,
)
from minigrad.llm_answerer import EvidencePassage
from minigrad.nli_verification import (
    NLIClaimVerifier,
    NLIModelUnavailable,
    NLIPrediction,
    NLIPredictor,
)

__all__ = [
    "WindowPrediction",
    "WindowedNLIResult",
    "WindowedNLIClaimVerifier",
    "evaluate_windowed",
    "truncation_summary",
]


@dataclass(frozen=True)
class WindowPrediction:
    """One raw NLI prediction tied to its source evidence window."""

    window_id: str
    source_id: str
    doc_id: str
    chunk_index: int
    strategy: str
    start_sentence: int
    end_sentence: int
    char_start: int
    char_end: int
    text: str
    label: str
    label_id: int
    scores: dict[str, float]
    raw_scores: tuple[float, ...]
    model_name: str
    device: str


@dataclass(frozen=True)
class WindowedNLIResult:
    """A claim decision plus all raw per-window predictions."""

    claim: Claim
    status: str
    explanation: str
    evidence_excerpts: tuple[tuple[str, str], ...]
    cited_passages_exist: bool
    label: str | None
    aggregate_scores: dict[str, float]
    decision_confidence: float | None
    per_window: tuple[WindowPrediction, ...]
    aggregation: str
    strategy: str
    adjacent_context: int
    model_metadata: dict[str, Any]
    error: str | None = None
    decision_policy: str = "conservative"
    policy_outcome: str = "conservative_aggregation"
    contradiction_threshold: float | None = None
    label_counts: dict[str, int] = field(default_factory=dict)
    max_contradiction_confidence: float | None = None
    abstained: bool = False
    policy_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": asdict(self.claim),
            "status": self.status,
            "explanation": self.explanation,
            "evidence_excerpts": [list(item) for item in self.evidence_excerpts],
            "cited_passages_exist": self.cited_passages_exist,
            "label": self.label,
            "aggregate_scores": dict(self.aggregate_scores),
            "decision_confidence": self.decision_confidence,
            "per_window": [asdict(item) for item in self.per_window],
            "aggregation": self.aggregation,
            "strategy": self.strategy,
            "adjacent_context": self.adjacent_context,
            "model_metadata": dict(self.model_metadata),
            "error": self.error,
            "decision_policy": self.decision_policy,
            "policy_outcome": self.policy_outcome,
            "contradiction_threshold": self.contradiction_threshold,
            "label_counts": dict(self.label_counts),
            "max_contradiction_confidence": self.max_contradiction_confidence,
            "abstained": self.abstained,
            "policy_details": dict(self.policy_details),
        }


def _window_record(
    window: EvidenceWindow, prediction: NLIPrediction
) -> WindowPrediction:
    return WindowPrediction(
        window_id=window.window_id,
        source_id=window.source_id,
        doc_id=window.doc_id,
        chunk_index=window.chunk_index,
        strategy=window.strategy,
        start_sentence=window.start_sentence,
        end_sentence=window.end_sentence,
        char_start=window.char_start,
        char_end=window.char_end,
        text=window.text,
        label=prediction.label,
        label_id=prediction.label_id,
        scores=dict(prediction.scores),
        raw_scores=tuple(prediction.raw_scores),
        model_name=prediction.model_name,
        device=prediction.device,
    )


class WindowedNLIClaimVerifier:
    """Verify claims against sentence or adjacent-sentence evidence windows."""

    def __init__(
        self,
        *,
        backend: NLIPredictor,
        strategy: str = SENTENCE,
        adjacent_context: int = 1,
        aggregation: str = "conservative",
        max_length: int = 512,
        model_name: str | None = None,
    ):
        if strategy not in WINDOW_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {WINDOW_STRATEGIES}; got {strategy!r}"
            )
        if adjacent_context < 0:
            raise ValueError("adjacent_context must be non-negative")
        self.backend = backend
        self.strategy = strategy
        self.adjacent_context = adjacent_context
        self.max_length = max_length
        self.aggregation = aggregation
        self.model_name = model_name or getattr(
            backend, "model_name", "injected-nli"
        )
        self.nli_verifier = NLIClaimVerifier(
            model_name=self.model_name,
            backend=backend,
            max_length=max_length,
            aggregation=aggregation,
        )

    @property
    def available(self) -> bool:
        return self.nli_verifier.available

    def is_available(self) -> bool:
        return self.available

    def load(self):
        return self.nli_verifier.load()

    def metadata(self) -> dict[str, Any]:
        metadata = self.nli_verifier.metadata()
        metadata.update(
            {
                "window_strategy": self.strategy,
                "adjacent_context": self.adjacent_context,
            }
        )
        return metadata

    def _empty_result(
        self,
        claim: Claim,
        status: str,
        explanation: str,
        *,
        cited_passages_exist: bool,
        excerpts: tuple[tuple[str, str], ...] = (),
        error: str | None = None,
    ) -> WindowedNLIResult:
        return WindowedNLIResult(
            claim=claim,
            status=status,
            explanation=explanation,
            evidence_excerpts=excerpts,
            cited_passages_exist=cited_passages_exist,
            label=None,
            aggregate_scores={},
            decision_confidence=None,
            per_window=(),
            aggregation=self.aggregation,
            strategy=self.strategy,
            adjacent_context=self.adjacent_context,
            model_metadata=self.metadata(),
            error=error,
        )

    def verify(
        self, claim: Claim, evidence_passages: Mapping[str, EvidencePassage]
    ) -> WindowedNLIResult:
        """Validate citations, build windows, and delegate NLI prediction."""

        supplied_ids = set(evidence_passages.keys())
        all_exist, existing_ids = check_citation_validity(claim, supplied_ids)

        if not claim.citation_ids:
            return self._empty_result(
                claim,
                "insufficient_evidence",
                "No citations provided.",
                cited_passages_exist=False,
            )
        if not all_exist:
            missing = [
                citation_id
                for citation_id in claim.citation_ids
                if citation_id not in supplied_ids
            ]
            return self._empty_result(
                claim,
                "invalid_citation",
                f"Invalid/missing citation IDs: {missing}",
                cited_passages_exist=False,
            )

        unique_ids = tuple(dict.fromkeys(existing_ids))
        cited = {
            citation_id: evidence_passages[citation_id]
            for citation_id in unique_ids
            if citation_id in evidence_passages
        }
        if not cited:
            return self._empty_result(
                claim,
                "insufficient_evidence",
                "Cited passages not found in supplied evidence.",
                cited_passages_exist=False,
            )

        excerpts = tuple(
            (
                citation_id,
                passage.text[:200] + ("..." if len(passage.text) > 200 else ""),
            )
            for citation_id, passage in cited.items()
        )
        windows = build_windows_for_passages(
            cited,
            self.strategy,
            adjacent_context=self.adjacent_context,
        )
        if not windows:
            return self._empty_result(
                claim,
                "insufficient_evidence",
                "No non-empty evidence windows were available.",
                cited_passages_exist=True,
                excerpts=excerpts,
            )

        window_passages = {
            window.window_id: EvidencePassage(
                source_id=window.window_id,
                doc_id=window.doc_id,
                chunk_index=window.chunk_index,
                source=f"{window.doc_id}.txt",
                text=window.text,
                score=1.0,
                rank=index + 1,
            )
            for index, window in enumerate(windows)
        }
        window_claim = Claim(
            text=claim.text,
            citation_ids=tuple(window_passages),
            sentence_index=claim.sentence_index,
            claim_index=claim.claim_index,
        )
        result = self.nli_verifier.verify(window_claim, window_passages)
        windows_by_id = {window.window_id: window for window in windows}
        per_window = tuple(
            _window_record(windows_by_id[prediction.evidence_id], prediction)
            for prediction in result.per_evidence
        )
        metadata = dict(result.model_metadata)
        metadata.update(
            {
                "window_strategy": self.strategy,
                "adjacent_context": self.adjacent_context,
                "windows_scored": len(per_window),
            }
        )
        return WindowedNLIResult(
            claim=claim,
            status=result.status,
            explanation=result.explanation,
            evidence_excerpts=excerpts,
            cited_passages_exist=result.cited_passages_exist,
            label=result.label,
            aggregate_scores=dict(result.aggregate_scores),
            decision_confidence=result.decision_confidence,
            per_window=per_window,
            aggregation=self.aggregation,
            strategy=self.strategy,
            adjacent_context=self.adjacent_context,
            model_metadata=metadata,
            error=result.error,
        )


def truncation_summary(
    verifier: WindowedNLIClaimVerifier,
    claim_text: str,
    windows: Sequence[WindowPrediction],
) -> dict[str, Any]:
    """Check whether any scored window exceeds the model's input limit."""

    counter = getattr(verifier.backend, "input_token_counts", None)
    if not callable(counter):
        return {
            "checked": False,
            "reason": "backend does not expose input_token_counts",
        }
    if not windows:
        return {"checked": True, "max_length": verifier.max_length, "windows": 0}
    pairs = [(window.text, claim_text) for window in windows]
    try:
        counts = list(counter(pairs))
    except Exception as error:
        return {
            "checked": False,
            "reason": f"token counting failed: {error}",
        }
    truncated = [
        window.window_id
        for window, count in zip(windows, counts)
        if count > verifier.max_length
    ]
    return {
        "checked": True,
        "max_length": verifier.max_length,
        "max_input_tokens": max(counts, default=0),
        "truncated_windows": len(truncated),
        "truncated_window_ids": truncated,
    }


def evaluate_windowed(cases, verifier: WindowedNLIClaimVerifier) -> list[dict]:
    """Run one window strategy over benchmark cases and retain raw windows."""

    results = []
    for case in cases:
        passages = {
            source_id: make_passage(source_id, text)
            for source_id, text in case["passages"].items()
        }
        claim = Claim(
            text=case["claim"],
            citation_ids=tuple(case["citations"]),
        )
        started = time.perf_counter()
        result = verifier.verify(claim, passages)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "id": case["id"],
                "type": case["type"],
                "claim": case["claim"],
                "expected": case["expected"],
                "predicted": result.status,
                "correct": result.status == case["expected"],
                "explanation": result.explanation,
                "rationale": case["rationale"],
                "nli_label": result.label,
                "aggregate_scores": dict(result.aggregate_scores),
                "decision_confidence": result.decision_confidence,
                "per_window": [asdict(window) for window in result.per_window],
                "windows_scored": len(result.per_window),
                "aggregation": result.aggregation,
                "strategy": result.strategy,
                "adjacent_context": result.adjacent_context,
                "model_metadata": dict(result.model_metadata),
                "truncation": truncation_summary(
                    verifier, claim.text, result.per_window
                ),
                "error": result.error,
                "decision_policy": result.decision_policy,
                "policy_outcome": result.policy_outcome,
                "contradiction_threshold": result.contradiction_threshold,
                "label_counts": dict(result.label_counts),
                "max_contradiction_confidence": (
                    result.max_contradiction_confidence
                ),
                "abstained": result.abstained,
                "policy_details": dict(result.policy_details),
                "runtime_seconds": elapsed,
            }
        )
    return results


def strategy_runtime(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise runtime and windows scored for one strategy."""

    total_seconds = sum(row["runtime_seconds"] for row in results)
    windows = sum(row["windows_scored"] for row in results)
    cases = len(results)
    return {
        "cases": cases,
        "inference_seconds": total_seconds,
        "milliseconds_per_case": (
            1000.0 * total_seconds / cases if cases else 0.0
        ),
        "windows_scored": windows,
        "average_windows_per_claim": windows / cases if cases else 0.0,
        "max_windows_per_claim": max(
            (row["windows_scored"] for row in results), default=0
        ),
    }


def find_strategy_disagreements(
    results_by_strategy: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict]:
    """Return cases where any two strategies chose different statuses."""

    by_strategy = {
        strategy: {row["id"]: row for row in rows}
        for strategy, rows in results_by_strategy.items()
    }
    case_ids = []
    for rows in results_by_strategy.values():
        for row in rows:
            if row["id"] not in case_ids:
                case_ids.append(row["id"])

    disagreements = []
    for case_id in case_ids:
        statuses = {}
        expected = None
        case_type = None
        for strategy, rows in by_strategy.items():
            row = rows.get(case_id)
            if row is None:
                continue
            statuses[strategy] = row["predicted"]
            expected = row["expected"]
            case_type = row["type"]
        if len(set(statuses.values())) <= 1:
            continue
        disagreements.append(
            {
                "id": case_id,
                "type": case_type,
                "expected": expected,
                "statuses": statuses,
                "correct_strategies": [
                    strategy
                    for strategy, status in statuses.items()
                    if status == expected
                ],
            }
        )
    return disagreements
