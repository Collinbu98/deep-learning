"""Reusable evaluation helpers for the Experiment 16 NLI benchmark.

These helpers do not import Transformers and do not load a model. They accept
an already constructed verifier, which may wrap a real model or an injected
fake. Metrics reuse the Experiment 15 implementation and the same four
benchmark labels, so lexical and NLI results are directly comparable.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict

from minigrad.claim_verification import Claim

from experiments.verification_utils.eval import make_passage

__all__ = [
    "evaluate_nli",
    "summarise_by_type",
    "find_disagreements",
]


def _prediction_record(prediction) -> dict:
    record = asdict(prediction)
    record["raw_scores"] = list(record["raw_scores"])
    return record


def evaluate_nli(cases, verifier) -> list[dict]:
    """Run the NLI verifier over benchmark cases and return inspectable rows."""

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
                "per_evidence": [
                    _prediction_record(prediction)
                    for prediction in result.per_evidence
                ],
                "aggregation": result.aggregation,
                "model_metadata": dict(result.model_metadata),
                "error": result.error,
                "runtime_seconds": elapsed,
            }
        )
    return results


def summarise_by_type(results) -> dict[str, dict]:
    """Compute accuracy and counts for each benchmark case type."""

    grouped = defaultdict(list)
    for result in results:
        grouped[result["type"]].append(result)
    summary = {}
    for case_type in sorted(grouped):
        rows = grouped[case_type]
        correct = sum(1 for row in rows if row["correct"])
        summary[case_type] = {
            "total": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
        }
    return summary


def find_disagreements(lexical_results, nli_results) -> list[dict]:
    """Return cases where the lexical baseline and NLI chose different statuses."""

    lexical_by_id = {result["id"]: result for result in lexical_results}
    disagreements = []
    for nli_result in nli_results:
        lexical_result = lexical_by_id.get(nli_result["id"])
        if lexical_result is None:
            continue
        if lexical_result["predicted"] == nli_result["predicted"]:
            continue
        disagreements.append(
            {
                "id": nli_result["id"],
                "type": nli_result["type"],
                "claim": nli_result["claim"],
                "expected": nli_result["expected"],
                "lexical_predicted": lexical_result["predicted"],
                "nli_predicted": nli_result["predicted"],
                "nli_label": nli_result["nli_label"],
                "aggregate_scores": nli_result["aggregate_scores"],
                "decision_confidence": nli_result["decision_confidence"],
                "rationale": nli_result["rationale"],
                "lexical_explanation": lexical_result["explanation"],
                "nli_explanation": nli_result["explanation"],
            }
        )
    return disagreements
