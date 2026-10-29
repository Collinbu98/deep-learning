import importlib
import re
import types

import pytest

from experiments import verification_utils  # noqa: F401
from experiments.verification_utils.eval import (
    compute_metrics,
    evaluate_lexical,
    load_benchmark,
)
from experiments.verification_utils.nli_eval import summarise_by_type
from minigrad.nli_verification import FakeNLIBackend

exp22 = importlib.import_module("experiments.22_generalization")

VALID_LABELS = {"supported", "contradicted", "insufficient_evidence", "invalid_citation"}
KNOWN_TYPES = {
    "direct",
    "paraphrase",
    "high_overlap_wrong",
    "negation",
    "contradiction",
    "irrelevant_citation",
    "missing_citation",
    "invalid_citation",
    "insufficient",
    "multi_component",
    "conflicting_evidence",
    "long_passage",
}
ABSTAIN_STATUS = "insufficient_evidence"


def _claim_key(text: str):
    return tuple(sorted(re.sub(r"[^a-z0-9 ]", "", text.lower()).split()))


def test_generalization_benchmark_loads():
    cases, metadata = exp22.load_generalization_benchmark()
    assert exp22.DEFAULT_GEN_BENCHMARK.exists()
    assert isinstance(cases, list)
    assert 90 <= len(cases) <= 150
    assert isinstance(metadata, dict)


def test_generalization_benchmark_case_fields_and_ids():
    cases, _ = exp22.load_generalization_benchmark()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert case["type"] in KNOWN_TYPES
        assert case["expected"] in VALID_LABELS
        assert isinstance(case["claim"], str) and case["claim"]
        assert isinstance(case["citations"], list)
        assert isinstance(case["passages"], dict)
        assert isinstance(case["rationale"], str) and case["rationale"]
        if case["citations"]:
            assert case["passages"]


def test_generalization_benchmark_expected_labels_valid():
    cases, _ = exp22.load_generalization_benchmark()
    assert all(case["expected"] in VALID_LABELS for case in cases)


def test_generalization_benchmark_types_are_known():
    cases, _ = exp22.load_generalization_benchmark()
    assert all(case["type"] in KNOWN_TYPES for case in cases)


def test_generalization_benchmark_has_no_cross_set_claim_duplicates_with_original():
    new_cases, _ = exp22.load_generalization_benchmark()
    original_cases = load_benchmark()
    original_keys = {_claim_key(case["claim"]) for case in original_cases}
    for case in new_cases:
        assert _claim_key(case["claim"]) not in original_keys


def test_generalization_metric_calculations_accuracy_macro_f1_and_false_supported():
    results = [
        {"correct": True, "expected": "supported", "predicted": "supported"},
        {"correct": False, "expected": "contradicted", "predicted": "supported"},
        {"correct": True, "expected": "invalid_citation", "predicted": "invalid_citation"},
    ]
    metrics = exp22.compute_metrics(results)
    assert metrics["total"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["false_supported_rate"] == pytest.approx(0.5)
    assert metrics["macro_f1"] > 0.0
    assert "confusion_matrix" in metrics
    assert "per_class" in metrics


def test_generalization_metric_details_incorrect_contradiction_and_coverage():
    results = [
        {
            "id": "g1",
            "type": "direct",
            "expected": "supported",
            "predicted": "supported",
            "correct": True,
        },
        {
            "id": "g2",
            "type": "insufficient",
            "expected": "insufficient_evidence",
            "predicted": "contradicted",
            "correct": False,
        },
    ]
    metrics = exp22.metrics_with_details(results)
    assert metrics["incorrect_contradiction"]["denominator"] == 1
    assert metrics["incorrect_contradiction"]["case_ids"] == ["g2"]
    assert metrics["coverage"]["abstention_rate"] == pytest.approx(0.0)


def test_generalization_by_type_summary_is_consistent():
    cases, _ = exp22.load_generalization_benchmark()
    results = evaluate_lexical(cases)
    summary = summarise_by_type(results)
    assert len(summary) > 0
    for case_type, values in summary.items():
        assert values["total"] >= 1
        assert values["accuracy"] == pytest.approx(values["correct"] / values["total"])


def test_frozen_policy_evaluation_plumbing_is_offline_and_deterministic():
    cases, _ = exp22.load_generalization_benchmark()
    backend = FakeNLIBackend()
    args = types.SimpleNamespace(max_length=512)
    first = exp22.evaluate_policy(cases[:4], backend, "exp21_paraphrase_aware", "fake-nli", args)
    second = exp22.evaluate_policy(cases[:4], backend, "exp21_paraphrase_aware", "fake-nli", args)
    assert [row["predicted"] for row in first] == [row["predicted"] for row in second]
    assert all(row["predicted"] in VALID_LABELS for row in first)
    assert all(row.get("abstained") is not None for row in first)


def test_frozen_policy_registry_contains_expected_policies():
    assert len(exp22.POLICY_ORDER) == 5
    assert "exp21_paraphrase_aware" in exp22.POLICY_ORDER
    assert "lexical_baseline" in exp22.POLICY_ORDER
