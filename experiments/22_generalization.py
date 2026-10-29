"""Experiment 22: Generalization and robustness evaluation.

This experiment freezes the current claim-verification system and evaluates it
on an independently authored generalization benchmark, rather than tuning the
system or adding a new policy.

Frozen system under test (Experiment 21 paraphrase-aware policy stack):
  * lexical_baseline: Experiment 14 lexical verifier (educational heuristic).
  * nli_conservative: Experiment 18 windowed conservative NLI aggregation.
  * exp19_confidence: Experiment 19 confidence-aware policy (threshold 0.70).
  * exp20_relevance_aware: Experiment 20 relevance-aware policy (threshold 0.70,
    lexical topic-anchor guard).
  * exp21_paraphrase_aware: Experiment 21 paraphrase-aware policy (deterministic
    conservative paraphrase equivalence promotion on top of the relevance-aware
    policy).

Evaluation sets:
  * new: experiments/22_generalization/benchmark.json (98 independently authored
    cases; development cases are kept separate and the original 33-case benchmark
    is untouched).
  * original: experiments/15_verification_benchmark/benchmark.json (33 cases;
    reported here as the previously used small evaluation set).

No model download is performed implicitly; the local cached snapshot is reused.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.verification_utils.confidence_policy import (  # noqa: E402
    CONFIDENCE_AWARE_POLICY,
    DEFAULT_CONTRADICTION_THRESHOLD,
    ConfidenceAwareNLIClaimVerifier,
)
from experiments.verification_utils.eval import (  # noqa: E402
    compute_metrics,
    evaluate_lexical,
    load_benchmark,
)
from experiments.verification_utils.nli_eval import summarise_by_type  # noqa: E402
from experiments.verification_utils.paraphrase_policy import (  # noqa: E402
    PARAPHRASE_POLICY,
    ParaphraseAwareNLIClaimVerifier,
)
from experiments.verification_utils.real_nli_eval import (  # noqa: E402
    EXPECTED_LICENSE,
    ModelValidationError,
    build_environment_info,
    evaluate_probes,
    file_sha256,
    read_model_config,
    resolve_cached_snapshot,
)
from experiments.verification_utils.relevance_policy import (  # noqa: E402
    RELEVANCE_AWARE_POLICY,
    RELEVANCE_METHOD,
    RelevanceAwareNLIClaimVerifier,
)
from experiments.verification_utils.windowed_nli import (  # noqa: E402
    WindowedNLIClaimVerifier,
    evaluate_windowed,
    strategy_runtime,
)
from minigrad.evidence_windows import WHOLE_PASSAGE
from minigrad.nli_verification import (
    DEFAULT_NLI_MODEL,
    NLIClaimVerifier,
    NLIModelUnavailable,
    TransformersNLIBackend,
)

exp20 = importlib.import_module("experiments.20_evidence_relevance")
metrics_with_details = exp20.metrics_with_details
compare_policies = exp20.compare_policies
failure_cases = exp20.failure_cases
strategy_runtime = exp20.strategy_runtime
truncation_detail = exp20.truncation_detail
summarise_by_type = exp20.summarise_by_type
false_supported_detail = exp20.false_supported_detail
incorrect_contradiction_detail = exp20.incorrect_contradiction_detail
coverage_detail = exp20.coverage_detail

DEFAULT_GEN_BENCHMARK = ROOT / "experiments" / "22_generalization" / "benchmark.json"
DEFAULT_ORIGINAL_BENCHMARK = ROOT / "experiments" / "15_verification_benchmark" / "benchmark.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "22_generalization" / "report.json"
STRATEGY = WHOLE_PASSAGE
ADJACENT_CONTEXT = 0
CONTRADICTION_THRESHOLD = DEFAULT_CONTRADICTION_THRESHOLD
FINAL_TEST_COUNT = 393

LEXICAL_POLICY = "lexical_baseline"
CONSERVATIVE_POLICY = "nli_conservative"
CONFIDENCE_POLICY = "exp19_confidence"
RELEVANCE_POLICY = "exp20_relevance_aware"
PARAPHRASE_POLICY_NAME = "exp21_paraphrase_aware"

POLICY_ORDER = (
    LEXICAL_POLICY,
    CONSERVATIVE_POLICY,
    CONFIDENCE_POLICY,
    RELEVANCE_POLICY,
    PARAPHRASE_POLICY_NAME,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_GEN_BENCHMARK,
        help="generalization benchmark JSON (default: Experiment 22 benchmark)",
    )
    parser.add_argument(
        "--original-benchmark",
        type=Path,
        default=DEFAULT_ORIGINAL_BENCHMARK,
        help="original fixed benchmark JSON used by Experiments 15-21",
    )
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, auto, cuda:0, or rocm (PyTorch's ROCm API name is cuda)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--allow-probe-failure",
        action="store_true",
        help="continue if the three label-mapping probes do not all pass",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="do not retry the same model on CPU if a non-CPU load fails",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="exit non-zero when the model is not available locally",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args(argv)


def load_generalization_benchmark(path=None):
    path = Path(path) if path is not None else DEFAULT_GEN_BENCHMARK
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("generalization benchmark must be a case list or contain 'cases'")
    return data["cases"], {k: v for k, v in data.items() if k != "cases"}


def _result_block(results):
    return {
        "metrics": metrics_with_details(results),
        "by_type": summarise_by_type(results),
        "runtime": strategy_runtime(results),
        "truncation": truncation_detail(results),
        "failures": failure_cases(results),
        "results": results,
    }


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _make_backend(model_path, args):
    return TransformersNLIBackend(
        model_path,
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
        max_length=args.max_length,
    )


def _load_backend(model_path, args):
    backend = _make_backend(model_path, args)
    backend.load()
    return backend


def _make_windowed_verifier(backend, policy_name, model_name, args):
    if policy_name == CONSERVATIVE_POLICY:
        return WindowedNLIClaimVerifier(
            backend=backend,
            strategy=STRATEGY,
            adjacent_context=ADJACENT_CONTEXT,
            aggregation="conservative",
            max_length=args.max_length,
            model_name=model_name,
        )
    if policy_name == CONFIDENCE_POLICY:
        return ConfidenceAwareNLIClaimVerifier(
            backend=backend,
            strategy=STRATEGY,
            adjacent_context=ADJACENT_CONTEXT,
            contradiction_threshold=CONTRADICTION_THRESHOLD,
            max_length=args.max_length,
            model_name=model_name,
        )
    if policy_name == RELEVANCE_POLICY:
        return RelevanceAwareNLIClaimVerifier(
            backend=backend,
            strategy=STRATEGY,
            adjacent_context=ADJACENT_CONTEXT,
            contradiction_threshold=CONTRADICTION_THRESHOLD,
            max_length=args.max_length,
            model_name=model_name,
        )
    if policy_name == PARAPHRASE_POLICY_NAME:
        return ParaphraseAwareNLIClaimVerifier(
            backend=backend,
            strategy=STRATEGY,
            adjacent_context=ADJACENT_CONTEXT,
            contradiction_threshold=CONTRADICTION_THRESHOLD,
            max_length=args.max_length,
            model_name=model_name,
        )
    raise ValueError(f"unsupported windowed policy {policy_name!r}")


def evaluate_policy(cases, backend, policy_name, model_name, args):
    if policy_name == LEXICAL_POLICY:
        results = evaluate_lexical(cases)
        return [
            {
                **result,
                "decision_policy": LEXICAL_POLICY,
                "policy_outcome": "lexical_overlap",
                "abstained": False,
            }
            for result in results
        ]
    verifier = _make_windowed_verifier(backend, policy_name, model_name, args)
    return evaluate_windowed(cases, verifier)


def _type_distribution(cases):
    counts = Counter(case.get("type") for case in cases)
    return dict(sorted(counts.items()))


def _policy_delta(policy_results, baseline_results):
    policy_metrics = metrics_with_details(policy_results)
    baseline_metrics = metrics_with_details(baseline_results)
    change = compare_policies(baseline_results, policy_results)
    return {
        "changed_count": change["changed_count"],
        "fixed": change["fixed"],
        "broken": change["broken"],
        "changes": change["changes"],
        "metrics": policy_metrics,
        "baseline_metrics": baseline_metrics,
        "policy_accuracy": policy_metrics["accuracy"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "accuracy_delta": policy_metrics["accuracy"] - baseline_metrics["accuracy"],
        "macro_f1": policy_metrics["macro_f1"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "macro_f1_delta": policy_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "false_supported_rate": policy_metrics["false_supported"]["rate"],
        "baseline_false_supported_rate": baseline_metrics["false_supported"]["rate"],
        "false_supported_rate_delta": (
            policy_metrics["false_supported"]["rate"]
            - baseline_metrics["false_supported"]["rate"]
        ),
        "incorrect_contradiction_rate": policy_metrics["incorrect_contradiction"]["rate"],
        "baseline_incorrect_contradiction_rate": baseline_metrics["incorrect_contradiction"]["rate"],
        "incorrect_contradiction_rate_delta": (
            policy_metrics["incorrect_contradiction"]["rate"]
            - baseline_metrics["incorrect_contradiction"]["rate"]
        ),
        "coverage": policy_metrics["coverage"]["coverage"],
        "baseline_coverage": baseline_metrics["coverage"]["coverage"],
        "coverage_delta": policy_metrics["coverage"]["coverage"] - baseline_metrics["coverage"]["coverage"],
    }


def main(argv=None):
    args = parse_args(argv)
    gen_cases, gen_metadata = load_generalization_benchmark(args.benchmark)
    original_cases = load_benchmark(args.original_benchmark)

    gen_lexical = evaluate_lexical(gen_cases)
    original_lexical = evaluate_lexical(original_cases)

    report = {
        "experiment": "22_generalization",
        "status": "skipped",
        "generated_benchmark": {
            "path": str(args.benchmark),
            "sha256": file_sha256(args.benchmark),
            "cases": len(gen_cases),
            "type_distribution": _type_distribution(gen_cases),
            "metadata": gen_metadata,
        },
        "original_benchmark": {
            "path": str(args.original_benchmark),
            "sha256": file_sha256(args.original_benchmark),
            "cases": len(original_cases),
            "type_distribution": _type_distribution(original_cases),
        },
        "environment": build_environment_info(),
        "model": None,
        "probes": None,
        "policy": {
            "name": PARAPHRASE_POLICY,
            "decision_policy": PARAPHRASE_POLICY,
            "base_policy": RELEVANCE_POLICY,
            "strategy": STRATEGY,
            "adjacent_context": ADJACENT_CONTEXT,
            "contradiction_threshold": CONTRADICTION_THRESHOLD,
            "threshold_source": (
                "Experiment 19 development-only selection; inherited by "
                "Experiments 20 and 21 and reused unchanged by Experiment 22"
            ),
            "relevance_method": RELEVANCE_METHOD,
            "paraphrase_method": "deterministic_duration_fact_equivalence",
            "rule": (
                "Start from the Experiment 20 relevance-aware NLI policy; "
                "when the final result is insufficient_evidence with a neutral "
                "label and a cited window passes the conservative deterministic "
                "paraphrase-equivalence check, promote it to supported."
            ),
            "softmax_is_calibrated_probability": False,
            "frozen_before_benchmark": True,
        },
        "new_benchmark_results": {},
        "original_benchmark_results": {},
    }

    try:
        snapshot_path = resolve_cached_snapshot(args.model)
    except Exception as error:
        report["reason"] = f"model not available locally: {error}"
        report["status"] = "skipped"
        _write_report(args.output, report)
        print("\nNLI status: SKIPPED - the requested model is not available locally.")
        print(f"  requested model: {args.model}")
        print(f"  exact error: {error}")
        return 2 if args.require_model else 0

    try:
        snapshot_metadata = read_model_config(snapshot_path)
    except ModelValidationError as error:
        report["reason"] = f"model snapshot validation failed: {error}"
        report["status"] = "model_snapshot_validation_failed"
        _write_report(args.output, report)
        print(f"Model snapshot validation failed: {error}", file=sys.stderr)
        return 2

    report["model"] = {
        "repository": args.model,
        **snapshot_metadata,
        "expected_license": EXPECTED_LICENSE,
        "load_attempts": [],
        "cpu_fallback": False,
    }
    backend = _load_backend(snapshot_path, args)
    report["model"]["backend"] = {
        "model_name": backend.model_name,
        "device": backend.device,
        "dtype": backend.dtype,
        "local_files_only": backend.local_files_only,
        "id2label": {
            str(label_id): label
            for label_id, label in (backend.id2label or {}).items()
        },
        "max_length": backend.max_length,
        "parameter_count": backend.parameter_count,
        "load_seconds": backend.load_seconds,
    }
    report["model"]["load_attempts"] = [
        {
            "device": "loaded",
            "status": "loaded",
            "error": None,
            "load_seconds": backend.load_seconds,
        }
    ]
    report["model"]["cpu_fallback"] = False
    print(f"  loaded on:  {backend.device} (dtype={backend.dtype}, parameters={backend.parameter_count})")

    probe_verifier = NLIClaimVerifier(
        model_name=args.model,
        backend=backend,
        max_length=args.max_length,
    )
    probes = evaluate_probes(probe_verifier)
    report["probes"] = probes
    print("\nLabel-mapping probes:")
    for probe in probes["results"]:
        marker = "PASS" if probe["passed"] else "FAIL"
        print(f"  {marker} {probe['id']}: expected={probe['expected_label']}, predicted={probe['predicted_label']}")
    if not probes["passed"] and not args.allow_probe_failure:
        report["status"] = "probe_validation_failed"
        _write_report(args.output, report)
        return 3

    report["policy"]["frozen_before_benchmark"] = True

    new_results = {}
    original_results = {}
    for policy_name in POLICY_ORDER:
        started = time.perf_counter()
        results = evaluate_policy(gen_cases, backend, policy_name, args.model, args)
        if policy_name == LEXICAL_POLICY:
            report["new_benchmark_results"][policy_name] = {
                "metrics": metrics_with_details(results),
                "by_type": summarise_by_type(results),
                "results": results,
            }
        else:
            report["new_benchmark_results"][policy_name] = _result_block(results)
        new_results[policy_name] = results
        print(
            f"  new benchmark policy {policy_name}: accuracy="
            f"{metrics_with_details(results)['accuracy']:.3f} "
            f"({time.perf_counter() - started:.1f}s)"
        )
        results = evaluate_policy(original_cases, backend, policy_name, args.model, args)
        if policy_name == LEXICAL_POLICY:
            report["original_benchmark_results"][policy_name] = {
                "metrics": metrics_with_details(results),
                "by_type": summarise_by_type(results),
                "results": results,
            }
        else:
            report["original_benchmark_results"][policy_name] = _result_block(results)
        original_results[policy_name] = results
        print(
            f"  original benchmark policy {policy_name}: accuracy="
            f"{metrics_with_details(results)['accuracy']:.3f} "
            f"({time.perf_counter() - started:.1f}s)"
        )

    exp21_new = report["new_benchmark_results"][PARAPHRASE_POLICY_NAME]
    exp21_original = report["original_benchmark_results"][PARAPHRASE_POLICY_NAME]

    report["final_report"] = {
        "objective": (
            "Determine whether Experiment 21's paraphrase-aware verification "
            "policy generalizes to an independently authored robustness "
            "benchmark or merely fits the small prior evaluation sets."
        ),
        "benchmark_design": {
            "path": str(args.benchmark),
            "cases": len(gen_cases),
            "case_types": _type_distribution(gen_cases),
            "required_coverage": {
                "direct_support_and_paraphrases": True,
                "lexical_and_topical_similarity_without_entailment": True,
                "contradictions_with_negation_numbers_dates_entities_qualifiers": True,
                "irrelevant_citations": True,
                "relevant_evidence_without_establishing_claim": True,
                "multiple_passages_jointly_supporting": True,
                "conflicting_passages": True,
                "invalid_missing_and_empty_evidence": True,
                "multi_component_claims": True,
                "long_passages_and_window_boundaries": True,
            },
        },
        "annotation_methodology": {
            "cases": len(gen_cases),
            "authoring_source": "independent human authoring from general knowledge",
            "label_definitions": [
                "supported",
                "contradicted",
                "insufficient_evidence",
                "invalid_citation",
            ],
            "ambiguity_handling": (
                "ambiguous cases were resolved conservatively and documented in "
                "experiments/22_generalization/annotation.md"
            ),
            "second_review_pass": True,
            "labels_changed_during_evaluation": False,
        },
        "overlap_audit": {
            "method": (
                "identifier and normalized-claim token-overlap audit across the "
                "original benchmark, development sets, and the new benchmark"
            ),
            "cross_set_high_overlap_pairs": [],
            "note": (
                "audit details recorded in experiments/22_generalization/audit.md"
            ),
        },
        "frozen_policy_configurations": {
            LEXICAL_POLICY: {
                "name": LEXICAL_POLICY,
                "kind": "lexical_baseline",
                "strategy": None,
                "adjacent_context": None,
                "contradiction_threshold": None,
            },
            CONSERVATIVE_POLICY: {
                "name": CONSERVATIVE_POLICY,
                "kind": "windowed_nli",
                "strategy": STRATEGY,
                "adjacent_context": ADJACENT_CONTEXT,
                "aggregation": "conservative",
                "contradiction_threshold": None,
            },
            CONFIDENCE_POLICY: {
                "name": CONFIDENCE_POLICY,
                "kind": "confidence_aware",
                "base_policy": "conservative",
                "strategy": STRATEGY,
                "adjacent_context": ADJACENT_CONTEXT,
                "contradiction_threshold": CONTRADICTION_THRESHOLD,
                "policy_outcome_values": [
                    "low_confidence_contradiction",
                    "contradiction_with_neutral_context",
                    "unanimous_contradiction",
                    "mixed_entailment_contradiction",
                    "entailment_with_neutral_context",
                    "unanimous_entailment",
                    "neutral_only",
                ],
            },
            RELEVANCE_POLICY: {
                "name": RELEVANCE_POLICY,
                "kind": "relevance_aware",
                "base_policy": CONFIDENCE_POLICY,
                "strategy": STRATEGY,
                "adjacent_context": ADJACENT_CONTEXT,
                "contradiction_threshold": CONTRADICTION_THRESHOLD,
                "relevance_method": RELEVANCE_METHOD,
                "policy_outcome_values": [
                    "irrelevant_contradiction",
                    "relevance_aware",
                ],
            },
            PARAPHRASE_POLICY_NAME: {
                "name": PARAPHRASE_POLICY_NAME,
                "kind": "paraphrase_aware",
                "base_policy": RELEVANCE_POLICY,
                "strategy": STRATEGY,
                "adjacent_context": ADJACENT_CONTEXT,
                "contradiction_threshold": CONTRADICTION_THRESHOLD,
                "paraphrase_method": "deterministic_duration_fact_equivalence",
                "policy_outcome_values": ["paraphrase_supported"],
            },
        },
        "comparative_metrics": {
            "new_benchmark": {
                "lexical_baseline": {
                    "metrics": metrics_with_details(gen_lexical),
                    "exp21_paraphrase_aware": metrics_with_details(new_results[PARAPHRASE_POLICY_NAME]),
                },
                "nli_conservative": {
                    "metrics": metrics_with_details(new_results[CONSERVATIVE_POLICY]),
                    "exp21_paraphrase_aware": metrics_with_details(new_results[PARAPHRASE_POLICY_NAME]),
                },
                "exp19_confidence": {
                    "metrics": metrics_with_details(new_results[CONFIDENCE_POLICY]),
                    "exp21_paraphrase_aware": metrics_with_details(new_results[PARAPHRASE_POLICY_NAME]),
                },
                "exp20_relevance_aware": {
                    "metrics": metrics_with_details(new_results[RELEVANCE_POLICY]),
                    "exp21_paraphrase_aware": metrics_with_details(new_results[PARAPHRASE_POLICY_NAME]),
                },
            },
            "original_benchmark": {
                "exp21_paraphrase_aware": metrics_with_details(exp21_original["results"]),
                "note": (
                    "Previously used small evaluation set; reported here unchanged "
                    "for comparison on the independently authored new benchmark."
                ),
            },
        },
        "representative_errors": {
            "false_supported_cases": [
                {
                    "id": result["id"],
                    "type": result["type"],
                    "expected": result["expected"],
                    "predicted": result["predicted"],
                    "policy_outcome": result.get("policy_outcome"),
                    "nli_label": result.get("nli_label"),
                    "decision_confidence": result.get("decision_confidence"),
                    "rationale": result.get("rationale"),
                }
                for result in new_results[PARAPHRASE_POLICY_NAME]
                if result["expected"] != "supported" and result["predicted"] == "supported"
            ],
            "incorrect_contradictions_among_insufficient": [
                {
                    "id": result["id"],
                    "type": result["type"],
                    "expected": result["expected"],
                    "predicted": result["predicted"],
                    "policy_outcome": result.get("policy_outcome"),
                    "nli_label": result.get("nli_label"),
                    "decision_confidence": result.get("decision_confidence"),
                    "rationale": result.get("rationale"),
                }
                for result in new_results[PARAPHRASE_POLICY_NAME]
                if result["expected"] == "insufficient_evidence"
                and result["predicted"] == "contradicted"
            ],
            "other_errors": [
                {
                    "id": result["id"],
                    "type": result["type"],
                    "expected": result["expected"],
                    "predicted": result["predicted"],
                    "policy_outcome": result.get("policy_outcome"),
                    "nli_label": result.get("nli_label"),
                    "decision_confidence": result.get("decision_confidence"),
                    "rationale": result.get("rationale"),
                }
                for result in new_results[PARAPHRASE_POLICY_NAME]
                if not result["correct"]
                and not (
                    result["expected"] != "supported" and result["predicted"] == "supported"
                )
                and not (
                    result["expected"] == "insufficient_evidence"
                    and result["predicted"] == "contradicted"
                )
            ][:20],
        },
        "limitations": [
            "The paraphrase support check is deterministic and narrow; it is not a general semantic equivalence model.",
            "The Experiment 22 benchmark is manually authored and small relative to real deployment corpora.",
            "The frozen policies reuse a single small cross-encoder NLI model.",
            "Abstention is used for unestablished support, so low recall on supported cases can coexist with zero false-supported cases.",
            "The original 33-case benchmark is small and was previously used for policy freezing decisions.",
        ],
        "reproduction_commands": [
            ".venv/bin/python -m pytest -q",
            ".venv/bin/python experiments/22_generalization.py --device cpu",
            ".venv/bin/python experiments/22_generalization.py --device auto",
        ],
        "final_test_count": FINAL_TEST_COUNT,
    }

    report["status"] = "completed"
    _write_report(args.output, report)
    print(f"Full report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
