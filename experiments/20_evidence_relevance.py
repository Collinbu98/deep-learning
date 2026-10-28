"""Experiment 20: separate lexical evidence relevance from contradiction.

Experiment 19 showed that a confidence threshold cannot repair high-confidence
false contradictions caused by citing evidence about another subject. This
experiment keeps the Experiment 19 whole-passage strategy and frozen 0.70
confidence threshold, then adds a small opt-in lexical topic-anchor guard. The
guard only demotes a contradiction when no contradiction window shares a
proper-noun anchor with the claim.

The fixed Experiment 19 development set is used for the inherited policy, and
a small relevance-focused extension was written before the Experiment 20 policy
was evaluated. The policy is frozen before the fixed Experiment 15 benchmark is
run. Nothing is downloaded implicitly and no benchmark label is modified.
"""
from __future__ import annotations

import argparse
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
from minigrad.evidence_windows import WHOLE_PASSAGE  # noqa: E402
from minigrad.nli_verification import (  # noqa: E402
    DEFAULT_NLI_MODEL,
    NLIModelUnavailable,
    NLIClaimVerifier,
    TransformersNLIBackend,
)

DEFAULT_DEV_SET = ROOT / "experiments" / "19_confidence_policy" / "dev_set.json"
DEFAULT_RELEVANCE_DEV = (
    ROOT / "experiments" / "20_evidence_relevance" / "dev_relevance.json"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "20_evidence_relevance" / "report.json"
STRATEGY = WHOLE_PASSAGE
ADJACENT_CONTEXT = 0
FROZEN_THRESHOLD = DEFAULT_CONTRADICTION_THRESHOLD


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-set", type=Path, default=DEFAULT_DEV_SET)
    parser.add_argument(
        "--relevance-dev", type=Path, default=DEFAULT_RELEVANCE_DEV
    )
    parser.add_argument("--benchmark", type=Path, default=None)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def load_development_set(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("development set must be a case list or contain 'cases'")
    return data["cases"], {
        key: value for key, value in data.items() if key != "cases"
    }


def combine_development_sets(primary_path, relevance_path):
    primary_cases, primary_metadata = load_development_set(primary_path)
    relevance_cases, relevance_metadata = load_development_set(relevance_path)
    cases = [*primary_cases, *relevance_cases]
    ids = [case["id"] for case in cases]
    duplicates = sorted(
        case_id for case_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate development case IDs: {duplicates}")
    return cases, {
        "primary": {
            "path": str(primary_path),
            "candidates": len(primary_cases),
            "metadata": primary_metadata,
        },
        "relevance_extension": {
            "path": str(relevance_path),
            "candidates": len(relevance_cases),
            "metadata": relevance_metadata,
        },
    }


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _make_backend(model_path, args, device):
    return TransformersNLIBackend(
        model_path,
        device=device,
        dtype=args.dtype,
        local_files_only=True,
        max_length=args.max_length,
    )


def _load_backend(model_path, args):
    attempts = []
    backend = _make_backend(model_path, args, args.device)
    started = time.perf_counter()
    try:
        backend.load()
        attempts.append(
            {
                "device": args.device,
                "status": "loaded",
                "error": None,
                "load_seconds": time.perf_counter() - started,
            }
        )
        return backend, attempts
    except NLIModelUnavailable as error:
        attempts.append(
            {
                "device": args.device,
                "status": "failed",
                "error": str(error),
                "load_seconds": time.perf_counter() - started,
            }
        )
        if args.device == "cpu" or args.no_cpu_fallback:
            raise

    fallback = _make_backend(model_path, args, "cpu")
    started = time.perf_counter()
    try:
        fallback.load()
    except NLIModelUnavailable as error:
        attempts.append(
            {
                "device": "cpu",
                "status": "failed",
                "error": str(error),
                "load_seconds": time.perf_counter() - started,
            }
        )
        raise
    attempts.append(
        {
            "device": "cpu",
            "status": "loaded",
            "error": None,
            "load_seconds": time.perf_counter() - started,
        }
    )
    return fallback, attempts


def false_supported_detail(results):
    not_supported = [
        result for result in results if result["expected"] != "supported"
    ]
    false_supported = [
        result for result in not_supported if result["predicted"] == "supported"
    ]
    return {
        "count": len(false_supported),
        "denominator": len(not_supported),
        "rate": (
            len(false_supported) / len(not_supported) if not_supported else 0.0
        ),
        "case_ids": [result["id"] for result in false_supported],
    }


def incorrect_contradiction_detail(results):
    insufficient = [
        result
        for result in results
        if result["expected"] == "insufficient_evidence"
    ]
    incorrect = [
        result
        for result in insufficient
        if result["predicted"] == "contradicted"
    ]
    return {
        "count": len(incorrect),
        "denominator": len(insufficient),
        "rate": len(incorrect) / len(insufficient) if insufficient else 0.0,
        "case_ids": [result["id"] for result in incorrect],
    }


def coverage_detail(results):
    definitive = [
        result
        for result in results
        if result["predicted"] in {"supported", "contradicted", "invalid_citation"}
    ]
    abstained = [
        result
        for result in results
        if result["predicted"] == "insufficient_evidence"
    ]
    errors = [
        result
        for result in results
        if result["predicted"] in {"backend_unavailable", "verification_error"}
    ]
    total = len(results)
    return {
        "total": total,
        "definitive": len(definitive),
        "abstained": len(abstained),
        "errors": len(errors),
        "coverage": len(definitive) / total if total else 0.0,
        "abstention_rate": len(abstained) / total if total else 0.0,
        "error_rate": len(errors) / total if total else 0.0,
        "definition": (
            "definitive = supported, contradicted, or invalid_citation; "
            "abstained = insufficient_evidence; backend/inference errors are "
            "counted separately and are not treated as abstention."
        ),
    }


def truncation_detail(results):
    truncated = []
    max_tokens = 0
    checked = 0
    for result in results:
        truncation = result["truncation"]
        if truncation.get("checked"):
            checked += 1
            max_tokens = max(max_tokens, truncation.get("max_input_tokens", 0))
            for window_id in truncation.get("truncated_window_ids", []):
                truncated.append(
                    {"case_id": result["id"], "window_id": window_id}
                )
    return {
        "checked_cases": checked,
        "truncated_window_count": len(truncated),
        "truncated": truncated,
        "max_input_tokens": max_tokens,
    }


def metrics_with_details(results):
    metrics = compute_metrics(results)
    metrics["false_supported"] = false_supported_detail(results)
    metrics["incorrect_contradiction"] = incorrect_contradiction_detail(results)
    metrics["coverage"] = coverage_detail(results)
    return metrics


def failure_cases(results):
    failures = []
    for result in results:
        if result["correct"]:
            continue
        failures.append(
            {
                "id": result["id"],
                "type": result["type"],
                "expected": result["expected"],
                "predicted": result["predicted"],
                "correct": result["correct"],
                "policy_outcome": result.get("policy_outcome"),
                "nli_label": result.get("nli_label"),
                "decision_confidence": result.get("decision_confidence"),
                "max_contradiction_confidence": result.get(
                    "max_contradiction_confidence"
                ),
                "label_counts": result.get("label_counts", {}),
                "policy_details": result.get("policy_details", {}),
                "explanation": result["explanation"],
                "rationale": result["rationale"],
                "windows": [
                    {
                        "window_id": window["window_id"],
                        "label": window["label"],
                        "confidence": max(window["scores"].values()),
                        "text": window["text"],
                    }
                    for window in result["per_window"]
                ],
            }
        )
    return failures


def compare_policies(baseline_results, policy_results):
    baseline_by_id = {row["id"]: row for row in baseline_results}
    changes = []
    for row in policy_results:
        baseline = baseline_by_id[row["id"]]
        if baseline["predicted"] == row["predicted"]:
            continue
        changes.append(
            {
                "id": row["id"],
                "type": row["type"],
                "expected": row["expected"],
                "baseline_predicted": baseline["predicted"],
                "policy_predicted": row["predicted"],
                "policy_outcome": row.get("policy_outcome"),
                "baseline_policy_outcome": baseline.get("policy_outcome"),
                "baseline_confidence": baseline.get("decision_confidence"),
                "policy_confidence": row.get("decision_confidence"),
                "baseline_correct": baseline["correct"],
                "policy_correct": row["correct"],
                "policy_details": row.get("policy_details", {}),
            }
        )
    return {
        "changed_count": len(changes),
        "fixed": [
            item
            for item in changes
            if not item["baseline_correct"] and item["policy_correct"]
        ],
        "broken": [
            item
            for item in changes
            if item["baseline_correct"] and not item["policy_correct"]
        ],
        "changes": changes,
    }


def _result_block(results):
    return {
        "metrics": metrics_with_details(results),
        "by_type": summarise_by_type(results),
        "runtime": strategy_runtime(results),
        "truncation": truncation_detail(results),
        "failures": failure_cases(results),
        "results": results,
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(
        f"  accuracy={metrics['accuracy']:.3f}, "
        f"macro-F1={metrics['macro_f1']:.3f}, "
        f"false-supported={metrics['false_supported']['rate']:.3f} "
        f"({metrics['false_supported']['count']}/"
        f"{metrics['false_supported']['denominator']})"
    )
    print(
        "  incorrect-contradiction="
        f"{metrics['incorrect_contradiction']['rate']:.3f} "
        f"({metrics['incorrect_contradiction']['count']}/"
        f"{metrics['incorrect_contradiction']['denominator']}), "
        f"coverage={metrics['coverage']['coverage']:.3f}, "
        f"abstention={metrics['coverage']['abstention_rate']:.3f}"
    )


def print_changes(title, changes):
    print(
        f"\n{title}: {changes['changed_count']} "
        f"(fixed={len(changes['fixed'])}, broken={len(changes['broken'])})"
    )
    for change in changes["changes"]:
        print(
            f"  {change['id']}: {change['baseline_predicted']} -> "
            f"{change['policy_predicted']} ({change['policy_outcome']})"
        )
    if not changes["changes"]:
        print("  None")


def _skip_message(model_name, error, output):
    print("\nNLI status: SKIPPED - the requested model is not available locally.")
    print(f"  requested model: {model_name}")
    print(f"  exact error: {error}")
    print("\nNothing was downloaded. To prepare the model explicitly, run:")
    print("  .venv/bin/python experiments/17_real_nli_evaluation.py --download-model")
    if output:
        print(f"\nSkipped report would be written to {output}")


def main(argv=None):
    args = parse_args(argv)
    benchmark_path = args.benchmark or (
        ROOT / "experiments" / "15_verification_benchmark" / "benchmark.json"
    )
    dev_cases, dev_sources = combine_development_sets(
        args.dev_set, args.relevance_dev
    )
    benchmark_cases = load_benchmark(benchmark_path)

    dev_types = Counter(case["type"] for case in dev_cases)
    benchmark_types = Counter(case["type"] for case in benchmark_cases)
    dev_lexical = evaluate_lexical(dev_cases)
    dev_lexical_metrics = metrics_with_details(dev_lexical)
    benchmark_lexical = evaluate_lexical(benchmark_cases)
    benchmark_lexical_metrics = metrics_with_details(benchmark_lexical)

    print("=" * 78)
    print("Experiment 20: Evidence Relevance and Contradiction")
    print("=" * 78)
    print(f"Development set: {len(dev_cases)} cases (2 sources)")
    print(f"Benchmark:       {len(benchmark_cases)} cases ({benchmark_path})")
    print_metrics("Development lexical baseline", dev_lexical_metrics)
    print_metrics("Benchmark lexical baseline", benchmark_lexical_metrics)

    report = {
        "experiment": "20_evidence_relevance",
        "status": "skipped",
        "development_set": {
            "path": str(args.dev_set),
            "relevance_extension_path": str(args.relevance_dev),
            "sha256": file_sha256(args.dev_set),
            "relevance_extension_sha256": file_sha256(args.relevance_dev),
            "cases": len(dev_cases),
            "type_distribution": dict(sorted(dev_types.items())),
            "sources": dev_sources,
            "lexical": {
                "metrics": dev_lexical_metrics,
                "by_type": summarise_by_type(dev_lexical),
                "results": dev_lexical,
            },
        },
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": file_sha256(benchmark_path),
            "cases": len(benchmark_cases),
            "type_distribution": dict(sorted(benchmark_types.items())),
            "labels_unchanged": True,
            "lexical": {
                "metrics": benchmark_lexical_metrics,
                "by_type": summarise_by_type(benchmark_lexical),
                "results": benchmark_lexical,
            },
        },
        "environment": build_environment_info(),
        "model": None,
        "probes": None,
        "policy": {
            "name": RELEVANCE_AWARE_POLICY,
            "base_policy": CONFIDENCE_AWARE_POLICY,
            "strategy": STRATEGY,
            "adjacent_context": ADJACENT_CONTEXT,
            "inherited_threshold": FROZEN_THRESHOLD,
            "threshold_source": (
                "Experiment 19 development-only selection; not re-tuned on the "
                "Experiment 20 extension or the benchmark"
            ),
            "relevance_method": RELEVANCE_METHOD,
            "rule": (
                "Start from the Experiment 19 confidence-aware whole-passage "
                "decision. If the result is contradicted and no contradiction "
                "window shares a capitalised claim topic anchor with the "
                "window text, source ID, or document ID, demote it to "
                "insufficient_evidence. Never promote neutral or uncertain "
                "cases, and never alter supported decisions."
            ),
            "softmax_is_calibrated_probability": False,
            "selection_source": (
                "Experiment 19 development set for the inherited threshold; "
                "Experiment 20 relevance extension written before policy "
                "evaluation"
            ),
            "benchmark_used_to_freeze_policy": False,
            "frozen_before_benchmark": False,
        },
        "development": {
            "nli_conservative": None,
            "exp19_confidence": None,
            "new_relevance_aware": None,
            "changes": {
                "confidence_vs_conservative": None,
                "relevance_vs_confidence": None,
                "relevance_vs_conservative": None,
            },
        },
        "benchmark_results": {
            "nli_conservative": None,
            "exp19_confidence": None,
            "new_relevance_aware": None,
            "changes": {
                "confidence_vs_conservative": None,
                "relevance_vs_confidence": None,
                "relevance_vs_conservative": None,
            },
        },
        "limitations": [
            "The combined development set is small, manually extracted from the corpus, and not a random sample.",
            "The relevance check is a lexical topic-anchor guard, not a semantic relevance model; synonyms and paraphrases without shared anchors are outside its scope.",
            "An incidental mention of the claim's subject in an otherwise irrelevant passage can make the guard regard the evidence as relevant; dev_r7 records this limitation.",
            "The guard cannot repair paraphrase failures in which the NLI model returns neutral for supported evidence, such as Experiment 19 p1 and p2.",
            "Pronoun-only evidence relies on source or document identifiers to establish topic; a generic source ID would leave the relevance assessment unknown.",
            "The Experiment 19 0.70 threshold is inherited rather than re-selected on the relevance extension, deliberately limiting additional development-set fitting.",
            "The fixed benchmark has only 33 cases, so differences of one case are not reliable population estimates.",
            "Citation validation and backend failure handling remain separate from both NLI confidence and relevance.",
        ],
    }

    try:
        snapshot_path = resolve_cached_snapshot(args.model)
    except Exception as error:
        report["reason"] = f"model not available locally: {error}"
        _skip_message(args.model, error, args.output)
        _write_report(args.output, report)
        return 2 if args.require_model else 0

    try:
        snapshot_metadata = read_model_config(snapshot_path)
    except ModelValidationError as error:
        report["reason"] = f"model snapshot validation failed: {error}"
        print(f"\nModel snapshot validation failed: {error}", file=sys.stderr)
        _write_report(args.output, report)
        return 2

    report["model"] = {
        "repository": args.model,
        **snapshot_metadata,
        "expected_license": EXPECTED_LICENSE,
        "load_attempts": [],
        "cpu_fallback": False,
    }
    print("\nModel snapshot:")
    print(f"  repository: {args.model}")
    print(f"  revision:   {snapshot_metadata['revision']}")
    print(f"  license:    {snapshot_metadata['license']}")
    print(f"  id2label:   {snapshot_metadata['id2label']}")

    try:
        backend, attempts = _load_backend(snapshot_path, args)
    except NLIModelUnavailable as error:
        report["reason"] = f"model could not be loaded: {error}"
        print(f"\nModel load failed: {error}", file=sys.stderr)
        _write_report(args.output, report)
        return 2

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
    report["model"]["load_attempts"] = attempts
    report["model"]["cpu_fallback"] = any(
        attempt["device"] != args.device and attempt["status"] == "loaded"
        for attempt in attempts
    )
    print(
        f"  loaded on:  {backend.device} "
        f"(dtype={backend.dtype}, parameters={backend.parameter_count})"
    )
    for attempt in attempts:
        print(
            f"    {attempt['device']}: {attempt['status']} "
            f"in {attempt['load_seconds']:.3f}s"
        )
        if attempt["error"]:
            print(f"      error: {attempt['error']}")

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
        print(
            f"  {marker} {probe['id']}: expected={probe['expected_label']}, "
            f"predicted={probe['predicted_label']}"
        )
    if not probes["passed"] and not args.allow_probe_failure:
        print(
            "\nProbe validation failed; stopping before any evaluation.",
            file=sys.stderr,
        )
        report["status"] = "probe_validation_failed"
        _write_report(args.output, report)
        return 3

    def verifiers():
        common = {
            "backend": backend,
            "strategy": STRATEGY,
            "adjacent_context": ADJACENT_CONTEXT,
            "max_length": args.max_length,
            "model_name": args.model,
        }
        return {
            "conservative": WindowedNLIClaimVerifier(**common),
            "confidence": ConfidenceAwareNLIClaimVerifier(
                contradiction_threshold=FROZEN_THRESHOLD, **common
            ),
            "relevance": RelevanceAwareNLIClaimVerifier(
                contradiction_threshold=FROZEN_THRESHOLD, **common
            ),
        }

    dev_verifiers = verifiers()
    conservative_dev = evaluate_windowed(
        dev_cases, dev_verifiers["conservative"]
    )
    confidence_dev = evaluate_windowed(dev_cases, dev_verifiers["confidence"])
    relevance_dev = evaluate_windowed(dev_cases, dev_verifiers["relevance"])

    confidence_vs_conservative_dev = compare_policies(
        conservative_dev, confidence_dev
    )
    relevance_vs_confidence_dev = compare_policies(
        confidence_dev, relevance_dev
    )
    relevance_vs_conservative_dev = compare_policies(
        conservative_dev, relevance_dev
    )

    report["development"]["nli_conservative"] = _result_block(
        conservative_dev
    )
    report["development"]["exp19_confidence"] = _result_block(confidence_dev)
    report["development"]["new_relevance_aware"] = _result_block(
        relevance_dev
    )
    report["development"]["changes"] = {
        "confidence_vs_conservative": confidence_vs_conservative_dev,
        "relevance_vs_confidence": relevance_vs_confidence_dev,
        "relevance_vs_conservative": relevance_vs_conservative_dev,
    }
    print_metrics(
        "Development conservative NLI",
        metrics_with_details(conservative_dev),
    )
    print_metrics(
        f"Development Experiment 19 confidence policy "
        f"(threshold={FROZEN_THRESHOLD:.2f})",
        metrics_with_details(confidence_dev),
    )
    print_metrics(
        "Development relevance-aware NLI",
        metrics_with_details(relevance_dev),
    )
    print_changes(
        "Development confidence vs conservative",
        confidence_vs_conservative_dev,
    )
    print_changes(
        "Development relevance vs confidence",
        relevance_vs_confidence_dev,
    )

    report["policy"]["frozen_before_benchmark"] = True
    report["policy"]["frozen_after"] = "development evaluation only"

    benchmark_verifiers = verifiers()
    conservative_benchmark = evaluate_windowed(
        benchmark_cases, benchmark_verifiers["conservative"]
    )
    confidence_benchmark = evaluate_windowed(
        benchmark_cases, benchmark_verifiers["confidence"]
    )
    relevance_benchmark = evaluate_windowed(
        benchmark_cases, benchmark_verifiers["relevance"]
    )

    confidence_vs_conservative_benchmark = compare_policies(
        conservative_benchmark, confidence_benchmark
    )
    relevance_vs_confidence_benchmark = compare_policies(
        confidence_benchmark, relevance_benchmark
    )
    relevance_vs_conservative_benchmark = compare_policies(
        conservative_benchmark, relevance_benchmark
    )

    report["benchmark_results"]["nli_conservative"] = _result_block(
        conservative_benchmark
    )
    report["benchmark_results"]["exp19_confidence"] = _result_block(
        confidence_benchmark
    )
    report["benchmark_results"]["new_relevance_aware"] = _result_block(
        relevance_benchmark
    )
    report["benchmark_results"]["changes"] = {
        "confidence_vs_conservative": confidence_vs_conservative_benchmark,
        "relevance_vs_confidence": relevance_vs_confidence_benchmark,
        "relevance_vs_conservative": relevance_vs_conservative_benchmark,
    }

    conservative_metrics = metrics_with_details(conservative_benchmark)
    confidence_metrics = metrics_with_details(confidence_benchmark)
    relevance_metrics = metrics_with_details(relevance_benchmark)
    report["benchmark_results"]["delta_vs_conservative"] = {
        "accuracy": (
            relevance_metrics["accuracy"] - conservative_metrics["accuracy"]
        ),
        "macro_f1": (
            relevance_metrics["macro_f1"] - conservative_metrics["macro_f1"]
        ),
        "false_supported_rate": (
            relevance_metrics["false_supported"]["rate"]
            - conservative_metrics["false_supported"]["rate"]
        ),
        "incorrect_contradiction_rate": (
            relevance_metrics["incorrect_contradiction"]["rate"]
            - conservative_metrics["incorrect_contradiction"]["rate"]
        ),
        "coverage": (
            relevance_metrics["coverage"]["coverage"]
            - conservative_metrics["coverage"]["coverage"]
        ),
    }
    report["benchmark_results"]["delta_vs_exp19"] = {
        "accuracy": (
            relevance_metrics["accuracy"] - confidence_metrics["accuracy"]
        ),
        "macro_f1": (
            relevance_metrics["macro_f1"] - confidence_metrics["macro_f1"]
        ),
        "false_supported_rate": (
            relevance_metrics["false_supported"]["rate"]
            - confidence_metrics["false_supported"]["rate"]
        ),
        "incorrect_contradiction_rate": (
            relevance_metrics["incorrect_contradiction"]["rate"]
            - confidence_metrics["incorrect_contradiction"]["rate"]
        ),
        "coverage": (
            relevance_metrics["coverage"]["coverage"]
            - confidence_metrics["coverage"]["coverage"]
        ),
    }

    print_metrics("Benchmark conservative NLI", conservative_metrics)
    print_metrics(
        f"Benchmark Experiment 19 confidence policy "
        f"(threshold={FROZEN_THRESHOLD:.2f})",
        confidence_metrics,
    )
    print_metrics("Benchmark relevance-aware NLI", relevance_metrics)
    print_changes(
        "Benchmark confidence vs conservative",
        confidence_vs_conservative_benchmark,
    )
    print_changes(
        "Benchmark relevance vs confidence",
        relevance_vs_confidence_benchmark,
    )

    report["status"] = "completed"
    _write_report(args.output, report)
    print(f"\nFull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
