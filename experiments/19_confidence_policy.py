"""Experiment 19: confidence-aware NLI decision policy.

A separate, manually labeled development set is used to choose one evidence
strategy and one contradiction-confidence threshold. The fixed Experiment 15
benchmark is evaluated only after those choices are frozen. The conservative
Experiment 17/18 policy remains available and is reported alongside the new
policy. No benchmark label is modified and no model is downloaded implicitly.
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
    ConfidenceAwareNLIClaimVerifier,
    select_threshold,
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
from experiments.verification_utils.windowed_nli import (  # noqa: E402
    WindowedNLIClaimVerifier,
    evaluate_windowed,
    strategy_runtime,
)
from minigrad.evidence_windows import (  # noqa: E402
    SENTENCE,
    SENTENCE_ADJACENT,
    WHOLE_PASSAGE,
)
from minigrad.nli_verification import (  # noqa: E402
    DEFAULT_NLI_MODEL,
    NLIModelUnavailable,
    NLIClaimVerifier,
    TransformersNLIBackend,
)

LABELS = [
    "supported",
    "contradicted",
    "insufficient_evidence",
    "invalid_citation",
]
DEFAULT_DEV_SET = ROOT / "experiments" / "19_confidence_policy" / "dev_set.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "19_confidence_policy" / "report.json"
THRESHOLD_GRID = tuple(round(0.50 + 0.05 * index, 2) for index in range(9))
STRATEGIES = (
    {
        "name": "whole_passage",
        "strategy": WHOLE_PASSAGE,
        "adjacent_context": 0,
        "description": "Existing whole-passage NLI baseline.",
    },
    {
        "name": "sentence",
        "strategy": SENTENCE,
        "adjacent_context": 0,
        "description": "One sentence per NLI premise with conservative aggregation.",
    },
    {
        "name": "sentence_adjacent",
        "strategy": SENTENCE_ADJACENT,
        "adjacent_context": 1,
        "description": "Sentence plus one neighbor on each side, deduplicated.",
    },
)
STRATEGY_ORDER = {config["name"]: index for index, config in enumerate(STRATEGIES)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-set", type=Path, default=DEFAULT_DEV_SET)
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
                "policy_outcome": row["policy_outcome"],
                "baseline_confidence": baseline["decision_confidence"],
                "policy_confidence": row["decision_confidence"],
                "baseline_correct": baseline["correct"],
                "policy_correct": row["correct"],
                "max_contradiction_confidence": row[
                    "max_contradiction_confidence"
                ],
                "label_counts": row["label_counts"],
            }
        )
    return {
        "changed_count": len(changes),
        "fixed": [item for item in changes if item["policy_correct"]],
        "broken": [
            item
            for item in changes
            if item["baseline_correct"] and not item["policy_correct"]
        ],
        "changes": changes,
    }


def strategy_candidates(dev_cases, backend, args):
    candidates = {}
    for config in STRATEGIES:
        verifier = WindowedNLIClaimVerifier(
            backend=backend,
            strategy=config["strategy"],
            adjacent_context=config["adjacent_context"],
            max_length=args.max_length,
            model_name=args.model,
        )
        started = time.perf_counter()
        results = evaluate_windowed(dev_cases, verifier)
        runtime = strategy_runtime(results)
        runtime["wall_seconds"] = time.perf_counter() - started
        metrics = metrics_with_details(results)
        candidates[config["name"]] = {
            "config": config,
            "metrics": metrics,
            "by_type": summarise_by_type(results),
            "runtime": runtime,
            "results": results,
        }
    selected = min(
        candidates,
        key=lambda name: (
            -candidates[name]["metrics"]["accuracy"],
            -candidates[name]["metrics"]["macro_f1"],
            candidates[name]["runtime"]["average_windows_per_claim"],
            STRATEGY_ORDER[name],
        ),
    )
    return selected, candidates


def threshold_candidates(dev_cases, backend, selected_strategy, args):
    candidates = []
    results_by_threshold = {}
    config = next(
        item for item in STRATEGIES if item["name"] == selected_strategy
    )
    for threshold in THRESHOLD_GRID:
        verifier = ConfidenceAwareNLIClaimVerifier(
            backend=backend,
            strategy=config["strategy"],
            adjacent_context=config["adjacent_context"],
            contradiction_threshold=threshold,
            max_length=args.max_length,
            model_name=args.model,
        )
        results = evaluate_windowed(dev_cases, verifier)
        metrics = metrics_with_details(results)
        candidates.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "contradicted_recall": metrics["per_class"]["contradicted"][
                    "recall"
                ],
                "incorrect_contradiction_rate": metrics[
                    "incorrect_contradiction"
                ]["rate"],
                "abstention_rate": metrics["coverage"]["abstention_rate"],
                "false_supported_rate": metrics["false_supported"]["rate"],
                "demoted_case_ids": [
                    row["id"]
                    for row in results
                    if row["policy_outcome"] == "low_confidence_contradiction"
                ],
            }
        )
        results_by_threshold[threshold] = results
    selected = select_threshold(candidates)
    return selected["threshold"], candidates, results_by_threshold


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
    print("  Per-class:")
    for label, values in metrics["per_class"].items():
        print(
            f"    {label:20s}: P={values['precision']:.3f} "
            f"R={values['recall']:.3f} F1={values['f1']:.3f} "
            f"(n={values['support']})"
        )


def print_by_type(summary):
    print("  By case type:")
    for case_type, values in summary.items():
        print(
            f"    {case_type}: {values['accuracy']:.3f} "
            f"({values['correct']}/{values['total']})"
        )


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
    dev_cases, dev_metadata = load_development_set(args.dev_set)
    benchmark_cases = load_benchmark(benchmark_path)

    dev_types = Counter(case["type"] for case in dev_cases)
    benchmark_types = Counter(case["type"] for case in benchmark_cases)
    dev_lexical = evaluate_lexical(dev_cases)
    dev_lexical_metrics = metrics_with_details(dev_lexical)
    benchmark_lexical = evaluate_lexical(benchmark_cases)
    benchmark_lexical_metrics = metrics_with_details(benchmark_lexical)

    print("=" * 78)
    print("Experiment 19: Confidence-Aware NLI Decision Policy")
    print("=" * 78)
    print(f"Development set: {len(dev_cases)} cases ({args.dev_set})")
    print(f"Benchmark:       {len(benchmark_cases)} cases ({benchmark_path})")
    print_metrics("Development lexical baseline", dev_lexical_metrics)
    print_metrics("Benchmark lexical baseline", benchmark_lexical_metrics)

    report = {
        "experiment": "19_confidence_policy",
        "status": "skipped",
        "development_set": {
            "path": str(args.dev_set),
            "sha256": file_sha256(args.dev_set),
            "cases": len(dev_cases),
            "type_distribution": dict(sorted(dev_types.items())),
            "metadata": dev_metadata,
            "lexical": {
                "metrics": dev_lexical_metrics,
                "by_type": summarise_by_type(dev_lexical),
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
            },
        },
        "environment": build_environment_info(),
        "model": None,
        "probes": None,
        "policy": {
            "name": CONFIDENCE_AWARE_POLICY,
            "rule": (
                "Preserve supported/neutral/mixed decisions; demote a "
                "contradiction to insufficient_evidence only when its "
                "strongest contradiction confidence is below the "
                "development-set threshold. Never use a threshold to promote "
                "an uncertain case to supported."
            ),
            "softmax_is_calibrated_probability": False,
            "selection_source": "development_set_only",
            "threshold_grid": list(THRESHOLD_GRID),
            "selected_threshold": None,
            "selected_strategy": None,
        },
        "development": {
            "strategy_comparison": {},
            "threshold_selection": {},
            "nli_conservative": None,
            "new_policy": None,
        },
        "benchmark_results": {
            "nli_conservative": None,
            "new_policy": None,
        },
        "limitations": [
            "The 26-case development set is manually extracted from the small corpus, is not a random sample, and gives only a rough threshold estimate.",
            "Softmax scores are not calibrated probabilities; the threshold is a model-confidence cutoff, not a guarantee that a claim is true.",
            "A confidence threshold cannot repair high-confidence false contradictions caused by citation irrelevance or domain mismatch.",
            "The fixed benchmark has only 33 cases, so differences of one case should not be treated as reliable population estimates.",
            "Citation validation and backend failure handling are evaluated separately from NLI confidence.",
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

    selected_strategy, strategies = strategy_candidates(
        dev_cases, backend, args
    )
    report["policy"]["selected_strategy"] = selected_strategy
    report["development"]["strategy_comparison"] = {
        "selection_criterion": (
            "highest development accuracy, then macro-F1, then fewer windows "
            "per claim, then fixed strategy order"
        ),
        "selected": selected_strategy,
        "candidates": {
            name: {
                "config": item["config"],
                "metrics": item["metrics"],
                "by_type": item["by_type"],
                "runtime": item["runtime"],
            }
            for name, item in strategies.items()
        },
    }
    print("\nDevelopment strategy comparison:")
    for name, item in strategies.items():
        metrics = item["metrics"]
        marker = "SELECTED" if name == selected_strategy else ""
        print(
            f"  {name}: accuracy={metrics['accuracy']:.3f}, "
            f"macro-F1={metrics['macro_f1']:.3f}, "
            f"incorrect-contradiction="
            f"{metrics['incorrect_contradiction']['rate']:.3f} {marker}"
        )

    selected_threshold, threshold_rows, dev_policy_results = (
        threshold_candidates(dev_cases, backend, selected_strategy, args)
    )
    report["policy"]["selected_threshold"] = selected_threshold
    report["development"]["threshold_selection"] = {
        "selection_source": "development_set_only",
        "selected": selected_threshold,
        "selection_rule": (
            "maximise accuracy, then contradiction recall, then minimise "
            "incorrect contradictions among expected-insufficient cases, "
            "then minimise abstention, then choose the lower threshold"
        ),
        "benchmark_used": False,
        "candidates": threshold_rows,
    }
    print("\nDevelopment threshold selection:")
    for candidate in threshold_rows:
        marker = (
            " SELECTED"
            if candidate["threshold"] == selected_threshold
            else ""
        )
        print(
            f"  {candidate['threshold']:.2f}: "
            f"accuracy={candidate['accuracy']:.3f}, "
            f"contradiction-recall={candidate['contradicted_recall']:.3f}, "
            f"incorrect-contradiction="
            f"{candidate['incorrect_contradiction_rate']:.3f}, "
            f"abstention={candidate['abstention_rate']:.3f}{marker}"
        )

    conservative_dev = strategies[selected_strategy]["results"]
    conservative_dev_metrics = strategies[selected_strategy]["metrics"]
    policy_dev = dev_policy_results[selected_threshold]
    policy_dev_metrics = metrics_with_details(policy_dev)
    dev_changes = compare_policies(conservative_dev, policy_dev)

    report["development"]["nli_conservative"] = {
        "strategy": selected_strategy,
        "metrics": conservative_dev_metrics,
        "by_type": strategies[selected_strategy]["by_type"],
        "runtime": strategies[selected_strategy]["runtime"],
        "truncation": truncation_detail(conservative_dev),
        "results": conservative_dev,
    }
    report["development"]["new_policy"] = {
        "strategy": selected_strategy,
        "threshold": selected_threshold,
        "metrics": policy_dev_metrics,
        "by_type": summarise_by_type(policy_dev),
        "runtime": strategy_runtime(policy_dev),
        "truncation": truncation_detail(policy_dev),
        "changes_vs_conservative": dev_changes,
        "failures": failure_cases(policy_dev),
        "results": policy_dev,
    }
    print_metrics("Development conservative NLI", conservative_dev_metrics)
    print_metrics(
        f"Development confidence-aware NLI (threshold={selected_threshold:.2f})",
        policy_dev_metrics,
    )
    print(
        f"  Changed cases: {dev_changes['changed_count']} "
        f"(fixed={len(dev_changes['fixed'])}, "
        f"broken={len(dev_changes['broken'])})"
    )
    for change in dev_changes["changes"]:
        print(
            f"    {change['id']}: {change['baseline_predicted']} -> "
            f"{change['policy_predicted']} ({change['policy_outcome']})"
        )

    config = next(
        item for item in STRATEGIES if item["name"] == selected_strategy
    )
    conservative_benchmark_verifier = WindowedNLIClaimVerifier(
        backend=backend,
        strategy=config["strategy"],
        adjacent_context=config["adjacent_context"],
        max_length=args.max_length,
        model_name=args.model,
    )
    conservative_benchmark = evaluate_windowed(
        benchmark_cases, conservative_benchmark_verifier
    )
    conservative_benchmark_metrics = metrics_with_details(
        conservative_benchmark
    )
    policy_benchmark_verifier = ConfidenceAwareNLIClaimVerifier(
        backend=backend,
        strategy=config["strategy"],
        adjacent_context=config["adjacent_context"],
        contradiction_threshold=selected_threshold,
        max_length=args.max_length,
        model_name=args.model,
    )
    policy_benchmark = evaluate_windowed(
        benchmark_cases, policy_benchmark_verifier
    )
    policy_benchmark_metrics = metrics_with_details(policy_benchmark)
    benchmark_changes = compare_policies(
        conservative_benchmark, policy_benchmark
    )

    report["benchmark_results"]["nli_conservative"] = {
        "strategy": selected_strategy,
        "metrics": conservative_benchmark_metrics,
        "by_type": summarise_by_type(conservative_benchmark),
        "runtime": strategy_runtime(conservative_benchmark),
        "truncation": truncation_detail(conservative_benchmark),
        "results": conservative_benchmark,
    }
    report["benchmark_results"]["new_policy"] = {
        "strategy": selected_strategy,
        "threshold": selected_threshold,
        "metrics": policy_benchmark_metrics,
        "by_type": summarise_by_type(policy_benchmark),
        "runtime": strategy_runtime(policy_benchmark),
        "truncation": truncation_detail(policy_benchmark),
        "changes_vs_conservative": benchmark_changes,
        "failures": failure_cases(policy_benchmark),
        "results": policy_benchmark,
    }
    report["benchmark_results"]["delta_vs_conservative"] = {
        "accuracy": (
            policy_benchmark_metrics["accuracy"]
            - conservative_benchmark_metrics["accuracy"]
        ),
        "macro_f1": (
            policy_benchmark_metrics["macro_f1"]
            - conservative_benchmark_metrics["macro_f1"]
        ),
        "false_supported_rate": (
            policy_benchmark_metrics["false_supported"]["rate"]
            - conservative_benchmark_metrics["false_supported"]["rate"]
        ),
        "incorrect_contradiction_rate": (
            policy_benchmark_metrics["incorrect_contradiction"]["rate"]
            - conservative_benchmark_metrics["incorrect_contradiction"][
                "rate"
            ]
        ),
        "coverage": (
            policy_benchmark_metrics["coverage"]["coverage"]
            - conservative_benchmark_metrics["coverage"]["coverage"]
        ),
    }

    print_metrics("Benchmark conservative NLI", conservative_benchmark_metrics)
    print_by_type(summarise_by_type(conservative_benchmark))
    print_metrics(
        f"Benchmark confidence-aware NLI (threshold={selected_threshold:.2f})",
        policy_benchmark_metrics,
    )
    print_by_type(summarise_by_type(policy_benchmark))
    print(
        f"\nBenchmark changed cases: {benchmark_changes['changed_count']} "
        f"(fixed={len(benchmark_changes['fixed'])}, "
        f"broken={len(benchmark_changes['broken'])})"
    )
    for change in benchmark_changes["changes"]:
        print(
            f"  {change['id']}: {change['baseline_predicted']} -> "
            f"{change['policy_predicted']} ({change['policy_outcome']})"
        )
    if not benchmark_changes["changes"]:
        print("  None")

    report["status"] = "completed"
    _write_report(args.output, report)
    print(f"\nFull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
