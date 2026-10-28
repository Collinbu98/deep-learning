"""Experiment 18: sentence-level evidence selection for NLI verification.

The Experiment 15 benchmark and lexical baseline are preserved. The same real
NLI model from Experiment 17 is evaluated with whole-passage, sentence, and
adjacent-sentence evidence windows. No threshold is tuned on the benchmark.
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
    find_strategy_disagreements,
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
DEFAULT_OUTPUT = ROOT / "experiments" / "18_evidence_windows" / "report.json"
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
        "description": "One sentence per NLI premise, conservative aggregation.",
    },
    {
        "name": "sentence_adjacent",
        "strategy": SENTENCE_ADJACENT,
        "adjacent_context": 1,
        "description": (
            "Each sentence plus one neighbor on each side, deduplicated, with "
            "conservative aggregation."
        ),
    },
)
EXPECTED_LABEL = {
    "supported": "entailment",
    "contradicted": "contradiction",
    "insufficient_evidence": "neutral",
}
FOCUS_GROUPS = (
    "irrelevant_citation",
    "paraphrase",
    "insufficient",
    "contradiction",
    "numbers",
    "negation",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="continue if the label-mapping probes do not all pass",
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
                truncated.append({"case_id": result["id"], "window_id": window_id})
    return {
        "checked_cases": checked,
        "truncated_window_count": len(truncated),
        "truncated": truncated,
        "max_input_tokens": max_tokens,
    }


def _label_counts(result):
    return dict(Counter(window["label"] for window in result["per_window"]))


def _diagnose_failure(result):
    if result["predicted"] in {"backend_unavailable", "verification_error"}:
        return "backend_or_inference_error"
    if result["expected"] == "invalid_citation":
        return "citation_validation"
    expected_label = EXPECTED_LABEL.get(result["expected"])
    labels = [window["label"] for window in result["per_window"]]
    if expected_label is not None and expected_label in labels:
        if "entailment" in labels and "contradiction" in labels:
            return "aggregation_mixed_conflict"
        if (
            result["expected"] == "insufficient_evidence"
            and "neutral" in labels
            and "contradiction" in labels
        ):
            return "aggregation_prefers_contradiction_over_neutral"
        return "aggregation"
    if result["expected"] == "supported":
        return "window_or_nli_missing_entailment"
    if result["expected"] == "contradicted":
        return "window_or_nli_missing_contradiction"
    if result["expected"] == "insufficient_evidence":
        if "contradiction" in labels:
            return "nli_false_contradiction"
        return "window_or_nli_other"
    return "unclassified"


def failure_analysis(results_by_strategy):
    analysis = {}
    for name, results in results_by_strategy.items():
        failures = []
        for result in results:
            if result["correct"]:
                continue
            failures.append(
                {
                    "id": result["id"],
                    "type": result["type"],
                    "claim": result["claim"],
                    "expected": result["expected"],
                    "predicted": result["predicted"],
                    "diagnosis": _diagnose_failure(result),
                    "window_label_counts": _label_counts(result),
                    "window_count": result["windows_scored"],
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
        analysis[name] = {
            "failure_count": len(failures),
            "diagnoses": dict(Counter(item["diagnosis"] for item in failures)),
            "failures": failures,
        }
    return analysis


def aggregation_diagnostics(results):
    single_entailment = []
    contradiction_over_neutral = []
    mixed_conflicts = []
    neutral_only_insufficient = []
    for result in results:
        labels = [window["label"] for window in result["per_window"]]
        if (
            result["predicted"] == "supported"
            and labels.count("entailment") == 1
            and labels.count("contradiction") == 0
        ):
            single_entailment.append(
                {
                    "id": result["id"],
                    "type": result["type"],
                    "correct": result["correct"],
                    "window_label_counts": _label_counts(result),
                }
            )
        if (
            result["predicted"] == "contradicted"
            and "neutral" in labels
            and "contradiction" in labels
        ):
            contradiction_over_neutral.append(
                {
                    "id": result["id"],
                    "type": result["type"],
                    "expected": result["expected"],
                    "correct": result["correct"],
                    "window_label_counts": _label_counts(result),
                }
            )
        if result["nli_label"] == "mixed":
            mixed_conflicts.append(
                {
                    "id": result["id"],
                    "type": result["type"],
                    "expected": result["expected"],
                    "correct": result["correct"],
                }
            )
        if result["predicted"] == "insufficient_evidence" and labels and set(
            labels
        ) == {"neutral"}:
            neutral_only_insufficient.append(result["id"])
    return {
        "single_entailment_without_contradiction": {
            "count": len(single_entailment),
            "cases": single_entailment,
        },
        "contradiction_prevails_over_neutral": {
            "count": len(contradiction_over_neutral),
            "cases": contradiction_over_neutral,
        },
        "mixed_entailment_contradiction": {
            "count": len(mixed_conflicts),
            "cases": mixed_conflicts,
        },
        "neutral_only_insufficient_case_ids": neutral_only_insufficient,
    }


def focus_results(results):
    grouped = {}
    for case_type in FOCUS_GROUPS:
        rows = [result for result in results if result["type"] == case_type]
        if not rows:
            continue
        grouped[case_type] = {
            "total": len(rows),
            "correct": sum(1 for row in rows if row["correct"]),
            "accuracy": sum(1 for row in rows if row["correct"]) / len(rows),
            "cases": [
                {
                    "id": row["id"],
                    "expected": row["expected"],
                    "predicted": row["predicted"],
                    "correct": row["correct"],
                    "window_label_counts": _label_counts(row),
                }
                for row in rows
            ],
        }
    return grouped


def evidence_effects(results_by_strategy):
    whole = {row["id"]: row for row in results_by_strategy["whole_passage"]}
    smaller = {}
    for name in ("sentence", "sentence_adjacent"):
        for row in results_by_strategy[name]:
            smaller.setdefault(row["id"], {})[name] = row

    helped = []
    lost_context = []
    adjacent_recovered = []
    for case_id, window_rows in smaller.items():
        baseline = whole[case_id]
        correct_names = [
            name for name, row in window_rows.items() if row["correct"]
        ]
        if not baseline["correct"] and correct_names:
            helped.append(
                {
                    "id": case_id,
                    "type": baseline["type"],
                    "expected": baseline["expected"],
                    "whole_predicted": baseline["predicted"],
                    "correct_smaller_strategies": correct_names,
                }
            )
        if baseline["correct"] and all(
            not row["correct"] for row in window_rows.values()
        ):
            lost_context.append(
                {
                    "id": case_id,
                    "type": baseline["type"],
                    "expected": baseline["expected"],
                    "smaller_predictions": {
                        name: row["predicted"] for name, row in window_rows.items()
                    },
                }
            )
        sentence = window_rows.get("sentence")
        adjacent = window_rows.get("sentence_adjacent")
        if (
            sentence is not None
            and adjacent is not None
            and not sentence["correct"]
            and adjacent["correct"]
        ):
            adjacent_recovered.append(
                {
                    "id": case_id,
                    "type": sentence["type"],
                    "expected": sentence["expected"],
                    "sentence_predicted": sentence["predicted"],
                    "adjacent_predicted": adjacent["predicted"],
                }
            )
    return {
        "smaller_evidence_helped": helped,
        "smaller_evidence_lost_whole_passage_context": lost_context,
        "adjacent_context_recovered_sentence_failure": adjacent_recovered,
    }


def metrics_with_details(results):
    metrics = compute_metrics(results)
    metrics["false_supported"] = false_supported_detail(results)
    return metrics


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(
        f"  accuracy={metrics['accuracy']:.3f}, "
        f"macro-F1={metrics['macro_f1']:.3f}, "
        f"false-supported={metrics['false_supported']['rate']:.3f} "
        f"({metrics['false_supported']['count']}/"
        f"{metrics['false_supported']['denominator']})"
    )
    print("  Per-class:")
    for label, values in metrics["per_class"].items():
        print(
            f"    {label:20s}: P={values['precision']:.3f} "
            f"R={values['recall']:.3f} F1={values['f1']:.3f} "
            f"(n={values['support']})"
        )
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"    {'':20s} " + " ".join(f"{label[:12]:>12s}" for label in LABELS))
    for label in LABELS:
        row = metrics["confusion_matrix"][label]
        print(
            f"    {label:20s} "
            + " ".join(f"{row.get(predicted, 0):>12d}" for predicted in LABELS)
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
    cases_path = args.benchmark or (
        ROOT / "experiments" / "15_verification_benchmark" / "benchmark.json"
    )
    cases = load_benchmark(cases_path)
    lexical_results = evaluate_lexical(cases)
    lexical_metrics = metrics_with_details(lexical_results)

    print("=" * 78)
    print("Experiment 18: Sentence-Level Evidence Selection")
    print("=" * 78)
    print(f"Benchmark: {len(cases)} cases ({cases_path})")
    print_metrics("Lexical baseline", lexical_metrics)
    print_by_type(summarise_by_type(lexical_results))

    base_report = {
        "experiment": "18_evidence_windows",
        "status": "skipped",
        "benchmark": {
            "path": str(cases_path),
            "sha256": file_sha256(cases_path),
            "cases": len(cases),
            "labels_unchanged": True,
        },
        "environment": build_environment_info(),
        "lexical": {
            "metrics": lexical_metrics,
            "by_type": summarise_by_type(lexical_results),
        },
        "model": None,
        "probes": None,
        "strategies": {},
        "disagreements": [],
        "evidence_effects": {},
        "failure_analysis": {},
        "calibration": {
            "softmax_is_calibrated_probability": False,
            "development_set_used": False,
            "note": (
                "No temperature scaling or threshold tuning was performed on "
                "the 33-case benchmark. Softmax values reflect only the "
                "model's premise/hypothesis relationship."
            ),
        },
    }

    try:
        snapshot_path = resolve_cached_snapshot(args.model)
    except Exception as error:
        base_report["reason"] = f"model not available locally: {error}"
        _skip_message(args.model, error, args.output)
        _write_report(args.output, base_report)
        print(f"Skipped report written to {args.output}")
        return 2 if args.require_model else 0

    try:
        snapshot_metadata = read_model_config(snapshot_path)
    except ModelValidationError as error:
        base_report["reason"] = f"model snapshot validation failed: {error}"
        print(f"\nModel snapshot validation failed: {error}", file=sys.stderr)
        _write_report(args.output, base_report)
        return 2

    base_report["model"] = {
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
        base_report["reason"] = f"model could not be loaded: {error}"
        base_report["model"]["load_attempts"] = []
        print(f"\nModel load failed: {error}", file=sys.stderr)
        _write_report(args.output, base_report)
        return 2

    base_report["model"]["backend"] = {
        "model_name": backend.model_name,
        "device": backend.device,
        "dtype": backend.dtype,
        "local_files_only": backend.local_files_only,
        "id2label": {
            str(label_id): label for label_id, label in (backend.id2label or {}).items()
        },
        "max_length": backend.max_length,
        "parameter_count": backend.parameter_count,
        "load_seconds": backend.load_seconds,
    }
    base_report["model"]["load_attempts"] = attempts
    base_report["model"]["cpu_fallback"] = any(
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
    base_report["probes"] = probes
    print("\nLabel-mapping probes:")
    for probe in probes["results"]:
        marker = "PASS" if probe["passed"] else "FAIL"
        print(
            f"  {marker} {probe['id']}: expected={probe['expected_label']}, "
            f"predicted={probe['predicted_label']}"
        )
    if not probes["passed"] and not args.allow_probe_failure:
        print(
            "\nProbe validation failed; stopping before benchmark evaluation.",
            file=sys.stderr,
        )
        base_report["status"] = "probe_validation_failed"
        _write_report(args.output, base_report)
        return 3

    results_by_strategy = {}
    for config in STRATEGIES:
        print("\n" + "-" * 78)
        print(f"Strategy: {config['name']} - {config['description']}")
        verifier = WindowedNLIClaimVerifier(
            backend=backend,
            strategy=config["strategy"],
            adjacent_context=config["adjacent_context"],
            max_length=args.max_length,
            model_name=args.model,
        )
        started = time.perf_counter()
        results = evaluate_windowed(cases, verifier)
        wall_seconds = time.perf_counter() - started
        metrics = metrics_with_details(results)
        by_type = summarise_by_type(results)
        runtime = strategy_runtime(results)
        runtime["wall_seconds"] = wall_seconds
        truncation = truncation_detail(results)
        results_by_strategy[config["name"]] = results
        base_report["strategies"][config["name"]] = {
            "config": config,
            "metrics": metrics,
            "by_type": by_type,
            "focus_results": focus_results(results),
            "aggregation_diagnostics": aggregation_diagnostics(results),
            "runtime": runtime,
            "truncation": truncation,
            "results": results,
        }
        print_metrics(config["name"], metrics)
        print_by_type(by_type)
        diagnostics = base_report["strategies"][config["name"]][
            "aggregation_diagnostics"
        ]
        print(
            "  Aggregation: "
            f"single-entailment={diagnostics['single_entailment_without_contradiction']['count']}, "
            f"contradiction-over-neutral={diagnostics['contradiction_prevails_over_neutral']['count']}, "
            f"mixed={diagnostics['mixed_entailment_contradiction']['count']}"
        )
        print(
            f"  Runtime: {runtime['inference_seconds']:.3f}s, "
            f"{runtime['milliseconds_per_case']:.1f} ms/case, "
            f"{runtime['average_windows_per_claim']:.2f} windows/claim"
        )
        if truncation["truncated_window_count"]:
            print(
                "  Truncated windows: "
                f"{truncation['truncated_window_count']} "
                f"(max tokens={truncation['max_input_tokens']})"
            )
        else:
            print(
                f"  Truncation: none "
                f"(max tokens={truncation['max_input_tokens']}, "
                f"limit={args.max_length})"
            )

    whole_metrics = base_report["strategies"]["whole_passage"]["metrics"]
    for name, strategy in base_report["strategies"].items():
        metrics = strategy["metrics"]
        strategy["delta_vs_whole_passage"] = {
            "accuracy": metrics["accuracy"] - whole_metrics["accuracy"],
            "macro_f1": metrics["macro_f1"] - whole_metrics["macro_f1"],
            "false_supported_rate": (
                metrics["false_supported"]["rate"]
                - whole_metrics["false_supported"]["rate"]
            ),
        }

    base_report["disagreements"] = find_strategy_disagreements(
        results_by_strategy
    )
    base_report["evidence_effects"] = evidence_effects(results_by_strategy)
    base_report["failure_analysis"] = failure_analysis(results_by_strategy)
    base_report["status"] = "completed"

    print("\nPer-case strategy disagreements:")
    if not base_report["disagreements"]:
        print("  None")
    for disagreement in base_report["disagreements"]:
        statuses = ", ".join(
            f"{name}={status}" for name, status in disagreement["statuses"].items()
        )
        print(
            f"  {disagreement['id']} [{disagreement['type']}]: "
            f"expected={disagreement['expected']}; {statuses}"
        )

    effects = base_report["evidence_effects"]
    print("\nEvidence-granularity effects:")
    print(
        "  smaller evidence helped: "
        f"{len(effects['smaller_evidence_helped'])} case(s)"
    )
    for item in effects["smaller_evidence_helped"]:
        print(
            f"    {item['id']}: whole={item['whole_predicted']} -> "
            f"{item['correct_smaller_strategies']}"
        )
    print(
        "  lost whole-passage context: "
        f"{len(effects['smaller_evidence_lost_whole_passage_context'])} case(s)"
    )
    for item in effects["smaller_evidence_lost_whole_passage_context"]:
        print(f"    {item['id']}: {item['smaller_predictions']}")
    print(
        "  adjacent context recovered sentence failure: "
        f"{len(effects['adjacent_context_recovered_sentence_failure'])} case(s)"
    )
    for item in effects["adjacent_context_recovered_sentence_failure"]:
        print(
            f"    {item['id']}: sentence={item['sentence_predicted']} -> "
            f"adjacent={item['adjacent_predicted']}"
        )

    _write_report(args.output, base_report)
    print(f"\nFull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
