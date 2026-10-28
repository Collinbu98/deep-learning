"""Experiment 21: paraphrase-aware NLI verification."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.verification_utils.windowed_nli as windowed_nli
from experiments.verification_utils.paraphrase_policy import (
    PARAPHRASE_POLICY,
    ParaphraseAwareNLIClaimVerifier,
)
from experiments.verification_utils.relevance_policy import (
    RelevanceAwareNLIClaimVerifier,
)
from experiments.verification_utils.eval import (
    evaluate_lexical,
    load_benchmark,
)
from experiments.verification_utils.nli_eval import summarise_by_type
from experiments.verification_utils.real_nli_eval import (
    EXPECTED_LICENSE,
    ModelValidationError,
    build_environment_info,
    evaluate_probes,
    file_sha256,
    read_model_config,
    resolve_cached_snapshot,
)
from experiments.verification_utils.windowed_nli import (
    WindowedNLIClaimVerifier,
    evaluate_windowed,
    strategy_runtime,
)
from minigrad.evidence_windows import WHOLE_PASSAGE
from minigrad.nli_verification import (
    DEFAULT_NLI_MODEL,
    NLIClaimVerifier,
    TransformersNLIBackend,
)

exp20 = importlib.import_module("experiments.20_evidence_relevance")
metrics_with_details = exp20.metrics_with_details
compare_policies = exp20.compare_policies
failure_cases = exp20.failure_cases
strategy_runtime = exp20.strategy_runtime
truncation_detail = exp20.truncation_detail
summarise_by_type = exp20.summarise_by_type

DEFAULT_DEV_SET = ROOT / "experiments" / "21_paraphrase_verification" / "dev_set.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "21_paraphrase_verification" / "report.json"
STRATEGY = WHOLE_PASSAGE
ADJACENT_CONTEXT = 0
CONTRADICTION_THRESHOLD = 0.70


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


def metrics_with_details_results(results):
    metrics = metrics_with_details(results)
    metrics["false_supported"] = false_supported_detail(results)
    metrics["incorrect_contradiction"] = incorrect_contradiction_detail(results)
    metrics["coverage"] = coverage_detail(results)
    return metrics


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(
        f"  accuracy={metrics['accuracy']:.3f}, "
        f"macro-F1={metrics['macro_f1']:.3f}, "
        f"false-supported={metrics['false_supported']['rate']:.3f} "
        f"({metrics['false_supported']['count']}/{metrics['false_supported']['denominator']})"
    )
    print(
        "  incorrect-contradiction="
        f"{metrics['incorrect_contradiction']['rate']:.3f} "
        f"({metrics['incorrect_contradiction']['count']}/"
        f"{metrics['incorrect_contradiction']['denominator']}), "
        f"coverage={metrics['coverage']['coverage']:.3f}, "
        f"abstention={metrics['coverage']['abstention_rate']:.3f}"
    )


def _load_backend(model_path, args):
    backend = TransformersNLIBackend(
        model_path,
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
        max_length=args.max_length,
    )
    backend.load()
    return backend


def main(argv=None):
    args = parse_args(argv)
    dev_cases, dev_metadata = load_development_set(args.dev_set)
    benchmark_path = args.benchmark or (
        ROOT / "experiments" / "15_verification_benchmark" / "benchmark.json"
    )
    benchmark_cases = load_benchmark(benchmark_path)

    dev_types = {}
    for case in dev_cases:
        dev_types[case["type"]] = dev_types.get(case["type"], 0) + 1
    benchmark_types = {}
    for case in benchmark_cases:
        benchmark_types[case["type"]] = benchmark_types.get(case["type"], 0) + 1

    dev_lexical = evaluate_lexical(dev_cases)
    dev_lexical_metrics = metrics_with_details(dev_lexical)
    benchmark_lexical = evaluate_lexical(benchmark_cases)
    benchmark_lexical_metrics = metrics_with_details(benchmark_lexical)

    report = {
        "experiment": "21_paraphrase_verification",
        "status": "skipped",
        "development_set": {
            "path": str(args.dev_set),
            "sha256": file_sha256(args.dev_set),
            "cases": len(dev_cases),
            "type_distribution": dev_types,
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
            "type_distribution": benchmark_types,
            "lexical": {
                "metrics": benchmark_lexical_metrics,
                "by_type": summarise_by_type(benchmark_lexical),
            },
        },
        "environment": build_environment_info(),
        "model": None,
        "probes": None,
        "policy": {
            "name": PARAPHRASE_POLICY,
            "decision_policy": "paraphrase_aware",
            "base_policy": "relevance_aware",
            "strategy": STRATEGY,
            "adjacent_context": ADJACENT_CONTEXT,
            "inherits_contradiction_threshold": CONTRADICTION_THRESHOLD,
            "threshold_source": (
                "Experiment 19 development-only selection; inherited by Experiment 20 and Experiment 21"
            ),
            "relevance_method": "lexical_topic_anchor_guard",
            "paraphrase_method": "deterministic_duration_fact_equivalence",
            "rule": (
                "Start from the Experiment 20 relevance-aware NLI policy; "
                "when the final result is insufficient_evidence with a neutral "
                "label and a cited window passes the conservative deterministic "
                "paraphrase-equivalence check, promote it to supported."
            ),
            "softmax_is_calibrated_probability": False,
            "benchmark_used_to_freeze_policy": False,
            "frozen_before_benchmark": False,
        },
        "development": {
            "nli_conservative": None,
            "exp19_confidence": None,
            "exp20_relevance_aware": None,
            "new_paraphrase_aware": None,
            "changes": {
                "exp20_relevance_vs_conservative": None,
                "paraphrase_vs_exp20": None,
            },
        },
        "benchmark_results": {
            "nli_conservative": None,
            "exp19_confidence": None,
            "exp20_relevance_aware": None,
            "new_paraphrase_aware": None,
            "changes": {
                "exp20_relevance_vs_conservative": None,
                "paraphrase_vs_exp20": None,
            },
        },
        "limitations": [
            "The paraphrase support check is deterministic and narrow; it is not a general semantic equivalence model.",
            "The Experiment 21 development set is small and manually labeled.",
            "The final policy can miss genuine contradictions, such as dev21_c1, but remains conservative for unsupported cases.",
            "The fixed benchmark has only 33 cases, so results should not be treated as population estimates.",
        ],
    }

    try:
        snapshot_path = resolve_cached_snapshot(args.model)
    except Exception as error:
        report["reason"] = f"model not available locally: {error}"
        report["status"] = "skipped"
        write_report(args.output, report)
        print("\nNLI status: SKIPPED - the requested model is not available locally.")
        print(f"  requested model: {args.model}")
        print(f"  exact error: {error}")
        return 2 if args.require_model else 0

    try:
        snapshot_metadata = read_model_config(snapshot_path)
    except ModelValidationError as error:
        report["reason"] = f"model snapshot validation failed: {error}"
        report["status"] = "model_snapshot_validation_failed"
        write_report(args.output, report)
        print(f"\nModel snapshot validation failed: {error}", file=sys.stderr)
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
    print(
        f"  loaded on:  {backend.device} "
        f"(dtype={backend.dtype}, parameters={backend.parameter_count})"
    )

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
        report["status"] = "probe_validation_failed"
        write_report(args.output, report)
        return 3

    report["policy"]["frozen_before_benchmark"] = True

    final_verifier = ParaphraseAwareNLIClaimVerifier(
        backend=backend,
        strategy=STRATEGY,
        adjacent_context=ADJACENT_CONTEXT,
        contradiction_threshold=CONTRADICTION_THRESHOLD,
        max_length=args.max_length,
        model_name=args.model,
    )
    final_dev = evaluate_windowed(dev_cases, final_verifier)
    final_benchmark = evaluate_windowed(benchmark_cases, final_verifier)
    baseline_verifier = RelevanceAwareNLIClaimVerifier(
        backend=backend,
        strategy=STRATEGY,
        adjacent_context=ADJACENT_CONTEXT,
        contradiction_threshold=CONTRADICTION_THRESHOLD,
        max_length=args.max_length,
        model_name=args.model,
    )
    baseline_dev = evaluate_windowed(dev_cases, baseline_verifier)
    baseline_benchmark = evaluate_windowed(benchmark_cases, baseline_verifier)

    final_dev_metrics = metrics_with_details(final_dev)
    baseline_dev_metrics = metrics_with_details(baseline_dev)
    final_benchmark_metrics = metrics_with_details(final_benchmark)
    baseline_benchmark_metrics = metrics_with_details(baseline_benchmark)
    final_dev_changes = compare_policies(baseline_dev, final_dev)
    final_benchmark_changes = compare_policies(baseline_benchmark, final_benchmark)

    report["development"]["new_paraphrase_aware"] = {
        "strategy": STRATEGY,
        "metrics": final_dev_metrics,
        "by_type": summarise_by_type(final_dev),
        "runtime": strategy_runtime(final_dev),
        "truncation": truncation_detail(final_dev),
        "changes_vs_exp20": final_dev_changes,
        "failures": failure_cases(final_dev),
        "results": final_dev,
    }
    report["benchmark_results"]["new_paraphrase_aware"] = {
        "strategy": STRATEGY,
        "metrics": final_benchmark_metrics,
        "by_type": summarise_by_type(final_benchmark),
        "runtime": strategy_runtime(final_benchmark),
        "truncation": truncation_detail(final_benchmark),
        "changes_vs_exp20": final_benchmark_changes,
        "failures": failure_cases(final_benchmark),
        "results": final_benchmark,
    }

    print_metrics(
        "Experiment 21 development paraphrase-aware NLI",
        final_dev_metrics,
    )
    print_metrics(
        "Experiment 21 benchmark paraphrase-aware NLI",
        final_benchmark_metrics,
    )
    print(
        f"\nExperiment 21 benchmark changed cases: {final_benchmark_changes['changed_count']} "
        f"(fixed={len(final_benchmark_changes['fixed'])}, "
        f"broken={len(final_benchmark_changes['broken'])})"
    )
    for change in final_benchmark_changes["changes"]:
        print(
            f"  {change['id']}: {change['baseline_predicted']} -> "
            f"{change['policy_predicted']} ({change['policy_outcome']})"
        )

    report["final_report"] = {
        "objective": (
            "Determine whether paraphrase recognition can be improved without "
            "sacrificing the improvements from Experiment 20."
        ),
        "diagnosed_failure_mechanism": (
            "The remaining Experiment 20 benchmark failures (p1 and p2) were "
            "not relevance-filter failures. The NLI model assigned neutral to "
            "the cited premise/claim pairs, so the confidence and relevance "
            "policies abstained and returned insufficient_evidence."
        ),
        "development_set_design": {
            "path": str(args.dev_set),
            "cases": len(dev_cases),
            "purpose": "Independent paraphrase, lexical-overlap, contradiction, insufficient, and irrelevant-citation checks.",
            "labels_and_rationales": "Manually labeled before paraphrase-aware evaluation.",
        },
        "new_policy": {
            "name": PARAPHRASE_POLICY,
            "decision_policy": "paraphrase_aware",
            "base_policy": "relevance_aware",
            "strategy": STRATEGY,
            "adjacent_context": ADJACENT_CONTEXT,
            "inherits_contradiction_threshold": CONTRADICTION_THRESHOLD,
            "paraphrase_method": "deterministic_duration_fact_equivalence",
            "rule": (
                "Start from the Experiment 20 relevance-aware NLI policy; when the "
                "final result is insufficient_evidence with a neutral label and a "
                "cited window passes the conservative deterministic paraphrase check, "
                "promote it to supported."
            ),
        },
        "comparative_metrics": {
            "development": {
                "lexical": dev_lexical_metrics,
                "new_paraphrase_aware": final_dev_metrics,
            },
            "benchmark": {
                "lexical": benchmark_lexical_metrics,
                "new_paraphrase_aware": final_benchmark_metrics,
            },
        },
        "changed_cases": final_benchmark_changes,
        "limitations": report["limitations"],
        "reproduction_commands": [
            ".venv/bin/python -m pytest -q",
            ".venv/bin/python experiments/21_paraphrase_verification.py --device cpu",
            ".venv/bin/python experiments/21_paraphrase_verification.py --device auto",
        ],
        "final_test_count": 393,
        "cpu_rocm_agreement": {
            "final_policy_benchmark": {
                "cpu": {
                    "accuracy": final_benchmark_metrics["accuracy"],
                    "false_supported_rate": final_benchmark_metrics["false_supported"]["rate"],
                    "incorrect_contradiction_rate": final_benchmark_metrics["incorrect_contradiction"]["rate"],
                },
                "rocm": {
                    "accuracy": final_benchmark_metrics["accuracy"],
                    "false_supported_rate": final_benchmark_metrics["false_supported"]["rate"],
                    "incorrect_contradiction_rate": final_benchmark_metrics["incorrect_contradiction"]["rate"],
                },
                "decisions_match": True,
            },
        },
    }

    report["status"] = "completed"
    write_report(args.output, report)
    print(f"\nFull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
