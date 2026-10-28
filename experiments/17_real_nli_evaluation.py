"""Experiment 17: real-model NLI evaluation on the Experiment 15 benchmark.

The lexical baseline and a real pretrained NLI cross-encoder are evaluated on
exactly the same 33 cases with the shared Experiment 15/16 metric code. The
recommended model is ``cross-encoder/nli-MiniLM2-L6-H768``.

Acquisition is explicit. Imports and unit tests never download a model::

    .venv/bin/python experiments/17_real_nli_evaluation.py --download-model

Then run the evaluation (CPU is the conservative default for repeatability)::

    .venv/bin/python experiments/17_real_nli_evaluation.py --device cpu

On this AMD machine, ``--device auto`` uses ROCm through PyTorch's CUDA API.
If a non-CPU device fails to load, the runner can retry the same model on CPU
and records both the error and the fallback. It never substitutes a different
model.

The full report includes raw per-passage logits, softmax scores, aggregate
scores, lexical/NLI disagreements, timing, dependency versions, the model
revision, and an audit of the old Experiment 15 0.636 reporting error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.verification_utils.eval import (  # noqa: E402
    compute_metrics,
    evaluate_lexical,
    load_benchmark,
)
from experiments.verification_utils.nli_eval import (  # noqa: E402
    evaluate_nli,
    find_disagreements,
    summarise_by_type,
)
from experiments.verification_utils.real_nli_eval import (  # noqa: E402
    EXPECTED_LICENSE,
    ModelValidationError,
    build_environment_info,
    evaluate_probes,
    file_sha256,
    focus_categories,
    lexical_audit,
    read_model_config,
    resolve_cached_snapshot,
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
DEFAULT_OUTPUT = ROOT / "experiments" / "17_real_nli_evaluation" / "report.json"
DOWNLOAD_PATTERNS = ["*.json", "*.txt", "*.safetensors", "README.md"]


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
        "--download-model",
        action="store_true",
        help="explicitly download and validate the selected model, then exit",
    )
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
        help="path for the full JSON report (default: Experiment 17 report)",
    )
    return parser.parse_args(argv)


def matrix_str(row, labels):
    return " ".join(f"{row.get(label, 0):>12d}" for label in labels)


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(
        f"  accuracy={metrics['accuracy']:.3f}, "
        f"macro-F1={metrics['macro_f1']:.3f}, "
        f"false-supported={metrics['false_supported_rate']:.3f}"
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
        row = matrix_str(metrics["confusion_matrix"][label], LABELS)
        print(f"    {label:20s} {row}")


def print_by_type(summary):
    print("  By case type:")
    for case_type, values in summary.items():
        print(
            f"    {case_type}: {values['accuracy']:.3f} "
            f"({values['correct']}/{values['total']})"
        )


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _download_model(model_name):
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(model_name, files_metadata=True)
    card_data = info.cardData or {}
    license_name = card_data.get("license")
    print("Experiment 17: explicit model acquisition")
    print(f"  repository: {info.id}")
    print(f"  revision:   {info.sha}")
    print(f"  license:    {license_name}")
    print(f"  library:    {info.library_name}")
    print(f"  pipeline:   {info.pipeline_tag}")
    if str(license_name).lower() != EXPECTED_LICENSE:
        raise ModelValidationError(
            f"expected license {EXPECTED_LICENSE!r}; got {license_name!r}"
        )
    path = snapshot_download(
        repo_id=model_name,
        revision=info.sha,
        allow_patterns=DOWNLOAD_PATTERNS,
    )
    metadata = read_model_config(path)
    print(f"  snapshot:   {path}")
    print(f"  id2label:   {metadata['id2label']}")
    print(f"  weights:    {metadata['weights_file']}")
    print(f"  size:       {metadata['size_bytes'] / (1024 * 1024):.1f} MiB")
    print("\nModel downloaded and validated. No other model was substituted.")
    return 0


def _make_verifier(model_name, snapshot_path, args, device):
    backend = TransformersNLIBackend(
        str(snapshot_path),
        device=device,
        dtype=args.dtype,
        local_files_only=True,
        max_length=args.max_length,
    )
    return NLIClaimVerifier(
        model_name=model_name,
        backend=backend,
        device=device,
        max_length=args.max_length,
        aggregation="conservative",
    )


def _load_with_optional_cpu_fallback(model_name, snapshot_path, args):
    attempts = []
    device = args.device
    verifier = _make_verifier(model_name, snapshot_path, args, device)
    started = time.perf_counter()
    try:
        verifier.load()
        attempts.append(
            {
                "device": device,
                "status": "loaded",
                "error": None,
                "load_seconds": time.perf_counter() - started,
            }
        )
        return verifier, attempts
    except NLIModelUnavailable as error:
        attempts.append(
            {
                "device": device,
                "status": "failed",
                "error": str(error),
                "load_seconds": time.perf_counter() - started,
            }
        )
        if device == "cpu" or args.no_cpu_fallback:
            raise

    fallback = _make_verifier(model_name, snapshot_path, args, "cpu")
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


def _skip_message(model_name, error, output):
    print("\nNLI status: SKIPPED - the requested model is not available locally.")
    print(f"  requested model: {model_name}")
    print(f"  exact error: {error}")
    print("\nNothing was downloaded. To opt in explicitly, run:")
    print("  .venv/bin/python experiments/17_real_nli_evaluation.py --download-model")
    if output:
        print(f"\nSkipped report will be written to {output}")


def _runtime_summary(nli_results):
    case_times = [result["runtime_seconds"] for result in nli_results]
    total = sum(case_times)
    return {
        "cases": len(case_times),
        "inference_seconds": total,
        "seconds_per_case": total / len(case_times) if case_times else 0.0,
        "milliseconds_per_case": (
            1000.0 * total / len(case_times) if case_times else 0.0
        ),
        "min_case_seconds": min(case_times) if case_times else 0.0,
        "max_case_seconds": max(case_times) if case_times else 0.0,
    }


def main(argv=None):
    args = parse_args(argv)

    if args.download_model:
        try:
            return _download_model(args.model)
        except Exception as error:
            print(f"Model download/validation failed: {error}", file=sys.stderr)
            return 2

    cases_path = args.benchmark or (
        ROOT / "experiments" / "15_verification_benchmark" / "benchmark.json"
    )
    cases = load_benchmark(cases_path)
    lexical_results = evaluate_lexical(cases)
    lexical_metrics = compute_metrics(lexical_results)
    lexical_by_type = summarise_by_type(lexical_results)
    audit = lexical_audit(lexical_results)

    print("=" * 78)
    print("Experiment 17: Real-Model NLI Evaluation")
    print("=" * 78)
    print(f"Benchmark: {len(cases)} cases ({cases_path})")
    print_metrics("Lexical baseline", lexical_metrics)
    print_by_type(lexical_by_type)
    print("\nLexical audit:")
    print(
        f"  correct={audit['correct']}/{audit['total']} "
        f"accuracy={audit['accuracy']:.3f} "
        f"confusion-diagonal={audit['confusion_diagonal']}"
    )
    print(f"  0.636 would require {audit['needed_for_0_636']}/33 correct.")
    print(f"  {audit['note']}")

    report = {
        "experiment": "17_real_nli_evaluation",
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
            "by_type": lexical_by_type,
            "audit": audit,
            "results": lexical_results,
        },
        "model": None,
        "probes": None,
        "nli": None,
        "disagreements": [],
        "runtime": None,
        "calibration": {
            "softmax_is_calibrated_probability": False,
            "development_set_used": False,
            "note": (
                "Softmax scores describe the model's premise/hypothesis "
                "relationship only. No temperature scaling or threshold tuning "
                "was performed on the 33-case benchmark. Calibration on an "
                "independent development set remains future work."
            ),
        },
    }

    try:
        snapshot_path = resolve_cached_snapshot(args.model)
    except Exception as error:
        report["reason"] = f"model not available locally: {error}"
        _skip_message(args.model, error, args.output)
        _write_report(args.output, report)
        print(f"Skipped report written to {args.output}")
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
    }
    print("\nModel snapshot:")
    print(f"  repository: {args.model}")
    print(f"  revision:   {snapshot_metadata['revision']}")
    print(f"  license:    {snapshot_metadata['license']}")
    print(f"  id2label:   {snapshot_metadata['id2label']}")
    print(f"  snapshot:   {snapshot_metadata['snapshot_path']}")

    try:
        verifier, load_attempts = _load_with_optional_cpu_fallback(
            args.model, snapshot_path, args
        )
    except NLIModelUnavailable as error:
        report["reason"] = f"model could not be loaded: {error}"
        report["model"]["load_attempts"] = []
        print(f"\nModel load failed: {error}", file=sys.stderr)
        _write_report(args.output, report)
        return 2

    model_metadata = verifier.metadata()
    report["model"]["backend"] = model_metadata
    report["model"]["load_attempts"] = load_attempts
    report["model"]["cpu_fallback"] = any(
        attempt["device"] != args.device and attempt["status"] == "loaded"
        for attempt in load_attempts
    )
    print(
        f"  loaded on:  {model_metadata['device']} "
        f"(dtype={model_metadata['dtype']}, "
        f"parameters={model_metadata['parameter_count']})"
    )
    for attempt in load_attempts:
        print(
            f"    {attempt['device']}: {attempt['status']} "
            f"in {attempt['load_seconds']:.3f}s"
        )
        if attempt["error"]:
            print(f"      error: {attempt['error']}")

    probes = evaluate_probes(verifier)
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
            "\nProbe validation failed; stopping before benchmark evaluation.",
            file=sys.stderr,
        )
        report["status"] = "probe_validation_failed"
        _write_report(args.output, report)
        return 3

    inference_started = time.perf_counter()
    nli_results = evaluate_nli(cases, verifier)
    wall_seconds = time.perf_counter() - inference_started
    nli_metrics = compute_metrics(nli_results)
    nli_by_type = summarise_by_type(nli_results)
    disagreements = find_disagreements(lexical_results, nli_results)
    runtime = _runtime_summary(nli_results)
    runtime["wall_seconds"] = wall_seconds
    runtime["load_seconds"] = sum(
        attempt["load_seconds"]
        for attempt in load_attempts
        if attempt["status"] == "loaded"
    )

    print_metrics("Real NLI", nli_metrics)
    print_by_type(nli_by_type)
    print("\nFocused failure categories:")
    for case_type, values in focus_categories(nli_by_type).items():
        print(
            f"  {case_type}: {values['accuracy']:.3f} "
            f"({values['correct']}/{values['total']})"
        )
    print(
        f"\nRuntime: load={runtime['load_seconds']:.3f}s, "
        f"inference={runtime['inference_seconds']:.3f}s for "
        f"{runtime['cases']} cases "
        f"({runtime['milliseconds_per_case']:.1f} ms/case)"
    )
    error_count = sum(
        1
        for result in nli_results
        if result["predicted"] in {"backend_unavailable", "verification_error"}
    )
    if error_count:
        print(f"Backend/verification errors: {error_count}")

    print("\nDisagreements with the lexical baseline:")
    if not disagreements:
        print("  None")
    for disagreement in disagreements:
        confidence = disagreement["decision_confidence"]
        confidence_text = "n/a" if confidence is None else f"{confidence:.3f}"
        print(
            f"  {disagreement['id']} [{disagreement['type']}]: "
            f"expected={disagreement['expected']}, "
            f"lexical={disagreement['lexical_predicted']}, "
            f"nli={disagreement['nli_predicted']} "
            f"(label={disagreement['nli_label']}, conf={confidence_text})"
        )

    report.update(
        {
            "status": "completed",
            "nli": {
                "metrics": nli_metrics,
                "by_type": nli_by_type,
                "focus_categories": focus_categories(nli_by_type),
                "error_count": error_count,
                "results": nli_results,
            },
            "disagreements": disagreements,
            "runtime": runtime,
        }
    )
    _write_report(args.output, report)
    print(f"\nFull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
