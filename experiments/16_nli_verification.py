"""Experiment 16: NLI-based evidence verification and benchmark comparison.

The lexical verifier from Experiment 14 remains the default in the library.
This experiment opts in to the optional NLI backend, validates citations before
any model call, and compares NLI with the lexical baseline on the exact same
33 manually labelled Experiment 15 cases.

No model is downloaded automatically. The recommended model is
``cross-encoder/nli-MiniLM2-L6-H768`` (Apache-2.0, about 82M parameters,
SNLI/MultiNLI cross-encoder, RoBERTa tokenizer). To opt in to obtaining it::

    .venv/bin/python experiments/16_nli_verification.py --download-model

Then run the benchmark (CPU is the default-safe alternative to ROCm)::

    .venv/bin/python experiments/16_nli_verification.py --device cpu

On this project's AMD PyTorch build, ``--device cuda:0`` uses ROCm through
PyTorch's CUDA-compatible API; ``--device rocm`` is accepted as an alias.
``--device auto`` also selects ``cuda:0`` when PyTorch reports it.

If the model is not present locally, the command still prints the lexical
baseline and exits cleanly with an explicit ``skipped`` status. Use
``--require-model`` to make a missing model exit non-zero instead.
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
from minigrad.nli_verification import (  # noqa: E402
    DEFAULT_NLI_MODEL,
    NLIModelUnavailable,
    NLIClaimVerifier,
    RECOMMENDED_NLI_MODELS,
    TransformersNLIBackend,
)

LABELS = [
    "supported",
    "contradicted",
    "insufficient_evidence",
    "invalid_citation",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="benchmark JSON (default: Experiment 15 benchmark)",
    )
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda:0, or rocm (PyTorch's ROCm API name is cuda)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--aggregation",
        choices=("conservative",),
        default="conservative",
        help="conservative: contradiction vetoes support and conflicts are mixed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the full report, raw scores, and per-case results as JSON",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="explicitly download the selected model, then exit",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow the benchmark load to access the network if model is missing",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="exit non-zero when the model is unavailable instead of skipping",
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


def _skip_message(model_name, output):
    print("\nNLI status: SKIPPED - no compatible NLI model is available locally.")
    print(f"  requested model: {model_name}")
    print(f"  recommended model: {DEFAULT_NLI_MODEL}")
    for key, details in RECOMMENDED_NLI_MODELS.items():
        print(
            f"  {key}: {details['model_name']} "
            f"(license={details['license']}, params={details['parameters']})"
        )
    print("\nNothing was downloaded. To opt in explicitly, run:")
    print("  .venv/bin/python experiments/16_nli_verification.py --download-model")
    print("Then rerun this benchmark:")
    print("  .venv/bin/python experiments/16_nli_verification.py --device cpu")
    if output:
        print(f"\nSkipped report written to {output}")


def main(argv=None):
    args = parse_args(argv)

    if args.download_model:
        backend = TransformersNLIBackend(
            args.model,
            device=args.device,
            dtype=args.dtype,
            local_files_only=False,
            max_length=args.max_length,
        )
        try:
            path = backend.download()
        except NLIModelUnavailable as error:
            print(f"Model download failed: {error}", file=sys.stderr)
            return 2
        print(f"Downloaded {args.model} into {path}")
        return 0

    cases = load_benchmark(args.benchmark)
    lexical_results = evaluate_lexical(cases)
    lexical_metrics = compute_metrics(lexical_results)

    print("=" * 78)
    print("Experiment 16: NLI Evidence Verification Benchmark")
    print("=" * 78)
    print(f"Benchmark: {len(cases)} cases (same labels as Experiment 15)")
    print_metrics("Lexical baseline", lexical_metrics)
    print_by_type(summarise_by_type(lexical_results))

    verifier = NLIClaimVerifier(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=not args.allow_download,
        max_length=args.max_length,
        aggregation=args.aggregation,
    )

    report = {
        "status": "skipped",
        "benchmark": str(args.benchmark or "experiments/15_verification_benchmark/benchmark.json"),
        "model": verifier.metadata(),
        "lexical": {
            "metrics": lexical_metrics,
            "by_type": summarise_by_type(lexical_results),
            "results": lexical_results,
        },
        "nli": None,
        "disagreements": [],
    }

    if not args.allow_download and not verifier.available:
        report["reason"] = "model not available in local path or Hugging Face cache"
        _skip_message(args.model, args.output)
        if args.output:
            _write_report(args.output, report)
        return 2 if args.require_model else 0

    load_started = time.perf_counter()
    try:
        verifier.load()
    except NLIModelUnavailable as error:
        report["reason"] = str(error)
        _skip_message(args.model, args.output)
        if args.output:
            _write_report(args.output, report)
        return 2 if args.require_model else 0
    load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    nli_results = evaluate_nli(cases, verifier)
    inference_seconds = time.perf_counter() - inference_started
    nli_metrics = compute_metrics(nli_results)
    by_type = summarise_by_type(nli_results)
    disagreements = find_disagreements(lexical_results, nli_results)

    model_metadata = verifier.metadata()
    model_metadata["load_seconds"] = load_seconds
    model_metadata["source_load_seconds"] = getattr(
        verifier.backend, "load_seconds", None
    )

    print_metrics("NLI verifier", nli_metrics)
    print_by_type(by_type)

    error_count = sum(
        1
        for result in nli_results
        if result["predicted"] in {"backend_unavailable", "verification_error"}
    )
    print(
        f"\nRuntime: load={load_seconds:.3f}s, "
        f"inference={inference_seconds:.3f}s for {len(cases)} cases"
    )
    print(
        f"Model: {model_metadata['model_name']}, "
        f"device={model_metadata['device']}, dtype={model_metadata['dtype']}, "
        f"parameters={model_metadata['parameter_count']}, "
        f"local_files_only={model_metadata['local_files_only']}"
    )
    if error_count:
        print(f"Backend/verification errors: {error_count}")

    print("\nDisagreements with the lexical baseline:")
    if not disagreements:
        print("  None")
    for disagreement in disagreements:
        confidence = disagreement["decision_confidence"]
        confidence_text = (
            "n/a" if confidence is None else f"{confidence:.3f}"
        )
        print(
            f"  {disagreement['id']} [{disagreement['type']}]: "
            f"expected={disagreement['expected']}, "
            f"lexical={disagreement['lexical_predicted']}, "
            f"nli={disagreement['nli_predicted']} "
            f"(label={disagreement['nli_label']}, conf={confidence_text})"
        )
        print(f"    claim: {disagreement['claim']}")
        print(f"    rationale: {disagreement['rationale']}")

    report.update(
        {
            "status": "completed",
            "runtime_seconds": {
                "load": load_seconds,
                "inference": inference_seconds,
                "total": load_seconds + inference_seconds,
            },
            "model": model_metadata,
            "nli": {
                "metrics": nli_metrics,
                "by_type": by_type,
                "error_count": error_count,
                "results": nli_results,
            },
            "disagreements": disagreements,
        }
    )
    if args.output:
        _write_report(args.output, report)
        print(f"\nFull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
