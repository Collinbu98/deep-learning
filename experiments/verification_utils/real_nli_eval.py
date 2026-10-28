"""Helpers for the Experiment 17 real-model NLI evaluation.

The helpers in this module are offline unless a caller explicitly resolves or
downloads a Hugging Face snapshot. They keep model-card/config validation,
lexical-result auditing, and probe validation separate from the CLI so the
evaluation logic can be unit tested with temporary files and fake predictors.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.verification_utils.eval import compute_metrics, make_passage
from minigrad.claim_verification import Claim
from minigrad.nli_verification import DEFAULT_NLI_MODEL, normalize_label_map

__all__ = [
    "EXPECTED_LICENSE",
    "FOCUS_CASE_TYPES",
    "ModelValidationError",
    "PROBES",
    "REQUIRED_SNAPSHOT_FILES",
    "build_environment_info",
    "evaluate_probes",
    "file_sha256",
    "focus_categories",
    "lexical_audit",
    "model_revision_from_snapshot",
    "read_model_config",
    "resolve_cached_snapshot",
]

EXPECTED_LICENSE = "apache-2.0"
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "README.md",
)
PROBES = (
    {
        "id": "probe_entailment",
        "premise": "Jupiter has 95 known moons.",
        "hypothesis": "Jupiter has 95 known moons.",
        "expected_label": "entailment",
        "expected_status": "supported",
    },
    {
        "id": "probe_contradiction",
        "premise": "Mars has two small moons, Phobos and Deimos.",
        "hypothesis": "Mars has one moon.",
        "expected_label": "contradiction",
        "expected_status": "contradicted",
    },
    {
        "id": "probe_neutral",
        "premise": "Jupiter has 95 known moons.",
        "hypothesis": "The Great Red Spot is a storm on Jupiter.",
        "expected_label": "neutral",
        "expected_status": "insufficient_evidence",
    },
)
FOCUS_CASE_TYPES = (
    "contradiction",
    "numbers",
    "negation",
    "irrelevant_citation",
    "high_overlap_wrong",
    "paraphrase",
    "insufficient",
)


class ModelValidationError(RuntimeError):
    """Raised when a downloaded model snapshot is incomplete or inconsistent."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_revision_from_snapshot(snapshot_path: str | Path) -> str | None:
    """Return the commit revision encoded in a Hugging Face snapshot path."""

    path = Path(snapshot_path)
    if path.parent.name != "snapshots":
        return None
    if re.fullmatch(r"[0-9a-f]{40}", path.name):
        return path.name
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelValidationError(f"could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ModelValidationError(f"{path.name} does not contain a JSON object")
    return value


def _read_license(readme_path: Path) -> str | None:
    try:
        text = readme_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ModelValidationError(f"could not read README.md: {error}") from error
    frontmatter = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)
    if frontmatter is None:
        return None
    match = re.search(
        r"^license\s*:\s*['\"]?([^'\"\n]+)['\"]?\s*$",
        frontmatter.group(1),
        re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def resolve_cached_snapshot(model_name: str) -> str:
    """Resolve a cached model snapshot without ever accessing the network.

    A revision-pinned download may leave the repository without a ``refs/main``
    entry. If the normal local snapshot lookup fails, fall back to the cache
    inventory and select the most recently modified cached revision.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelValidationError(
            "huggingface_hub is not installed"
        ) from error

    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception as lookup_error:
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            raise ModelValidationError(
                f"could not resolve cached model {model_name!r}: {lookup_error}"
            ) from lookup_error
        try:
            cache = scan_cache_dir()
        except Exception as cache_error:
            raise ModelValidationError(
                f"could not inspect Hugging Face cache: {cache_error}; "
                f"initial lookup failed with: {lookup_error}"
            ) from cache_error
        matches = [repo for repo in cache.repos if repo.repo_id == model_name]
        if not matches:
            raise ModelValidationError(
                f"no cached snapshot for {model_name!r}; "
                f"initial lookup failed with: {lookup_error}"
            ) from lookup_error
        revisions = sorted(
            matches[0].revisions,
            key=lambda revision: revision.last_modified,
            reverse=True,
        )
        if not revisions:
            raise ModelValidationError(
                f"cache entry for {model_name!r} has no revisions"
            )
        return str(revisions[0].snapshot_path)


def read_model_config(snapshot_path: str | Path) -> dict[str, Any]:
    """Read and validate a local snapshot's config, labels, tokenizer, license."""

    snapshot = Path(snapshot_path)
    if not snapshot.exists():
        raise ModelValidationError(f"snapshot path does not exist: {snapshot}")

    missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (snapshot / name).exists()]
    weights = next(
        (
            name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (snapshot / name).exists()
        ),
        None,
    )
    if weights is None:
        missing.append("model.safetensors or pytorch_model.bin")
    if missing:
        raise ModelValidationError(
            "snapshot is missing required files: " + ", ".join(missing)
        )

    config = _read_json(snapshot / "config.json")
    try:
        labels = normalize_label_map(config.get("id2label", {}))
    except Exception as error:
        raise ModelValidationError(
            f"model config does not expose a usable NLI id2label map: {error}"
        ) from error

    architectures = config.get("architectures") or []
    if "RobertaForSequenceClassification" not in architectures:
        raise ModelValidationError(
            "expected architectures to include "
            f"RobertaForSequenceClassification; got {architectures}"
        )

    license_name = _read_license(snapshot / "README.md")
    if license_name is None:
        raise ModelValidationError("model card does not declare a license")
    if license_name.lower() != EXPECTED_LICENSE:
        raise ModelValidationError(
            f"expected license {EXPECTED_LICENSE!r}; got {license_name!r}"
        )

    revision = model_revision_from_snapshot(snapshot)
    return {
        "snapshot_path": str(snapshot),
        "revision": revision,
        "model_name": config.get("_name_or_path"),
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "id2label": {str(key): value for key, value in labels.items()},
        "max_position_embeddings": config.get("max_position_embeddings"),
        "license": license_name,
        "weights_file": weights,
        "size_bytes": sum(
            path.stat().st_size
            for path in snapshot.iterdir()
            if path.is_file()
        ),
    }


def lexical_audit(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit the lexical result counts behind the old 0.636 report.

    Experiment 15's narrative summary printed 0.636, but its own frozen
    confusion matrix has a diagonal of 13 + 0 + 3 + 2 = 18 correct out of 33.
    That is 0.545, not 0.636. The latter would require 21 correct. This is a
    reporting discrepancy, not a label-mapping or evaluation change.
    """

    metrics = compute_metrics(results)
    labels = ("supported", "contradicted", "insufficient_evidence", "invalid_citation")
    diagonal = sum(
        metrics["confusion_matrix"][label].get(label, 0) for label in labels
    )
    correct = sum(1 for result in results if result["correct"])
    total = len(results)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "confusion_diagonal": diagonal,
        "needed_for_0_636": round(0.636 * total),
        "note": (
            "Experiment 15's hand-written final report printed accuracy 0.636, "
            "but its own confusion matrix already contained only 18/33 correct "
            "(0.545). The 0.636 figure would require 21/33 correct and is not "
            "reproduced by any stored result or by the current shared evaluator. "
            "The discrepancy is a reporting error, not a label-mapping, "
            "invalid-citation, or insufficient-evidence convention change."
        ),
    }


def evaluate_probes(verifier) -> dict[str, Any]:
    """Run deterministic label-mapping probes through a constructed verifier."""

    results = []
    for probe in PROBES:
        source_id = f"{probe['id']}#0"
        passage = make_passage(source_id, probe["premise"])
        claim = Claim(probe["hypothesis"], (source_id,))
        result = verifier.verify(claim, {source_id: passage})
        results.append(
            {
                **probe,
                "predicted_label": result.label,
                "predicted_status": result.status,
                "passed": (
                    result.label == probe["expected_label"]
                    and result.status == probe["expected_status"]
                ),
                "scores": dict(result.aggregate_scores),
                "per_evidence": [
                    {
                        "label": prediction.label,
                        "scores": dict(prediction.scores),
                        "raw_scores": list(prediction.raw_scores),
                    }
                    for prediction in result.per_evidence
                ],
                "error": result.error,
            }
        )
    return {
        "passed": all(result["passed"] for result in results),
        "results": results,
    }


def focus_categories(by_type: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Select the failure categories requested by Experiment 17."""

    return {
        case_type: dict(by_type[case_type])
        for case_type in FOCUS_CASE_TYPES
        if case_type in by_type
    }


def build_environment_info() -> dict[str, Any]:
    """Record dependency and accelerator details without loading a model."""

    packages = {}
    for package in (
        "torch",
        "transformers",
        "sentence-transformers",
        "huggingface_hub",
        "tokenizers",
        "numpy",
        "sentencepiece",
        "protobuf",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    accelerator: dict[str, Any] = {
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_devices": [],
        "rocm_version": None,
    }
    try:
        import torch
    except ImportError:
        return {"packages": packages, "accelerator": accelerator}

    accelerator["torch_cuda_available"] = bool(torch.cuda.is_available())
    accelerator["torch_cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        accelerator["torch_cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    accelerator["rocm_version"] = getattr(torch.version, "hip", None)
    return {"packages": packages, "accelerator": accelerator}
