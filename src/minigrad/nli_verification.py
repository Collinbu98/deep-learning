"""Optional natural-language-inference verification of claims against evidence.

This module is deliberately separate from citation-ID validation. A citation
ID can be valid while the cited passage is irrelevant, and an invalid citation
must never be repaired by a model that happens to predict entailment. The
verifier therefore validates citations before any model call.

NLI treats each cited passage as the premise and the claim as the hypothesis.
The model's raw label ids are mapped to ``entailment``, ``contradiction``, and
``neutral`` using the model configuration (``id2label``). Label order is never
assumed. A softmax over the model logits is reported as a confidence
distribution; it is **not** a calibrated probability that the claim is true in
the world. It only describes the model's evidence/claim relationship.

Multiple passages
-----------------
Each cited passage is scored independently. The default aggregation is
conservative:

* any entailment plus any contradiction -> ``mixed`` / ``insufficient_evidence``;
* any contradiction and no entailment -> ``contradicted``;
* any entailment and no contradiction -> ``supported``;
* only neutral results -> ``insufficient_evidence``.

Separately neutral passages are never combined into an entailment. A single
entailed premise is enough for ``supported``; other neutral premises do not
add joint support. Raw per-passage predictions remain available in the result.

Model availability
------------------
The recommended default is ``cross-encoder/nli-MiniLM2-L6-H768``: a small
(about 82M parameter) English cross-encoder trained on SNLI and MultiNLI,
released under Apache-2.0. It uses a RoBERTa tokenizer, so it does not require
``sentencepiece``. ``cross-encoder/nli-deberta-v3-small`` (Apache-2.0, about
142M parameters) is a higher-accuracy alternative, but its DeBERTa-v3 tokenizer
needs ``sentencepiece``.

Model loading is lazy. Automatic downloads are disabled by default: the
Transformers backend uses ``local_files_only=True``. To opt in later, run::

    python experiments/16_nli_verification.py --download-model

All unit tests use injected fakes and never download or load a real model.
"""
from __future__ import annotations

import inspect
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from minigrad.claim_verification import Claim, check_citation_validity
from minigrad.llm_answerer import EvidencePassage

__all__ = [
    "DEFAULT_NLI_MODEL",
    "RECOMMENDED_NLI_MODELS",
    "CANONICAL_NLI_LABELS",
    "NLIModelUnavailable",
    "NLIInferenceError",
    "NLIOutputError",
    "NLIPrediction",
    "NLIAggregation",
    "NLIResult",
    "NLIPredictor",
    "FakeNLIBackend",
    "TransformersNLIBackend",
    "NLIClaimVerifier",
    "normalize_label_map",
    "scores_from_logits",
    "make_prediction",
    "aggregate_predictions",
]

DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
RECOMMENDED_NLI_MODELS = {
    "default": {
        "model_name": DEFAULT_NLI_MODEL,
        "license": "apache-2.0",
        "parameters": 82_121_221,
        "notes": (
            "Small SNLI/MultiNLI cross-encoder with a RoBERTa tokenizer; "
            "does not require sentencepiece."
        ),
    },
    "higher_accuracy": {
        "model_name": "cross-encoder/nli-deberta-v3-small",
        "license": "apache-2.0",
        "parameters": 141_897_731,
        "notes": (
            "Higher-accuracy DeBERTa-v3-small alternative; requires the "
            "sentencepiece package for its tokenizer."
        ),
    },
}
CANONICAL_NLI_LABELS = ("entailment", "contradiction", "neutral")


class NLIModelUnavailable(RuntimeError):
    """Raised when the optional NLI model or its dependencies are unavailable."""


class NLIInferenceError(RuntimeError):
    """Raised when the model fails during inference."""


class NLIOutputError(RuntimeError):
    """Raised when labels or scores from a model are malformed."""


class NLIPredictor(Protocol):
    """Minimal injected-predictor interface used by :class:`NLIClaimVerifier`.

    ``predict_batch`` receives ``(premise, hypothesis)`` pairs and returns one
    row of raw logits per pair. Rows are ordered by the model's ``id2label``
    mapping, which the predictor must expose as ``id2label``.
    """

    model_name: str
    id2label: Mapping[int, str]

    def predict_batch(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class NLIPrediction:
    """NLI prediction for one evidence/claim pair."""

    label: str
    label_id: int
    scores: dict[str, float]
    raw_scores: tuple[float, ...]
    evidence_id: str = ""
    premise_excerpt: str = ""
    model_name: str = ""
    device: str = ""

    @property
    def confidence(self) -> float:
        return self.scores[self.label]


@dataclass(frozen=True)
class NLIAggregation:
    """Aggregated decision over one or more per-evidence NLI predictions."""

    label: str
    status: str
    explanation: str
    scores: dict[str, float]
    decision_confidence: float | None


@dataclass(frozen=True)
class NLIResult:
    """Structured claim-verification result from the NLI backend."""

    claim: Claim
    status: str
    explanation: str
    evidence_excerpts: tuple[tuple[str, str], ...]
    cited_passages_exist: bool
    label: str | None
    aggregate_scores: dict[str, float]
    decision_confidence: float | None
    per_evidence: tuple[NLIPrediction, ...]
    aggregation: str
    model_metadata: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": asdict(self.claim),
            "status": self.status,
            "explanation": self.explanation,
            "evidence_excerpts": [list(item) for item in self.evidence_excerpts],
            "cited_passages_exist": self.cited_passages_exist,
            "label": self.label,
            "aggregate_scores": dict(self.aggregate_scores),
            "decision_confidence": self.decision_confidence,
            "per_evidence": [asdict(item) for item in self.per_evidence],
            "aggregation": self.aggregation,
            "model_metadata": dict(self.model_metadata),
            "error": self.error,
        }


def _canonical_label(label: Any) -> str:
    if not isinstance(label, str):
        raise NLIOutputError(f"model label {label!r} is not a string")
    key = label.strip().lower()
    if key not in CANONICAL_NLI_LABELS:
        raise NLIOutputError(
            f"model label {label!r} cannot be mapped to entailment, "
            "contradiction, or neutral; check the model's id2label metadata"
        )
    return key


def normalize_label_map(id2label: Mapping[Any, Any]) -> dict[int, str]:
    """Validate and normalise a model's ``id2label`` mapping.

    Keys may be integers or numeric strings, as produced by different
    Transformers configuration loaders. The labels themselves must be the
    canonical NLI names; ``LABEL_0``-style placeholders are rejected rather
    than guessed.
    """

    if not isinstance(id2label, Mapping) or not id2label:
        raise NLIOutputError("model does not provide an id2label mapping")
    normalized: dict[int, str] = {}
    for raw_id, raw_label in id2label.items():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise NLIOutputError(
                f"model label id {raw_id!r} is not an integer"
            ) from error
        if label_id in normalized:
            raise NLIOutputError(f"duplicate model label id {label_id}")
        normalized[label_id] = _canonical_label(raw_label)
    if len(normalized) != 3 or set(normalized.values()) != set(
        CANONICAL_NLI_LABELS
    ):
        raise NLIOutputError(
            "id2label must contain exactly entailment, contradiction, and "
            f"neutral; got {normalized}"
        )
    return normalized


def scores_from_logits(
    raw_scores: Sequence[float], id2label: Mapping[Any, Any]
) -> tuple[str, int, dict[str, float], tuple[float, ...]]:
    """Map raw logits to a canonical label and a softmax confidence distribution."""

    label_map = normalize_label_map(id2label)
    if isinstance(raw_scores, (str, bytes)):
        raise NLIOutputError("model scores must be a sequence of numbers")
    try:
        values = tuple(float(value) for value in raw_scores)
    except (TypeError, ValueError) as error:
        raise NLIOutputError("model scores contain a non-numeric value") from error
    if len(values) != len(label_map):
        raise NLIOutputError(
            f"expected {len(label_map)} scores, received {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise NLIOutputError("model scores contain a non-finite value")

    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise NLIOutputError("model scores cannot be normalised")
    probabilities = {
        label_map[label_id]: exponentials[label_id] / total
        for label_id in sorted(label_map)
    }
    label_id = max(
        sorted(label_map), key=lambda candidate: probabilities[label_map[candidate]]
    )
    return label_map[label_id], label_id, probabilities, values


def make_prediction(
    raw_scores: Sequence[float],
    id2label: Mapping[Any, Any],
    *,
    evidence_id: str = "",
    premise_excerpt: str = "",
    model_name: str = "",
    device: str = "",
) -> NLIPrediction:
    """Build an :class:`NLIPrediction` and validate the raw score row."""

    label, label_id, scores, values = scores_from_logits(raw_scores, id2label)
    return NLIPrediction(
        label=label,
        label_id=label_id,
        scores=scores,
        raw_scores=values,
        evidence_id=evidence_id,
        premise_excerpt=premise_excerpt,
        model_name=model_name,
        device=device,
    )


def aggregate_predictions(
    predictions: Sequence[NLIPrediction], aggregation: str = "conservative"
) -> NLIAggregation:
    """Aggregate independent evidence/claim predictions conservatively.

    ``conservative`` is currently the only policy. Conflicting entailment and
    contradiction produce ``mixed`` rather than silently selecting the most
    favourable evidence. Neutral predictions never add up to entailment.
    """

    if aggregation != "conservative":
        raise ValueError(f"unsupported aggregation strategy {aggregation!r}")
    prediction_list = tuple(predictions)
    if not prediction_list:
        raise ValueError("at least one prediction is required")

    labels = {prediction.label for prediction in prediction_list}
    scores = {
        label: sum(prediction.scores[label] for prediction in prediction_list)
        / len(prediction_list)
        for label in CANONICAL_NLI_LABELS
    }

    if "entailment" in labels and "contradiction" in labels:
        return NLIAggregation(
            label="mixed",
            status="insufficient_evidence",
            explanation=(
                "Cited passages disagree: the model predicted both entailment "
                "and contradiction, so the claim is not accepted."
            ),
            scores=scores,
            decision_confidence=None,
        )
    if "contradiction" in labels:
        confidence = max(
            prediction.scores["contradiction"]
            for prediction in prediction_list
            if prediction.label == "contradiction"
        )
        return NLIAggregation(
            label="contradiction",
            status="contradicted",
            explanation=(
                "At least one cited passage contradicts the claim and none "
                "entails it."
            ),
            scores=scores,
            decision_confidence=confidence,
        )
    if "entailment" in labels:
        confidence = max(
            prediction.scores["entailment"]
            for prediction in prediction_list
            if prediction.label == "entailment"
        )
        return NLIAggregation(
            label="entailment",
            status="supported",
            explanation=(
                "At least one cited passage entails the claim and none "
                "contradicts it."
            ),
            scores=scores,
            decision_confidence=confidence,
        )
    confidence = max(
        prediction.scores["neutral"] for prediction in prediction_list
    )
    return NLIAggregation(
        label="neutral",
        status="insufficient_evidence",
        explanation=(
            "The model found neither entailment nor contradiction in the "
            "cited evidence."
        ),
        scores=scores,
        decision_confidence=confidence,
    )


class FakeNLIBackend:
    """Deterministic in-memory predictor for offline tests.

    ``outputs`` may be a sequence of raw score rows (one per pair) or a
    callable that receives the list of pairs and returns score rows. With no
    outputs every pair receives a neutral logit row.
    """

    def __init__(
        self,
        outputs: Sequence[Sequence[float]] | None = None,
        id2label: Mapping[Any, Any] | None = None,
        model_name: str = "fake-nli",
    ):
        self.model_name = model_name
        self.device = "fake"
        self.dtype = "fake"
        self.id2label = normalize_label_map(
            id2label
            or {0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        self._outputs = outputs
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def predict_batch(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[Sequence[float]]:
        pair_list = tuple(pairs)
        self.calls.append(pair_list)
        if self._outputs is not None:
            if callable(self._outputs):
                return self._outputs(list(pair_list))
            return self._outputs
        neutral_id = next(
            label_id
            for label_id, label in self.id2label.items()
            if label == "neutral"
        )
        return [
            [1.0 if label_id == neutral_id else 0.0 for label_id in sorted(self.id2label)]
            for _ in pair_list
        ]


class TransformersNLIBackend:
    """Lazy Hugging Face sequence-classification backend.

    The model is loaded only on first use. ``local_files_only`` defaults to
    ``True``, so constructing or running this backend never downloads a model.
    The PyTorch ``cuda`` device name is used for both CUDA and ROCm builds;
    on this project's AMD machine that means ROCm. ``device='rocm'`` is
    accepted as an explicit alias for ``cuda:0``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = True,
        max_length: int = 512,
    ):
        if not model_name:
            raise ValueError("model_name must not be empty")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self.model_name = model_name
        self.requested_device = device
        self.requested_dtype = dtype
        self.local_files_only = local_files_only
        self.max_length = max_length
        self.id2label: dict[int, str] | None = None
        self.parameter_count: int | None = None
        self.load_seconds: float | None = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device: str | None = None
        self._dtype_name: str | None = None

    @property
    def device(self) -> str:
        return self._device or self.requested_device

    @property
    def dtype(self) -> str:
        return self._dtype_name or self.requested_dtype

    def is_available(self) -> bool:
        """Return whether the model can be loaded from a local path/cache."""

        if self._model is not None:
            return True
        local_path = Path(self.model_name).expanduser()
        if local_path.exists():
            return True
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return False
        try:
            snapshot_download(
                repo_id=self.model_name,
                local_files_only=True,
            )
            return True
        except Exception:
            return False

    def download(self) -> str:
        """Explicitly download the model into the Hugging Face cache."""

        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise NLIModelUnavailable(
                "huggingface_hub is not installed; install transformers or "
                "huggingface_hub to download a model"
            ) from error
        try:
            return snapshot_download(repo_id=self.model_name)
        except Exception as error:
            raise NLIModelUnavailable(
                f"could not download NLI model {self.model_name!r}: {error}"
            ) from error

    def _resolve_device(self, torch) -> str:
        requested = self.requested_device
        if requested in {"rocm", "hip"}:
            requested = "cuda:0"
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise NLIModelUnavailable(
                    f"requested {requested!r}, but PyTorch reports no "
                    "CUDA/ROCm device; use device='cpu' for CPU inference"
                )
            try:
                index = int(requested.split(":", 1)[1]) if ":" in requested else 0
            except ValueError as error:
                raise NLIModelUnavailable(
                    f"invalid device {requested!r}"
                ) from error
            if index >= torch.cuda.device_count():
                raise NLIModelUnavailable(
                    f"requested {requested!r}, but only "
                    f"{torch.cuda.device_count()} device(s) are available"
                )
        return requested

    def _resolve_dtype(self, torch, device: str):
        if self.requested_dtype == "auto":
            return torch.float16 if device.startswith("cuda") else torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.requested_dtype]

    def _load_error(self, error: Exception) -> NLIModelUnavailable:
        return NLIModelUnavailable(
            f"could not load NLI model {self.model_name!r} on "
            f"{self.device}: {error}. Automatic downloads are disabled by "
            "default. To opt in later, run "
            "`python experiments/16_nli_verification.py --download-model`."
        )

    def load(self):
        """Load tokenizer and model on first use; return the model."""

        if self._model is not None:
            return self._model
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise NLIModelUnavailable(
                "the optional NLI dependencies are unavailable; install "
                "`transformers` and a PyTorch build for this machine"
            ) from error

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)
        load_kwargs = {"local_files_only": self.local_files_only}
        started = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **load_kwargs
            )
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, dtype=dtype, **load_kwargs
                )
            except TypeError:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, torch_dtype=dtype, **load_kwargs
                )
            model.to(device)
            model.eval()
        except Exception as error:
            raise self._load_error(error) from error

        try:
            id2label = normalize_label_map(model.config.id2label)
        except Exception as error:
            raise NLIModelUnavailable(
                f"model {self.model_name!r} does not expose usable NLI "
                f"id2label metadata: {error}"
            ) from error

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype_name = str(dtype).replace("torch.", "")
        self.id2label = id2label
        self.parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        self.load_seconds = time.perf_counter() - started
        return model

    def input_token_counts(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[int]:
        """Return untruncated token counts for each premise/hypothesis pair."""

        pair_list = [(str(premise), str(hypothesis)) for premise, hypothesis in pairs]
        if not pair_list:
            return []
        self.load()
        try:
            encoded = self._tokenizer(
                [premise for premise, _ in pair_list],
                [hypothesis for _, hypothesis in pair_list],
                padding=False,
                truncation=False,
            )
        except Exception as error:
            raise NLIInferenceError(
                f"token counting failed for {self.model_name!r}: {error}"
            ) from error
        return [len(ids) for ids in encoded["input_ids"]]

    def predict_batch(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[Sequence[float]]:
        """Return raw logits for each ``(premise, hypothesis)`` pair."""

        pair_list = [(str(premise), str(hypothesis)) for premise, hypothesis in pairs]
        if not pair_list:
            return []
        self.load()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        max_length = self.max_length
        model_max = getattr(tokenizer, "model_max_length", max_length)
        if isinstance(model_max, int) and 0 < model_max < 1_000_000:
            max_length = min(max_length, model_max)
        try:
            inputs = tokenizer(
                [premise for premise, _ in pair_list],
                [hypothesis for _, hypothesis in pair_list],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self._device) for key, value in inputs.items()
            }
        except Exception as error:
            raise NLIInferenceError(
                f"tokenisation failed for {self.model_name!r}: {error}"
            ) from error

        try:
            with torch.inference_mode():
                logits = model(**inputs).logits
            rows = logits.detach().float().cpu().tolist()
        except Exception as error:
            raise NLIInferenceError(
                f"NLI inference failed for {self.model_name!r} on "
                f"{self._device}: {error}"
            ) from error
        if not isinstance(rows, list) or len(rows) != len(pair_list):
            raise NLIInferenceError(
                "model returned a different number of score rows than pairs"
            )
        return rows


def _excerpt(text: str, limit: int = 200) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


class NLIClaimVerifier:
    """Validate citations, then verify a claim against cited evidence with NLI.

    ``backend`` may be any object implementing :class:`NLIPredictor`; this is
    how tests inject deterministic fakes. If no backend is supplied, a lazy
    :class:`TransformersNLIBackend` is created with downloads disabled unless
    ``local_files_only=False`` is passed explicitly.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        backend: NLIPredictor | None = None,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = True,
        max_length: int = 512,
        aggregation: str = "conservative",
    ):
        if aggregation != "conservative":
            raise ValueError(f"unsupported aggregation strategy {aggregation!r}")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if backend is None:
            self.model_name = model_name or DEFAULT_NLI_MODEL
            self.backend = TransformersNLIBackend(
                self.model_name,
                device=device,
                dtype=dtype,
                local_files_only=local_files_only,
                max_length=max_length,
            )
        else:
            self.backend = backend
            self.model_name = model_name or getattr(
                backend, "model_name", "injected-nli"
            )
        self.device = device
        self.max_length = max_length
        self.aggregation = aggregation

    @property
    def available(self) -> bool:
        checker = getattr(self.backend, "is_available", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return True

    def is_available(self) -> bool:
        return self.available

    def load(self):
        loader = getattr(self.backend, "load", None)
        if callable(loader):
            return loader()
        return self.backend

    def metadata(self) -> dict[str, Any]:
        backend = self.backend
        raw_mapping = getattr(backend, "id2label", None)
        label_map: dict[str, str] | None = None
        if raw_mapping:
            try:
                label_map = {
                    str(label_id): label
                    for label_id, label in normalize_label_map(raw_mapping).items()
                }
            except NLIOutputError:
                label_map = {
                    str(label_id): str(label)
                    for label_id, label in raw_mapping.items()
                }
        return {
            "model_name": getattr(backend, "model_name", self.model_name),
            "device": getattr(backend, "device", ""),
            "dtype": getattr(backend, "dtype", ""),
            "local_files_only": getattr(backend, "local_files_only", None),
            "id2label": label_map,
            "max_length": self.max_length,
            "parameter_count": getattr(backend, "parameter_count", None),
            "load_seconds": getattr(backend, "load_seconds", None),
            "aggregation": self.aggregation,
        }

    def _id2label(self) -> dict[int, str]:
        mapping = getattr(self.backend, "id2label", None)
        if mapping is None:
            mapping = getattr(self.backend, "label_map", None)
        if mapping is None:
            raise NLIOutputError(
                "NLI backend does not expose an id2label label mapping"
            )
        return normalize_label_map(mapping)

    def _predict(self, pairs: Sequence[tuple[str, str]]):
        predictor = getattr(self.backend, "predict_batch", None)
        if predictor is None:
            predictor = getattr(self.backend, "predict", None)
        if not callable(predictor):
            raise NLIInferenceError(
                "NLI backend must provide predict_batch(pairs) or predict(pairs)"
            )
        try:
            signature = inspect.signature(predictor)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_varargs = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            if not has_varargs and len(positional) == 2:
                return [predictor(premise, hypothesis) for premise, hypothesis in pairs]
        try:
            return predictor(pairs)
        except (NLIModelUnavailable, NLIInferenceError, NLIOutputError):
            raise
        except Exception as error:
            raise NLIInferenceError(
                f"NLI predictor failed: {error}"
            ) from error

    def _empty_result(
        self,
        claim: Claim,
        status: str,
        explanation: str,
        *,
        cited_passages_exist: bool,
        error: str | None = None,
    ) -> NLIResult:
        return NLIResult(
            claim=claim,
            status=status,
            explanation=explanation,
            evidence_excerpts=(),
            cited_passages_exist=cited_passages_exist,
            label=None,
            aggregate_scores={},
            decision_confidence=None,
            per_evidence=(),
            aggregation=self.aggregation,
            model_metadata=self.metadata(),
            error=error,
        )

    def verify(
        self, claim: Claim, evidence_passages: Mapping[str, EvidencePassage]
    ) -> NLIResult:
        """Verify one claim. Citation validation always happens before NLI."""

        supplied_ids = set(evidence_passages.keys())
        all_exist, existing_ids = check_citation_validity(claim, supplied_ids)

        if not claim.citation_ids:
            return self._empty_result(
                claim,
                "insufficient_evidence",
                "No citations provided.",
                cited_passages_exist=False,
            )
        if not all_exist:
            missing = [
                citation_id
                for citation_id in claim.citation_ids
                if citation_id not in supplied_ids
            ]
            return self._empty_result(
                claim,
                "invalid_citation",
                f"Invalid/missing citation IDs: {missing}",
                cited_passages_exist=False,
            )

        unique_ids = tuple(dict.fromkeys(existing_ids))
        cited = [
            (citation_id, evidence_passages[citation_id])
            for citation_id in unique_ids
            if citation_id in evidence_passages
        ]
        if not cited:
            return self._empty_result(
                claim,
                "insufficient_evidence",
                "Cited passages not found in supplied evidence.",
                cited_passages_exist=False,
            )

        excerpts = tuple(
            (citation_id, _excerpt(passage.text)) for citation_id, passage in cited
        )
        pairs = [(passage.text, claim.text) for _, passage in cited]

        try:
            raw_outputs = list(self._predict(pairs))
            if len(raw_outputs) != len(cited):
                raise NLIOutputError(
                    f"expected {len(cited)} prediction rows, received "
                    f"{len(raw_outputs)}"
                )
            id2label = self._id2label()
            predictions = tuple(
                make_prediction(
                    row,
                    id2label,
                    evidence_id=citation_id,
                    premise_excerpt=_excerpt(passage.text),
                    model_name=getattr(self.backend, "model_name", self.model_name),
                    device=getattr(self.backend, "device", ""),
                )
                for (citation_id, passage), row in zip(cited, raw_outputs)
            )
            aggregated = aggregate_predictions(predictions, self.aggregation)
        except NLIModelUnavailable as error:
            return NLIResult(
                claim=claim,
                status="backend_unavailable",
                explanation=str(error),
                evidence_excerpts=excerpts,
                cited_passages_exist=True,
                label=None,
                aggregate_scores={},
                decision_confidence=None,
                per_evidence=(),
                aggregation=self.aggregation,
                model_metadata=self.metadata(),
                error=str(error),
            )
        except (NLIInferenceError, NLIOutputError) as error:
            return NLIResult(
                claim=claim,
                status="verification_error",
                explanation=f"NLI verification failed: {error}",
                evidence_excerpts=excerpts,
                cited_passages_exist=True,
                label=None,
                aggregate_scores={},
                decision_confidence=None,
                per_evidence=(),
                aggregation=self.aggregation,
                model_metadata=self.metadata(),
                error=str(error),
            )
        except Exception as error:
            return NLIResult(
                claim=claim,
                status="verification_error",
                explanation=f"Unexpected NLI verification failure: {error}",
                evidence_excerpts=excerpts,
                cited_passages_exist=True,
                label=None,
                aggregate_scores={},
                decision_confidence=None,
                per_evidence=(),
                aggregation=self.aggregation,
                model_metadata=self.metadata(),
                error=str(error),
            )

        return NLIResult(
            claim=claim,
            status=aggregated.status,
            explanation=aggregated.explanation,
            evidence_excerpts=excerpts,
            cited_passages_exist=True,
            label=aggregated.label,
            aggregate_scores=aggregated.scores,
            decision_confidence=aggregated.decision_confidence,
            per_evidence=predictions,
            aggregation=self.aggregation,
            model_metadata=self.metadata(),
            error=None,
        )
