import math

import pytest

from experiments.verification_utils.eval import compute_metrics, evaluate_lexical, load_benchmark
from experiments.verification_utils.nli_eval import (
    evaluate_nli,
    find_disagreements,
    summarise_by_type,
)
from minigrad.claim_verification import Claim, verify_claim
from minigrad.llm_answerer import EvidencePassage
from minigrad.nli_verification import (
    DEFAULT_NLI_MODEL,
    FakeNLIBackend,
    NLIClaimVerifier,
    NLIInferenceError,
    NLIModelUnavailable,
    NLIOutputError,
    TransformersNLIBackend,
    normalize_label_map,
    scores_from_logits,
)

DEFAULT_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def make_passage(source_id, text):
    doc_id = source_id.split("#")[0]
    chunk_idx = int(source_id.split("#")[1])
    return EvidencePassage(
        source_id=source_id,
        doc_id=doc_id,
        chunk_index=chunk_idx,
        source=f"{doc_id}.txt",
        text=text,
        score=1.0,
        rank=1,
    )


def logits_for(label, id2label=None, high=5.0, low=-2.0):
    mapping = id2label or DEFAULT_ID2LABEL
    return [
        high if mapping[label_id] == label else low
        for label_id in sorted(mapping)
    ]


class ScriptedBackend:
    model_name = "scripted-nli"
    device = "fake"
    dtype = "fake"

    def __init__(self, rows=None, id2label=None, error=None, rows_fn=None):
        self.id2label = dict(id2label or DEFAULT_ID2LABEL)
        self.rows = rows
        self.error = error
        self.rows_fn = rows_fn
        self.calls = []

    def predict_batch(self, pairs):
        pair_list = tuple(pairs)
        self.calls.append(pair_list)
        if self.error is not None:
            raise self.error
        if self.rows_fn is not None:
            return self.rows_fn(list(pair_list))
        if self.rows is not None:
            return self.rows
        return [logits_for("neutral", self.id2label) for _ in pair_list]


def test_label_mapping_uses_model_id_order_not_positional_assumption():
    id2label = {0: "neutral", 1: "contradiction", 2: "entailment"}
    label, label_id, scores, raw = scores_from_logits(
        [6.0, 0.0, -1.0], id2label
    )
    assert label == "neutral"
    assert label_id == 0
    assert raw == (6.0, 0.0, -1.0)
    assert set(scores) == {"entailment", "contradiction", "neutral"}
    assert math.isclose(sum(scores.values()), 1.0)

    label, label_id, _, _ = scores_from_logits([-1.0, 0.0, 6.0], id2label)
    assert label == "entailment"
    assert label_id == 2


def test_label_map_accepts_string_ids_and_rejects_placeholder_labels():
    mapping = normalize_label_map(
        {"0": "ENTAILMENT", "1": "neutral", "2": "contradiction"}
    )
    assert mapping == {0: "entailment", 1: "neutral", 2: "contradiction"}

    with pytest.raises(NLIOutputError):
        normalize_label_map({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(NLIOutputError):
        normalize_label_map({0: "entailment", 1: "entailment", 2: "neutral"})


@pytest.mark.parametrize(
    "label,status",
    [
        ("entailment", "supported"),
        ("contradiction", "contradicted"),
        ("neutral", "insufficient_evidence"),
    ],
)
def test_single_evidence_predictions_map_to_statuses(label, status):
    backend = ScriptedBackend(rows=[logits_for(label)])
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("jupiter#0",))

    result = verifier.verify(claim, {"jupiter#0": passage})

    assert result.status == status
    assert result.label == label
    assert result.cited_passages_exist is True
    assert len(result.per_evidence) == 1
    assert result.per_evidence[0].evidence_id == "jupiter#0"
    assert result.decision_confidence is not None
    assert backend.calls == [(("Jupiter has 95 known moons.", claim.text),)]


def test_invalid_citation_is_rejected_before_inference():
    backend = ScriptedBackend(rows=[logits_for("entailment")])
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("fake#99",))

    result = verifier.verify(claim, {"jupiter#0": passage})

    assert result.status == "invalid_citation"
    assert result.label is None
    assert result.per_evidence == ()
    assert backend.calls == []


def test_missing_citations_do_not_call_the_model():
    backend = ScriptedBackend(rows=[logits_for("entailment")])
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("venus#0", "Venus is the hottest planet.")
    claim = Claim("Venus is hot.", ())

    result = verifier.verify(claim, {"venus#0": passage})

    assert result.status == "insufficient_evidence"
    assert result.cited_passages_exist is False
    assert backend.calls == []


def test_conflicting_passages_are_mixed_and_not_supported():
    backend = ScriptedBackend(
        rows=[logits_for("entailment"), logits_for("contradiction")]
    )
    verifier = NLIClaimVerifier(backend=backend)
    passages = {
        "a#0": make_passage("a#0", "The claim is supported here."),
        "b#0": make_passage("b#0", "The claim is contradicted here."),
    }
    claim = Claim("A factual claim.", ("a#0", "b#0"))

    result = verifier.verify(claim, passages)

    assert result.status == "insufficient_evidence"
    assert result.label == "mixed"
    assert result.decision_confidence is None
    assert len(result.per_evidence) == 2
    assert {prediction.label for prediction in result.per_evidence} == {
        "entailment",
        "contradiction",
    }


def test_separate_neutral_passages_do_not_jointly_entail():
    backend = ScriptedBackend(
        rows=[logits_for("neutral"), logits_for("neutral")]
    )
    verifier = NLIClaimVerifier(backend=backend)
    passages = {
        "a#0": make_passage("a#0", "Some related context."),
        "b#0": make_passage("b#0", "More related context."),
    }
    claim = Claim("A claim requiring both parts.", ("a#0", "b#0"))

    result = verifier.verify(claim, passages)

    assert result.status == "insufficient_evidence"
    assert result.label == "neutral"


def test_one_entailment_with_neutral_evidence_is_supported():
    backend = ScriptedBackend(
        rows=[logits_for("entailment"), logits_for("neutral")]
    )
    verifier = NLIClaimVerifier(backend=backend)
    passages = {
        "a#0": make_passage("a#0", "Direct support."),
        "b#0": make_passage("b#0", "Unrelated context."),
    }
    claim = Claim("A factual claim.", ("a#0", "b#0"))

    result = verifier.verify(claim, passages)

    assert result.status == "supported"
    assert result.label == "entailment"


@pytest.mark.parametrize(
    "rows",
    [
        [[1.0, 2.0]],
        [[1.0, float("nan"), 0.0]],
        [[1.0, float("inf"), 0.0]],
        [["not-a-number", 0.0, 0.0]],
        [],
    ],
)
def test_malformed_scores_become_verification_errors(rows):
    backend = ScriptedBackend(rows=rows)
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("jupiter#0",))

    result = verifier.verify(claim, {"jupiter#0": passage})

    assert result.status == "verification_error"
    assert result.status != "supported"
    assert result.label is None
    assert result.error


def test_inference_exception_becomes_verification_error():
    backend = ScriptedBackend(error=RuntimeError("simulated inference crash"))
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("jupiter#0",))

    result = verifier.verify(claim, {"jupiter#0": passage})

    assert result.status == "verification_error"
    assert "simulated inference crash" in result.error


def test_unavailable_backend_becomes_backend_unavailable():
    backend = ScriptedBackend(error=NLIModelUnavailable("model is not cached"))
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("jupiter#0",))

    result = verifier.verify(claim, {"jupiter#0": passage})

    assert result.status == "backend_unavailable"
    assert result.label is None
    assert "not cached" in result.error


def test_transformers_backend_checks_cache_without_downloading(monkeypatch):
    import huggingface_hub

    calls = []

    def missing(*args, **kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    backend = TransformersNLIBackend(
        "definitely-not-a-real-nli-model-for-this-test",
        local_files_only=True,
    )

    assert backend.is_available() is False
    assert calls
    assert calls[0]["local_files_only"] is True


def test_benchmark_metric_integration_and_disagreements():
    wanted = {"s1", "c1", "inv1"}
    cases = [case for case in load_benchmark() if case["id"] in wanted]

    def rows_for(pairs):
        rows = []
        for _, hypothesis in pairs:
            if "79" in hypothesis:
                rows.append(logits_for("contradiction"))
            else:
                rows.append(logits_for("entailment"))
        return rows

    backend = ScriptedBackend(rows_fn=rows_for)
    verifier = NLIClaimVerifier(backend=backend)
    nli_results = evaluate_nli(cases, verifier)
    metrics = compute_metrics(nli_results)

    assert len(nli_results) == 3
    assert metrics["total"] == 3
    assert metrics["accuracy"] == 1.0
    assert metrics["false_supported_rate"] == 0.0
    assert "confusion_matrix" in metrics
    assert summarise_by_type(nli_results)["direct"]["accuracy"] == 1.0
    assert len(backend.calls) == 2

    lexical_results = evaluate_lexical(cases)
    disagreements = find_disagreements(lexical_results, nli_results)
    assert isinstance(disagreements, list)
    assert all("lexical_predicted" in item for item in disagreements)
    assert all("nli_predicted" in item for item in disagreements)


def test_deterministic_fake_predictions():
    backend = ScriptedBackend(
        rows=[logits_for("entailment"), logits_for("neutral")]
    )
    verifier = NLIClaimVerifier(backend=backend)
    passages = {
        "a#0": make_passage("a#0", "Direct support."),
        "b#0": make_passage("b#0", "Some context."),
    }
    claim = Claim("A factual claim.", ("a#0", "b#0"))

    first = verifier.verify(claim, passages)
    second = verifier.verify(claim, passages)

    assert first.to_dict() == second.to_dict()
    assert first.status == second.status == "supported"


def test_default_fake_backend_is_neutral():
    backend = FakeNLIBackend()
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("jupiter#0",))

    result = verifier.verify(claim, {"jupiter#0": passage})

    assert result.status == "insufficient_evidence"
    assert result.label == "neutral"


def test_dispatch_backend_selection_preserves_lexical_default():
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("jupiter#0",))
    verifier = NLIClaimVerifier(backend=ScriptedBackend(rows=[[-1.0, 5.0, -1.0]]))

    lexical_result = verify_claim(
        claim, {"jupiter#0": passage}, backend="lexical"
    )
    nli_result = verify_claim(
        claim,
        {"jupiter#0": passage},
        backend="nli",
        nli_verifier=verifier,
    )

    assert lexical_result.status == "supported"
    assert nli_result.status == "supported"
    assert nli_result.label == "entailment"
    with pytest.raises(ValueError):
        verify_claim(claim, {"jupiter#0": passage}, backend="unknown")


def test_metadata_describes_model_and_label_mapping():
    backend = ScriptedBackend(rows=[logits_for("entailment")])
    verifier = NLIClaimVerifier(backend=backend)
    passage = make_passage("jupiter#0", "Jupiter has 95 known moons.")
    claim = Claim("Jupiter has 95 known moons.", ("jupiter#0",))

    result = verifier.verify(claim, {"jupiter#0": passage})
    metadata = result.model_metadata

    assert metadata["model_name"] == "scripted-nli"
    assert metadata["id2label"] == {
        "0": "contradiction",
        "1": "entailment",
        "2": "neutral",
    }
    assert metadata["aggregation"] == "conservative"
    assert DEFAULT_NLI_MODEL
