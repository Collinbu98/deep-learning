import json

import pytest

from experiments.verification_utils.eval import evaluate_lexical, load_benchmark
from experiments.verification_utils.real_nli_eval import (
    ModelValidationError,
    evaluate_probes,
    focus_categories,
    lexical_audit,
    model_revision_from_snapshot,
    read_model_config,
)
from minigrad.nli_verification import FakeNLIBackend, NLIClaimVerifier

DEFAULT_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def logits_for(label, high=5.0, low=-2.0):
    return [
        high if DEFAULT_ID2LABEL[label_id] == label else low
        for label_id in sorted(DEFAULT_ID2LABEL)
    ]


def test_lexical_audit_explains_the_old_0636_reporting_error():
    results = evaluate_lexical(load_benchmark())

    audit = lexical_audit(results)

    assert audit["total"] == 33
    assert audit["correct"] == 18
    assert audit["confusion_diagonal"] == 18
    assert audit["accuracy"] == pytest.approx(18 / 33)
    assert audit["needed_for_0_636"] == 21
    assert "reporting error" in audit["note"]


def make_snapshot(tmp_path, revision="a" * 40, license_name="apache-2.0"):
    snapshot = tmp_path / "models--cross-encoder--nli-test" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large",
                "architectures": ["RobertaForSequenceClassification"],
                "model_type": "roberta",
                "max_position_embeddings": 514,
                "id2label": {
                    "0": "contradiction",
                    "1": "entailment",
                    "2": "neutral",
                },
            }
        )
    )
    (snapshot / "README.md").write_text(f"---\nlicense: {license_name}\n---\n")
    for name in ("tokenizer.json", "vocab.json", "merges.txt"):
        (snapshot / name).write_text("{}" if name.endswith(".json") else "")
    (snapshot / "model.safetensors").write_bytes(b"test")
    return snapshot


def test_snapshot_validation_reads_labels_license_and_revision(tmp_path):
    snapshot = make_snapshot(tmp_path)

    metadata = read_model_config(snapshot)

    assert model_revision_from_snapshot(snapshot) == "a" * 40
    assert metadata["revision"] == "a" * 40
    assert metadata["license"] == "apache-2.0"
    assert metadata["id2label"] == {
        "0": "contradiction",
        "1": "entailment",
        "2": "neutral",
    }
    assert metadata["weights_file"] == "model.safetensors"


def test_snapshot_validation_rejects_missing_files_and_wrong_license(tmp_path):
    missing = make_snapshot(tmp_path)
    (missing / "merges.txt").unlink()
    with pytest.raises(ModelValidationError, match="merges.txt"):
        read_model_config(missing)

    wrong_license = make_snapshot(tmp_path / "other", license_name="mit")
    with pytest.raises(ModelValidationError, match="expected license"):
        read_model_config(wrong_license)


def test_probe_validation_uses_config_label_mapping():
    labels_by_pair = {
        ("Jupiter has 95 known moons.", "Jupiter has 95 known moons."): "entailment",
        (
            "Mars has two small moons, Phobos and Deimos.",
            "Mars has one moon.",
        ): "contradiction",
        (
            "Jupiter has 95 known moons.",
            "The Great Red Spot is a storm on Jupiter.",
        ): "neutral",
    }

    def rows(pairs):
        label = labels_by_pair[pairs[0]]
        return [logits_for(label)]

    verifier = NLIClaimVerifier(backend=FakeNLIBackend(outputs=rows))
    result = evaluate_probes(verifier)

    assert result["passed"] is True
    assert [probe["predicted_label"] for probe in result["results"]] == [
        "entailment",
        "contradiction",
        "neutral",
    ]


def test_probe_validation_can_fail_without_claiming_success():
    verifier = NLIClaimVerifier(
        backend=FakeNLIBackend(outputs=[logits_for("neutral")])
    )
    result = evaluate_probes(verifier)

    assert result["passed"] is False
    assert all(probe["predicted_label"] == "neutral" for probe in result["results"])


def test_focus_categories_selects_requested_case_types():
    summary = {
        "contradiction": {"total": 5, "correct": 1, "accuracy": 0.2},
        "direct": {"total": 5, "correct": 5, "accuracy": 1.0},
        "numbers": {"total": 2, "correct": 1, "accuracy": 0.5},
    }

    focused = focus_categories(summary)

    assert set(focused) == {"contradiction", "numbers"}
    assert focused["contradiction"]["accuracy"] == 0.2
