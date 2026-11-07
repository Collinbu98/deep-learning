import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

from minigrad.bpe_tokenizer import BPETokenizer


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp28 = load_module("early_generalization_exp28_tests", "28_early_generalization.py")
exp23 = exp28.exp23


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp28_tiny")
    rng = random.Random(0)
    text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(600))
    settings = exp28.experiment_settings(
        vocab_size=64,
        n_layers=2,
        d_model=16,
        n_heads=2,
        dff=32,
        context_len=8,
        batch_size=4,
        val_split=0.1,
        use_mixed_precision=False,
    )
    result = exp28.run_experiment(
        horizons=(12, 24),
        text=text,
        output_dir=base / "out",
        checkpoint_root=base / "ckpt",
        validation_interval=4,
        periodic_val_batches=1,
        train_subset_batches=2,
        settings=settings,
        device="cpu",
        prompts=["the quick brown fox"],
        generation_settings={"max_new_tokens": 4, "temperature": 0.8, "top_k": None},
    )
    return result, base


def test_horizon_warmup_scales_with_experiment_27_fraction():
    assert exp28.warmup_steps_for_horizon(3000) == 23
    assert exp28.warmup_steps_for_horizon(6000) == 46
    assert exp28.warmup_steps_for_horizon(12000) == 93
    for horizon in (3000, 6000, 12000):
        plan = exp28.horizon_plan(horizon)
        assert plan["horizon"] == horizon
        assert 0 < plan["warmup_steps"] < horizon
        assert plan["lr_at_step_0"] == pytest.approx(exp28.PEAK_LR / plan["warmup_steps"])
        assert plan["lr_at_warmup_end"] == pytest.approx(exp28.PEAK_LR)
        assert plan["lr_at_final_step"] == pytest.approx(exp28.MIN_LR)
        assert plan["sample_points"]["25%"] >= plan["sample_points"]["50%"]
        assert plan["sample_points"]["50%"] >= plan["sample_points"]["75%"]
        assert plan["sample_points"]["100%"] == pytest.approx(exp28.MIN_LR)


def test_horizon_plan_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        exp28.warmup_steps_for_horizon(1)
    with pytest.raises(ValueError):
        exp28.warmup_steps_for_horizon(0)
    with pytest.raises(ValueError):
        exp28._validate_horizons([])
    with pytest.raises(ValueError):
        exp28._validate_horizons((1,))
    with pytest.raises(ValueError):
        exp28.experiment_settings(min_lr=2e-3)
    with pytest.raises(ValueError):
        exp28.experiment_settings(lr=0.0)
    with pytest.raises(ValueError):
        exp28.experiment_settings(batch_size=0)
    with pytest.raises(ValueError):
        exp28.experiment_settings(val_split=1.0)


def test_tiny_end_to_end_records_required_artifacts(tiny_run):
    result, base = tiny_run
    assert result["status"] == "completed"
    assert result["num_parameters"] == result["expected_parameter_count"]
    assert result["parameter_count_matches"] is True
    assert [run["horizon"] for run in result["runs"]] == [12, 24]
    assert result["comparison"]["best_horizon"] in (12, 24)
    saved = result["comparison"]["best_saved_checkpoint"]
    any_best = result["comparison"]["best_any_checkpoint"]
    assert saved["kind"] == "best"
    assert any_best["full_val_loss"] <= saved["full_val_loss"]
    assert any_best["checkpoint"]
    assert any_best["kind"] in ("best", "final")
    assert any_best["horizon"] in (12, 24)

    out = base / "out"
    for name in ("metrics.json", "samples.json", "report.md"):
        assert (out / name).exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["experiment"] == exp28.EXPERIMENT_NAME
    assert len(metrics["runs"]) == 2


def test_each_horizon_selects_best_by_validation_and_saves_checkpoints(tiny_run):
    result, base = tiny_run
    for run in result["runs"]:
        horizon = run["horizon"]
        steps = run["steps"]
        trajectory = run["trajectory"]
        assert steps == horizon
        assert len(trajectory) == horizon // 4
        assert trajectory[-1]["step"] == horizon
        assert all(point["step"] % 4 == 0 for point in trajectory)
        best_point = min(trajectory, key=lambda p: p["val_loss"])
        assert run["best"]["step"] == best_point["step"]
        assert run["best"]["val_loss"] == pytest.approx(best_point["val_loss"])
        assert run["all_finite"] is True

        best_path = base / "ckpt" / f"h{horizon}" / "best.pt"
        final_path = base / "ckpt" / f"h{horizon}" / "final.pt"
        assert best_path.exists() and final_path.exists()
        payload = exp23.load_checkpoint(best_path)
        state = payload["training_state"]
        assert state["step"] == run["best"]["step"]
        scheduler_state = state["scheduler_state_dict"]
        assert scheduler_state["total_steps"] == horizon
        assert scheduler_state["warmup_steps"] == run["plan"]["warmup_steps"]
        assert scheduler_state["min_lr"] == pytest.approx(exp28.MIN_LR)
        assert BPETokenizer.from_dict(payload["tokenizer"]).vocab_size == 64

        assert math.isfinite(run["best"]["full_val_loss"])
        assert math.isfinite(run["final"]["full_val_loss"])
        assert math.isfinite(run["best"]["train_subset_loss"])
        assert math.isfinite(run["best"]["generalization_gap"])
        assert isinstance(run["deteriorated_after_best"], bool)


def test_samples_contain_initial_best_and_final_for_each_horizon(tiny_run):
    result, base = tiny_run
    stages = result["samples"]["stages"]
    assert stages == ["initial", "h12_best", "h12_final", "h24_best", "h24_final"]
    payload = json.loads((base / "out" / "samples.json").read_text())
    assert payload["stages"] == stages
    assert len(payload["samples"]) == 1
    row = payload["samples"][0]
    assert row["prompt"] == "the quick brown fox"
    for stage in stages:
        assert isinstance(row[stage], str)
        assert row[stage]


def test_report_documents_audit_trajectories_and_recommendation(tiny_run):
    result, base = tiny_run
    report = (base / "out" / "report.md").read_text()
    assert "Phase 1 audit" in report
    assert "Experiment 27" in report
    assert "Complete validation trajectories" in report
    assert "Recommendation" in report
    assert "Pending: run the full test suite" in report
    assert "warmup 2,000" in report or "warmup" in report
