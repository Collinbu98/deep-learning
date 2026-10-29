import importlib.util
from pathlib import Path

import pytest
import torch


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp24 = load_module("training_benchmarks_exp24", "24_training_benchmarks.py")
EXP23 = exp24.exp23


def _tiny_model_config():
    return {
        "vocab_size": 64,
        "n_layers": 1,
        "d_model": 8,
        "n_heads": 1,
        "dff": 16,
        "context_len": 8,
        "dropout": 0.0,
    }


def test_gpu_memory_stats_report_keys():
    stats = exp24.gpu_memory_stats(device="cpu")
    assert stats["supported"] is True
    assert stats["device"] == "cpu"
    assert "allocated_mb" in stats
    assert "reserved_mb" in stats
    assert "peak_allocated_mb" in stats
    assert "peak_reserved_mb" in stats
    assert "free_mb" in stats
    assert "total_mb" in stats


def test_estimate_duration_math():
    est = exp24.estimate_duration(total_tokens=1000, tokens_per_second=10.0, epochs=2.0)
    assert est["tokens_processed_estimate"] == 2000.0
    assert est["seconds_per_token"] == pytest.approx(0.1)
    assert est["estimated_train_seconds"] == pytest.approx(200.0)
    assert est["estimated_train_minutes"] == pytest.approx(200.0 / 60.0)
    assert est["assumptions"]["epochs"] == 2.0


def test_run_benchmark_short_cpu_run():
    result = exp24.run_benchmark(
        _tiny_model_config(),
        text=EXP23.SMOKE_TEXT,
        val_split=0.1,
        batch_size=1,
        context_len=8,
        steps=2,
        warmup_steps=1,
        lr=1e-3,
        clip_grad=1.0,
        device="cpu",
        seed=0,
        use_mixed_precision=False,
    )
    assert result["status"] == "completed"
    assert result["error"] is None
    assert len(result["losses"]) == 2
    assert all(torch.isfinite(torch.tensor([loss])).item() for loss in result["losses"])
    assert result["final_loss_finite"] is True
    assert result["median_step_seconds"] > 0
    assert result["tokens_per_second"] > 0
    assert result["tokens_per_step"] > 0
    assert result["model_params"] > 0
    assert "memory_before" in result and "memory_after" in result
    assert result["val_loss"] >= 0.0
    assert result["val_perplexity"] > 0.0


def test_get_corpus_text_smoke_and_missing():
    text, path = exp24.get_corpus_text(smoke=True, data=None)
    assert isinstance(text, str) and len(text) > 0
    assert path == "smoke-builtin"
    assert isinstance(EXP23.SMOKE_TEXT, str)
