import importlib.util
import math
import random
from pathlib import Path

import pytest
import torch
from torch import nn

from minigrad.bpe_tokenizer import BPETokenizer


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp23 = load_module("scalable_gpt_training_exp23_for_sched", "23_scalable_gpt_training.py")
exp27 = load_module("scheduled_gpt_training_exp27_tests", "27_scheduled_gpt_training.py")
DEVICE = torch.device("cpu")


def _tiny_config(**overrides):
    config = {
        "vocab_size": 12,
        "n_layers": 2,
        "d_model": 16,
        "n_heads": 2,
        "dff": 32,
        "context_len": 8,
        "dropout": 0.0,
    }
    config.update(overrides)
    return config


def _make_model(**overrides):
    return exp23.GPTLanguageModel(**exp23.validate_gpt_config(_tiny_config(**overrides)))


def _tiny_loader(context_len=4, batch_size=2):
    ids = [0, 1, 2, 3, 2, 1, 0, 3, 2, 1, 0, 2, 3, 0, 1, 2, 0, 1, 2, 3]
    ds = exp23.GPTTokenDataset(ids, context_len=context_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Scheduler behavior
# ---------------------------------------------------------------------------


def test_warmup_is_linear_to_peak_then_cosine_to_nonzero_floor():
    model = nn.Linear(2, 2)
    opt = exp23.make_optimizer(model, lr=1e-3)
    sched = exp23.make_warmup_cosine_scheduler(
        opt, total_steps=100, warmup_steps=10, min_lr=1e-4, peak_lr=1e-3
    )
    warmup = [sched.lr_for_step(i) for i in range(10)]
    assert all(warmup[i] < warmup[i + 1] for i in range(len(warmup) - 1))
    assert warmup[0] == pytest.approx(1e-3 / 10)
    assert warmup[-1] == pytest.approx(1e-3)
    assert sched.lr_for_step(10) == pytest.approx(1e-3)

    decay = [sched.lr_for_step(i) for i in range(10, 100)]
    assert all(decay[i] >= decay[i + 1] for i in range(len(decay) - 1))
    assert sched.lr_for_step(99) == pytest.approx(1e-4)
    assert min(sched.lr_for_step(i) for i in range(100)) == pytest.approx(1e-4)


def test_scheduler_steps_once_per_update_and_state_roundtrip():
    model = nn.Linear(2, 2)
    opt = exp23.make_optimizer(model, lr=1e-2)
    sched = exp23.make_warmup_cosine_scheduler(
        opt, total_steps=20, warmup_steps=4, min_lr=1e-4, peak_lr=1e-2
    )
    assert sched.completed_steps == 0
    assert opt.param_groups[0]["lr"] == pytest.approx(sched.lr_for_step(0))
    observed = []
    for _ in range(5):
        observed.append(opt.param_groups[0]["lr"])
        sched.step()
    assert observed == [pytest.approx(sched.lr_for_step(i)) for i in range(5)]
    assert sched.completed_steps == 5

    state = sched.state_dict()
    restored = exp23.make_warmup_cosine_scheduler(
        opt, total_steps=20, warmup_steps=4, min_lr=1e-4, peak_lr=1e-2
    )
    restored.load_state_dict(state)
    assert restored.completed_steps == 5
    assert restored.get_last_lr()[0] == pytest.approx(sched.lr_for_step(5))
    with pytest.raises(ValueError):
        restored.load_state_dict({**state, "total_steps": 999})


def test_scheduler_rejects_invalid_configuration():
    opt = exp23.make_optimizer(nn.Linear(2, 2), lr=1e-3)
    with pytest.raises(ValueError):
        exp23.make_warmup_cosine_scheduler(opt, total_steps=0)
    with pytest.raises(ValueError):
        exp23.make_warmup_cosine_scheduler(opt, total_steps=10, warmup_steps=10)
    with pytest.raises(ValueError):
        exp23.make_warmup_cosine_scheduler(opt, total_steps=10, warmup_steps=-1)
    with pytest.raises(ValueError):
        exp23.make_warmup_cosine_scheduler(opt, total_steps=10, min_lr=2e-3, peak_lr=1e-3)


# ---------------------------------------------------------------------------
# Compatibility with the existing training pipeline
# ---------------------------------------------------------------------------


def test_train_epoch_advances_scheduler_once_per_optimizer_update():
    loader = _tiny_loader()
    n_steps = len(loader)
    model = _make_model()
    criterion = nn.CrossEntropyLoss()
    opt = exp23.make_optimizer(model, lr=1e-2)
    sched = exp23.make_warmup_cosine_scheduler(
        opt, total_steps=n_steps, warmup_steps=2, min_lr=1e-4, peak_lr=1e-2
    )
    out = exp23.train_epoch(
        model, loader, criterion, opt, DEVICE, clip_grad=1.0, diagnostics=True, scheduler=sched
    )
    assert out["steps"] == n_steps
    assert sched.completed_steps == n_steps
    train_diags = [d for d in out["diagnostics"] if "val_loss" not in d]
    assert [d["lr"] for d in train_diags] == [
        pytest.approx(sched.lr_for_step(i)) for i in range(n_steps)
    ]


def test_train_epoch_without_scheduler_preserves_default_behavior():
    loader = _tiny_loader()
    model = _make_model()
    criterion = nn.CrossEntropyLoss()
    opt = exp23.make_optimizer(model, lr=1e-3)
    out = exp23.train_epoch(model, loader, criterion, opt, DEVICE, clip_grad=1.0)
    assert set(out.keys()) == {"loss", "tokens", "steps"}


def test_train_model_scheduler_advances_across_epochs():
    loader = _tiny_loader()
    val_loader = _tiny_loader()
    n_steps = len(loader) * 2
    model = _make_model()
    criterion = nn.CrossEntropyLoss()
    opt = exp23.make_optimizer(model, lr=1e-2)
    sched = exp23.make_warmup_cosine_scheduler(
        opt, total_steps=n_steps, warmup_steps=2, min_lr=1e-4, peak_lr=1e-2
    )
    result = exp23.train_model(
        model, loader, val_loader, criterion, opt, DEVICE, epochs=2,
        clip_grad=1.0, diagnostics=True, scheduler=sched,
    )
    assert result["steps"] == n_steps
    assert sched.completed_steps == n_steps
    assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# Checkpoint restoration
# ---------------------------------------------------------------------------


def test_checkpoint_restores_optimizer_scheduler_and_rng(tmp_path):
    model = _make_model()
    opt = exp23.make_optimizer(model, lr=1e-2)
    sched = exp23.make_warmup_cosine_scheduler(
        opt, total_steps=20, warmup_steps=4, min_lr=1e-4, peak_lr=1e-2
    )
    loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()
    exp23.train_epoch(model, loader, criterion, opt, DEVICE, scheduler=sched)
    torch.manual_seed(123)
    saved_rng = torch.get_rng_state()
    saved_python = random.getstate()
    path = exp23.save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        config=model.config_dict(),
        tokenizer=BPETokenizer(vocab_size=8).fit("abcdefgh"),
        training_state={
            "step": sched.completed_steps,
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "torch_rng_state": saved_rng,
            "python_rng_state": saved_python,
        },
        args={"test": True},
    )
    payload = exp23.load_checkpoint(path)
    model2 = _make_model()
    opt2 = exp23.make_optimizer(model2, lr=1e-2)
    sched2 = exp23.make_warmup_cosine_scheduler(
        opt2, total_steps=20, warmup_steps=4, min_lr=1e-4, peak_lr=1e-2
    )
    torch.manual_seed(999)
    restored = exp23.restore_training_state(
        payload, model2, opt2, DEVICE, scheduler=sched2, restore_rng=True
    )
    assert restored["step"] == sched.completed_steps
    assert restored["scheduler_state"] is not None
    assert sched2.completed_steps == sched.completed_steps
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert "state" in restored["optimizer_state"]


# ---------------------------------------------------------------------------
# Experiment 27 configuration and end-to-end CPU smoke test
# ---------------------------------------------------------------------------


def test_exp27_config_defaults_and_validation():
    cfg = exp27.scheduled_training_config()
    assert cfg["vocab_size"] == 256
    assert cfg["n_layers"] == 16
    assert cfg["epochs"] == 3
    assert cfg["lr"] == 1e-3
    assert cfg["warmup_steps"] == exp27.WARMUP_STEPS
    assert cfg["min_lr"] == exp27.MIN_LR
    assert cfg["schedule"] == "warmup-cosine"
    assert cfg["validation_points_per_epoch"] == exp27.VALIDATION_POINTS_PER_EPOCH
    assert exp23.gpt_parameter_count(256, 16, 512, 8, 768, 256) == 29_840_640
    with pytest.raises(ValueError):
        exp27.scheduled_training_config(warmup_steps=-1)
    with pytest.raises(ValueError):
        exp27.scheduled_training_config(min_lr=2e-3)
    with pytest.raises(ValueError):
        exp27.scheduled_training_config(validation_points_per_epoch=0)
    with pytest.raises(ValueError):
        exp27.scheduled_training_config(periodic_val_batches=0)
    with pytest.raises(ValueError):
        exp27.scheduled_training_config(epochs=0)


def test_exp27_checkpoint_paths_and_limited_loader(tmp_path):
    d = tmp_path / "ckpts"
    assert exp27.best_checkpoint_path(d) == d / "best.pt"
    assert exp27.latest_checkpoint_path(d) == d / "latest.pt"
    assert exp27.epoch_checkpoint_path(d, 2) == d / "epoch_2.pt"
    loader = _tiny_loader()
    limited = exp27._LimitedLoader(loader, 2)
    assert len(limited) == 2
    assert len(list(limited)) == 2


def test_exp27_tiny_end_to_end_cpu(tmp_path):
    rng = random.Random(0)
    text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(600))
    result = exp27.run_scheduled_gpt_training(
        text=text,
        vocab_size=64,
        n_layers=2,
        d_model=16,
        n_heads=2,
        dff=32,
        context_len=8,
        dropout=0.0,
        batch_size=4,
        epochs=2,
        lr=1e-2,
        warmup_steps=2,
        min_lr=1e-3,
        clip_grad=1.0,
        val_split=0.1,
        seed=0,
        device="cpu",
        use_mixed_precision=False,
        validation_points_per_epoch=2,
        periodic_val_batches=1,
        checkpoint_dir=tmp_path,
        prompts=["the quick brown fox"],
        generation_settings={"max_new_tokens": 8, "temperature": 0.8, "top_k": None},
    )
    assert result["status"] == "completed"
    assert result["num_parameters"] == result["expected_parameter_count"]
    assert result["steps"] == result["steps_per_epoch"] * 2
    assert result["steps"] == result["planned_total_steps"]
    assert len(result["history"]) == 2
    assert len(result["periodic"]) >= 2
    assert result["schedule_info"]["total_steps"] == result["planned_total_steps"]
    assert result["schedule_info"]["lr_at_final_step"] == pytest.approx(1e-3)
    assert result["schedule_info"]["peak_lr"] == pytest.approx(1e-2)
    assert result["best_epoch"] in (1, 2)
    for point in result["periodic"]:
        assert math.isfinite(point["train_loss"])
        assert math.isfinite(point["val_loss"])
        assert point["params_finite"]
        assert point["loss_finite"]
        assert point["grad_finite"]
        assert 1e-3 - 1e-12 <= point["lr"] <= 1e-2 + 1e-12
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "epoch_1.pt").exists()
    assert (tmp_path / "epoch_2.pt").exists()
    assert "initial" in result["samples"]
    assert "epoch_1" in result["samples"]
    assert "epoch_2" in result["samples"]
    assert "best" in result["samples"]
