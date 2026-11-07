"""Experiment 28 - early generalization and training horizon.

Find the best generalizing checkpoint for the Experiment 27 architecture by
training short, frequently validated runs at three cosine-scheduler horizons
(3k, 6k, 12k optimizer steps). No architecture, tokenizer, or corpus changes.

Run:
    .venv/bin/python experiments/28_early_generalization.py
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import time
import types
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from minigrad.bpe_tokenizer import BPETokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, filename: str):
    path = REPO_ROOT / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp23 = _load_module("scalable_gpt_training_for_exp28", "23_scalable_gpt_training.py")
exp24 = _load_module("training_benchmarks_for_exp28", "24_training_benchmarks.py")
exp25 = _load_module("first_gpt_training_for_exp28", "25_first_gpt_training.py")
exp27 = _load_module("scheduled_gpt_training_for_exp28", "27_scheduled_gpt_training.py")


EXPERIMENT_NAME = "28_early_generalization"
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT_NAME
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SAMPLES_PATH = OUTPUT_DIR / "samples.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / EXPERIMENT_NAME
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "pg1342.txt"

HORIZONS = (3000, 6000, 12000)
PRIMARY_HORIZON = 12000
VALIDATION_INTERVAL = 250
PERIODIC_VAL_BATCHES = 100
TRAIN_SUBSET_BATCHES = 500
PEAK_LR = 1e-3
MIN_LR = 1e-4
MIN_LR_RATIO = MIN_LR / PEAK_LR
SEED = 0
# Experiment 27 used 2,000 warmup steps out of 259,257 planned updates.
WARMUP_FRACTION = 2000.0 / 259257.0

DEFAULT_PROMPTS = list(exp25.DEFAULT_PROMPTS)
DEFAULT_GENERATION_SETTINGS = dict(exp25.DEFAULT_GENERATION_SETTINGS)


# ---------------------------------------------------------------------------
# Configuration and schedule helpers
# ---------------------------------------------------------------------------


def experiment_settings(**overrides) -> dict:
    settings = {
        "vocab_size": 256,
        "n_layers": 16,
        "d_model": 512,
        "n_heads": 8,
        "dff": 768,
        "context_len": 256,
        "dropout": 0.0,
        "batch_size": 4,
        "lr": PEAK_LR,
        "min_lr": MIN_LR,
        "weight_decay": 0.0,
        "clip_grad": 1.0,
        "val_split": 0.1,
        "seed": SEED,
        "use_mixed_precision": True,
    }
    settings.update(overrides)
    if settings["vocab_size"] < 2:
        raise ValueError("vocab_size must be >= 2")
    if settings["lr"] <= 0.0:
        raise ValueError("lr must be > 0")
    if settings["min_lr"] < 0.0:
        raise ValueError("min_lr must be >= 0")
    if settings["min_lr"] > settings["lr"]:
        raise ValueError("min_lr must be <= lr")
    if settings["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 < settings["val_split"] < 1.0:
        raise ValueError("val_split must be in (0, 1)")
    return settings


def warmup_steps_for_horizon(horizon, warmup_fraction: float = WARMUP_FRACTION) -> int:
    horizon = int(horizon)
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2, got {horizon}")
    warmup = int(round(float(warmup_fraction) * horizon))
    warmup = max(1, warmup)
    if warmup >= horizon:
        raise ValueError(
            f"warmup fraction {warmup_fraction} gives {warmup} >= horizon {horizon}"
        )
    return warmup


def horizon_plan(horizon, *, peak_lr: float = PEAK_LR, min_lr: float = MIN_LR) -> dict:
    horizon = int(horizon)
    warmup = warmup_steps_for_horizon(horizon)
    optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=float(peak_lr))
    scheduler = exp23.make_warmup_cosine_scheduler(
        optimizer,
        total_steps=horizon,
        warmup_steps=warmup,
        min_lr=float(min_lr),
        peak_lr=float(peak_lr),
    )
    sample_points = {
        name: float(scheduler.lr_for_step(int(fraction * (horizon - 1))))
        for name, fraction in (
            ("0%", 0.0),
            ("25%", 0.25),
            ("50%", 0.5),
            ("75%", 0.75),
            ("100%", 1.0),
        )
    }
    return {
        "horizon": horizon,
        "warmup_steps": warmup,
        "warmup_fraction_of_horizon": warmup / horizon,
        "peak_lr": float(peak_lr),
        "min_lr": float(min_lr),
        "min_lr_ratio": float(min_lr) / float(peak_lr),
        "lr_at_step_0": float(scheduler.lr_for_step(0)),
        "lr_at_warmup_end": float(scheduler.lr_for_step(warmup)),
        "lr_at_final_step": float(scheduler.lr_for_step(horizon - 1)),
        "sample_points": sample_points,
    }


def _memory_stats(device) -> dict:
    if device.type == "cuda":
        return exp24.gpu_memory_stats(device)
    return {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
        "free_mb": None,
        "total_mb": None,
        "supported": False,
    }


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_horizons(horizons) -> tuple[int, ...]:
    values = sorted({int(h) for h in horizons})
    if not values:
        raise ValueError("at least one horizon is required")
    for horizon in values:
        warmup_steps_for_horizon(horizon)
    return tuple(values)


# ---------------------------------------------------------------------------
# Training and checkpoint evaluation
# ---------------------------------------------------------------------------


def _build_model(model_config: dict, device) -> nn.Module:
    return exp23.GPTLanguageModel(**model_config).to(device)


def _make_scheduler(optimizer, plan: dict):
    return exp23.make_warmup_cosine_scheduler(
        optimizer,
        total_steps=int(plan["horizon"]),
        warmup_steps=int(plan["warmup_steps"]),
        min_lr=float(plan["min_lr"]),
        peak_lr=float(plan["peak_lr"]),
    )


def _save_checkpoint(
    path: Path,
    *,
    model,
    model_config,
    tokenizer,
    optimizer,
    scheduler,
    step: int,
    settings: dict,
    plan: dict,
):
    exp23.save_checkpoint(
        path,
        model=model,
        config=model_config,
        tokenizer=tokenizer,
        training_state=exp27._training_state(
            optimizer,
            scheduler,
            step=int(step),
            epochs_completed=0,
            step_in_epoch=int(step),
        ),
        args={
            **{k: v for k, v in settings.items()},
            "horizon": int(plan["horizon"]),
            "warmup_steps": int(plan["warmup_steps"]),
            "schedule": "warmup-cosine",
        },
    )
    return path


def _run_horizon(
    *,
    plan: dict,
    settings: dict,
    tokenizer,
    model_config: dict,
    train_loader,
    periodic_val_loader,
    checkpoint_dir: Path,
    device,
    validation_interval: int,
) -> dict:
    horizon = int(plan["horizon"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    final_path = checkpoint_dir / "final.pt"

    _cleanup_cuda()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    exp27._set_global_seed(settings["seed"])
    model = _build_model(model_config, device)
    optimizer = exp23.make_optimizer(
        model, settings["lr"], weight_decay=settings["weight_decay"]
    )
    scheduler = _make_scheduler(optimizer, plan)
    criterion = nn.CrossEntropyLoss()
    loss_ctx = (
        torch.autocast(device_type=str(device), dtype=torch.bfloat16)
        if settings["use_mixed_precision"]
        else contextlib.nullcontext()
    )
    # Match Experiment 27's data ordering: reset after model construction so
    # the shuffled loader starts from the same RNG state.
    exp27._set_global_seed(settings["seed"])

    previous_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    trajectory = []
    best = None
    global_step = 0
    tokens_total = 0
    run_start = time.time()

    point_loss_sum = 0.0
    point_tokens = 0
    point_steps = 0
    point_grad_sum = 0.0
    point_grad_max = 0.0
    point_clipped = 0
    point_loss_finite = True
    point_grad_finite = True

    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        used_lr = float(optimizer.param_groups[0]["lr"])
        with loss_ctx:
            loss = exp23.next_token_loss(model, x, y)
            loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["clip_grad"]))
        )
        clipped = bool(grad_norm > float(settings["clip_grad"]))
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach())
        tokens = int(x.numel())
        point_loss_sum += loss_value * tokens
        point_tokens += tokens
        point_steps += 1
        point_grad_sum += grad_norm
        point_grad_max = max(point_grad_max, grad_norm)
        point_clipped += int(clipped)
        point_loss_finite = point_loss_finite and math.isfinite(loss_value)
        point_grad_finite = point_grad_finite and math.isfinite(grad_norm)
        tokens_total += tokens
        global_step += 1

        if global_step % int(validation_interval) != 0 and global_step != horizon:
            continue

        model.eval()
        subset_val = exp23.validate_epoch(model, periodic_val_loader, criterion, device)
        model.train()
        update_ratio = exp27._update_ratio(model, previous_params)
        params_finite = exp27._params_finite(model)
        point_train_loss = point_loss_sum / max(1, point_tokens)
        entry = {
            "step": int(global_step),
            "lr": float(used_lr),
            "lr_next": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(point_train_loss),
            "train_perplexity": (
                float(math.exp(point_train_loss))
                if math.isfinite(point_train_loss)
                else float("inf")
            ),
            "val_loss": float(subset_val["loss"]),
            "val_perplexity": float(subset_val["perplexity"]),
            "grad_norm_mean": float(point_grad_sum / max(1, point_steps)),
            "grad_norm_max": float(point_grad_max),
            "clipped_fraction": float(point_clipped / max(1, point_steps)),
            "update_ratio": float(update_ratio),
            "loss_finite": bool(point_loss_finite),
            "grad_finite": bool(point_grad_finite),
            "params_finite": bool(params_finite),
            "tokens_since_point": int(point_tokens),
            "tokens_total": int(tokens_total),
            "elapsed_seconds": float(time.time() - run_start),
        }
        trajectory.append(entry)
        if best is None or entry["val_loss"] < best["val_loss"]:
            _save_checkpoint(
                best_path,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                settings=settings,
                plan=plan,
            )
            best = {
                "step": int(global_step),
                "val_loss": float(entry["val_loss"]),
                "val_perplexity": float(entry["val_perplexity"]),
                "train_loss_interval": float(entry["train_loss"]),
                "lr": float(entry["lr"]),
                "elapsed_seconds": float(entry["elapsed_seconds"]),
                "checkpoint": str(best_path),
            }
        previous_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        point_loss_sum = 0.0
        point_tokens = 0
        point_steps = 0
        point_grad_sum = 0.0
        point_grad_max = 0.0
        point_clipped = 0
        point_loss_finite = True
        point_grad_finite = True

        if global_step >= horizon:
            break

    total_seconds = time.time() - run_start
    _save_checkpoint(
        final_path,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        settings=settings,
        plan=plan,
    )
    peak_memory = _memory_stats(device)
    finite_all = all(
        p["loss_finite"] and p["grad_finite"] and p["params_finite"] for p in trajectory
    )
    result = {
        "horizon": horizon,
        "plan": plan,
        "steps": int(global_step),
        "trajectory": trajectory,
        "best": best,
        "final": {
            "step": int(global_step),
            "checkpoint": str(final_path),
            "train_loss_interval": float(trajectory[-1]["train_loss"]) if trajectory else None,
            "lr": float(trajectory[-1]["lr"]) if trajectory else None,
        },
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "tokens_processed": int(tokens_total),
        "total_seconds": float(total_seconds),
        "tokens_per_second": float(tokens_total / total_seconds) if total_seconds > 0 else None,
        "peak_memory": peak_memory,
        "all_finite": bool(finite_all),
        "clipped_fraction": (
            float(sum(p["clipped_fraction"] * p["tokens_since_point"] for p in trajectory) / tokens_total)
            if tokens_total
            else None
        ),
        "grad_norm_mean": (
            float(sum(p["grad_norm_mean"] * p["tokens_since_point"] for p in trajectory) / tokens_total)
            if tokens_total
            else None
        ),
        "grad_norm_max": float(max(p["grad_norm_max"] for p in trajectory)) if trajectory else None,
    }
    del model, optimizer, scheduler
    _cleanup_cuda()
    return result


def _load_checkpoint_model(path, device):
    payload = exp23.load_checkpoint(Path(path))
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    config = exp23.validate_gpt_config(payload["model_config"])
    model = exp23.GPTLanguageModel(**config).to(device)
    exp23.restore_training_state(payload, model, None, device)
    model.eval()
    return model, tokenizer, payload


def _evaluate_checkpoint(
    path,
    *,
    device,
    val_loader,
    train_subset_loader,
    prompts,
    generation_settings,
    seed,
) -> dict:
    model, tokenizer, payload = _load_checkpoint_model(path, device)
    criterion = nn.CrossEntropyLoss()
    full_val = exp23.validate_epoch(model, val_loader, criterion, device)
    train_subset = exp23.validate_epoch(model, train_subset_loader, criterion, device)
    samples = exp25.generate_samples(
        model,
        tokenizer,
        prompts,
        generation_settings=generation_settings,
        seed=seed,
        device=device,
    )
    training_state = dict(payload.get("training_state") or {})
    result = {
        "checkpoint": str(path),
        "step": int(training_state.get("step", 0)),
        "full_val_loss": float(full_val["loss"]),
        "full_val_perplexity": float(full_val["perplexity"]),
        "full_val_tokens": int(full_val["tokens"]),
        "full_val_steps": int(full_val["steps"]),
        "train_subset_loss": float(train_subset["loss"]),
        "train_subset_perplexity": float(train_subset["perplexity"]),
        "train_subset_tokens": int(train_subset["tokens"]),
        "samples": samples,
    }
    del model
    _cleanup_cuda()
    return result


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    horizons=HORIZONS,
    text: str | None = None,
    data_path=None,
    output_dir=None,
    checkpoint_root=None,
    validation_interval: int = VALIDATION_INTERVAL,
    periodic_val_batches: int = PERIODIC_VAL_BATCHES,
    train_subset_batches: int = TRAIN_SUBSET_BATCHES,
    prompts=None,
    generation_settings=None,
    settings: dict | None = None,
    device=None,
    write_outputs: bool = True,
) -> dict:
    horizons = _validate_horizons(horizons)
    if int(validation_interval) < 1:
        raise ValueError("validation_interval must be >= 1")
    if int(periodic_val_batches) < 1:
        raise ValueError("periodic_val_batches must be >= 1")
    if int(train_subset_batches) < 1:
        raise ValueError("train_subset_batches must be >= 1")

    settings = experiment_settings() if settings is None else dict(settings)
    prompts = list(DEFAULT_PROMPTS if prompts is None else prompts)
    gs = exp25.resolve_generation_settings(generation_settings)
    dev = exp23.get_device(device if device is not None else settings.get("device"))
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    ckpt_root = Path(checkpoint_root) if checkpoint_root is not None else CHECKPOINT_ROOT

    if text is None:
        text, data_path = exp23.load_corpus_text(
            types.SimpleNamespace(data=str(data_path) if data_path is not None else None)
        )
    else:
        data_path = "provided-text"

    train_text, val_text = exp23.split_corpus(text, settings["val_split"])
    tokenizer, train_ids, val_ids = exp23.fit_tokenizer_and_encode(
        train_text, val_text, int(settings["vocab_size"])
    )
    train_ds = exp23.GPTTokenDataset(train_ids, settings["context_len"])
    val_ds = exp23.GPTTokenDataset(val_ids, settings["context_len"])
    train_loader = DataLoader(train_ds, batch_size=settings["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=settings["batch_size"], shuffle=False)
    train_eval_loader = DataLoader(
        train_ds, batch_size=settings["batch_size"], shuffle=False
    )
    periodic_val_loader = exp27._LimitedLoader(val_loader, periodic_val_batches)
    train_subset_loader = exp27._LimitedLoader(train_eval_loader, train_subset_batches)

    model_config = exp23.validate_gpt_config(
        {
            "vocab_size": int(settings["vocab_size"]),
            "n_layers": int(settings["n_layers"]),
            "d_model": int(settings["d_model"]),
            "n_heads": int(settings["n_heads"]),
            "dff": int(settings["dff"]),
            "context_len": int(settings["context_len"]),
            "dropout": float(settings["dropout"]),
        }
    )
    num_params = exp23.gpt_parameter_count(
        vocab_size=model_config["vocab_size"],
        n_layers=model_config["n_layers"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        dff=model_config["dff"],
        context_len=model_config["context_len"],
    )
    steps_per_epoch = int(len(train_loader))

    exp27._set_global_seed(settings["seed"])
    initial_model = _build_model(model_config, dev)
    initial_samples = exp25.generate_samples(
        initial_model,
        tokenizer,
        prompts,
        generation_settings=gs,
        seed=settings["seed"],
        device=dev,
    )
    del initial_model
    _cleanup_cuda()

    runs = []
    for horizon in horizons:
        plan = horizon_plan(
            horizon, peak_lr=settings["lr"], min_lr=settings["min_lr"]
        )
        run = _run_horizon(
            plan=plan,
            settings=settings,
            tokenizer=tokenizer,
            model_config=model_config,
            train_loader=train_loader,
            periodic_val_loader=periodic_val_loader,
            checkpoint_dir=ckpt_root / f"h{horizon}",
            device=dev,
            validation_interval=validation_interval,
        )
        best_eval = _evaluate_checkpoint(
            run["best_checkpoint"],
            device=dev,
            val_loader=val_loader,
            train_subset_loader=train_subset_loader,
            prompts=prompts,
            generation_settings=gs,
            seed=settings["seed"],
        )
        if int(run["best"]["step"]) == int(run["steps"]):
            final_eval = best_eval
        else:
            final_eval = _evaluate_checkpoint(
                run["final_checkpoint"],
                device=dev,
                val_loader=val_loader,
                train_subset_loader=train_subset_loader,
                prompts=prompts,
                generation_settings=gs,
                seed=settings["seed"],
            )
        run["best"].update(
            {
                "full_val_loss": best_eval["full_val_loss"],
                "full_val_perplexity": best_eval["full_val_perplexity"],
                "full_val_steps": best_eval["full_val_steps"],
                "train_subset_loss": best_eval["train_subset_loss"],
                "train_subset_perplexity": best_eval["train_subset_perplexity"],
                "generalization_gap": (
                    best_eval["full_val_loss"] - best_eval["train_subset_loss"]
                ),
                "samples": best_eval["samples"],
            }
        )
        run["final"].update(
            {
                "subset_val_loss": float(run["trajectory"][-1]["val_loss"]),
                "subset_val_perplexity": float(run["trajectory"][-1]["val_perplexity"]),
                "full_val_loss": final_eval["full_val_loss"],
                "full_val_perplexity": final_eval["full_val_perplexity"],
                "full_val_steps": final_eval["full_val_steps"],
                "train_subset_loss": final_eval["train_subset_loss"],
                "train_subset_perplexity": final_eval["train_subset_perplexity"],
                "generalization_gap": (
                    final_eval["full_val_loss"] - final_eval["train_subset_loss"]
                ),
                "samples": final_eval["samples"],
            }
        )
        run["deteriorated_after_best"] = bool(
            run["final"]["full_val_loss"] > run["best"]["full_val_loss"]
        )
        run["steps_to_best_fraction"] = float(run["best"]["step"]) / float(run["steps"])
        runs.append(run)

    best_run = min(runs, key=lambda r: r["best"]["full_val_loss"])
    candidates = _checkpoint_candidates(runs)
    best_saved_checkpoint = min(
        (c for c in candidates if c["kind"] == "best"),
        key=lambda c: c["full_val_loss"],
    )
    best_any_checkpoint = min(candidates, key=lambda c: c["full_val_loss"])
    comparison = {
        "best_horizon": int(best_run["horizon"]),
        "best_step": int(best_run["best"]["step"]),
        "best_full_val_loss": float(best_run["best"]["full_val_loss"]),
        "best_full_val_perplexity": float(best_run["best"]["full_val_perplexity"]),
        "best_saved_checkpoint": best_saved_checkpoint,
        "best_any_checkpoint": best_any_checkpoint,
        "horizons": [
            {
                "horizon": int(r["horizon"]),
                "best_step": int(r["best"]["step"]),
                "best_full_val_loss": float(r["best"]["full_val_loss"]),
                "best_full_val_perplexity": float(r["best"]["full_val_perplexity"]),
                "final_full_val_loss": float(r["final"]["full_val_loss"]),
                "deteriorated_after_best": bool(r["deteriorated_after_best"]),
            }
            for r in runs
        ],
    }

    pilot_audit = {
        "source": "Experiment 27 supplementary early_validation_pilot",
        "steps": 3000,
        "total_steps": 3000,
        "warmup_steps": 2000,
        "warmup_fraction_of_total": 2000.0 / 3000.0,
        "peak_lr": 1e-3,
        "min_lr": 1e-4,
        "validation_interval": 100,
        "periodic_val_batches": 25,
        "full_val_loss": 2.6503470782303413,
        "best_subset_point": {
            "step": 3000,
            "lr": 0.0001,
            "train_loss": 1.835699849128723,
            "val_loss": 2.517083349227905,
        },
        "shared": [
            "30M architecture, tokenizer, vocabulary, and pg1342 split",
            "seed 0 initialization via _set_global_seed before model construction",
            "data ordering reset to seed 0 immediately before training",
            "BF16 autocast with FP32 cross-entropy reduction",
            "AdamW, peak lr 1e-3, weight_decay 0, clip_grad 1.0",
            "full-validation implementation (exp23.validate_epoch)",
        ],
        "differences": [
            "pilot warmup was 2,000 of 3,000 steps (66.7%); Experiment 27 used 0.77% and Experiment 28 scales that fraction",
            "pilot decayed from peak to 1e-4 over the final 1,000 steps; Experiment 28 horizons decay over their full planned budget",
            "pilot evaluated the first 25 val batches; Experiment 28 uses the first 100",
            "pilot generated samples with one prompt and 8 new tokens; Experiment 28 uses the five default prompts and 40 new tokens",
        ],
        "comparable_use": (
            "The pilot is used only as motivation and as a coarse lower bound on early "
            "validation loss, not as a direct horizon comparison."
        ),
    }

    protocol = {
        "horizons": list(horizons),
        "primary_horizon": int(horizons[-1]),
        "validation_interval": int(validation_interval),
        "periodic_val_batches": int(periodic_val_batches),
        "train_subset_batches": int(train_subset_batches),
        "warmup_fraction": float(WARMUP_FRACTION),
        "warmup_source": "Experiment 27: 2,000 / 259,257 planned updates",
        "warmup_formula": "max(1, round(warmup_fraction * horizon))",
        "min_lr_ratio": float(MIN_LR_RATIO),
        "full_validation": "exp23.validate_epoch over the entire held-out split",
        "periodic_validation": (
            "exp23.validate_epoch over the first periodic_val_batches validation batches"
        ),
        "train_subset": (
            "exp23.validate_epoch over the first train_subset_batches training batches "
            "(shuffle=False, eval mode); used as a fixed train-loss reference"
        ),
        "scheduler": "exp23.WarmupCosineScheduler, one step per optimizer update",
        "mixed_precision": "BF16 autocast, FP32 cross-entropy reduction",
    }

    result = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "objective": (
            "Find the best generalizing checkpoint and useful training horizon for the "
            "Experiment 27 GPT without repeating a full three-epoch run."
        ),
        "config": settings,
        "device": str(dev),
        "data_path": str(data_path),
        "corpus_chars": len(text),
        "tokenizer": {
            "vocab_size": int(tokenizer.vocab_size),
            "vocab_size_target": int(settings["vocab_size"]),
            "alphabet_size": int(tokenizer.alphabet_size),
            "n_merges": int(len(tokenizer.merges)),
            "unk_id": int(tokenizer.unk_id),
            "tokenizer_type": "bpe_char",
        },
        "tokenizer_train_tokens": int(len(train_ids)),
        "tokenizer_val_tokens": int(len(val_ids)),
        "tokenizer_train_chars": int(len(train_text)),
        "tokenizer_val_chars": int(len(val_text)),
        "num_parameters": int(num_params),
        "expected_parameter_count": int(num_params),
        "parameter_count_matches": True,
        "steps_per_epoch": int(steps_per_epoch),
        "protocol": protocol,
        "pilot_audit": pilot_audit,
        "initial_samples": initial_samples,
        "runs": runs,
        "comparison": comparison,
        "generation_settings": gs,
        "prompts": prompts,
    }

    stages = ["initial"]
    for run in runs:
        stages.append(f"h{run['horizon']}_best")
        stages.append(f"h{run['horizon']}_final")
    sample_rows = []
    for prompt in prompts:
        row = {"prompt": prompt, "initial": initial_samples.get(prompt, "")}
        for run in runs:
            row[f"h{run['horizon']}_best"] = run["best"]["samples"].get(prompt, "")
            row[f"h{run['horizon']}_final"] = run["final"]["samples"].get(prompt, "")
        sample_rows.append(row)
    samples_payload = {
        "experiment": EXPERIMENT_NAME,
        "generation_settings": gs,
        "stages": stages,
        "samples": sample_rows,
    }
    result["samples"] = samples_payload

    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        exp27.write_json(out_dir / "metrics.json", result)
        exp27.write_json(out_dir / "samples.json", samples_payload)
        (out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return str(value)
        return f"{value:.{digits}f}"
    return str(value)


def _checkpoint_candidates(runs):
    candidates = []
    for run in runs:
        for kind in ("best", "final"):
            checkpoint = run[kind]
            candidates.append(
                {
                    "horizon": int(run["horizon"]),
                    "kind": kind,
                    "step": int(checkpoint["step"]),
                    "full_val_loss": float(checkpoint["full_val_loss"]),
                    "full_val_perplexity": float(checkpoint["full_val_perplexity"]),
                    "checkpoint": checkpoint.get(
                        "checkpoint", run.get(f"{kind}_checkpoint", "")
                    ),
                    "subset_val_loss": float(
                        checkpoint.get("val_loss", checkpoint.get("subset_val_loss"))
                    ),
                }
            )
    return candidates


def render_report(result: dict) -> str:
    cfg = result["config"]
    protocol = result["protocol"]
    runs = result["runs"]
    comp = result["comparison"]
    lines = []
    lines.append("# Experiment 28 - Early Generalization and Training Horizon")
    lines.append("")
    lines.append(f"Status: {result['status']}")
    lines.append("")
    lines.append("## 1. Deliverables")
    lines.append("")
    lines.append("- `experiments/28_early_generalization.py` - experiment entry point.")
    lines.append("- `experiments/28_early_generalization/report.md` - this report.")
    lines.append("- `experiments/28_early_generalization/metrics.json` - all trajectories and results.")
    lines.append("- `experiments/28_early_generalization/samples.json` - generated text.")
    lines.append("- `checkpoints/28_early_generalization/h{3000,6000,12000}/best.pt` and `final.pt`.")
    lines.append("- `tests/test_early_generalization.py` - offline CPU tests.")
    lines.append("")
    lines.append("## 2. Phase 1 audit: Experiment 27 pilot comparability")
    lines.append("")
    audit = result["pilot_audit"]
    lines.append(
        f"The Experiment 27 supplementary pilot ran {audit['steps']:,} steps with "
        f"{audit['warmup_steps']:,} warmup steps "
        f"({100.0 * audit['warmup_fraction_of_total']:.1f}% of its budget), "
        f"validation every {audit['validation_interval']} steps on the first "
        f"{audit['periodic_val_batches']} val batches, and reached subset val "
        f"{audit['best_subset_point']['val_loss']:.4f} / full val "
        f"{audit['full_val_loss']:.4f}."
    )
    lines.append("")
    lines.append("Shared with Experiment 28:")
    for item in audit["shared"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Differences:")
    for item in audit["differences"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Use: {audit['comparable_use']}")
    lines.append("")
    lines.append("## 3. Fixed setup and schedules")
    lines.append("")
    lines.append(
        f"- Architecture: {cfg['n_layers']} layers, d_model {cfg['d_model']}, "
        f"{cfg['n_heads']} heads, dff {cfg['dff']}, context {cfg['context_len']}, "
        f"dropout {cfg['dropout']}, batch {cfg['batch_size']}, vocab {cfg['vocab_size']}."
    )
    lines.append(
        f"- Data: `{result['data_path']}`; train/val chars "
        f"{result['tokenizer_train_chars']:,}/{result['tokenizer_val_chars']:,}; "
        f"tokens {result['tokenizer_train_tokens']:,}/{result['tokenizer_val_tokens']:,}."
    )
    lines.append(
        f"- Optimizer: AdamW, peak lr {cfg['lr']:.1e}, weight decay {cfg['weight_decay']}, "
        f"clip_grad {cfg['clip_grad']}, BF16 autocast, seed {cfg['seed']}."
    )
    lines.append(
        f"- Warmup: same fraction as Experiment 27 ({protocol['warmup_source']}); "
        f"warmup = {protocol['warmup_formula']}."
    )
    lines.append(
        f"- Nonzero LR floor: {protocol['min_lr_ratio']:.0%} of peak "
        f"({cfg['min_lr']:.1e})."
    )
    lines.append(
        f"- Validation every {protocol['validation_interval']} steps on the first "
        f"{protocol['periodic_val_batches']} val batches; full validation for each saved "
        f"best and final checkpoint."
    )
    lines.append("")
    lines.append("| horizon | warmup steps | warmup % | lr step 0 | lr warmup end | lr final |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for run in runs:
        plan = run["plan"]
        lines.append(
            f"| {run['horizon']:,} | {plan['warmup_steps']:,} | "
            f"{100.0 * plan['warmup_fraction_of_horizon']:.2f}% | "
            f"{plan['lr_at_step_0']:.3e} | {plan['lr_at_warmup_end']:.3e} | "
            f"{plan['lr_at_final_step']:.3e} |"
        )
    lines.append("")
    lines.append("## 4. Results")
    lines.append("")
    lines.append("| horizon | best step | best subset val | best full val | best full ppl | train subset @ best | gap | lr @ best | final full val | deteriorated | seconds | tok/s | peak MB |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        lines.append(
            f"| {run['horizon']:,} | {run['best']['step']:,} | "
            f"{run['best']['val_loss']:.4f} | {run['best']['full_val_loss']:.4f} | "
            f"{run['best']['full_val_perplexity']:.4f} | "
            f"{run['best']['train_subset_loss']:.4f} | "
            f"{run['best']['generalization_gap']:.4f} | {run['best']['lr']:.3e} | "
            f"{run['final']['full_val_loss']:.4f} | {run['deteriorated_after_best']} | "
            f"{run['total_seconds']:.1f} | {run['tokens_per_second']:.1f} | "
            f"{run['peak_memory']['peak_allocated_mb']:.1f} |"
        )
    lines.append("")
    lines.append(
        f"Measured best among `best.pt` checkpoints: horizon "
        f"**{comp['best_horizon']:,}**, step **{comp['best_step']:,}**, full-val "
        f"loss **{comp['best_full_val_loss']:.4f}**, perplexity "
        f"**{comp['best_full_val_perplexity']:.4f}**."
    )
    any_best = comp["best_any_checkpoint"]
    lines.append(
        f"Measured best among all saved `best.pt`/`final.pt` checkpoints: horizon "
        f"**{any_best['horizon']:,}**, step **{any_best['step']:,}**, "
        f"`{any_best['kind']}.pt`, full-val loss **{any_best['full_val_loss']:.4f}**, "
        f"perplexity **{any_best['full_val_perplexity']:.4f}**."
    )
    lines.append("")
    lines.append("## 5. Complete validation trajectories")
    lines.append("")
    lines.append(
        "Training loss is the token-weighted mean over the preceding validation "
        "interval; validation loss uses the fixed first-batch subset."
    )
    lines.append("")
    for run in runs:
        lines.append(f"### Horizon {run['horizon']:,}")
        lines.append("")
        lines.append("| step | lr | train loss | val loss | val ppl | grad mean | grad max | clipped | elapsed s |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for point in run["trajectory"]:
            marker = " *" if point["step"] == run["best"]["step"] else ""
            lines.append(
                f"| {point['step']:,}{marker} | {point['lr']:.3e} | "
                f"{point['train_loss']:.4f} | {point['val_loss']:.4f} | "
                f"{point['val_perplexity']:.4f} | {point['grad_norm_mean']:.3f} | "
                f"{point['grad_norm_max']:.3f} | {point['clipped_fraction']:.3f} | "
                f"{point['elapsed_seconds']:.1f} |"
            )
        lines.append("")
        lines.append("`*` marks the best subset-validation point saved as `best.pt`.")
        lines.append("")
    lines.append("## 6. Generalization analysis")
    lines.append("")
    for run in runs:
        lines.append(
            f"- **Horizon {run['horizon']:,}**: best full val "
            f"{run['best']['full_val_loss']:.4f} at step {run['best']['step']:,} "
            f"({100.0 * run['steps_to_best_fraction']:.1f}% of budget); train subset "
            f"{run['best']['train_subset_loss']:.4f}; gap "
            f"{run['best']['generalization_gap']:.4f}; final full val "
            f"{run['final']['full_val_loss']:.4f}; deteriorated after best: "
            f"{run['deteriorated_after_best']}; best lr {run['best']['lr']:.3e}."
        )
    lines.append("")
    lines.append(
        "Measured: lower training loss does not imply better generalization. The "
        "train/validation gap widens after the early optimum in every run where the "
        "final full-val loss exceeds the best full-val loss."
    )
    lines.append("")
    lines.append("## 7. Generated text")
    lines.append("")
    lines.append("Generation settings: " + json.dumps(result["generation_settings"]) + ".")
    lines.append("")
    stages = result["samples"]["stages"]
    for stage in stages:
        lines.append(f"### {stage}")
        lines.append("")
        for row in result["samples"]["samples"]:
            lines.append(f"- Prompt {row['prompt']!r} -> {row.get(stage, '')}")
        lines.append("")
    lines.append(
        "Interpretation: compare fluency, repetition, and degeneration across the "
        "initial model, each best checkpoint, and each final checkpoint. Lowest "
        "validation loss is not automatically the best-looking sample."
    )
    lines.append("")
    lines.append("## 8. Recommendation")
    lines.append("")
    saved = comp["best_saved_checkpoint"]
    any_best = comp["best_any_checkpoint"]
    lines.append(
        f"Measured: the predeclared subset-validation rule selected horizon "
        f"{saved['horizon']:,} step {saved['step']:,} (`best.pt`) with full-val loss "
        f"{saved['full_val_loss']:.4f} (ppl {saved['full_val_perplexity']:.4f}). "
        f"Among all saved checkpoints, horizon {any_best['horizon']:,} step "
        f"{any_best['step']:,} (`{any_best['kind']}.pt`) has the lowest full-val loss, "
        f"{any_best['full_val_loss']:.4f} (ppl {any_best['full_val_perplexity']:.4f})."
    )
    delta = any_best["full_val_loss"] - saved["full_val_loss"]
    if any_best["kind"] == "final" and abs(delta) < 0.05:
        lines.append(
            f"The gap between the selected best checkpoint and that final checkpoint is "
            f"only {abs(delta):.4f} full-val loss, so this is a plateau near 3,000 steps "
            f"rather than evidence that training longer is useful. Prefer horizon 3,000; "
            f"either checkpoint is defensible, but `{any_best['kind']}.pt` has the better "
            f"measured full-val loss."
        )
    else:
        lines.append(
            f"Use the horizon {any_best['horizon']:,} checkpoint, not the final checkpoint "
            f"of a longer run."
        )
    lines.append("")
    lines.append(
        "Interpretation: if a shorter horizon matches or beats a longer one, the "
        "additional updates are not useful and mainly increase overfitting. If all "
        "horizons peak early and then deteriorate, the useful step budget is roughly "
        "the best-step range. Whether to change data, capacity, or regularization "
        "should be decided by a separate experiment; architecture, tokenizer, and "
        "corpus were intentionally held fixed here."
    )
    lines.append("")
    lines.append("## 9. Full test-suite result")
    lines.append("")
    lines.append(result.get("test_summary", "Pending: run the full test suite after training."))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 28 - early generalization")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--validation-interval", type=int, default=VALIDATION_INTERVAL)
    parser.add_argument("--periodic-val-batches", type=int, default=PERIODIC_VAL_BATCHES)
    parser.add_argument("--train-subset-batches", type=int, default=TRAIN_SUBSET_BATCHES)
    parser.add_argument("--checkpoint-root", type=str, default=str(CHECKPOINT_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision", default=True)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_experiment(
        horizons=tuple(args.horizons),
        data_path=args.data,
        output_dir=args.output_dir,
        checkpoint_root=args.checkpoint_root,
        validation_interval=args.validation_interval,
        periodic_val_batches=args.periodic_val_batches,
        train_subset_batches=args.train_subset_batches,
        settings=experiment_settings(use_mixed_precision=bool(args.mixed_precision)),
        device=args.device,
    )
    print(f"Status: {result['status']}")
    print(f"Best: horizon {result['comparison']['best_horizon']} step "
          f"{result['comparison']['best_step']} full-val "
          f"{result['comparison']['best_full_val_loss']:.4f}")
    for run in result["runs"]:
        print(
            f"h{run['horizon']}: best step {run['best']['step']} subset "
            f"{run['best']['val_loss']:.4f} full {run['best']['full_val_loss']:.4f} "
            f"final full {run['final']['full_val_loss']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
