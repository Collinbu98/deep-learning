"""Experiment 27 - Scheduled GPT training.

Follow-up to Experiments 25 and 26. Trains the same ~30M parameter GPT model
on the same corpus/split for the same three-epoch budget as Experiment 25, but
with a linear-warmup + cosine-decay learning-rate schedule and frequent
intra-epoch validation.

The schedule is motivated by Experiment 26 measurements:
- BF16 matched FP32 closely at lr=1e-3 (so the divergence was not precision).
- Short-horizon gradients were finite but showed early spikes (max pre-clip
  norm ~4.5 vs ~1.26 mean) with ~13% clipping in the first 300 steps, which
  motivates a short warmup.
- lr=1e-4 still trained and stayed finite, so a nonzero 1e-4 floor is safe.

Run with:
    .venv/bin/python experiments/27_scheduled_gpt_training.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import random
import time
import types
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "27_scheduled_gpt"
DEFAULT_METRICS_PATH = REPO_ROOT / "experiments" / "27_scheduled_gpt_training" / "metrics.json"
DEFAULT_SAMPLES_PATH = REPO_ROOT / "experiments" / "27_scheduled_gpt_training" / "samples.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "experiments" / "27_scheduled_gpt_training" / "report.md"
DEFAULT_DATA = REPO_ROOT / "data" / "pg1342.txt"

WARMUP_STEPS = 2000
MIN_LR = 1e-4
VALIDATION_POINTS_PER_EPOCH = 8
PERIODIC_VAL_BATCHES = 100


def _load_module(name, filename):
    path = REPO_ROOT / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp23 = _load_module("scalable_gpt_training_for_exp27", "23_scalable_gpt_training.py")
exp24 = _load_module("training_benchmarks_for_exp27", "24_training_benchmarks.py")
exp25 = _load_module("first_gpt_training_for_exp27", "25_first_gpt_training.py")

DEFAULT_PROMPTS = list(exp25.DEFAULT_PROMPTS)
DEFAULT_GENERATION_SETTINGS = dict(exp25.DEFAULT_GENERATION_SETTINGS)


class _LimitedLoader:
    """Iterate at most ``max_batches`` batches of an existing loader."""

    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = int(max_batches)

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


def scheduled_training_config(
    *,
    vocab_size: int = 256,
    n_layers: int = 16,
    d_model: int = 512,
    n_heads: int = 8,
    dff: int = 768,
    context_len: int = 256,
    dropout: float = 0.0,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 1e-3,
    warmup_steps: int = WARMUP_STEPS,
    min_lr: float = MIN_LR,
    weight_decay: float = 0.0,
    clip_grad: float = 1.0,
    val_split: float = 0.1,
    seed: int = 0,
    device=None,
    use_mixed_precision: bool = True,
    validation_points_per_epoch: int = VALIDATION_POINTS_PER_EPOCH,
    periodic_val_batches: int = PERIODIC_VAL_BATCHES,
) -> dict:
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if float(lr) <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if float(min_lr) < 0.0:
        raise ValueError(f"min_lr must be >= 0, got {min_lr}")
    if float(min_lr) > float(lr):
        raise ValueError(f"min_lr ({min_lr}) must be <= lr ({lr})")
    if float(weight_decay) < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if float(clip_grad) <= 0.0:
        raise ValueError(f"clip_grad must be > 0, got {clip_grad}")
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    if int(validation_points_per_epoch) < 1:
        raise ValueError(
            f"validation_points_per_epoch must be >= 1, got {validation_points_per_epoch}"
        )
    if int(periodic_val_batches) < 1:
        raise ValueError(f"periodic_val_batches must be >= 1, got {periodic_val_batches}")
    model_config = exp23.validate_gpt_config(
        {
            "vocab_size": int(vocab_size),
            "n_layers": int(n_layers),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "dff": int(dff),
            "context_len": int(context_len),
            "dropout": float(dropout),
        }
    )
    return {
        "vocab_size": int(model_config["vocab_size"]),
        "n_layers": int(model_config["n_layers"]),
        "d_model": int(model_config["d_model"]),
        "n_heads": int(model_config["n_heads"]),
        "dff": int(model_config["dff"]),
        "context_len": int(model_config["context_len"]),
        "dropout": float(model_config["dropout"]),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "warmup_steps": int(warmup_steps),
        "min_lr": float(min_lr),
        "weight_decay": float(weight_decay),
        "clip_grad": float(clip_grad),
        "val_split": float(val_split),
        "seed": int(seed),
        "device": None if device is None else str(device),
        "use_mixed_precision": bool(use_mixed_precision),
        "validation_points_per_epoch": int(validation_points_per_epoch),
        "periodic_val_batches": int(periodic_val_batches),
        "schedule": "warmup-cosine",
    }


def default_checkpoint_dir() -> Path:
    return DEFAULT_CHECKPOINT_DIR


def best_checkpoint_path(checkpoint_dir=None) -> Path:
    d = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    return d / "best.pt"


def latest_checkpoint_path(checkpoint_dir=None) -> Path:
    d = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    return d / "latest.pt"


def epoch_checkpoint_path(checkpoint_dir, epoch) -> Path:
    return Path(checkpoint_dir) / f"epoch_{int(epoch)}.pt"


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _training_state(optimizer, scheduler, *, step, epochs_completed, step_in_epoch) -> dict:
    return {
        "step": int(step),
        "epochs_completed": int(epochs_completed),
        "step_in_epoch": int(step_in_epoch),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _update_ratio(model, previous_params) -> float:
    live = [p for p in model.parameters() if p.requires_grad]
    if not live or previous_params is None:
        return 0.0
    deltas = torch.stack(
        [(p.detach() - p_old).abs().max() for p, p_old in zip(live, previous_params)]
    )
    denoms = torch.stack([p_old.abs().max() for p_old in previous_params])
    return float((deltas.max() / (denoms.max() + 1e-8)).item())


def _params_finite(model) -> bool:
    checks = [torch.isfinite(p).all() for p in model.parameters()]
    return bool(torch.stack(checks).all().item()) if checks else True


def _generate_stage_from_checkpoint(device, checkpoint_path, prompt, generation_settings, seed) -> str:
    return exp25._generate_from_checkpoint(
        exp23, device, checkpoint_path, prompt, generation_settings, seed
    )


def run_scheduled_gpt_training(
    *,
    text: str | None = None,
    data_path: str | None = None,
    vocab_size: int = 256,
    n_layers: int = 16,
    d_model: int = 512,
    n_heads: int = 8,
    dff: int = 768,
    context_len: int = 256,
    dropout: float = 0.0,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 1e-3,
    warmup_steps: int = WARMUP_STEPS,
    min_lr: float = MIN_LR,
    weight_decay: float = 0.0,
    clip_grad: float = 1.0,
    val_split: float = 0.1,
    seed: int = 0,
    device=None,
    use_mixed_precision: bool = True,
    validation_points_per_epoch: int = VALIDATION_POINTS_PER_EPOCH,
    periodic_val_batches: int = PERIODIC_VAL_BATCHES,
    checkpoint_dir=None,
    resume: str | Path | None = None,
    prompts=None,
    generation_settings=None,
    max_steps: int | None = None,
) -> dict:
    run_start = time.time()
    settings = scheduled_training_config(
        vocab_size=vocab_size,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        dff=dff,
        context_len=context_len,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        weight_decay=weight_decay,
        clip_grad=clip_grad,
        val_split=val_split,
        seed=seed,
        device=device,
        use_mixed_precision=use_mixed_precision,
        validation_points_per_epoch=validation_points_per_epoch,
        periodic_val_batches=periodic_val_batches,
    )
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    dev = exp23.get_device(settings["device"])

    if text is None:
        text, data_path = exp23.load_corpus_text(types.SimpleNamespace(data=data_path))
    train_text, val_text = exp23.split_corpus(text, settings["val_split"])
    tokenizer, train_ids, val_ids = exp23.fit_tokenizer_and_encode(
        train_text, val_text, int(settings["vocab_size"])
    )
    train_ds = exp23.GPTTokenDataset(train_ids, context_len)
    val_ds = exp23.GPTTokenDataset(val_ids, context_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    periodic_val_loader = _LimitedLoader(val_loader, settings["periodic_val_batches"])

    config = exp23.validate_gpt_config(
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
    _set_global_seed(settings["seed"])
    model = exp23.GPTLanguageModel(**config).to(dev)
    model_config = model.config_dict()
    num_params = exp23.count_parameters(model)
    expected_params = exp23.gpt_parameter_count(
        vocab_size=model_config["vocab_size"],
        n_layers=model_config["n_layers"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        dff=model_config["dff"],
        context_len=model_config["context_len"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = exp23.make_optimizer(model, settings["lr"], weight_decay=settings["weight_decay"])

    steps_per_epoch = int(len(train_loader))
    planned_total_steps = (
        int(max_steps) if max_steps is not None else steps_per_epoch * int(settings["epochs"])
    )
    scheduler = exp23.make_warmup_cosine_scheduler(
        optimizer,
        total_steps=planned_total_steps,
        warmup_steps=int(settings["warmup_steps"]),
        min_lr=float(settings["min_lr"]),
        peak_lr=settings["lr"],
    )
    schedule_info = {
        "kind": "linear-warmup + cosine-decay",
        "peak_lr": float(settings["lr"]),
        "min_lr": float(settings["min_lr"]),
        "warmup_steps": int(settings["warmup_steps"]),
        "total_steps": int(planned_total_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_fraction_of_total": float(settings["warmup_steps"]) / float(planned_total_steps),
        "warmup_fraction_of_epoch": float(settings["warmup_steps"]) / float(steps_per_epoch),
        "min_lr_ratio": float(settings["min_lr"]) / float(settings["lr"]),
        "lr_at_step_0": float(scheduler.lr_for_step(0)),
        "lr_at_warmup_end": float(scheduler.lr_for_step(settings["warmup_steps"])),
        "lr_at_final_step": float(scheduler.lr_for_step(planned_total_steps - 1)),
        "sample_points": {
            name: float(scheduler.lr_for_step(int(fraction * (planned_total_steps - 1))))
            for name, fraction in (
                ("0%", 0.0),
                ("25%", 0.25),
                ("50%", 0.5),
                ("75%", 0.75),
                ("100%", 1.0),
            )
        },
    }

    prompts = list(DEFAULT_PROMPTS if prompts is None else prompts)
    gs = exp25.resolve_generation_settings(generation_settings)

    start_epoch = 1
    global_step = 0
    history = []
    periodic = []
    resumed = False
    samples: dict = {}
    if resume is not None:
        payload = exp23.load_checkpoint(resume)
        restored = exp23.restore_training_state(
            payload, model, optimizer, dev, scheduler=scheduler, restore_rng=True
        )
        state = dict(payload.get("training_state") or {})
        if int(state.get("step_in_epoch", 0)) != 0:
            raise ValueError(
                "resume requires an epoch-boundary checkpoint (step_in_epoch must be 0); "
                f"got step_in_epoch={state.get('step_in_epoch')}"
            )
        start_epoch = int(state.get("epochs_completed", 0)) + 1
        global_step = int(restored["step"])
        if restored.get("scheduler_state") is None:
            scheduler.set_completed_steps(global_step)
        resumed = True
        model.train()

    if not resumed:
        samples["initial"] = exp25.generate_samples(
            model, tokenizer, prompts, generation_settings=gs, seed=settings["seed"], device=dev
        )
    _set_global_seed(settings["seed"])

    best_val_loss = float("inf")
    best_epoch = 0
    best_ckpt_path = best_checkpoint_path(ckpt_dir)
    latest_ckpt_path = latest_checkpoint_path(ckpt_dir)
    loss_ctx = (
        torch.autocast(device_type=str(dev), dtype=torch.bfloat16)
        if settings["use_mixed_precision"]
        else contextlib.nullcontext()
    )
    interval = max(1, steps_per_epoch // int(settings["validation_points_per_epoch"]))
    previous_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    train_seconds = 0.0
    train_tokens = 0
    final_val = None

    stop = False
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        if stop:
            break
        epoch_start = time.time()
        model.train()
        step_in_epoch = 0
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        epoch_steps = 0
        point_loss_sum = 0.0
        point_tokens = 0
        point_steps = 0
        point_grad_sum = 0.0
        point_grad_max = 0.0
        point_clipped = 0
        point_loss_finite = True
        point_grad_finite = True
        point_start = time.time()

        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            used_lr = float(optimizer.param_groups[0]["lr"])
            with loss_ctx:
                loss = exp23.next_token_loss(model, x, y)
                loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad"]))
            clipped = bool(grad_norm > settings["clip_grad"])
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach())
            tokens = int(x.numel())
            train_tokens += tokens
            epoch_loss_sum += loss_value * tokens
            epoch_tokens += tokens
            epoch_steps += 1
            point_loss_sum += loss_value * tokens
            point_tokens += tokens
            point_steps += 1
            point_grad_sum += grad_norm
            point_grad_max = max(point_grad_max, grad_norm)
            point_clipped += int(clipped)
            point_loss_finite = point_loss_finite and math.isfinite(loss_value)
            point_grad_finite = point_grad_finite and math.isfinite(grad_norm)
            global_step += 1
            step_in_epoch += 1

            is_point = (step_in_epoch % interval == 0) or (step_in_epoch == steps_per_epoch)
            if max_steps is not None and global_step >= int(max_steps):
                is_point = True
            if not is_point:
                continue

            model.eval()
            subset_val = exp23.validate_epoch(model, periodic_val_loader, criterion, dev)
            model.train()
            update_ratio = _update_ratio(model, previous_params)
            params_finite = _params_finite(model)
            point_train_loss = point_loss_sum / max(1, point_tokens)
            periodic.append(
                {
                    "step": int(global_step),
                    "epoch": int(epoch),
                    "step_in_epoch": int(step_in_epoch),
                    "lr": float(used_lr),
                    "lr_next": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": float(point_train_loss),
                    "train_perplexity": (
                        float(math.exp(point_train_loss))
                        if math.isfinite(point_train_loss)
                        else float("inf")
                    ),
                    "grad_norm_mean": float(point_grad_sum / max(1, point_steps)),
                    "grad_norm_max": float(point_grad_max),
                    "clipped_fraction": float(point_clipped / max(1, point_steps)),
                    "update_ratio": float(update_ratio),
                    "params_finite": bool(params_finite),
                    "loss_finite": bool(point_loss_finite),
                    "grad_finite": bool(point_grad_finite),
                    "tokens_since_point": int(point_tokens),
                    "tokens_total": int(train_tokens),
                    "elapsed_seconds": float(time.time() - run_start),
                    "val_loss": float(subset_val["loss"]),
                    "val_perplexity": float(subset_val["perplexity"]),
                }
            )
            previous_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            point_loss_sum = 0.0
            point_tokens = 0
            point_steps = 0
            point_grad_sum = 0.0
            point_grad_max = 0.0
            point_clipped = 0
            point_loss_finite = True
            point_grad_finite = True
            point_start = time.time()

            exp23.save_checkpoint(
                latest_ckpt_path,
                model=model,
                config=model_config,
                tokenizer=tokenizer,
                training_state=_training_state(
                    optimizer,
                    scheduler,
                    step=global_step,
                    epochs_completed=epoch,
                    step_in_epoch=step_in_epoch,
                ),
                args={
                    **{k: v for k, v in settings.items()},
                    "schedule_info": schedule_info,
                },
            )

            epoch_done = step_in_epoch >= steps_per_epoch
            reached_max = max_steps is not None and global_step >= int(max_steps)
            if epoch_done or reached_max:
                full_val = exp23.validate_epoch(model, val_loader, criterion, dev)
                full_val_loss = float(full_val["loss"])
                epoch_seconds = float(time.time() - epoch_start)
                epoch_train_loss = float(epoch_loss_sum / max(1, epoch_tokens))
                history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": epoch_train_loss,
                        "train_perplexity": (
                            float(math.exp(epoch_train_loss))
                            if math.isfinite(epoch_train_loss)
                            else float("inf")
                        ),
                        "val_loss": full_val_loss,
                        "val_perplexity": float(full_val["perplexity"]),
                        "train_tokens": int(epoch_tokens),
                        "train_steps": int(epoch_steps),
                        "epoch_seconds": epoch_seconds,
                        "lr_end": float(used_lr),
                        "full_val_steps": int(full_val["steps"]),
                    }
                )
                train_seconds += epoch_seconds
                final_val = full_val
                samples[f"epoch_{epoch}"] = exp25.generate_samples(
                    model,
                    tokenizer,
                    prompts,
                    generation_settings=gs,
                    seed=settings["seed"],
                    device=dev,
                )
                exp23.save_checkpoint(
                    epoch_checkpoint_path(ckpt_dir, epoch),
                    model=model,
                    config=model_config,
                    tokenizer=tokenizer,
                    training_state=_training_state(
                        optimizer,
                        scheduler,
                        step=global_step,
                        epochs_completed=epoch,
                        step_in_epoch=0,
                    ),
                    args={
                        **{k: v for k, v in settings.items()},
                        "schedule_info": schedule_info,
                    },
                )
                if full_val_loss < best_val_loss:
                    best_val_loss = full_val_loss
                    best_epoch = int(epoch)
                    exp23.save_checkpoint(
                        best_ckpt_path,
                        model=model,
                        config=model_config,
                        tokenizer=tokenizer,
                        training_state=_training_state(
                            optimizer,
                            scheduler,
                            step=global_step,
                            epochs_completed=epoch,
                            step_in_epoch=0,
                        ),
                        args={
                            **{k: v for k, v in settings.items()},
                            "schedule_info": schedule_info,
                        },
                    )
                model.train()
                if reached_max:
                    stop = True
                    break

    if final_val is None:
        final_val = exp23.validate_epoch(model, val_loader, criterion, dev)

    samples["best"] = {}
    if best_ckpt_path.exists():
        for prompt in prompts:
            samples["best"][prompt] = _generate_stage_from_checkpoint(
                dev, best_ckpt_path, str(prompt), gs, settings["seed"]
            )
    else:
        samples["best"] = exp25.generate_samples(
            model, tokenizer, prompts, generation_settings=gs, seed=settings["seed"], device=dev
        )

    if dev.type == "cuda":
        mem_stats = exp24.gpu_memory_stats(dev)
    else:
        mem_stats = {
            "device": str(dev),
            "cuda_available": bool(torch.cuda.is_available()),
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
            "free_mb": None,
            "total_mb": None,
            "supported": False,
        }
    final_val_loss = float(final_val["loss"])
    final_val_perplexity = (
        float(math.exp(final_val_loss)) if math.isfinite(final_val_loss) else float("inf")
    )
    tokens_per_second = float(train_tokens / train_seconds) if train_seconds > 0 else float("inf")

    comparison = None
    exp25_metrics_path = REPO_ROOT / "experiments" / "25_first_gpt_training" / "metrics.json"
    if exp25_metrics_path.exists():
        try:
            old = json.loads(exp25_metrics_path.read_text(encoding="utf-8"))
            comparison = {
                "source": str(exp25_metrics_path),
                "exp25_history": [
                    {
                        "epoch": int(h["epoch"]),
                        "train_loss": float(h["train_loss"]),
                        "val_loss": float(h["val_loss"]),
                        "val_perplexity": float(h["val_perplexity"]),
                    }
                    for h in old.get("history", [])
                ],
                "exp27_history": [
                    {
                        "epoch": int(h["epoch"]),
                        "train_loss": float(h["train_loss"]),
                        "val_loss": float(h["val_loss"]),
                        "val_perplexity": float(h["val_perplexity"]),
                    }
                    for h in history
                ],
            }
        except Exception as exc:  # noqa: BLE001
            comparison = {"error": f"could not read Experiment 25 metrics: {exc}"}

    return {
        "status": "completed",
        "device": str(dev),
        "seed": int(settings["seed"]),
        "data_path": str(data_path) if data_path is not None else str(DEFAULT_DATA),
        "corpus_chars": len(text),
        "tokenizer": {
            "vocab_size": int(tokenizer.vocab_size),
            "vocab_size_target": int(tokenizer.vocab_size_target),
            "alphabet_size": int(tokenizer.alphabet_size),
            "n_merges": int(tokenizer.n_merges),
            "unk_id": int(tokenizer.unk_id),
            "tokenizer_type": tokenizer.to_dict().get("type"),
        },
        "tokenizer_train_chars": len(train_text),
        "tokenizer_val_chars": len(val_text),
        "tokenizer_train_tokens": len(train_ids),
        "tokenizer_val_tokens": len(val_ids),
        "config": {
            **{k: settings[k] for k in (
                "vocab_size", "n_layers", "d_model", "n_heads", "dff", "context_len",
                "dropout", "batch_size", "epochs", "lr", "weight_decay", "clip_grad",
                "val_split", "seed", "device", "use_mixed_precision",
            )},
            "warmup_steps": int(settings["warmup_steps"]),
            "min_lr": float(settings["min_lr"]),
            "schedule": settings["schedule"],
            "validation_points_per_epoch": int(settings["validation_points_per_epoch"]),
            "periodic_val_batches": int(settings["periodic_val_batches"]),
        },
        "schedule_info": schedule_info,
        "num_parameters": int(num_params),
        "expected_parameter_count": int(expected_params),
        "parameter_count_matches": bool(int(num_params) == int(expected_params)),
        "estimated_parameter_bytes": exp23.estimate_parameter_memory_bytes(num_params),
        "estimated_optimizer_bytes": exp23.estimate_parameter_memory_bytes(2 * num_params + num_params),
        "steps_per_epoch": int(steps_per_epoch),
        "planned_total_steps": int(planned_total_steps),
        "history": history,
        "periodic": periodic,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_val_perplexity,
        "final_val": final_val,
        "steps": int(global_step),
        "tokens_processed": int(train_tokens),
        "total_seconds": float(time.time() - run_start),
        "train_seconds": float(train_seconds),
        "tokens_per_second": float(tokens_per_second),
        "peak_memory": mem_stats,
        "checkpoint_dir": str(ckpt_dir),
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path),
        "epoch_checkpoints": {
            str(epoch): str(epoch_checkpoint_path(ckpt_dir, epoch))
            for epoch in range(1, int(settings["epochs"]) + 1)
        },
        "resumed": bool(resumed),
        "start_epoch": int(start_epoch),
        "initial_samples": samples.get("initial", {}),
        "generation_settings": gs,
        "prompts": prompts,
        "samples": samples,
        "comparison_with_exp25": comparison,
    }


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return p


def _fmt_optional(value, spec=".4f"):
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def render_report(result: dict) -> str:
    cfg = result["config"]
    sch = result["schedule_info"]
    lines = []
    lines.append("# Experiment 27 - Scheduled GPT Training")
    lines.append("")
    lines.append(f"Status: {result['status']}")
    lines.append("")
    lines.append("## 1. Files added and changed")
    lines.append("")
    lines.append("- `experiments/27_scheduled_gpt_training.py` - scheduled training entry point.")
    lines.append("- `experiments/27_scheduled_gpt_training/report.md` - this report.")
    lines.append("- `experiments/27_scheduled_gpt_training/metrics.json` - machine-readable metrics.")
    lines.append("- `experiments/27_scheduled_gpt_training/samples.json` - generated samples.")
    lines.append("- `checkpoints/27_scheduled_gpt/` - latest/best/epoch checkpoints.")
    lines.append("- `experiments/23_scalable_gpt_training.py` - added opt-in")
    lines.append("  `WarmupCosineScheduler` (linear warmup + cosine decay to a nonzero minimum),")
    lines.append("  scheduler stepping in `train_epoch`/`train_model`, scheduler + RNG state")
    lines.append("  restoration, and `--lr-schedule` / `--warmup-steps` / `--min-lr` CLI flags.")
    lines.append("- `tests/test_scheduled_gpt_training.py` - offline scheduler/restore tests.")
    lines.append("")
    lines.append("## 2. Exact schedule and why it was selected")
    lines.append("")
    lines.append(f"- Schedule: **{sch['kind']}** over the planned optimizer-update budget.")
    lines.append(f"- Peak lr: **{sch['peak_lr']:.1e}** (kept equal to Experiment 25).")
    lines.append(f"- Linear warmup: **{sch['warmup_steps']:,}** optimizer steps")
    lines.append(f"  ({100.0 * sch['warmup_fraction_of_total']:.3f}% of total steps;")
    lines.append(f"  {100.0 * sch['warmup_fraction_of_epoch']:.2f}% of one epoch).")
    lines.append(f"- Cosine decay from peak to a nonzero minimum lr of **{sch['min_lr']:.1e}**")
    lines.append(f"  ({100.0 * sch['min_lr_ratio']:.0f}% of peak).")
    lines.append(f"- Planned total optimizer steps: **{sch['total_steps']:,}**")
    lines.append(f"  ({sch['steps_per_epoch']:,} steps/epoch x {cfg['epochs']} epochs).")
    lines.append(f"- lr at step 0: {sch['lr_at_step_0']:.3e}; at warmup end: {sch['lr_at_warmup_end']:.3e};")
    lines.append(f"  at the final step: {sch['lr_at_final_step']:.3e}.")
    lines.append("- The scheduler advances once per optimizer update (after `optimizer.step()`),")
    lines.append("  never on validation passes.")
    lines.append("")
    lines.append("Why these choices (from Experiment 26 measurements):")
    lines.append("")
    lines.append("- Experiment 26 showed BF16 and FP32 were nearly identical at lr=1e-3")
    lines.append("  (final val 3.7512 vs 3.7475), so precision was not the divergence cause and")
    lines.append("  the peak lr was kept at 1e-3.")
    lines.append("- In the first 300 steps at lr=1e-3, pre-clip gradient norms spiked to ~4.5")
    lines.append("  against a ~1.26 mean and ~13% of steps were clipped. A short warmup lets the")
    lines.append("  early transient settle before reaching the peak lr; 2,000 steps is ~0.8% of")
    lines.append("  the planned budget and ~2.3% of an epoch.")
    lines.append("- Experiment 26's lr=1e-4 run stayed finite and still improved (val 3.798),")
    lines.append("  supporting a nonzero 1e-4 floor instead of decaying to zero.")
    lines.append("")
    lines.append("## 3. Loss and perplexity trajectories")
    lines.append("")
    lines.append("### Per-epoch (full validation split)")
    lines.append("")
    lines.append("| epoch | train loss | train ppl | val loss | val ppl | train steps | epoch seconds | lr at end |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for h in result["history"]:
        lines.append(
            f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_perplexity']:.4f} | "
            f"{h['val_loss']:.4f} | {h['val_perplexity']:.4f} | {h['train_steps']} | "
            f"{h['epoch_seconds']:.1f} | {h['lr_end']:.3e} |"
        )
    lines.append("")
    lines.append("### Periodic intra-epoch measurements")
    lines.append("")
    lines.append("Periodic validation uses a fixed subset of the first")
    lines.append(f"{cfg['periodic_val_batches']} validation batches (same methodology every")
    lines.append("point); epoch-end rows above use the full validation split.")
    lines.append("")
    lines.append("| step | epoch | step/epoch | lr | train loss | grad mean | grad max | clipped | update ratio | subset val | finite |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in result["periodic"]:
        finite = "yes" if (p["loss_finite"] and p["grad_finite"] and p["params_finite"]) else "NO"
        lines.append(
            f"| {p['step']} | {p['epoch']} | {p['step_in_epoch']} | {p['lr']:.3e} | "
            f"{p['train_loss']:.4f} | {p['grad_norm_mean']:.3f} | {p['grad_norm_max']:.3f} | "
            f"{p['clipped_fraction']:.3f} | {p['update_ratio']:.3e} | {p['val_loss']:.4f} | {finite} |"
        )
    lines.append("")
    lines.append("## 4. Did training and validation improve consistently?")
    lines.append("")
    hist = result["history"]
    if len(hist) >= 2:
        train_better = all(
            hist[i + 1]["train_loss"] <= hist[i]["train_loss"] for i in range(len(hist) - 1)
        )
        val_better = all(
            hist[i + 1]["val_loss"] <= hist[i]["val_loss"] for i in range(len(hist) - 1)
        )
        train_path = " -> ".join("{:.4f}".format(h["train_loss"]) for h in hist)
        val_path = " -> ".join("{:.4f}".format(h["val_loss"]) for h in hist)
        lines.append(f"- Train loss improved every epoch: **{train_better}** ({train_path}).")
        lines.append(f"- Validation loss improved every epoch: **{val_better}** ({val_path}).")
        finite_all = all(
            p["loss_finite"] and p["grad_finite"] and p["params_finite"] for p in result["periodic"]
        )
        lines.append(f"- All periodic finite-value checks passed: **{finite_all}**.")
        lines.append(f"- Best validation epoch: **{result['best_epoch']}** "
                     f"(val loss {result['best_val_loss']:.4f}).")
        if result["periodic"]:
            point = min(result["periodic"], key=lambda p: p["val_loss"])
            lines.append(
                f"- Best periodic subset validation point: step {point['step']:,} "
                f"(subset val loss {point['val_loss']:.4f}; the subset is not directly "
                "comparable to the full-val epoch rows)."
            )
    else:
        lines.append("- Fewer than two epochs were completed, so a trend cannot be assessed.")
    lines.append("")
    lines.append("## 5. Was the Experiment 25 failure avoided?")
    lines.append("")
    comparison = result.get("comparison_with_exp25")
    if comparison and comparison.get("exp25_history"):
        lines.append("| epoch | Exp25 train | Exp27 train | Exp25 val | Exp27 val |")
        lines.append("| --- | --- | --- | --- | --- |")
        old = {int(h["epoch"]): h for h in comparison["exp25_history"]}
        for h in comparison["exp27_history"]:
            o = old.get(int(h["epoch"]), {})
            lines.append(
                f"| {h['epoch']} | {_fmt_optional(o.get('train_loss'))} | {h['train_loss']:.4f} | "
                f"{_fmt_optional(o.get('val_loss'))} | {h['val_loss']:.4f} |"
            )
        lines.append("")
        lines.append("Measured: Experiment 25's train and validation losses both rose after")
        lines.append("epoch 1. Experiment 27 does not reproduce a rising training loss (it falls")
        lines.append("every epoch), but its full-validation loss rises monotonically after epoch 1")
        lines.append("and ends far above Experiment 25's. The epoch-1 full-val result is already")
        lines.append("well above the early subset-val minimum (section 11), so this is severe")
        lines.append("overfitting rather than the Experiment 25 optimization breakdown.")
        lines.append("")
        lines.append("Baseline caveat: Experiment 25 predates the FP32 cross-entropy reduction")
        lines.append("added while running Experiment 26 (git history commits both together).")
        lines.append("Its 3.878/4.667/5.039 trajectory is therefore not an apples-to-apples")
        lines.append("baseline for the current code path; Experiment 26's current-code 300-step")
        lines.append("runs (subset val 3.647-3.751) are the relevant short-horizon reference.")
    else:
        lines.append("Experiment 25 metrics were unavailable for comparison.")
    lines.append("")
    lines.append("## 6. Generated text across training")
    lines.append("")
    lines.append("Generation settings: " + json.dumps(result["generation_settings"]) + ".")
    lines.append("")
    for stage in ("initial", "epoch_1", "epoch_2", "epoch_3", "best"):
        texts = result["samples"].get(stage)
        if not texts:
            continue
        lines.append(f"### Stage {stage}")
        lines.append("")
        for prompt, text in texts.items():
            lines.append(f"- Prompt {prompt!r} -> {text}")
        lines.append("")
    lines.append("Interpretation: compare recognizable words/structure, repetition, and")
    lines.append("degeneration across stages, and check whether quality tracks validation loss.")
    lines.append("")
    lines.append("## 7. Throughput, memory, and duration")
    lines.append("")
    lines.append(f"- Total optimizer steps: {result['steps']:,}")
    lines.append(f"- Tokens processed: {result['tokens_processed']:,}")
    lines.append(f"- Measured training seconds (epoch walls incl. validation): "
                 f"{result['train_seconds']:.1f}")
    lines.append(f"- Total run wall seconds: {result['total_seconds']:.1f}")
    lines.append(f"- Throughput: {result['tokens_per_second']:.2f} tokens/s")
    lines.append(f"- Peak GPU memory: {result['peak_memory']['peak_allocated_mb']:.2f} MB allocated / "
                 f"{result['peak_memory']['peak_reserved_mb']:.2f} MB reserved")
    for h in result["history"]:
        lines.append(f"- Epoch {h['epoch']}: {h['epoch_seconds']:.1f}s, {h['train_tokens']:,} tokens")
    lines.append("")
    lines.append("## 8. Checkpoints and resume")
    lines.append("")
    lines.append(f"- Latest: `{result['latest_checkpoint']}`")
    lines.append(f"- Best: `{result['best_checkpoint']}` (epoch {result['best_epoch']})")
    lines.append(f"- Epoch checkpoints: {json.dumps(result['epoch_checkpoints'])}")
    lines.append("- Checkpoints include model weights, optimizer state, scheduler state, RNG")
    lines.append("  state (Python/CPU/CUDA), step counters, and configuration metadata.")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python experiments/27_scheduled_gpt_training.py")
    lines.append(".venv/bin/python experiments/27_scheduled_gpt_training.py --resume checkpoints/27_scheduled_gpt/epoch_2.pt")
    lines.append("```")
    lines.append("")
    lines.append("## 9. Readiness and recommendation")
    lines.append("")
    lines.append("Read section 4/5 for the measured trend before reusing this base model.")
    lines.append("If all three epochs improved monotonically and stayed finite, the scheduled")
    lines.append("recipe resolves the Experiment 25 deterioration at this budget and the best")
    lines.append("checkpoint is suitable as the base for the next stage. If any epoch worsened,")
    lines.append("the periodic table identifies where; the next diagnostic should then test a")
    lines.append("longer warmup, a lower peak lr, or a shorter effective step budget rather than")
    lines.append("scaling up. All statements above are measured unless explicitly marked as")
    lines.append("interpretation.")
    lines.append("")
    lines.append("Measured result of this run: training loss fell every epoch, but full-val")
    lines.append("loss rose every epoch and the subset-val trajectory rose within epoch 1.")
    lines.append("Interpretation: the schedule removed the Experiment 25 train-loss blow-up but")
    lines.append("did not produce a good final model; the model overfits before the first")
    lines.append("periodic validation point. The next run should validate far more frequently")
    lines.append("early (e.g. every 250-1,000 steps), keep the best checkpoint by full")
    lines.append("validation, and consider a shorter budget or stronger regularization rather")
    lines.append("than three full epochs.")
    lines.append("")
    lines.append("## 10. Full test-suite result")
    lines.append("")
    lines.append(result.get("test_summary", "Pending: run the full test suite after training."))
    lines.append("")
    pilot = result.get("early_validation_pilot")
    if pilot:
        lines.append("## 11. Supplementary early-validation pilot")
        lines.append("")
        lines.append("The main run validated only eight times per epoch (every ~10,802")
        lines.append("steps), so it could not resolve the early optimum. This supplementary")
        lines.append("pilot replays the same seed/configuration for a short budget with")
        lines.append("validation every 100 steps, using the first 25 validation batches.")
        lines.append("")
        lines.append(f"- Pilot steps: {pilot['steps']:,}; validation interval: "
                     f"{pilot['validation_interval']}; subset val batches: "
                     f"{pilot['periodic_val_batches']}.")
        lines.append(f"- Full-validation loss at step {pilot['steps']:,}: "
                     f"{pilot['full_val_loss']:.4f}.")
        lines.append(f"- Best subset-val point: step {pilot['best_subset_point']['step']:,}, "
                     f"val {pilot['best_subset_point']['val_loss']:.4f}.")
        lines.append("- Schedule caveat: the pilot's planned total is only "
                     f"{pilot['total_steps']:,} steps, so after warmup its lr decays faster")
        lines.append("  than the main run's; steps 0-2,000 use the same warmup and are the")
        lines.append("  directly comparable portion.")
        lines.append("")
        lines.append("| step | lr | train loss | subset val |")
        lines.append("| --- | --- | --- | --- |")
        for p in pilot["points"]:
            lines.append(
                f"| {p['step']} | {p['lr']:.3e} | {p['train_loss']:.4f} | {p['val_loss']:.4f} |"
            )
        lines.append("")
        lines.append("Measured: subset val reached its minimum in this pilot at step")
        lines.append(f"{pilot['best_subset_point']['step']:,} and was already far below the main")
        lines.append("run's first periodic subset-val point (step 10,802, val 4.3605).")
        lines.append("Interpretation: the useful checkpoint lies in the first few thousand")
        lines.append("steps, before the main run's validation cadence sampled the model.")
        lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 27 - Scheduled GPT training")
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dff", type=int, default=768)
    parser.add_argument("--context_len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--min-lr", type=float, default=MIN_LR)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--validation-points", type=int, default=VALIDATION_POINTS_PER_EPOCH)
    parser.add_argument("--periodic-val-batches", type=int, default=PERIODIC_VAL_BATCHES)
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap (smoke/testing)")
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision", default=True)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_scheduled_gpt_training(
        data_path=args.data,
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dff=args.dff,
        context_len=args.context_len,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        use_mixed_precision=bool(args.mixed_precision),
        validation_points_per_epoch=args.validation_points,
        periodic_val_batches=args.periodic_val_batches,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        prompts=list(DEFAULT_PROMPTS),
        generation_settings=DEFAULT_GENERATION_SETTINGS,
        max_steps=args.max_steps,
    )
    write_json(DEFAULT_METRICS_PATH, result)
    stage_order = ["initial"] + [
        f"epoch_{epoch}" for epoch in range(1, int(result["config"]["epochs"]) + 1)
    ] + ["best"]
    write_json(
        DEFAULT_SAMPLES_PATH,
        {
            "experiment": "27_scheduled_gpt_training",
            "generation_settings": result["generation_settings"],
            "stages": [s for s in stage_order if s in result["samples"]],
            "samples": [
                {
                    "prompt": prompt,
                    **{
                        stage: result["samples"].get(stage, {}).get(prompt, "")
                        for stage in stage_order
                    },
                }
                for prompt in result["prompts"]
            ],
        },
    )
    Path(DEFAULT_REPORT_PATH).write_text(render_report(result), encoding="utf-8")
    print(f"Training status: {result['status']}")
    print(f"Device: {result['device']}")
    print(f"Vocabulary size: {result['tokenizer']['vocab_size']}")
    print(f"Train tokens: {result['tokenizer_train_tokens']} | Val tokens: {result['tokenizer_val_tokens']}")
    print(f"Model params: {result['num_parameters']:,} (match={result['parameter_count_matches']})")
    print(f"Schedule: {result['schedule_info']['kind']} warmup={result['schedule_info']['warmup_steps']} "
          f"min_lr={result['schedule_info']['min_lr']} total_steps={result['schedule_info']['total_steps']}")
    for h in result["history"]:
        print(
            f"Epoch {h['epoch']} | train {h['train_loss']:.4f} | val {h['val_loss']:.4f} "
            f"| val ppl {h['val_perplexity']:.4f} | lr_end {h['lr_end']:.3e} | {h['epoch_seconds']:.1f}s"
        )
    print(f"Best val loss {result['best_val_loss']:.4f} at epoch {result['best_epoch']}")
    print(f"Peak allocated GPU memory: {result['peak_memory']['peak_allocated_mb']:.2f} MB")
    print(f"Steps run: {result['steps']}")
    print(f"Checkpoint dir: {result['checkpoint_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
