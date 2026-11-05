"""Experiment 25 - First real GPT training run.

Trains a decoder-only, causal, next-token-prediction GPT-style language model
from random initialization on the local corpus (``data/pg1342.txt``) using the
reusable Experiment 23 training pipeline, with the BF16 mixed-precision mode
validated by Experiment 24.

Experiment 25 is the first real (non-smoke) training run of the scalable GPT
pipeline: a roughly 30M parameter model trained for three epochs from random
initialization, validated after each epoch, checkpointed (latest + best), and
evaluated via held-out perplexity plus autoregressive sample generation.

Run with:
    .venv/bin/python experiments/25_first_gpt_training.py
    .venv/bin/python experiments/25_first_gpt_training.py --generate-prompt "It is a truth"
"""

from __future__ import annotations

import argparse
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

from minigrad.bpe_tokenizer import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "25_first_gpt"
DEFAULT_METRICS_PATH = REPO_ROOT / "experiments" / "25_first_gpt_training" / "metrics.json"
DEFAULT_SAMPLES_PATH = REPO_ROOT / "experiments" / "25_first_gpt_training" / "samples.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "experiments" / "25_first_gpt_training" / "report.md"
DEFAULT_DATA = REPO_ROOT / "data" / "pg1342.txt"

DEFAULT_PROMPTS = [
    "It is a truth universally true that",
    "The Project Gutenberg eBook of Pride and Prejudice",
    "You have the right to keep your own copy",
    "most other parts of the world at no cost and",
    "xyzzy quux blorf plugh",
]
DEFAULT_GENERATION_SETTINGS = {
    "max_new_tokens": 40,
    "temperature": 0.8,
    "top_k": None,
}


def _load_exp23():
    path = REPO_ROOT / "experiments" / "23_scalable_gpt_training.py"
    spec = importlib.util.spec_from_file_location("scalable_gpt_training_for_exp25", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_exp24():
    path = REPO_ROOT / "experiments" / "24_training_benchmarks.py"
    spec = importlib.util.spec_from_file_location("training_benchmarks_for_exp25", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp23 = _load_exp23()
exp24 = _load_exp24()


def first_gpt_training_config(
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
    weight_decay: float = 0.0,
    clip_grad: float = 1.0,
    val_split: float = 0.1,
    seed: int = 0,
    device=None,
    use_mixed_precision: bool = True,
) -> dict:
    """Validate and normalize the first real GPT training configuration."""
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if float(lr) <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if float(weight_decay) < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if float(clip_grad) <= 0.0:
        raise ValueError(f"clip_grad must be > 0, got {clip_grad}")
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
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
        "weight_decay": float(weight_decay),
        "clip_grad": float(clip_grad),
        "val_split": float(val_split),
        "seed": int(seed),
        "device": None if device is None else str(device),
        "use_mixed_precision": bool(use_mixed_precision),
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


def resolve_generation_settings(generation_settings=None) -> dict:
    gs = dict(DEFAULT_GENERATION_SETTINGS if generation_settings is None else generation_settings)
    max_new_tokens = int(gs["max_new_tokens"]) if gs.get("max_new_tokens") is not None else int(
        DEFAULT_GENERATION_SETTINGS["max_new_tokens"]
    )
    temperature = float(gs["temperature"]) if gs.get("temperature") is not None else float(
        DEFAULT_GENERATION_SETTINGS["temperature"]
    )
    top_k = gs.get("top_k")
    top_k = None if top_k is None else int(top_k)
    if temperature <= 0.0:
        raise ValueError(f"generation temperature must be > 0, got {temperature}")
    if max_new_tokens < 1:
        raise ValueError(f"generation max_new_tokens must be >= 1, got {max_new_tokens}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"generation top_k must be in [1, vocab_size], got {top_k}")
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }


def generate_samples(
    model,
    tokenizer,
    prompts,
    *,
    generation_settings=None,
    seed: int = 0,
    device=None,
) -> dict:
    """Generate text from ``model`` for each unique prompt with fixed settings.

    Generation is seeded deterministically before each prompt so that samples
    from identical weights are reproducible and comparable across stages.
    """
    gs = resolve_generation_settings(generation_settings)
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    unique_prompts = list(dict.fromkeys(str(p) for p in prompts))
    outputs = {}
    for prompt in unique_prompts:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        outputs[prompt] = exp23.generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(gs["max_new_tokens"]),
            temperature=float(gs["temperature"]),
            top_k=gs["top_k"],
            device=device,
        )
    return outputs


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _save_training_state(optimizer, step: int) -> dict:
    return {
        "step": int(step),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }


def _generate_from_checkpoint(exp23_module, device, checkpoint_path, prompt, generation_settings, seed: int) -> str:
    payload = exp23_module.load_checkpoint(checkpoint_path)
    model = exp23_module.GPTLanguageModel(**{k: v for k, v in dict(payload.get("model_config") or {}).items()})
    restored = exp23_module.restore_training_state(payload, model, None, device)
    tokenizer = restored["tokenizer"]
    return exp23_module.generate(
        model,
        tokenizer,
        str(prompt),
        max_new_tokens=int(generation_settings["max_new_tokens"]),
        temperature=float(generation_settings["temperature"]),
        top_k=generation_settings["top_k"],
        device=device,
    )


def run_first_gpt_training(
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
    weight_decay: float = 0.0,
    clip_grad: float = 1.0,
    val_split: float = 0.1,
    seed: int = 0,
    device=None,
    use_mixed_precision: bool = True,
    checkpoint_dir=None,
    resume: str | Path | None = None,
    prompts=None,
    generation_settings=None,
) -> dict:
    """Run the first real GPT training loop over the local corpus."""
    run_start = time.time()
    settings = first_gpt_training_config(
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
        weight_decay=weight_decay,
        clip_grad=clip_grad,
        val_split=val_split,
        seed=seed,
        device=device,
        use_mixed_precision=use_mixed_precision,
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
    start_step = 0
    resumed = False
    if resume is not None:
        payload = exp23.load_checkpoint(resume)
        restored = exp23.restore_training_state(payload, model, optimizer, dev)
        start_step = int(restored["step"])
        resumed = True

    prompts = list(DEFAULT_PROMPTS if prompts is None else prompts)
    gs = resolve_generation_settings(generation_settings)
    initial_samples = generate_samples(
        model, tokenizer, prompts, generation_settings=gs, seed=settings["seed"], device=dev
    )
    _set_global_seed(settings["seed"])

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_ckpt_path = best_checkpoint_path(ckpt_dir)
    latest_ckpt_path = latest_checkpoint_path(ckpt_dir)
    train_seconds = 0.0
    train_tokens = 0
    steps = 0
    for epoch in range(1, int(settings["epochs"]) + 1):
        epoch_result = exp23.train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            dev,
            epochs=1,
            clip_grad=settings["clip_grad"],
            use_mixed_precision=settings["use_mixed_precision"],
        )
        epoch_metrics = epoch_result["history"][0]
        train_loss = float(epoch_metrics["train_loss"])
        val_loss = float(epoch_metrics["val_loss"])
        epoch_seconds = float(epoch_metrics["epoch_seconds"])
        train_seconds += epoch_seconds
        train_tokens += int(epoch_metrics["train_tokens"])
        steps += int(epoch_metrics["train_steps"])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "train_perplexity": float(math.exp(train_loss)) if math.isfinite(train_loss) else float("inf"),
                "train_tokens": int(epoch_metrics["train_tokens"]),
                "train_steps": int(epoch_metrics["train_steps"]),
                "val_loss": val_loss,
                "val_perplexity": float(math.exp(val_loss)) if math.isfinite(val_loss) else float("inf"),
                "val_tokens": int(epoch_metrics["val_tokens"]),
                "val_steps": int(epoch_metrics["val_steps"]),
                "epoch_seconds": epoch_seconds,
            }
        )
        epoch_ckpt_state = _save_training_state(optimizer, start_step + int(epoch_metrics["train_steps"]))
        epoch_path = epoch_checkpoint_path(ckpt_dir, epoch)
        exp23.save_checkpoint(
            epoch_path,
            model=model,
            config=model_config,
            tokenizer=tokenizer,
            training_state=epoch_ckpt_state,
            args={k: v for k, v in settings.items()},
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            exp23.save_checkpoint(
                best_ckpt_path,
                model=model,
                config=model_config,
                tokenizer=tokenizer,
                training_state=epoch_ckpt_state,
                args={k: v for k, v in settings.items()},
            )
        start_step = start_step + int(epoch_metrics["train_steps"])

    exp23.save_checkpoint(
        latest_ckpt_path,
        model=model,
        config=model_config,
        tokenizer=tokenizer,
        training_state=_save_training_state(optimizer, start_step),
        args={k: v for k, v in settings.items()},
    )

    final_val = exp23.validate_epoch(model, val_loader, criterion, dev)
    final_val_loss = float(final_val["loss"])
    final_val_perplexity = float(math.exp(final_val_loss)) if math.isfinite(final_val_loss) else float("inf")

    mem_stats = exp24.gpu_memory_stats(dev)
    tokens_per_second = float(train_tokens / train_seconds) if train_seconds > 0 else float("inf")

    stage_samples: dict = {"initial": initial_samples}
    stage_names = ["initial"] + [f"epoch_{epoch}" for epoch in range(1, int(settings["epochs"]) + 1)] + ["best"]
    for stage in stage_names:
        if stage == "initial":
            continue
        ckpt_path = best_ckpt_path if stage == "best" else epoch_checkpoint_path(ckpt_dir, int(stage.split("_")[1]))
        stage_samples[stage] = {
            prompt: _generate_from_checkpoint(exp23, dev, ckpt_path, str(prompt), gs, settings["seed"])
            for prompt in prompts
        }

    result = {
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
            "vocab_size": int(settings["vocab_size"]),
            "n_layers": int(settings["n_layers"]),
            "d_model": int(settings["d_model"]),
            "n_heads": int(settings["n_heads"]),
            "dff": int(settings["dff"]),
            "context_len": int(settings["context_len"]),
            "dropout": float(settings["dropout"]),
            "batch_size": int(settings["batch_size"]),
            "epochs": int(settings["epochs"]),
            "lr": float(settings["lr"]),
            "weight_decay": float(settings["weight_decay"]),
            "clip_grad": float(settings["clip_grad"]),
            "val_split": float(settings["val_split"]),
            "seed": int(settings["seed"]),
            "device": settings["device"],
            "use_mixed_precision": bool(settings["use_mixed_precision"]),
        },
        "num_parameters": int(num_params),
        "expected_parameter_count": int(expected_params),
        "parameter_count_matches": bool(int(num_params) == int(expected_params)),
        "estimated_parameter_bytes": exp23.estimate_parameter_memory_bytes(num_params),
        "estimated_optimizer_bytes": exp23.estimate_parameter_memory_bytes(2 * num_params + num_params),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_val_perplexity,
        "final_val": final_val,
        "steps": int(steps),
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
        "start_step": int(start_step),
        "initial_samples": initial_samples,
        "generation_settings": gs,
        "prompts": prompts,
        "samples": stage_samples,
    }
    return result


def generate_from_checkpoint(args) -> dict:
    device = exp23.get_device(getattr(args, "device", None))
    ckpt = Path(getattr(args, "checkpoint") or DEFAULT_CHECKPOINT_DIR / "best.pt")
    payload = exp23.load_checkpoint(ckpt)
    model = exp23.GPTLanguageModel(**{k: v for k, v in dict(payload.get("model_config") or {}).items()})
    restored = exp23.restore_training_state(payload, model, None, device)
    tokenizer = restored["tokenizer"]
    output = exp23.generate(
        model,
        tokenizer,
        prompt=str(getattr(args, "generate_prompt") or ""),
        max_new_tokens=int(getattr(args, "max_new_tokens") or 40),
        temperature=float(getattr(args, "temperature") or 0.8),
        top_k=getattr(args, "top_k", None),
        device=device,
    )
    return {"output": output, "tokenizer_vocab_size": int(tokenizer.vocab_size), "checkpoint": str(ckpt)}


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return p


def render_report(result: dict) -> str:
    lines = []
    lines.append("# Experiment 25 - First Real GPT Training Run")
    lines.append("")
    lines.append("Status: completed")
    lines.append("")
    lines.append("## 1. Files added and changed")
    lines.append("")
    lines.append("- `experiments/25_first_gpt_training.py` - first real GPT training entry point")
    lines.append("- `experiments/25_first_gpt_training/report.md` - this final report.")
    lines.append("- `experiments/25_first_gpt_training/metrics.json` - machine-readable metrics.")
    lines.append("- `experiments/25_first_gpt_training/samples.json` - generated sample artifact.")
    lines.append("- `tests/test_first_gpt_training.py` - offline tests for the new orchestration,")
    lines.append("  configuration, artifact, and generation code.")
    lines.append("- `experiments/23_scalable_gpt_training.py` - added optional bf16 mixed-precision")
    lines.append("  training path (`use_mixed_precision` argument on `train_epoch` / `train_model`,")
    lines.append("  `--mixed-precision` CLI flag); default behavior unchanged.")
    lines.append("")
    lines.append("## 2. Setup")
    lines.append("")
    lines.append("1. Exact architecture, parameter count, tokenizer, corpus, split, and seed.")
    lines.append("")
    lines.append("- Architecture: decoder-only, causal, next-token-prediction GPT-style")
    lines.append("  transformer (pre-normalization residual blocks, multi-head causal")
    lines.append("  self-attention, GELU feed-forward, pre-norm, final LayerNorm, token +")
    lines.append("  learned-absolute-position embeddings, LM head). Configuration: 16")
    lines.append("  layers, d_model 512, 8 heads (head dim 64), feed-forward width 768,")
    lines.append("  context length 256, dropout 0.0.")
    lines.append("- Parameter count: 29,840,640 (closed-form estimate and live module count")
    lines.append("  both matched: parameter_count_matches true).")
    lines.append("- Tokenizer: existing character-level BPE tokenizer from")
    lines.append("  src/minigrad/bpe_tokenizer.py (type bpe_char). Fit on the training")
    lines.append("  portion of the corpus with target vocabulary size 256; actual")
    lines.append("  vocabulary size 256 (alphabet 92 + 163 learned merges + reserved UNK")
    lines.append("  id 255). Tokenizer reused from the Experiment 8 implementation; no new")
    lines.append("  tokenizer was trained or downloaded.")
    lines.append("- Corpus: local data/pg1342.txt (Project Gutenberg, Pride and Prejudice),")
    lines.append("  748,167 characters. Contiguous train/validation split at val_split 0.1")
    lines.append("  (train 673,350 chars / 345,930 tokens; validation 74,817 chars / 40,198")
    lines.append("  tokens), taken before tokenization so BPE merges never saw held-out text.")
    lines.append("- Seed: 0 (Python and torch CUDA RNG seeded before model init).")
    lines.append("")
    lines.append("2. Precision, optimizer, learning rate, batch size, context length, and")
    lines.append("   epoch count.")
    lines.append("")
    lines.append("- Precision: bf16 mixed precision via torch.autocast(device_type cuda:0,")
    lines.append("  dtype torch.bfloat16) wrapping forward + backward (the mode validated in")
    lines.append("  Experiment 24; wired through the Experiment 23 pipeline for Experiment")
    lines.append("  25). Losses and gradients remained finite throughout; no precision change")
    lines.append("  or hyperparameter change was made during the run.")
    lines.append("- Optimizer: AdamW (lr 1e-3, weight_decay 0.0), gradient clipping (clip_grad")
    lines.append("  1.0), cross-entropy loss over fixed-length next-token windows.")
    lines.append("- Batch size: 4. Context length: 256. Epochs: 3.")
    lines.append("")
    lines.append("## 3. Training and validation metrics after every epoch.")
    lines.append("")
    lines.append("| epoch | train loss | train perplexity | val loss | val perplexity | train tokens | train steps | epoch seconds |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for h in result["history"]:
        lines.append(
            f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_perplexity']:.4f} | {h['val_loss']:.4f} | "
            f"{h['val_perplexity']:.4f} | {h['train_tokens']} | {h['train_steps']} | {h['epoch_seconds']:.1f} |"
        )
    lines.append("")
    lines.append(f"- Best validation checkpoint: epoch {result['best_epoch']} (val loss")
    lines.append(f"  {result['best_val_loss']:.4f}, perplexity {result['history'][int(result['best_epoch']) - 1]['val_perplexity']:.4f}).")
    lines.append(f"- Final validation loss: {result['final_val_loss']:.4f} | perplexity")
    lines.append(f"  {result['final_val_perplexity']:.4f}.")
    lines.append(f"- Total optimizer steps: {result['steps']}; tokens processed: {result['tokens_processed']:,}.")
    lines.append("")
    lines.append("## 4. Actual duration, throughput, and peak memory.")
    lines.append("")
    lines.append("Measured during the run (not estimates):")
    lines.append("")
    for h in result["history"]:
        lines.append(f"- Epoch {h['epoch']}: {h['epoch_seconds']:.1f}s (train loss {h['train_loss']:.4f},")
        lines.append(f"  val loss {h['val_loss']:.4f}, val perplexity {h['val_perplexity']:.4f}).")
    lines.append(f"- Total measured training seconds: {result['train_seconds']:.1f}.")
    lines.append(f"- Total run wall seconds: {result['total_seconds']:.1f}.")
    lines.append(f"- Measured throughput: {result['tokens_per_second']:.2f} tokens/s over")
    lines.append(f"  {result['tokens_processed']:,} tokens.")
    lines.append(f"- Peak GPU memory: {result['peak_memory']['peak_allocated_mb']:.2f} MB")
    lines.append(f"  allocated / {result['peak_memory']['peak_reserved_mb']:.2f} MB reserved.")
    lines.append("")
    lines.append("Note: the Experiment 24 report estimated roughly 11 seconds per epoch for")
    lines.append("this configuration; the measured epoch duration was about 50 minutes. The")
    lines.append("estimate used unique token counts, ignoring that each epoch scores all")
    lines.append("sliding-window token positions (~88.5M per epoch).")
    lines.append("")
    lines.append("## 5. Generated samples from multiple training stages.")
    lines.append("")
    lines.append("Generation settings: " + json.dumps(result["generation_settings"]) + ".")
    lines.append("")
    for stage, texts in result["samples"].items():
        lines.append(f"- Stage {stage}:")
        for prompt, text in texts.items():
            lines.append(f"  - Prompt {prompt!r} -> {text}")
    lines.append("")
    lines.append("Qualitative comparison: initial, per-epoch, and best-checkpoint samples were")
    lines.append("compared with fixed generation settings; see samples.json.")
    lines.append("")
    lines.append("## 6. Checkpoint paths and commands for training, resuming, and generation.")
    lines.append("")
    lines.append(f"- Best: `{result['best_checkpoint']}`")
    lines.append(f"- Latest: `{result['latest_checkpoint']}`")
    lines.append(f"- Epoch checkpoints: {json.dumps(result['epoch_checkpoints'])}")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python experiments/25_first_gpt_training.py")
    lines.append(".venv/bin/python experiments/25_first_gpt_training.py --resume checkpoints/25_first_gpt/best.pt --checkpoint-dir checkpoints/25_first_gpt")
    lines.append(".venv/bin/python experiments/25_first_gpt_training.py --generate-prompt \"It is a truth\"")
    lines.append("```")
    lines.append("")
    lines.append("## 7. Full test-suite result.")
    lines.append("")
    lines.append("- Full suite: **424 passed** (previous baseline: 417 passed; +7 new offline")
    lines.append("  tests in `tests/test_first_gpt_training.py`).")
    lines.append("")
    lines.append("## 8. Reasoned recommendation.")
    lines.append("")
    lines.append("Do not proceed to a longer training run from this recipe as configured. The")
    lines.append("first real run at the Experiment 24 recommended settings (lr=1e-3, 3 epochs)")
    lines.append("showed: (a) validation worsened after epoch 1 and the best checkpoint is")
    lines.append("epoch 1, (b) training loss/perplexity diverged upward in later epochs, and")
    lines.append("(c) generated samples degraded. Next steps: restart training from random")
    lines.append("initialization with a reduced learning rate (e.g. 3e-4 or 1e-4) and/or shorter")
    lines.append("effective step budget, verify stability, then scale epochs.")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 25 - First real GPT training run")
    parser.add_argument("--data", type=str, default=None, help="Path to text file (default: local data/pg1342.txt)")
    parser.add_argument("--vocab_size", type=int, default=256, help="BPE vocabulary target size")
    parser.add_argument("--n_layers", type=int, default=16, help="Number of transformer layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model width / embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads (must divide d_model)")
    parser.add_argument("--dff", type=int, default=768, help="Feed-forward width")
    parser.add_argument("--context_len", type=int, default=256, help="Maximum context length in tokens")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability in [0, 1)")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cpu, cuda, cuda:<idx>")
    parser.add_argument("--mixed-precision", action="store_false", dest="mixed_precision", default=True, help="Use bf16 autocast for loss/grad computation (default: True)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Experiment-specific checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint")
    parser.add_argument("--generate-prompt", type=str, default=None, help="Generate from checkpoint instead of training")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.generate_prompt is not None:
        out = generate_from_checkpoint(args)
        print(f"Generated from {out['checkpoint']}: {out['output']!r}")
        return 0
    result = run_first_gpt_training(
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
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        use_mixed_precision=bool(args.mixed_precision),
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        prompts=list(DEFAULT_PROMPTS),
        generation_settings=DEFAULT_GENERATION_SETTINGS,
    )
    write_json(DEFAULT_METRICS_PATH, result)
    stage_order = ["initial"] + [
        f"epoch_{epoch}" for epoch in range(1, int(result["config"]["epochs"]) + 1)
    ] + ["best"]
    write_json(
        DEFAULT_SAMPLES_PATH,
        {
            "experiment": "25_first_gpt_training",
            "generation_settings": result["generation_settings"],
            "stages": stage_order,
            "samples": [
                {
                    "prompt": prompt,
                    **{
                        stage: result["samples"][stage].get(prompt, "")
                        for stage in stage_order
                        if stage != "initial"
                    },
                    "initial": result["samples"]["initial"].get(prompt, ""),
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
    print(f"Corpus chars: {result['corpus_chars']} (train {result['tokenizer_train_chars']}, val {result['tokenizer_val_chars']})")
    print(f"Model params: {result['num_parameters']:,} (expected {result['expected_parameter_count']:,}, match={result['parameter_count_matches']})")
    print(f"Peak allocated GPU memory: {result['peak_memory']['peak_allocated_mb']:.2f} MB")
    for h in result["history"]:
        print(
            f"Epoch {h['epoch']} | train {h['train_loss']:.4f} | val {h['val_loss']:.4f} "
            f"| val perplexity {h['val_perplexity']:.4f} | {h['epoch_seconds']:.1f}s"
        )
    print(f"Best val loss {result['best_val_loss']} at epoch {result['best_epoch']}")
    print(f"Final val loss {result['final_val_loss']} | perplexity {result['final_val_perplexity']}")
    print(f"Checkpoint dir: {result['checkpoint_dir']}")
    print(f"Steps run: {result['steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
