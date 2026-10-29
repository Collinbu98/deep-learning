"""Experiment 24 - GPT training performance and scale feasibility.

Benchmarks short GPU training runs for representative GPT-style model sizes,
using the reusable pipeline from Experiment 23 (model, dataset, loss, optimizer,
checkpointing) without re-implementing it.

Each benchmark runs a few optimizer steps (warm-up + timed) and records measured
throughput, loss, gradient/optimizer behavior, and GPU memory. No long training
runs are performed here; recommendations are produced from measurements.

Run with, e.g.:
    .venv/bin/python experiments/24_training_benchmarks.py --smoke --steps 3
    .venv/bin/python experiments/24_training_benchmarks.py --data data/pg1342.txt \
        --vocab_size 256 --n_layers 16 --d_model 512 --n_heads 8 --dff 768 \
        --context_len 256 --batch_size 4 --steps 5
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "experiments" / "24_training_benchmarks" / "benchmark_report.json"
DEFAULT_DATA = REPO_ROOT / "data" / "pg1342.txt"


def _load_exp23():
    path = REPO_ROOT / "experiments" / "23_scalable_gpt_training.py"
    spec = importlib.util.spec_from_file_location("scalable_gpt_training_for_exp24", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp23 = _load_exp23()


def gpu_memory_stats(device=None) -> dict:
    dev = torch.device(device) if device is not None else torch.device("cuda:0")
    allocated = 0
    reserved = 0
    peak_allocated = 0
    peak_reserved = 0
    free_mb = None
    total_mb = None
    if torch.cuda.is_available():
        allocated = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
        peak_allocated = int(torch.cuda.max_memory_allocated(dev))
        peak_reserved = int(torch.cuda.max_memory_reserved(dev))
        try:
            free, total = torch.cuda.mem_get_info(dev)
            free_mb = float(free) / 1e6
            total_mb = float(total) / 1e6
        except Exception:
            free_mb = None
            total_mb = None
    return {
        "device": str(dev),
        "cuda_available": bool(torch.cuda.is_available()),
        "allocated_mb": allocated / 1e6,
        "reserved_mb": reserved / 1e6,
        "peak_allocated_mb": peak_allocated / 1e6,
        "peak_reserved_mb": peak_reserved / 1e6,
        "free_mb": free_mb,
        "total_mb": total_mb,
        "supported": True,
    }


def get_corpus_text(smoke: bool, data: str | None) -> tuple[str, str]:
    if smoke:
        return exp23.SMOKE_TEXT, "smoke-builtin"
    if data is not None:
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        return p.read_text(encoding="utf-8", errors="replace"), str(p)
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA.read_text(encoding="utf-8", errors="replace"), str(DEFAULT_DATA)
    raise RuntimeError("No corpus provided and default local corpus not found; pass --data.")


def estimate_duration(total_tokens: int, tokens_per_second: float, epochs: float = 1.0, val_every_steps: int = 100, ckpt_every_steps: int = 500) -> dict:
    seconds_per_token = 1.0 / tokens_per_second if tokens_per_second and tokens_per_second > 0 else float("inf")
    train_seconds = float(total_tokens) * seconds_per_token * float(epochs)
    return {
        "tokens_processed_estimate": float(total_tokens * epochs),
        "seconds_per_token": seconds_per_token,
        "estimated_train_seconds": train_seconds,
        "estimated_train_minutes": train_seconds / 60.0,
        "assumptions": {
            "epochs": epochs,
            "tokens_per_second": tokens_per_second,
            "val_every_steps": val_every_steps,
            "ckpt_every_steps": ckpt_every_steps,
            "note": "Validation and checkpoint overhead are not included in the estimate.",
        },
    }


def run_benchmark(
    model_config,
    *,
    text,
    val_split=0.1,
    batch_size=2,
    context_len=None,
    steps=3,
    warmup_steps=2,
    lr=1e-3,
    clip_grad=1.0,
    device=None,
    seed=0,
    use_mixed_precision=False,
) -> dict:
    device = exp23.get_device(device)
    ctx_len = int(context_len if context_len is not None else model_config["context_len"])
    train_text, val_text = exp23.split_corpus(text, val_split)
    tokenizer, train_ids, val_ids = exp23.fit_tokenizer_and_encode(train_text, val_text, int(model_config["vocab_size"]))
    train_ds = exp23.GPTTokenDataset(train_ids, ctx_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = exp23.GPTLanguageModel(**model_config).to(device)
    model.train()
    optimizer = exp23.make_optimizer(model, lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    batch = None
    for bx, by in train_loader:
        batch = (bx, by)
        break
    if batch is None:
        return {"status": "failed", "error": "empty train loader", "device": str(device)}
    x, y = batch[0].to(device), batch[1].to(device)

    mem_before = gpu_memory_stats(device)
    losses = []
    step_seconds = []
    mixed_error = None
    loss = None
    use_mp = bool(use_mixed_precision)
    for i in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        if use_mp:
            try:
                with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
                    loss = exp23.next_token_loss(model, x, y)
                    loss.backward()
            except Exception as e:  # noqa: BLE001
                mixed_error = f"mixed precision (bf16 autocast) failed: {e}"
                use_mp = False
                loss = exp23.next_token_loss(model, x, y)
                loss.backward()
        else:
            loss = exp23.next_token_loss(model, x, y)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
        optimizer.step()
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        step_seconds.append(float(time.perf_counter() - t0))
        losses.append(float(loss.detach()))

    median_step = sorted(step_seconds)[len(step_seconds) // 2] if step_seconds else float("nan")
    tokens = int(x.numel())
    tokens_per_sec = tokens / median_step if median_step > 0 else float("inf")
    val_metrics = exp23.validate_epoch(model, DataLoader(exp23.GPTTokenDataset(val_ids, ctx_len), batch_size=batch_size, shuffle=False), criterion, device)
    mem_after = gpu_memory_stats(device)

    return {
        "status": "completed",
        "error": None,
        "device": str(device),
        "model_config": dict(model_config),
        "model_params": int(exp23.count_parameters(model)),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_train_tokens": len(train_ids),
        "tokenizer_val_tokens": len(val_ids),
        "context_len": int(ctx_len),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "warmup_steps": int(warmup_steps),
        "lr": float(lr),
        "clip_grad": float(clip_grad),
        "mixed_precision": bool(use_mp),
        "mixed_precision_error": mixed_error,
        "losses": losses,
        "step_seconds": step_seconds,
        "median_step_seconds": float(median_step),
        "tokens_per_second": float(tokens_per_sec),
        "tokens_per_step": int(tokens),
        "memory_before": mem_before,
        "memory_after": mem_after,
        "peak_allocated_mb": mem_after["peak_allocated_mb"],
        "peak_reserved_mb": mem_after["peak_reserved_mb"],
        "final_loss": float(loss.detach()) if loss is not None else None,
        "final_loss_finite": bool(loss is not None and torch.isfinite(loss).item()),
        "val_loss": val_metrics["loss"],
        "val_perplexity": val_metrics["perplexity"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 24 - GPT training performance and scale feasibility")
    parser.add_argument("--smoke", action="store_true", help="Use the tiny built-in smoke corpus and tiny model config")
    parser.add_argument("--data", type=str, default=None, help="Path to text file (default: data/pg1342.txt)")
    parser.add_argument("--vocab_size", type=int, default=None, help="BPE vocabulary target size")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--d_model", type=int, default=16, help="Model width / embedding dimension")
    parser.add_argument("--n_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--dff", type=int, default=32, help="Feed-forward width")
    parser.add_argument("--context_len", type=int, default=16, help="Maximum context length in tokens")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--batch_size", type=int, default=2, help="Benchmark batch size")
    parser.add_argument("--steps", type=int, default=3, help="Number of timed optimizer steps")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Warm-up steps before timed measurement")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cpu, cuda, cuda:<idx>")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT), help="Path to write the benchmark JSON report")
    parser.add_argument("--mixed-precision", action="store_true", help="Run the benchmark step under bf16 autocast")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    model_config = {
        "vocab_size": int(args.vocab_size if args.vocab_size is not None else (exp23.SMOKE_VOCAB_SIZE if args.smoke else 256)),
        "n_layers": int(args.n_layers),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "dff": int(args.dff),
        "context_len": int(args.context_len),
        "dropout": float(args.dropout),
    }
    validated = exp23.validate_gpt_config(model_config)
    text, data_path = get_corpus_text(args.smoke, args.data)
    result = run_benchmark(
        validated,
        text=text,
        val_split=float(args.val_split),
        batch_size=int(args.batch_size),
        context_len=int(args.context_len),
        steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
        lr=float(args.lr),
        clip_grad=float(args.clip_grad),
        device=args.device,
        seed=0,
        use_mixed_precision=bool(args.mixed_precision),
    )
    report = dict(result)
    report["smoke"] = bool(args.smoke)
    report["data_path"] = data_path
    report["corpus_chars"] = len(text)
    report["timestamp"] = time.time()
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Benchmark status: {report['status']} error={report['error']}")
    print(f"Device: {report['device']}")
    print(f"Model params: {report['model_params']:,} (config {report['model_config']})")
    print(f"Tokens: train={report['tokenizer_train_tokens']} val={report['tokenizer_val_tokens']} batch={report['batch_size']} ctx={report['context_len']}")
    print(f"Losses: {report['losses']} final_finite={report['final_loss_finite']}")
    print(f"Median step: {report['median_step_seconds']:.4f}s | tokens/s: {report['tokens_per_second']:.2f}")
    print(f"Memory before/after: {report['memory_before']['allocated_mb']:.4f}/{report['memory_after']['allocated_mb']:.4f} MB allocated; peak {report['peak_allocated_mb']:.4f}/{report['peak_reserved_mb']:.4f} MB")
    if report["mixed_precision_error"]:
        print(f"Mixed precision: {report['mixed_precision_error']}")
    print(f"Report written to {out}")
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
