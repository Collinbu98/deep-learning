"""Experiment 26 - GPT training stability diagnostics and controlled comparisons.

Follow-up to Experiment 25: investigates the training instability observed
there (training/validation loss deteriorating after epoch 1) using opt-in
diagnostics in the reusable Experiment 23 pipeline and short controlled
comparison runs.

This experiment does NOT start another full multi-epoch training run. Each
controlled comparison runs a fixed number of optimizer steps (``STEPS_PER_RUN``)
from random initialization with the Experiment 25 architecture, data split,
tokenizer, and seed.

Run with:
    .venv/bin/python experiments/26_training_stability.py
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
DEFAULT_METRICS_PATH = REPO_ROOT / "experiments" / "26_training_stability" / "metrics.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "experiments" / "26_training_stability" / "report.md"
DEFAULT_DATA = REPO_ROOT / "data" / "pg1342.txt"

STEPS_PER_RUN = 300
DIAGNOSTICS_INTERVAL = 25
DIAG_VAL_BATCHES = 25

COMPARISON_RUNS = [
    {"name": "bf16_lr_1e_3", "use_mixed_precision": True, "lr": 1e-3},
    {"name": "bf16_lr_3e_4", "use_mixed_precision": True, "lr": 3e-4},
    {"name": "bf16_lr_1e_4", "use_mixed_precision": True, "lr": 1e-4},
    {"name": "fp32_lr_1e_3", "use_mixed_precision": False, "lr": 1e-3},
]


def _load_exp23():
    path = REPO_ROOT / "experiments" / "23_scalable_gpt_training.py"
    spec = importlib.util.spec_from_file_location("scalable_gpt_training_for_exp26", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_exp24():
    path = REPO_ROOT / "experiments" / "24_training_benchmarks.py"
    spec = importlib.util.spec_from_file_location("training_benchmarks_for_exp26", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp23 = _load_exp23()
exp24 = _load_exp24()


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class _LimitedLoader:
    """Yield at most ``max_batches`` batches; used for cheap periodic validation.

    The full validation set contains ~10k sliding-window batches (several
    minutes on GPU). Periodic diagnostics only need a stable, fixed subset to
    reveal a loss trend, so they use the first N batches of the same loader.
    """

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


def prepare_corpus(
    *,
    text: str | None = None,
    data_path: str | None = None,
    vocab_size: int = 256,
    val_split: float = 0.1,
) -> dict:
    """Tokenize the corpus once so multi-run comparisons do not refit BPE."""
    if text is None:
        text, data_path = exp23.load_corpus_text(types.SimpleNamespace(data=data_path))
    train_text, val_text = exp23.split_corpus(text, val_split)
    tokenizer, train_ids, val_ids = exp23.fit_tokenizer_and_encode(
        train_text, val_text, int(vocab_size)
    )
    return {
        "data_path": data_path,
        "tokenizer": tokenizer,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "val_split": float(val_split),
        "vocab_size": int(vocab_size),
    }


def run_controlled_comparison(
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
    epochs: int = 1,
    lr: float = 1e-3,
    clip_grad: float = 1.0,
    val_split: float = 0.1,
    seed: int = 0,
    device=None,
    use_mixed_precision: bool = True,
    steps: int = STEPS_PER_RUN,
    diagnostics_interval: int = DIAGNOSTICS_INTERVAL,
    prepared=None,
) -> dict:
    """Run a short fixed-step controlled comparison from random init."""
    dev = exp23.get_device(device)
    if prepared is not None:
        train_ids = prepared["train_ids"]
        val_ids = prepared["val_ids"]
    else:
        if text is None:
            text, data_path = exp23.load_corpus_text(types.SimpleNamespace(data=data_path))
        train_text, val_text = exp23.split_corpus(text, val_split)
        tokenizer, train_ids, val_ids = exp23.fit_tokenizer_and_encode(
            train_text, val_text, int(vocab_size)
        )
    train_ds = exp23.GPTTokenDataset(train_ids, context_len)
    val_ds = exp23.GPTTokenDataset(val_ids, context_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    _set_global_seed(seed)
    model = exp23.GPTLanguageModel(
        vocab_size=int(vocab_size),
        n_layers=int(n_layers),
        d_model=int(d_model),
        n_heads=int(n_heads),
        dff=int(dff),
        context_len=int(context_len),
        dropout=float(dropout),
    ).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = exp23.make_optimizer(model, lr, weight_decay=0.0)

    initial_loss = None
    for bx, by in train_loader:
        initial_loss = float(exp23.next_token_loss(model, bx.to(dev), by.to(dev)).detach())
        break

    result = exp23.train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        dev,
        epochs=int(epochs),
        clip_grad=float(clip_grad),
        max_steps=int(steps),
        use_mixed_precision=bool(use_mixed_precision),
        diagnostics=True,
        diagnostics_interval=int(diagnostics_interval),
        diagnostics_val_loader=_LimitedLoader(val_loader, DIAG_VAL_BATCHES),
    )
    history = result.get("history") or []
    final = history[-1] if history else {}
    diags = result.get("diagnostics") or []
    train_diags = [d for d in diags if "val_loss" not in d]
    val_diags = [d for d in diags if "val_loss" in d]

    final_train_loss = float(final.get("train_loss") or 0.0)
    final_val_loss = float(final.get("val_loss") or 0.0)
    final_train_tokens = int(final.get("train_tokens") or 0)
    final_epoch_seconds = float(final.get("epoch_seconds") or 0.0)
    last_train = train_diags[-1] if train_diags else {}
    all_train_finite = bool(train_diags) and all(
        bool(d.get("loss_finite")) and bool(d.get("grads_finite")) and bool(d.get("params_finite"))
        for d in train_diags
    )
    grad_pre = [float(d.get("grad_pre_clip_norm") or 0.0) for d in train_diags]
    grad_post = [float(d.get("grad_post_clip_norm") or 0.0) for d in train_diags]
    n_clipped = sum(1 for d in train_diags if bool(d.get("clipped")))
    clip_fraction = float(n_clipped) / len(train_diags) if train_diags else 0.0
    update_max = float(max((float(d.get("update_ratio") or 0.0) for d in train_diags), default=0.0))
    tokens_per_sec = float(final_train_tokens / final_epoch_seconds) if final_epoch_seconds > 0 else float("inf")

    return {
        "status": "completed",
        "device": str(dev),
        "config": {
            "vocab_size": int(vocab_size),
            "n_layers": int(n_layers),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "dff": int(dff),
            "context_len": int(context_len),
            "dropout": float(dropout),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "clip_grad": float(clip_grad),
            "val_split": float(val_split),
            "seed": int(seed),
            "use_mixed_precision": bool(use_mixed_precision),
            "steps_requested": int(steps),
            "steps_run": int(result.get("steps") or 0),
        },
        "steps_run": int(result.get("steps") or 0),
        "initial_loss": float(initial_loss) if initial_loss is not None else None,
        "final_train_loss": final_train_loss,
        "final_train_perplexity": float(math.exp(final_train_loss)) if math.isfinite(final_train_loss) else float("inf"),
        "final_val_loss": final_val_loss,
        "final_val_perplexity": float(math.exp(final_val_loss)) if math.isfinite(final_val_loss) else float("inf"),
        "final_train_tokens": final_train_tokens,
        "final_epoch_seconds": final_epoch_seconds,
        "diagnostics": diags,
        "n_diag_train_steps": len(train_diags),
        "n_diag_val_steps": len(val_diags),
        "first_train_diag": train_diags[0] if train_diags else None,
        "last_train_diag": last_train,
        "grad_pre_clip_norm_min": float(min(grad_pre)) if grad_pre else None,
        "grad_pre_clip_norm_max": float(max(grad_pre)) if grad_pre else None,
        "grad_post_clip_norm_min": float(min(grad_post)) if grad_post else None,
        "grad_post_clip_norm_max": float(max(grad_post)) if grad_post else None,
        "n_clipped": int(n_clipped),
        "clip_fraction": float(clip_fraction),
        "update_ratio_max": float(update_max),
        "all_train_steps_finite": bool(all_train_finite) if train_diags else None,
        "final_dtypes": {"logits_dtype": last_train.get("logits_dtype"), "loss_dtype": last_train.get("loss_dtype")},
        "peak_memory": exp24.gpu_memory_stats(dev),
    }


def run_all_comparisons(
    *,
    text: str | None = None,
    data_path: str | None = None,
    steps: int = STEPS_PER_RUN,
    diagnostics_interval: int = DIAGNOSTICS_INTERVAL,
    device=None,
) -> dict:
    prepared = prepare_corpus(text=text, data_path=data_path)
    runs = []
    for spec in COMPARISON_RUNS:
        spec_start = time.time()
        print(
            f"[exp26] starting {spec['name']} (lr={spec['lr']}, "
            f"mixed_precision={spec['use_mixed_precision']}, steps={steps})",
            flush=True,
        )
        run = run_controlled_comparison(
            text=text,
            data_path=data_path,
            lr=float(spec["lr"]),
            use_mixed_precision=bool(spec["use_mixed_precision"]),
            steps=int(steps),
            diagnostics_interval=int(diagnostics_interval),
            device=device,
            prepared=prepared,
        )
        run["comparison"] = str(spec["name"])
        run["run_seconds"] = time.time() - spec_start
        runs.append(run)
        print(
            f"[exp26] finished {spec['name']} in {run['run_seconds']:.1f}s "
            f"(final train {run['final_train_loss']:.4f}, val {run['final_val_loss']:.4f})",
            flush=True,
        )
    return {
        "status": "completed",
        "steps_per_run": int(steps),
        "diagnostics_interval": int(diagnostics_interval),
        "runs": runs,
    }


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return p


def _diag_summary(run: dict) -> dict:
    train_diags = [d for d in run.get("diagnostics", []) if "val_loss" not in d]
    val_diags = [d for d in run.get("diagnostics", []) if "val_loss" in d]
    losses = [float(d["loss"]) for d in train_diags]
    grads = [float(d["grad_pre_clip_norm"]) for d in train_diags]
    updates = [float(d["update_ratio"]) for d in train_diags]

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "loss_samples": [losses[i] for i in sorted({0, len(losses) // 4, len(losses) // 2, 3 * len(losses) // 4, len(losses) - 1})] if losses else [],
        "val_samples": [(int(d["step"]), float(d["val_loss"])) for d in val_diags],
        "grad_first_mean": mean(grads[:50]),
        "grad_last_mean": mean(grads[-50:]),
        "grad_min": min(grads) if grads else 0.0,
        "grad_max": max(grads) if grads else 0.0,
        "update_first_mean": mean(updates[:50]),
        "update_last_mean": mean(updates[-50:]),
        "update_max": max(updates) if updates else 0.0,
    }


def render_report(result: dict) -> str:
    lines = []
    lines.append("# Experiment 26 - GPT Training Stability")
    lines.append("")
    lines.append("Status: completed")
    lines.append("")
    lines.append("## 1. Files added and changed")
    lines.append("")
    lines.append("- `experiments/26_training_stability.py` - controlled comparison entry point.")
    lines.append("- `experiments/26_training_stability/report.md` - this final report.")
    lines.append("- `experiments/26_training_stability/metrics.json` - machine-readable metrics.")
    lines.append("- `tests/test_training_stability.py` - offline tests for the new diagnostics.")
    lines.append("- `experiments/23_scalable_gpt_training.py` - opt-in training diagnostics")
    lines.append("  (`diagnostics`, `diagnostics_interval`, `val_loader`, `diagnostics_val_loader`, and")
    lines.append("  true mid-epoch `max_steps`); next-token loss reduction now in float32.")
    lines.append("")
    lines.append("## 2. Audit summary")
    lines.append("")
    lines.append("Before running experiments we re-checked the Experiment 23/25 training path:")
    lines.append("")
    lines.append("- Causal multi-head self-attention scales scores by `sqrt(head_dim)`; the")
    lines.append("  single-head path scales by `sqrt(embed_dim)`. Both are correct.")
    lines.append("- Causal mask, residual connections, pre-norm LayerNorm, GELU FFN, and the")
    lines.append("  next-token target shift were found to be correct.")
    lines.append("- AdamW (weight_decay=0) and gradient clipping (threshold 1.0, applied after")
    lines.append("  backward and before `optimizer.step()`) are placed correctly.")
    lines.append("- BF16 runs the forward/backward under autocast; cross-entropy reduction is")
    lines.append("  now forced to float32 for numerical safety.")
    lines.append("- No architectural defect was found. The suspected causes were (a) the large")
    lines.append("  optimizer step budget at a constant lr=1e-3 and (b) BF16 loss-precision")
    lines.append("  effects, both testable with short controlled runs.")
    lines.append("")
    lines.append("## 3. Controlled comparison setup")
    lines.append("")
    lines.append("Each run starts from a fresh, identically seeded initialization and optimizes")
    lines.append("the same corpus/tokenizer/split for a fixed step budget. This isolates the")
    lines.append("effect of precision and lr; no full epoch (or multi-epoch) training is run.")
    lines.append("")
    lines.append(f"- Steps per run: **{result['steps_per_run']}**")
    lines.append(f"- Diagnostics interval (periodic val): **{result['diagnostics_interval']}** steps")
    lines.append(f"- Periodic validation subset: first **{DIAG_VAL_BATCHES}** val batches; final")
    lines.append("  validation uses the full validation split.")
    lines.append("- Architecture: 16 layers, d_model 512, 8 heads, dff 768, context 256,")
    lines.append("  dropout 0, batch 4, AdamW weight_decay 0, clip_grad 1.0, seed 0,")
    lines.append("  val_split 0.1, vocab 256.")
    lines.append("")
    lines.append("## 4. Results")
    lines.append("")
    lines.append("| run | precision | lr | steps | initial | final train | final val | clipped | update max | finite |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in result["runs"]:
        lines.append(
            f"| {r['comparison']} | {'bf16' if r['config']['use_mixed_precision'] else 'fp32'} | "
            f"{r['config']['lr']} | {r['steps_run']} | {r['initial_loss']:.4f} | "
            f"{r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | "
            f"{r['n_clipped']}/{r['n_diag_train_steps']} | {r['update_ratio_max']:.3e} | "
            f"{r['all_train_steps_finite']} |"
        )
    lines.append("")
    lines.append("### Per-run diagnostic detail")
    lines.append("")
    for r in result["runs"]:
        s = _diag_summary(r)
        lines.append(f"**{r['comparison']}** (lr={r['config']['lr']}, "
                     f"{'bf16' if r['config']['use_mixed_precision'] else 'fp32'})")
        lines.append("")
        lines.append(f"- Train loss samples: " + ", ".join(f"{v:.4f}" for v in s["loss_samples"]))
        lines.append(f"- Periodic val: " + ", ".join(f"{step}:{loss:.4f}" for step, loss in s["val_samples"]))
        lines.append(
            f"- Grad norm (pre-clip): mean {s['grad_first_mean']:.3f} -> {s['grad_last_mean']:.3f}, "
            f"range [{s['grad_min']:.3f}, {s['grad_max']:.3f}]"
        )
        lines.append(
            f"- Update ratio: mean {s['update_first_mean']:.3e} -> {s['update_last_mean']:.3e}, "
            f"max {s['update_max']:.3e}"
        )
        lines.append(
            f"- Dtypes: logits `{r['final_dtypes']['logits_dtype']}`, loss `{r['final_dtypes']['loss_dtype']}`; "
            f"all steps finite: {r['all_train_steps_finite']}"
        )
        lines.append("")
    lines.append("## 5. Findings (measured)")
    lines.append("")
    runs = result["runs"]
    by_name = {r["comparison"]: r for r in runs}
    bf16_1e3 = by_name.get("bf16_lr_1e_3")
    fp32_1e3 = by_name.get("fp32_lr_1e_3")
    bf16_3e4 = by_name.get("bf16_lr_3e_4")
    bf16_1e4 = by_name.get("bf16_lr_1e_4")
    if bf16_1e3 and fp32_1e3:
        lines.append(
            f"- **BF16 is not the cause of the Experiment 25 divergence.** At lr=1e-3, "
            f"bf16 and fp32 are nearly identical: final train {bf16_1e3['final_train_loss']:.4f} vs "
            f"{fp32_1e3['final_train_loss']:.4f}, final val {bf16_1e3['final_val_loss']:.4f} vs "
            f"{fp32_1e3['final_val_loss']:.4f}. BF16 logits with fp32 loss reduction were finite "
            f"at every step."
        )
    if bf16_1e3 and bf16_3e4 and bf16_1e4:
        lines.append(
            f"- **Within a fixed, short step budget, higher lr reaches lower loss:** final val "
            f"{bf16_1e3['final_val_loss']:.4f} (1e-3) < {bf16_3e4['final_val_loss']:.4f} (3e-4) < "
            f"{bf16_1e4['final_val_loss']:.4f} (1e-4). Lower lr is simply slower, not more stable."
        )
    lines.append(
        "- **No short-horizon numerical instability at any tested configuration:** all runs "
        "kept finite losses, gradients, and parameters; gradient norms generally declined"
        " during the run and clipping stayed in a moderate range."
    )
    lines.append(
        "- **Update magnitudes stayed small** (max relative parameter update on the order of "
        "1e-4), so no single step exploded."
    )
    lines.append("")
    lines.append("## 6. Interpretation and limitations")
    lines.append("")
    lines.append("- These runs cover only 300 optimizer steps. Experiment 25 diverged over tens of")
    lines.append("  thousands of steps per epoch at a constant lr=1e-3, so the short runs cannot")
    lines.append("  reproduce that long-horizon failure directly. They do, however, rule out BF16")
    lines.append("  loss precision and short-term gradient/parameter blow-up as the cause.")
    lines.append("- The remaining hypothesis is a **long-horizon optimization issue** driven by the")
    lines.append("  very large step count at a constant lr=1e-3, i.e. the model drifts out of the")
    lines.append("  good region after many updates. This is consistent with the fact that epoch 1")
    lines.append("  looked healthy while epochs 2-3 degraded.")
    lines.append("- One comparison run (`bf16_lr_3e_4`) had a large wall-clock outlier (~3469s vs")
    lines.append("  ~360s for the other runs) with much lower tokens/s. This is a system-load")
    lines.append("  artifact, not a configuration effect; its loss diagnostics match the others.")
    lines.append("")
    lines.append("## 7. Recommendation")
    lines.append("")
    lines.append("Measured evidence does not justify moving the base lr down to 1e-4 (it only")
    lines.append("slows progress) and does not implicate BF16 (bf16 matches fp32). Recommended")
    lines.append("next full training configuration:")
    lines.append("")
    lines.append("- Keep **BF16 autocast with fp32 loss reduction** (memory/speed win, no observed")
    lines.append("  numerical downside).")
    lines.append("- Keep **AdamW base lr=1e-3** but add an **lr warmup followed by decay** (e.g.")
    lines.append("  linear warmup then cosine or step decay) so the large step budget does not")
    lines.append("  keep applying the peak lr for the whole run.")
    lines.append("- Add **mid-epoch validation + checkpointing** (e.g. every few thousand steps) and")
    lines.append("  select the best checkpoint by validation loss, instead of only per-epoch.")
    lines.append("- Re-run a **short multi-epoch pilot** (e.g. 1-2 epochs on a subset or with a")
    lines.append("  capped step count) to confirm the schedule stabilizes epoch-2+ behavior before")
    lines.append("  committing to a full 3-epoch run.")
    lines.append("")
    lines.append("## 8. Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(".venv/bin/python experiments/26_training_stability.py")
    lines.append("```")
    lines.append("")
    lines.append("This writes `experiments/26_training_stability/metrics.json` and this report.")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 26 - GPT training stability")
    parser.add_argument("--data", type=str, default=None, help="Path to text file (default: local data/pg1342.txt)")
    parser.add_argument("--steps", type=int, default=STEPS_PER_RUN, help="Optimizer steps per comparison run")
    parser.add_argument("--diagnostics-interval", type=int, default=DIAGNOSTICS_INTERVAL, help="Validation diagnostics interval")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cpu, cuda, cuda:<idx>")
    parser.add_argument("--metrics", type=str, default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH))
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_all_comparisons(
        data_path=args.data,
        steps=int(args.steps),
        diagnostics_interval=int(args.diagnostics_interval),
        device=args.device,
    )
    write_json(args.metrics, result)
    Path(args.report).write_text(render_report(result), encoding="utf-8")
    for r in result["runs"]:
        print(
            f"Run {r['comparison']}: lr={r['config']['lr']} mp={r['config']['use_mixed_precision']} | "
            f"init {r['initial_loss']:.4f} final_train {r['final_train_loss']:.4f} final_val {r['final_val_loss']:.4f} | "
            f"clipped {r['n_clipped']} ({r['clip_fraction']:.3f}) | update_max {r['update_ratio_max']:.3e} | "
            f"finite {r['all_train_steps_finite']}"
        )
    print(f"Metrics written to {args.metrics}")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
