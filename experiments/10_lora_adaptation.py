"""
Experiment 10 - LoRA (Low-Rank Adaptation) for parameter-efficient domain adaptation.

Experiment 09 continued training the Experiment 08 BPE Transformer on Faraday's
"The Chemical History of a Candle" by updating **all 161,920** parameters.
That adapts the model to the new domain but also moves every weight of the
pretrained model, which is what degraded the original-domain (Pride and
Prejudice) score.

This experiment repeats the *same* adaptation task with LoRA: the pretrained
weights stay frozen and only a small low-rank adapter is learned

    output = base_linear(x) + (alpha / rank) * lora_B(lora_A(x))

Design decisions
----------------
* **Same base model, same data, same protocol.** The Experiment 08 checkpoint
  (``checkpoints/08_bpe_best.pt``) is loaded with Experiment 09's
  ``load_checkpoint``; the tokenizer is reused unchanged (no new merges are
  fitted, the vocabulary never changes); the domain corpus is split
  sequentially into domain-train / domain-validation with the same rule as
  Experiment 09; checkpoint selection uses the domain-validation per-token
  loss only. The original-domain (Pride and Prejudice) validation segment is
  evaluated every epoch as a *forgetting monitor* and never influences
  selection or gradients.
* **Target layers: the attention projections.** The pretrained checkpoint uses
  ``n_heads=1``, so each of the 2 Transformer blocks contains a
  ``SelfAttention`` module with the projections ``query``, ``key`` and
  ``value`` (a single-head attention has no output projection; with
  ``n_heads>1`` the multi-head module adds ``out_proj``, which can be targeted
  with ``--target_layers attention.out_proj``). Adapting the attention
  projections is the standard LoRA starting point and keeps the adapter
  small: 6 wrapped layers x 2 x (r x 64) = 3,072 trainable parameters at
  rank 4, i.e. ~1.9% of the pretrained model.
* **Everything else stays frozen.** After wrapping, every parameter of the
  model (embeddings, positional embeddings, layer norms, the LM head, the
  frozen base weights and their biases) has ``requires_grad = False``; only
  ``lora_A``/``lora_B`` are trainable. This is asserted at runtime and
  reported in the parameter budget.
* **Learning rate.** A learning-rate sweep on the *domain validation* split
  (1e-4, 3e-4, 1e-3 - the original-domain segment was never used to make any
  decision) showed that 1e-4 is the best rate for LoRA on this model, and
  that 1e-3 is actively harmful: it degrades the domain-validation loss for
  LoRA (rank 4) *and* for full fine-tuning. The LoRA default is therefore
  1e-4, the same rate Experiment 09 used for full fine-tuning; the LoRA
  branch additionally multiplies its update by ``alpha / rank = 2``, so the
  effective step is already larger than the full fine-tuning step at the same
  learning rate. The full fine-tuning baseline in this experiment is run at
  1e-4 (Experiment 09 parity) and at 1e-3, so the comparison is not decided
  by a single learning rate.
* **A real comparison, not just a claim.** ``--method full`` runs the same
  script, the same split, the same seed and the same evaluation as the LoRA
  run, with every parameter trainable - the only difference is *which*
  parameters are updated and the learning rate. The Experiment 09 checkpoint
  is additionally reported as clearly-labelled *historical* context (a
  separate earlier run, not a controlled comparison).

Checkpoint format
-----------------
``checkpoints/10_lora_best.pt`` stores the frozen pretrained weights
(``base_state_dict``), the adapter tensors only (``lora_state_dict``), the
architecture (``model_config``), the tokenizer, the LoRA configuration
(rank / alpha / target layers) and the base checkpoint path. The Experiment
08 and Experiment 09 checkpoints are never overwritten. ``--eval_only`` loads
a saved adapted checkpoint and only evaluates / generates from it.

Observed results (seed 0, 8-epoch schedule, checkpoint = best domain-val
per-token loss, lr 1e-4, batch 32, context 64, device cuda:0)
------------------------------------------------------------------
============================  ========  ==================  ==============
method                        trainable  domain (Candle)     original (P&P)
============================  ========  ==================  ==============
base (exp08 checkpoint)              0  2.407 nats/char     1.511 nats/char
LoRA attn q/k/v (r=4, a=8)       3,072  2.385 (-0.022)      1.567 (+0.056)
LoRA attn+ffn (r=4, a=8)        8,192  2.252 (-0.154)      1.664 (+0.154)
full fine-tuning (exp09 parity) 161,920  2.030 (-0.377)      1.675 (+0.165)
============================  ========  ==================  ==============

Full fine-tuning adapts the new domain more strongly but also forgets the
original domain more; LoRA moves the model less in both directions. The
original-domain loss increased in *every* configuration, so these numbers do
**not** support a claim that LoRA prevents catastrophic forgetting at this
scale - it only reduces the size of both the gain and the loss.

Run with:
    # LoRA (default: attention q/k/v, rank 4, alpha 8)
    .venv/bin/python experiments/10_lora_adaptation.py --epochs 8

    # LoRA with more capacity (attention + feed-forward projections)
    .venv/bin/python experiments/10_lora_adaptation.py --epochs 8 \
        --target_layers attention.query,attention.key,attention.value,ffn.0,ffn.2 \
        --adapt_checkpoint checkpoints/10_lora_attn_ffn_best.pt

    # Full fine-tuning baseline with the identical protocol
    .venv/bin/python experiments/10_lora_adaptation.py --method full --epochs 8 \
        --adapt_checkpoint checkpoints/10_fullft_best.pt

    # Evaluate / generate from a saved adapted checkpoint
    .venv/bin/python experiments/10_lora_adaptation.py --eval_only \
        --adapt_checkpoint checkpoints/10_lora_best.pt
"""

import argparse
import importlib.util
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from minigrad.bpe_tokenizer import BPETokenizer
from minigrad.lm_eval import nats_per_character
from minigrad.lora import (
    LoRALinear,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    mark_only_lora_trainable,
)


def _load_module(name, relpath):
    path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp09 = _load_module("domain_adaptation_experiment", "experiments/09_domain_adaptation.py")
exp08 = exp09.exp08


# Attention projections are the default LoRA targets. With n_heads == 1 the
# pretrained model has no attention output projection ("attention.out_proj"
# only exists for n_heads > 1).
DEFAULT_TARGET_LAYERS = ["attention.query", "attention.key", "attention.value"]

# Learning-rate defaults per method. Both were chosen on the DOMAIN
# VALIDATION split only (a sweep over 1e-4 / 3e-4 / 1e-3); the
# original-domain segment never influenced any decision.
DEFAULT_LR = {"lora": 1e-4, "full": 1e-4}

REQUIRED_CKPT_KEYS = ("model_config", "tokenizer", "base_state_dict", "method")
REQUIRED_MODEL_CONFIG_KEYS = (
    "vocab_size", "d_model", "n_layers", "context_len", "n_heads", "dropout",
)


def get_device():
    # cuda:0 only when available (ROCm reports the AMD GPU as cuda:0);
    # never a higher-index device.
    return exp09.get_device()


def count_parameters(model):
    """Return (total, trainable, trainable_percent) for ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return total, trainable, pct


def build_lora_model(base_model, target_layers, rank, alpha, dropout=0.0):
    """Wrap the targeted linears with LoRA and freeze everything else.

    Returns the qualified names of the wrapped modules.
    """
    replaced = apply_lora(base_model, target_layers, rank=rank, alpha=alpha, dropout=dropout)
    mark_only_lora_trainable(base_model)
    return replaced


def adapter_summary(model):
    """Per-layer report of the learned weight update (LoRA only)."""
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            rows.append(
                {
                    "layer": name,
                    "delta_norm": module.delta_weight_norm(),
                    "base_norm": module.base_weight_norm(),
                    "rel_delta": module.delta_weight_norm() / max(module.base_weight_norm(), 1e-12),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def save_adapted_checkpoint(
    path,
    *,
    base_state,
    lora_state,
    model_config,
    tokenizer,
    lora_config,
    method,
    history,
    best_epoch,
    best_val_loss,
    base_checkpoint,
    args,
):
    """Save the adapted model.

    ``base_state`` holds the weights of the *unwrapped* model (the frozen
    pretrained weights for LoRA, the fine-tuned weights for full
    fine-tuning), ``lora_state`` holds the adapter tensors only (empty for
    full fine-tuning). That separation is what makes the adapter small and
    independently loadable.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "lora_adaptation" if method == "lora" else "full_finetune",
            "method": method,
            "base_checkpoint": str(base_checkpoint),
            "model_config": model_config,
            "tokenizer": tokenizer.to_dict(),
            "lora_config": lora_config,
            "base_state_dict": base_state,
            "lora_state_dict": lora_state,
            "history": history,
            "best_epoch": best_epoch,
            "best_domain_val_loss": best_val_loss,
            "args": {k: v for k, v in vars(args).items()},
        },
        ckpt_path,
    )
    return str(ckpt_path)


def load_adapted_checkpoint(path, device):
    """Rebuild (model, tokenizer, checkpoint) from an Experiment 10 file.

    The model is rebuilt exactly as it was trained: the base weights are
    loaded first, then the LoRA wrappers are applied, then the adapter
    tensors are loaded.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint {ckpt_path} is not a dict; cannot load")
    missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        raise ValueError(
            f"checkpoint {ckpt_path} missing required key(s) {missing}; "
            f"expected the Experiment 10 format {list(REQUIRED_CKPT_KEYS)}"
        )
    missing_cfg = [k for k in REQUIRED_MODEL_CONFIG_KEYS if k not in ckpt["model_config"]]
    if missing_cfg:
        raise ValueError(
            f"checkpoint {ckpt_path} model_config missing key(s) {missing_cfg}; "
            f"refusing to guess architecture settings"
        )
    tokenizer = BPETokenizer.from_dict(ckpt["tokenizer"])
    if ckpt["model_config"]["vocab_size"] != tokenizer.vocab_size:
        raise ValueError(
            f"checkpoint {ckpt_path}: model vocab_size "
            f"{ckpt['model_config']['vocab_size']} != tokenizer vocab_size "
            f"{tokenizer.vocab_size}; checkpoint and tokenizer are incompatible"
        )
    model = exp08.BPETokenLanguageModel(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["base_state_dict"])
    if ckpt["method"] == "lora":
        lora_cfg = ckpt["lora_config"]
        apply_lora(
            model,
            lora_cfg["target_layers"],
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg.get("dropout", 0.0),
        )
        load_lora_state_dict(model, ckpt["lora_state_dict"])
        # The reloaded model has the same trainability as the trained one:
        # frozen base weights, trainable adapter only.
        mark_only_lora_trainable(model)
    model.eval()
    return model, tokenizer, ckpt


# --------------------------------------------------------------------------
# Training (same protocol as Experiment 09, different parameter subset)
# --------------------------------------------------------------------------

def train_adaptation(
    model,
    tokenizer,
    train_ids,
    domain_val_ids,
    *,
    epochs,
    batch_size,
    lr,
    device,
    context_len=None,
    orig_val_ids=None,
):
    """Train ``model`` on domain text with a FRESH AdamW optimizer.

    Identical protocol to Experiment 09 (``train_adaptation`` there):
    sequential sliding windows over the domain-TRAIN token stream only, fresh
    AdamW, and checkpoint selection on the minimum domain-validation
    per-token loss. The original-domain validation segment is evaluated each
    epoch as a forgetting monitor only.

    The difference to Experiment 09 is the optimizer's parameter group: only
    parameters with ``requires_grad=True`` are updated, so the same function
    trains a LoRA adapter or a fully fine-tuned model.
    """
    if context_len is None:
        context_len = model.context_len
    if not 1 <= context_len <= model.context_len:
        raise ValueError(
            f"context_len {context_len} outside [1, {model.context_len}] "
            f"(pretrained positional embeddings)"
        )
    train_ds = exp08.TokenDataset(train_ids, context_len)
    if len(train_ds) <= 0:
        raise ValueError(
            f"domain training set too small: {len(train_ids)} tokens with context_len {context_len}"
        )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters: nothing would be optimized")
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = exp08.train_epoch(model, train_dl, criterion, optimizer, device)
        dom = exp09.evaluate(model, domain_val_ids, tokenizer, device, batch_size=batch_size)
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "domain_val_per_token": dom["per_token_loss"],
            "domain_val_nats_per_char": dom["nats_per_char"],
            "seconds": time.time() - epoch_start,
        }
        if orig_val_ids is not None:
            orig = exp09.evaluate(model, orig_val_ids, tokenizer, device, batch_size=batch_size)
            entry["orig_val_per_token"] = orig["per_token_loss"]
            entry["orig_val_nats_per_char"] = orig["nats_per_char"]
        if dom["per_token_loss"] < best_val:
            best_val = dom["per_token_loss"]
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        history.append(entry)

    elapsed = time.time() - start
    if best_state is None:
        raise RuntimeError("no checkpoint selected (no epochs run?)")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_domain_val_loss": best_val,
        "elapsed_seconds": elapsed,
        "best_state": best_state,
    }


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def print_metrics(label, metrics):
    print(
        f"  {label:<18}: {metrics['per_token_loss']:.4f} per token | "
        f"{metrics['nats_per_char']:.3f} nats/char"
    )


def report_domain_table(rows):
    """Print a table of per-domain metrics (base vs adapted)."""
    print("\n{:<12} {:>10} {:>14} {:>14}".format("domain", "model", "per token", "nats/char"))
    for domain, model_name, metrics in rows:
        print(
            "{:<12} {:>10} {:>14.4f} {:>14.3f}".format(
                domain, model_name, metrics["per_token_loss"], metrics["nats_per_char"]
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "08_bpe_best.pt"),
        help="Experiment 08 checkpoint to adapt",
    )
    parser.add_argument(
        "--adapt_checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "10_lora_best.pt"),
        help="Where to save the best adapted checkpoint",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to the domain text file")
    parser.add_argument("--orig_data", type=str, default=None, help="Path to Pride and Prejudice text")
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="'lora' = train adapters only, 'full' = full fine-tuning baseline",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--lr", type=float, default=None,
        help=f"learning rate (default: {DEFAULT_LR['lora']} for lora, {DEFAULT_LR['full']} for full)",
    )
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha (scale = alpha/rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA-branch input dropout")
    parser.add_argument(
        "--target_layers",
        type=str,
        default=",".join(DEFAULT_TARGET_LAYERS),
        help="Comma-separated module-name patterns to wrap with LoRA",
    )
    parser.add_argument("--context_len", type=int, default=None,
                        help="Training window length (default: pretrained context_len)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1, help="Sequential domain train/val split")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training: only evaluate/generate from --adapt_checkpoint",
    )
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--compare_checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "09_adapted_best.pt"),
        help="Historical Experiment 09 checkpoint to report as context (optional)",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")
    if args.lr is None:
        args.lr = DEFAULT_LR[args.method]
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}")
    if args.lora_rank < 1:
        raise ValueError(f"lora_rank must be >= 1, got {args.lora_rank}")
    if args.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {args.lora_alpha}")
    target_layers = [t.strip() for t in args.target_layers.split(",") if t.strip()]
    if not target_layers:
        raise ValueError("target_layers must not be empty")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # 1. Load the pretrained model + the fixed tokenizer (Experiment 09 helper).
    model, tokenizer, base_ckpt = exp09.load_checkpoint(args.checkpoint, device)
    print(f"Loaded base checkpoint: {args.checkpoint}")
    print(f"  stage: {base_ckpt.get('stage', 'pretraining (exp08)')}")
    print(f"  vocab_size {tokenizer.vocab_size} | d_model {model.d_model} "
          f"| n_layers {model.n_layers} | n_heads {model.n_heads} "
          f"| context_len {model.context_len} | dropout {model.dropout}")
    tokenizer_frozen = tokenizer.to_dict()
    # Snapshot of the pretrained weights BEFORE any adapter is introduced:
    # this is what the adapter is applied to and what stays frozen.
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    total_base, trainable_base, pct_base = count_parameters(model)
    print(f"  base parameters: {total_base:,} total, {trainable_base:,} trainable ({pct_base:.1f}%)")

    # 2. Domain corpus + deterministic sequential split (identical rule to 09).
    domain_text, domain_path = exp09.load_text(args)
    domain_train_text, domain_val_text = exp09.split_sequence(domain_text, args.val_split)
    print(f"\nDomain corpus: {domain_path}")
    print(f"  {len(domain_text):,} chars | train {len(domain_train_text):,} | "
          f"val {len(domain_val_text):,} (sequential {1 - args.val_split:.0%}/{args.val_split:.0%} split)")

    # 3. Encode with the FIXED tokenizer (no refitting - that would change the
    #    token ids and invalidate the embedding/output weights).
    t0 = time.time()
    domain_train_ids = exp09.encode_text(tokenizer, domain_train_text)
    domain_val_ids = exp09.encode_text(tokenizer, domain_val_text)
    n_unk = int((domain_train_ids == tokenizer.unk_id).sum().item()) + int(
        (domain_val_ids == tokenizer.unk_id).sum().item()
    )
    print(f"  encoded with fixed tokenizer in {time.time() - t0:.1f}s | "
          f"train {len(domain_train_ids):,} tokens "
          f"({len(domain_train_text) / max(1, len(domain_train_ids)):.2f} chars/token) | "
          f"val {len(domain_val_ids):,} tokens | {n_unk} UNK token(s) total")

    # 4. Original-domain held-out segment (same rule as experiments 07/08/09).
    orig_val_text = exp09.load_orig_val_text(args)
    orig_val_ids = exp09.encode_text(tokenizer, orig_val_text)
    print(f"  original-domain val segment: {len(orig_val_text):,} chars / "
          f"{len(orig_val_ids):,} tokens")

    # 5. BEFORE metrics on both domains (pretrained checkpoint).
    print("\n=== Before adaptation (pretrained checkpoint) ===")
    before_domain = exp09.evaluate(model, domain_val_ids, tokenizer, device, args.batch_size)
    before_orig = exp09.evaluate(model, orig_val_ids, tokenizer, device, args.batch_size)
    print_metrics("domain (Candle)", before_domain)
    print_metrics("original (P&P)", before_orig)

    # 6. Build the adapted model, train it and save it -- or, in eval-only
    #    mode, load a previously saved adapted checkpoint.
    lora_config = None
    if args.eval_only:
        print(f"\n=== Eval-only mode: loading {args.adapt_checkpoint} ===")
        adapted, adapted_tokenizer, adapted_ckpt = load_adapted_checkpoint(
            args.adapt_checkpoint, device
        )
        print(f"  method: {adapted_ckpt['method']} | base checkpoint: "
              f"{adapted_ckpt.get('base_checkpoint')}")
        if adapted_ckpt.get("lora_config"):
            print(f"  lora config: {adapted_ckpt['lora_config']}")
        total_params, trainable_params, trainable_pct = count_parameters(adapted)
        print(f"  parameters: {total_params:,} total, {trainable_params:,} trainable "
              f"({trainable_pct:.2f}%)")
    else:
        if args.method == "lora":
            replaced = build_lora_model(
                model, target_layers, rank=args.lora_rank, alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
            lora_config = {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "target_layers": target_layers,
                "dropout": args.lora_dropout,
            }
            print(f"\nLoRA applied to {len(replaced)} layer(s): {replaced}")
            print(f"  rank {args.lora_rank} | alpha {args.lora_alpha} "
                  f"(scale alpha/rank = {args.lora_alpha / args.lora_rank:.2f})")
        else:
            print("\nFull fine-tuning baseline: every parameter stays trainable.")

        total_params, trainable_params, trainable_pct = count_parameters(model)
        print(f"  parameters: {total_params:,} total, {trainable_params:,} trainable "
              f"({trainable_pct:.2f}%)")
        if args.method == "lora":
            n_lora_tensors = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
            assert n_lora_tensors == len(replaced)
            # Safety check: nothing but the adapter may be trainable.
            for name, param in model.named_parameters():
                is_lora = name.endswith("lora_A") or name.endswith("lora_B")
                if is_lora != param.requires_grad:
                    raise RuntimeError(
                        f"unexpected trainability for {name}: requires_grad={param.requires_grad}"
                    )

        # 7. Adaptation training (same protocol as Experiment 09).
        print("\n=== Adaptation training ===")
        print(f"Config: method={args.method}, epochs={args.epochs}, batch_size={args.batch_size}, "
              f"lr={args.lr} (fresh AdamW), context_len={args.context_len or model.context_len}, "
              f"seed={args.seed}, selection=domain-val per-token loss")
        result = train_adaptation(
            model,
            tokenizer,
            domain_train_ids,
            domain_val_ids,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            context_len=args.context_len,
            orig_val_ids=orig_val_ids,
        )
        for e in result["history"]:
            line = (f"Epoch {e['epoch']}/{args.epochs} | train {e['train_loss']:.4f} | "
                    f"domain val {e['domain_val_per_token']:.4f} "
                    f"({e['domain_val_nats_per_char']:.3f} nats/char)")
            if "orig_val_per_token" in e:
                line += (f" | orig val {e['orig_val_per_token']:.4f} "
                         f"({e['orig_val_nats_per_char']:.3f} nats/char)")
            print(line + f" | {e['seconds']:.1f}s")
        print(f"Best domain val {result['best_domain_val_loss']:.4f} at epoch "
              f"{result['best_epoch']} (adapted checkpoint selected)")
        print(f"Adaptation time: {result['elapsed_seconds']:.1f}s")

        assert tokenizer.to_dict() == tokenizer_frozen, "tokenizer must remain fixed during adaptation"

        # 8. Save the adapted checkpoint (base weights + adapter only).
        best_state = result["best_state"]
        if args.method == "lora":
            saved_lora_state = {k: v for k, v in best_state.items()
                                if k.endswith("lora_A") or k.endswith("lora_B")}
            saved_base_state = base_state
        else:
            saved_lora_state = {}
            saved_base_state = best_state
        ckpt_path = save_adapted_checkpoint(
            args.adapt_checkpoint,
            base_state=saved_base_state,
            lora_state=saved_lora_state,
            model_config=model.config_dict(),
            tokenizer=tokenizer,
            lora_config=lora_config,
            method=args.method,
            history=result["history"],
            best_epoch=result["best_epoch"],
            best_val_loss=result["best_domain_val_loss"],
            base_checkpoint=args.checkpoint,
            args=args,
        )
        print(f"Saved adapted checkpoint to {ckpt_path}")

        # 9. Reload the saved checkpoint (verifies the round-trip).
        adapted, adapted_tokenizer, adapted_ckpt = load_adapted_checkpoint(
            args.adapt_checkpoint, device
        )
    print("\n=== After adaptation (best adapted checkpoint) ===")
    after_domain = exp09.evaluate(adapted, domain_val_ids, adapted_tokenizer, device, args.batch_size)
    after_orig = exp09.evaluate(adapted, orig_val_ids, adapted_tokenizer, device, args.batch_size)
    print_metrics("domain (Candle)", after_domain)
    print_metrics("original (P&P)", after_orig)
    print("  (the two domains have different distributions; compare before/after "
          "within a domain, not across domains)")

    delta_dom = after_domain["nats_per_char"] - before_domain["nats_per_char"]
    delta_orig = after_orig["nats_per_char"] - before_orig["nats_per_char"]
    print(f"  delta: domain {delta_dom:+.3f} nats/char | original {delta_orig:+.3f} nats/char")

    report_domain_table(
        [
            ("Candle", "base", before_domain),
            ("Candle", "adapted", after_domain),
            ("P&P", "base", before_orig),
            ("P&P", "adapted", after_orig),
        ]
    )

    if adapted_ckpt["method"] == "lora":
        print("\n=== Learned weight update per adapted layer ===")
        for row in adapter_summary(adapted):
            print(f"  {row['layer']:<28} ||dW||_F {row['delta_norm']:.4f} | "
                  f"||W||_F {row['base_norm']:.4f} | relative {row['rel_delta']:.4f}")

    # 10. Historical context: the Experiment 09 full-fine-tuning checkpoint.
    compare_path = Path(args.compare_checkpoint)
    if compare_path.exists() and adapted_ckpt["method"] == "lora":
        print(f"\n=== Historical reference (NOT a controlled comparison): Experiment 09 "
              f"full fine-tuning, {compare_path.name} ===")
        print("  Separate earlier run: same base checkpoint, same corpus and split, "
              "all 161,920 parameters trainable, lr 1e-4.")
        try:
            hist_model, hist_tokenizer, _ = exp09.load_checkpoint(compare_path, device)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  could not load {compare_path}: {e}")
        else:
            hist_domain = exp09.evaluate(hist_model, domain_val_ids, hist_tokenizer, device, args.batch_size)
            hist_orig = exp09.evaluate(hist_model, orig_val_ids, hist_tokenizer, device, args.batch_size)
            print_metrics("domain (Candle)", hist_domain)
            print_metrics("original (P&P)", hist_orig)
            print(f"  delta vs base: domain "
                  f"{hist_domain['nats_per_char'] - before_domain['nats_per_char']:+.3f} nats/char | "
                  f"original {hist_orig['nats_per_char'] - before_orig['nats_per_char']:+.3f} nats/char")

    # 11. Generation: identical prompts, seed and settings for both models.
    base_model, base_tokenizer, _ = exp09.load_checkpoint(args.checkpoint, device)
    print("\n=== Generation (fixed prompts, seed, temperature) ===")
    for prompt in exp09.GENERATION_PROMPTS:
        print(f"\n-- prompt {prompt!r} (seed {exp09.GENERATION_SEED}, "
              f"temperature {args.temperature}, {args.max_new_tokens} tokens)")
        base_out = exp09.seeded_generate(
            base_model, base_tokenizer, prompt, seed=exp09.GENERATION_SEED,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device,
        )
        adapt_out = exp09.seeded_generate(
            adapted, adapted_tokenizer, prompt, seed=exp09.GENERATION_SEED,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device,
        )
        print(f"[base    ] {base_out}")
        print(f"[adapted] {adapt_out}")


if __name__ == "__main__":
    main()
