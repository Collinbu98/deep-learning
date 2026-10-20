"""
Experiment 09 - Domain adaptation through continued LM training.

Continues training an Experiment 08 BPE Transformer (pretrained on Pride and
Prejudice) on a small, different public-domain corpus: Michael Faraday's
"The Chemical History of a Candle" (Project Gutenberg ebook #14474,
https://www.gutenberg.org/cache/epub/14474/pg14474.txt). Faraday died in
1867 and the text is in the public domain in the United States; Project
Gutenberg distributes it under its public-domain terms. The raw Gutenberg
plain-text file is used as-is (header/license boilerplate included), with no
other preprocessing - the same convention as experiments 07/08.

Design decisions:

* The Experiment 08 checkpoint format already contains everything needed to
  reconstruct the model: ``model_config`` (architecture), ``model_state_dict``
  (weights), and ``tokenizer`` (alphabet + merges + UNK). No guessing is
  needed; ``load_checkpoint`` validates every required key and the
  tokenizer/vocabulary agreement, and fails loudly otherwise.
* The tokenizer and vocabulary are kept **fixed**. Domain text is encoded
  with the pretrained tokenizer (unseen characters become UNK); no new BPE
  merges are learned, because re-fitting would change token ids and
  invalidate the embedding and output-layer weights.
* A **fresh AdamW optimizer** is used for adaptation (no restored optimizer
  state) with a conservative default learning rate (1e-4 vs 1e-3 for
  pretraining), to isolate the effect of continued training itself.
* Domain text is split sequentially (deterministic) into domain-train and
  domain-validation segments; validation text never produces gradient
  updates and is used only for evaluation and best-checkpoint selection.
* Best checkpoint selection uses the **domain-validation** per-token loss
  only. Original-domain (Pride and Prejudice) validation is evaluated each
  epoch purely as a forgetting monitor; it never influences selection.
* Both the original and the adapted checkpoint are scored on both held-out
  segments with the same implementations (Experiment 08 ``eval_epoch`` for
  per-token loss, ``minigrad.lm_eval.nats_per_character`` for nats/char).
  Metrics of the two domains are kept separate: the data distributions
  differ, so their raw losses are not comparable to each other - only
  before/after differences within one domain are meaningful.
* Generation uses fixed prompts, a fixed sampling seed, and identical
  temperature/token counts for both checkpoints to avoid cherry-picking.
"""

import argparse
import importlib.util
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from minigrad.bpe_tokenizer import BPETokenizer
from minigrad.lm_eval import nats_per_character

# Fixed generation settings shared by both checkpoints (anti cherry-picking).
GENERATION_PROMPTS = ["The ", "A candle is"]
GENERATION_SEED = 1234

DOMAIN_DATA_URLS = [
    "https://www.gutenberg.org/cache/epub/14474/pg14474.txt",  # Faraday, Candle
]

REQUIRED_CKPT_KEYS = ("model_state_dict", "model_config", "tokenizer")
REQUIRED_MODEL_CONFIG_KEYS = ("vocab_size", "d_model", "n_layers", "context_len", "n_heads", "dropout")


def _load_module(name, relpath):
    path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp08 = _load_module("bpe_lm_experiment", "experiments/08_bpe_language_model.py")


def get_device():
    # cuda:0 only when available; never a higher-index device.
    return exp08.get_device()


def load_checkpoint(ckpt_path, device):
    """Rebuild (model, tokenizer, raw checkpoint) from an Experiment 08 file.

    Fails with an explicit error instead of guessing architecture settings
    when the checkpoint lacks required keys or the stored vocabulary
    disagrees with the model configuration.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint {ckpt_path} is not a dict; cannot load")
    missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        raise ValueError(
            f"checkpoint {ckpt_path} missing required key(s) {missing}; "
            f"expected the Experiment 08 format {list(REQUIRED_CKPT_KEYS)}"
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt


def split_sequence(text, val_frac):
    """Deterministic sequential split into (domain_train_text, domain_val_text)."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if not text:
        raise ValueError("cannot split empty text")
    split = int(len(text) * (1 - val_frac))
    if split <= 0 or split >= len(text):
        raise ValueError(f"val_frac={val_frac} leaves an empty segment for text of length {len(text)}")
    return text[:split], text[split:]


def encode_text(tokenizer, text):
    """Encode with the FIXED tokenizer; never fits new merges."""
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    covered = sum(tokenizer.token_char_length(int(t)) for t in ids.tolist())
    assert covered == len(text), (covered, len(text))
    return ids


def evaluate(model, token_ids, tokenizer, device, batch_size=32):
    """Per-token validation loss (sliding windows) + nats/character metric.

    Uses Experiment 08's eval_epoch and the shared nats_per_character, so
    all numbers stay comparable with earlier experiments.
    """
    ds = exp08.TokenDataset(token_ids, model.context_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    per_token = exp08.eval_epoch(model, dl, nn.CrossEntropyLoss(), device)
    nats = nats_per_character(
        model, token_ids, exp08.bpe_char_weights(tokenizer), device=device, batch_size=batch_size
    )
    return {
        "per_token_loss": per_token,
        "nats_per_char": nats["nats_per_char"],
        "n_tokens": int(token_ids.numel()),
        "n_tokens_scored": nats["n_tokens_scored"],
        "sum_chars": nats["sum_chars"],
    }


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
    checkpoint_path,
    context_len=None,
    orig_val_ids=None,
    orig_tokenizer=None,
):
    """Continue training ``model`` on domain text with a FRESH AdamW optimizer.

    Best checkpoint selection: minimum domain-validation per-token loss.
    If ``orig_val_ids`` is given it is evaluated each epoch as a forgetting
    monitor only (never used for selection or updates).

    Saves a checkpoint in the same format as Experiment 08 (loadable by
    ``load_checkpoint``) plus provenance fields. Returns a result dict with
    ``history``, ``best_epoch``, ``best_domain_val_loss``, ``elapsed_seconds``.
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

    # Fresh optimizer state for the adaptation stage (documented choice).
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = exp08.train_epoch(model, train_dl, criterion, optimizer, device)
        dom = evaluate(model, domain_val_ids, tokenizer, device, batch_size=batch_size)
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "domain_val_per_token": dom["per_token_loss"],
            "domain_val_nats_per_char": dom["nats_per_char"],
            "seconds": time.time() - epoch_start,
        }
        if orig_val_ids is not None:
            orig = evaluate(
                model,
                orig_val_ids,
                orig_tokenizer if orig_tokenizer is not None else tokenizer,
                device,
                batch_size=batch_size,
            )
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

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model.config_dict(),
            "tokenizer": tokenizer.to_dict(),
            "stage": "domain_adaptation",
            "best_epoch": best_epoch,
            "best_domain_val_loss": best_val,
            "epochs_run": epochs,
            "adaptation_lr": lr,
        },
        ckpt_path,
    )
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_domain_val_loss": best_val,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(ckpt_path),
    }


def seeded_generate(model, tokenizer, prompt, *, seed, max_new_tokens, temperature, device):
    """Generation with an explicit seed so both checkpoints sample identically."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return exp08.generate(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, device=device
    )


def load_text(args):
    """Load the domain corpus from --data or download it into data/."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.data:
        p = Path(args.data)
        if not p.exists():
            raise FileNotFoundError(f"Domain data file not found: {p}")
        return p.read_text(encoding="utf-8", errors="replace"), str(p)
    for url in DOMAIN_DATA_URLS:
        try:
            import urllib.request

            out = data_dir / (url.split("/")[-1] or "domain.txt")
            if not out.exists():
                print(f"Downloading domain corpus from {url}...", file=sys.stderr)
                urllib.request.urlretrieve(url, out)
            return out.read_text(encoding="utf-8", errors="replace"), str(out)
        except Exception as e:
            print(f"Failed to download {url}: {e}", file=sys.stderr)
            continue
    raise RuntimeError(
        "Domain corpus download failed; supply --data <path> to a local text file. "
        "No synthetic substitute is used."
    )


def load_orig_val_text(args):
    """Held-out original-domain (Pride and Prejudice) segment, same rule as 07/08."""
    if args.orig_data:
        p = Path(args.orig_data)
        if not p.exists():
            raise FileNotFoundError(f"Original data file not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
    else:
        p = Path(__file__).resolve().parents[1] / "data" / "pg1342.txt"
        if not p.exists():
            raise FileNotFoundError(
                f"Original corpus {p} not found; it is required for the "
                f"original-domain evaluation (or pass --orig_data)."
            )
        text = p.read_text(encoding="utf-8", errors="replace")
    split = int(len(text) * (1 - args.val_split))  # identical rule to experiments 07/08
    return text[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "08_bpe_best.pt"),
        help="Experiment 08 checkpoint to continue training from",
    )
    parser.add_argument(
        "--adapt_checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "09_adapted_best.pt"),
        help="Where to save the best adapted checkpoint",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to the domain text file")
    parser.add_argument("--orig_data", type=str, default=None, help="Path to Pride and Prejudice text")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Fresh AdamW lr for adaptation (conservative)")
    parser.add_argument("--context_len", type=int, default=None,
                        help="Training window length (default: pretrained context_len)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1, help="Sequential domain train/val split")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # 1. Load pretrained model + fixed tokenizer.
    model, tokenizer, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  stage: {ckpt.get('stage', 'pretraining (exp08)')}")
    print(f"  vocab_size {tokenizer.vocab_size} | d_model {model.d_model} "
          f"| n_layers {model.n_layers} | n_heads {model.n_heads} "
          f"| context_len {model.context_len} | dropout {model.dropout}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {total_params:,} total, {trainable_params:,} trainable")
    tokenizer_frozen = tokenizer.to_dict()

    # 2. Domain corpus + deterministic sequential split BEFORE evaluation use.
    domain_text, domain_path = load_text(args)
    domain_train_text, domain_val_text = split_sequence(domain_text, args.val_split)
    print(f"\nDomain corpus: {domain_path}")
    print(f"  {len(domain_text):,} chars | train {len(domain_train_text):,} | "
          f"val {len(domain_val_text):,} (sequential {1 - args.val_split:.0%}/{args.val_split:.0%} split)")

    # 3. Encode with the FIXED tokenizer (no refitting - this would invalidate
    #    the embedding/output weights).
    t0 = time.time()
    domain_train_ids = encode_text(tokenizer, domain_train_text)
    domain_val_ids = encode_text(tokenizer, domain_val_text)
    n_unk = int((domain_train_ids == tokenizer.unk_id).sum().item()) + int(
        (domain_val_ids == tokenizer.unk_id).sum().item()
    )
    print(f"  encoded with fixed tokenizer in {time.time() - t0:.1f}s | "
          f"train {len(domain_train_ids):,} tokens "
          f"({len(domain_train_text) / max(1, len(domain_train_ids)):.2f} chars/token) | "
          f"val {len(domain_val_ids):,} tokens | {n_unk} UNK token(s) total")

    # 4. Original-domain held-out segment (same tokenizer, same metric code).
    orig_val_text = load_orig_val_text(args)
    orig_val_ids = encode_text(tokenizer, orig_val_text)
    print(f"  original-domain val segment: {len(orig_val_text):,} chars / "
          f"{len(orig_val_ids):,} tokens")

    # 5. BEFORE metrics on both domains (original checkpoint).
    print("\n=== Before adaptation (original checkpoint) ===")
    before_domain = evaluate(model, domain_val_ids, tokenizer, device, args.batch_size)
    before_orig = evaluate(model, orig_val_ids, tokenizer, device, args.batch_size)
    print(f"  domain      : {before_domain['per_token_loss']:.4f} per token | "
          f"{before_domain['nats_per_char']:.3f} nats/char")
    print(f"  original (PP): {before_orig['per_token_loss']:.4f} per token | "
          f"{before_orig['nats_per_char']:.3f} nats/char")

    # 6. Continuation training (fresh optimizer).
    print("\n=== Domain adaptation training ===")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr} "
          f"(fresh AdamW), context_len={args.context_len or model.context_len}, "
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
        checkpoint_path=args.adapt_checkpoint,
        context_len=args.context_len,
        orig_val_ids=orig_val_ids,
        orig_tokenizer=tokenizer,
    )
    for e in result["history"]:
        line = (f"Epoch {e['epoch']}/{args.epochs} | train {e['train_loss']:.4f} | "
                f"domain val {e['domain_val_per_token']:.4f} ({e['domain_val_nats_per_char']:.3f} nats/char)")
        if "orig_val_per_token" in e:
            line += (f" | orig val {e['orig_val_per_token']:.4f} "
                     f"({e['orig_val_nats_per_char']:.3f} nats/char)")
        print(line + f" | {e['seconds']:.1f}s")
    print(f"Best domain val {result['best_domain_val_loss']:.4f} at epoch "
          f"{result['best_epoch']} (adapted checkpoint selected)")
    print(f"Adaptation time: {result['elapsed_seconds']:.1f}s")
    print(f"Saved adapted checkpoint + tokenizer to {result['checkpoint_path']}")

    assert tokenizer.to_dict() == tokenizer_frozen, "tokenizer must remain fixed during adaptation"

    # 7. AFTER metrics: reload the saved checkpoint (verifies format round-trip).
    adapted, adapted_tokenizer, adapted_ckpt = load_checkpoint(args.adapt_checkpoint, device)
    print("\n=== After adaptation (best adapted checkpoint) ===")
    after_domain = evaluate(adapted, domain_val_ids, adapted_tokenizer, device, args.batch_size)
    after_orig = evaluate(adapted, orig_val_ids, adapted_tokenizer, device, args.batch_size)
    print(f"  domain      : {after_domain['per_token_loss']:.4f} per token | "
          f"{after_domain['nats_per_char']:.3f} nats/char")
    print(f"  original (PP): {after_orig['per_token_loss']:.4f} per token | "
          f"{after_orig['nats_per_char']:.3f} nats/char")
    print("  (the two domains have different distributions; compare before/after "
          "within a domain, not across domains)")

    delta_dom = after_domain["nats_per_char"] - before_domain["nats_per_char"]
    delta_orig = after_orig["nats_per_char"] - before_orig["nats_per_char"]
    print(f"  delta: domain {delta_dom:+.3f} nats/char | original {delta_orig:+.3f} nats/char")

    # 8. Generation: identical prompts, seed, and settings for both models.
    # Reload the original checkpoint for a clean comparison (in-memory `model`
    # was mutated by adaptation).
    original_model, original_tokenizer, _ = load_checkpoint(args.checkpoint, device)
    print("\n=== Generation (fixed prompts, seed, temperature) ===")
    for prompt in GENERATION_PROMPTS:
        print(f"\n-- prompt {prompt!r} (seed {GENERATION_SEED}, "
              f"temperature {args.temperature}, {args.max_new_tokens} tokens)")
        orig_out = seeded_generate(
            original_model, original_tokenizer, prompt, seed=GENERATION_SEED,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device,
        )
        adapt_out = seeded_generate(
            adapted, adapted_tokenizer, prompt, seed=GENERATION_SEED,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device,
        )
        print(f"[original ] {orig_out}")
        print(f"[adapted  ] {adapt_out}")


if __name__ == "__main__":
    main()
