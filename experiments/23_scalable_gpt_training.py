"""Experiment 23 - Scalable GPT-style training pipeline.

A configurable, reusable decoder-only Transformer language model (GPT-style,
causal, next-token prediction) trained with AdamW, validated, checkpointed, and
used for autoregressive generation.

Experiment 23 is primarily infrastructure and validation: the pipeline is
exercised end to end with a small, inexpensive smoke run. The design supports
later training of roughly 30M-100M parameter models from random initialization.

Existing implementations reused here:
- experiments/05_self_attention.py: causal multi-head self-attention primitives.
- src/minigrad/bpe_tokenizer.py: deterministic BPE tokenizer + checkpoint dict.
- src/minigrad/lm_eval.py: shared held-out NLL utilities.

The production training path uses PyTorch autograd / nn modules directly and
never routes through the educational scalar autograd engine.

Run with:
    .venv/bin/python experiments/23_scalable_gpt_training.py --smoke
    .venv/bin/python experiments/23_scalable_gpt_training.py --data data/pg1342.txt
    .venv/bin/python experiments/23_scalable_gpt_training.py --resume <ckpt> --checkpoint <ckpt2>
    .venv/bin/python experiments/23_scalable_gpt_training.py --checkpoint <ckpt> --generate-prompt "To "
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from minigrad.bpe_tokenizer import BPETokenizer
from minigrad.lm_eval import nats_per_character

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "23_scalable_gpt_training_best.pt"
DEFAULT_DATA = REPO_ROOT / "data" / "pg1342.txt"

SMOKE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump."
)
SMOKE_VOCAB_SIZE = 64
SMOKE_N_LAYERS = 1
SMOKE_D_MODEL = 16
SMOKE_N_HEADS = 1
SMOKE_DFF = 32
SMOKE_CONTEXT_LEN = 16
SMOKE_DROPOUT = 0.0
SMOKE_BATCH_SIZE = 2
SMOKE_EPOCHS = 2
SMOKE_LR = 3e-3
SMOKE_VAL_SPLIT = 0.2
SMOKE_CLIP_GRAD = 1.0
SMOKE_MAX_NEW_TOKENS = 40
SMOKE_TEMPERATURE = 0.8


def _load_self_attention():
    path = REPO_ROOT / "experiments" / "05_self_attention.py"
    spec = importlib.util.spec_from_file_location("self_attention_exp23", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SelfAttentionExp = _load_self_attention()


def validate_dropout(dropout):
    """Reject dropout probabilities outside [0, 1)."""
    return SelfAttentionExp.validate_dropout(dropout)


def get_device(requested=None):
    """Resolve the device.

    ``requested`` may be None (auto), "cpu", "cuda", or "cuda:<idx>". Auto
    resolves to cuda:0 when a CUDA device is available (ROCm included) and to
    CPU otherwise. An explicit CUDA request raises clearly when no CUDA device
    is available; there is never a silent fallback from an explicit GPU request.
    """
    req = None if requested is None else str(requested)
    if req is None or req.strip().lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    key = req.strip().lower()
    if key.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA was explicitly requested (device={req}) but no CUDA device is "
                "available in this environment; refusing to silently fall back to CPU."
            )
        idx = 0
        if ":" in key:
            try:
                idx = int(key.split(":", 1)[1])
            except ValueError as e:
                raise ValueError(f"invalid CUDA device string: {req!r}") from e
        if torch.cuda.device_count() < idx + 1:
            raise RuntimeError(
                f"CUDA device cuda:{idx} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) available."
            )
        return torch.device(f"cuda:{idx}")
    return torch.device(req)


def validate_gpt_config(config):
    """Validate a GPT model configuration dict; raises ValueError when invalid."""
    required = (
        "vocab_size",
        "n_layers",
        "d_model",
        "n_heads",
        "context_len",
    )
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"GPT config missing required keys: {missing}")
    vocab_size = int(config["vocab_size"])
    n_layers = int(config["n_layers"])
    d_model = int(config["d_model"])
    n_heads = int(config["n_heads"])
    context_len = int(config["context_len"])
    dff = int(config["dff"]) if config.get("dff") is not None else 4 * d_model
    dropout = float(config.get("dropout") or 0.0)
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if d_model < 1:
        raise ValueError(f"d_model must be >= 1, got {d_model}")
    if n_heads < 1 or d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    if dff < 1:
        raise ValueError(f"dff (feed-forward width) must be >= 1, got {dff}")
    if context_len < 2:
        raise ValueError(f"context_len must be >= 2, got {context_len}")
    validate_dropout(dropout)
    return {
        "vocab_size": vocab_size,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "dff": dff,
        "context_len": context_len,
        "dropout": dropout,
    }


class GPTBlock(nn.Module):
    """Decoder-style Transformer block (pre-normalization), GPT-style FFN."""

    def __init__(self, d_model, n_heads=1, dff=None, dropout=0.0):
        super().__init__()
        validate_dropout(dropout)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.dff = int(4 * self.d_model) if dff is None else int(dff)
        if self.n_heads < 1 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.dff < 1:
            raise ValueError(f"dff must be >= 1, got {self.dff}")
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        if self.n_heads == 1:
            self.attention = SelfAttentionExp.SelfAttention(self.d_model, dropout=dropout)
        else:
            self.attention = SelfAttentionExp.MultiHeadSelfAttention(
                self.d_model, self.n_heads, dropout=dropout
            )
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.dff),
            nn.GELU(),
            nn.Linear(self.dff, self.d_model),
        )
        self.ffn_hidden_dropout = nn.Dropout(dropout)
        self.ffn_output_dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        norm_x1 = self.layer_norm1(x)
        attn_out = self.attention(norm_x1, causal=causal)
        x = x + attn_out
        norm_x2 = self.layer_norm2(x)
        ff_out = self.ffn[0](norm_x2)
        ff_out = self.ffn[1](ff_out)
        ff_out = self.ffn_hidden_dropout(ff_out)
        ff_out = self.ffn[2](ff_out)
        ff_out = self.ffn_output_dropout(ff_out)
        return x + ff_out


class GPTLanguageModel(nn.Module):
    """Configurable decoder-only GPT-style language model for next-token prediction."""

    def __init__(
        self,
        vocab_size,
        n_layers=2,
        d_model=128,
        n_heads=1,
        dff=None,
        context_len=64,
        dropout=0.0,
    ):
        super().__init__()
        config = validate_gpt_config(
            {
                "vocab_size": vocab_size,
                "n_layers": n_layers,
                "d_model": d_model,
                "n_heads": n_heads,
                "dff": dff,
                "context_len": context_len,
                "dropout": dropout,
            }
        )
        self.vocab_size = config["vocab_size"]
        self.n_layers = config["n_layers"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.dff = config["dff"]
        self.context_len = config["context_len"]
        self.dropout = config["dropout"]

        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embed = nn.Embedding(self.context_len, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(self.d_model, n_heads=self.n_heads, dff=self.dff, dropout=self.dropout)
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        if not torch.is_tensor(x):
            raise TypeError(f"expected a LongTensor of token ids, got {type(x)}")
        if x.dim() != 2:
            raise ValueError(f"expected [B, T] token ids, got shape {tuple(x.shape)}")
        B, T = x.shape
        if T > self.context_len:
            raise ValueError(f"Input length {T} exceeds context_len {self.context_len}")
        pos = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        h = self.token_embed(x) + self.pos_embed(pos)
        h = self.embed_dropout(h)
        for block in self.blocks:
            h = block(h, causal=True)
        h = self.final_norm(h)
        return self.lm_head(h)

    def config_dict(self):
        return {
            "vocab_size": int(self.vocab_size),
            "n_layers": int(self.n_layers),
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "dff": int(self.dff),
            "context_len": int(self.context_len),
            "dropout": float(self.dropout),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config))


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def gpt_parameter_count(vocab_size, n_layers, d_model, n_heads, dff, context_len):
    """Closed-form parameter count for the configurable GPT architecture.

    Parameter counts include Linear biases (weight + bias per layer).

    Per block: attention projections (3 or 4 Linear layers of d -> d, plus
    bias) + feed-forward (Linear d -> dff, GELU, Linear dff -> d, plus bias) +
    two LayerNorms (2 params each). Plus token embedding, positional embedding,
    final LayerNorm, and LM head.
    """
    if n_heads == 1:
        attn = 3 * (d_model * d_model + d_model)
    else:
        attn = 4 * (d_model * d_model + d_model)
    ffn = (dff * d_model + dff) + (d_model * dff + d_model)
    block = attn + ffn + 4 * d_model
    embed = vocab_size * d_model + context_len * d_model
    final_norm = 2 * d_model
    head = vocab_size * d_model + vocab_size
    return int(embed + n_layers * block + final_norm + head)


def estimate_parameter_memory_bytes(num_params, dtype=None) -> int:
    dtype = torch.float32 if dtype is None else dtype
    return int(num_params * dtype.itemsize)


def make_optimizer(model, lr, weight_decay=0.0) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


class GPTTokenDataset(Dataset):
    """Sliding-window next-token dataset over a flat token-id sequence."""

    def __init__(self, token_ids, context_len):
        self.token_ids = [int(t) for t in list(token_ids)]
        self.context_len = int(context_len)
        self.n_windows = max(0, len(self.token_ids) - self.context_len)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = int(idx)
        T = self.context_len
        x = self.token_ids[start : start + T]
        y = self.token_ids[start + 1 : start + 1 + T]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def next_token_loss(model, x, y):
    logits = model(x)
    return nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))


def train_epoch(model, loader, criterion, optimizer, device, clip_grad=None) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    steps = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, x, y)
        loss.backward()
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
        optimizer.step()
        total_loss += loss.item() * int(x.numel())
        total_tokens += int(x.numel())
        steps += 1
    return {
        "loss": float(total_loss / max(1, total_tokens)),
        "tokens": int(total_tokens),
        "steps": int(steps),
    }


def validate_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = next_token_loss(model, x, y)
            total_loss += loss.item() * int(x.numel())
            total_tokens += int(x.numel())
            steps += 1
    loss = float(total_loss / max(1, total_tokens))
    return {
        "loss": loss,
        "perplexity": float(math.exp(loss)) if math.isfinite(loss) else float("inf"),
        "tokens": int(total_tokens),
        "steps": int(steps),
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, clip_grad=1.0, max_steps=None):
    model.train()
    history = []
    step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=clip_grad)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        step += train_metrics["steps"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_tokens": train_metrics["tokens"],
                "train_steps": train_metrics["steps"],
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "val_tokens": val_metrics["tokens"],
                "val_steps": val_metrics["steps"],
                "epoch_seconds": time.time() - epoch_start,
            }
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if max_steps is not None and step >= max_steps:
            break
    return {
        "status": "completed",
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "steps": step,
        "total_seconds": time.time() - start,
    }


def save_checkpoint(
    path,
    *,
    model,
    config,
    tokenizer,
    training_state,
    args,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": dict(config),
        "tokenizer": tokenizer.to_dict(),
        "training_state": dict(training_state),
        "args": dict(args),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def restore_training_state(payload, model, optimizer=None, device=None) -> dict:
    config = dict(payload.get("model_config") or {})
    validated = validate_gpt_config(config)
    model_config = {k: validated[k] for k in ("vocab_size", "n_layers", "d_model", "n_heads", "dff", "context_len", "dropout")}
    if (
        int(model.vocab_size) != int(validated["vocab_size"])
        or int(model.n_layers) != int(validated["n_layers"])
        or int(model.d_model) != int(validated["d_model"])
        or int(model.n_heads) != int(validated["n_heads"])
        or int(model.dff) != int(validated["dff"])
        or int(model.context_len) != int(validated["context_len"])
        or float(model.dropout) != float(validated["dropout"])
    ):
        raise ValueError(
            "Checkpoint model_config is incompatible with the current model: "
            f"{model_config} vs current "
            f"{model.config_dict()}"
        )
    model.load_state_dict(payload["model_state_dict"])
    tokenizer = None
    if payload.get("tokenizer") is not None:
        tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    training_state = dict(payload.get("training_state") or {})
    step = int(training_state.get("step", 0))
    optimizer_state = training_state.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if device is not None:
        model.to(device)
    return {"step": step, "tokenizer": tokenizer, "optimizer_state": optimizer_state}


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    device=None,
):
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if top_k is not None and (top_k < 1 or top_k > model.vocab_size):
        raise ValueError(f"top_k must be in [1, vocab_size], got {top_k}")
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ids = list(tokenizer.encode(prompt))
    if not ids:
        ids = [tokenizer.unk_id]
    if len(ids) > model.context_len:
        ids = ids[-model.context_len :]
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = 0
    while generated < int(max_new_tokens):
        ctx = ids_t[:, -model.context_len :] if ids_t.size(1) > model.context_len else ids_t
        with torch.no_grad():
            logits = model(ctx)
        next_logits = logits[0, -1, :] / float(temperature)
        if top_k is not None:
            kth = torch.topk(next_logits, k=min(int(top_k), int(next_logits.numel())))
            threshold = kth.values[..., -1:]
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids_t = torch.cat([ids_t, next_id.unsqueeze(0)], dim=1)
        generated += 1
    return tokenizer.decode(ids_t[0].cpu())


def load_corpus_text(args) -> tuple[str, str]:
    if args.data is not None:
        p = Path(args.data)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        return p.read_text(encoding="utf-8", errors="replace"), str(p)
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA.read_text(encoding="utf-8", errors="replace"), str(DEFAULT_DATA)
    raise RuntimeError("No corpus provided and default local corpus not found; pass --data.")


def split_corpus(text: str, val_split: float) -> tuple[str, str]:
    split = int(len(text) * (1.0 - float(val_split)))
    return text[:split], text[split:]


def fit_tokenizer_and_encode(train_text: str, val_text: str, vocab_size: int):
    tokenizer = BPETokenizer(vocab_size=vocab_size).fit(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    return tokenizer, train_ids, val_ids


def run_training(args) -> dict:
    device = get_device(getattr(args, "device", None))
    if bool(getattr(args, "smoke", False)):
        text = SMOKE_TEXT
        data_path = "smoke-builtin"
        vocab_size = int(getattr(args, "vocab_size") or SMOKE_VOCAB_SIZE)
        n_layers = int(getattr(args, "n_layers") or SMOKE_N_LAYERS)
        d_model = int(getattr(args, "d_model") or SMOKE_D_MODEL)
        n_heads = int(getattr(args, "n_heads") or SMOKE_N_HEADS)
        dff = int(getattr(args, "dff") or SMOKE_DFF)
        context_len = int(getattr(args, "context_len") or SMOKE_CONTEXT_LEN)
        dropout = float(getattr(args, "dropout") or SMOKE_DROPOUT)
        batch_size = int(getattr(args, "batch_size") or SMOKE_BATCH_SIZE)
        epochs = int(getattr(args, "epochs") or SMOKE_EPOCHS)
        lr = float(getattr(args, "lr") or SMOKE_LR)
        val_split = float(getattr(args, "val_split") or SMOKE_VAL_SPLIT)
        clip_grad = float(getattr(args, "clip_grad") or SMOKE_CLIP_GRAD)
        max_steps = getattr(args, "max_steps", None)
    else:
        text, data_path = load_corpus_text(args)
        vocab_size = int(getattr(args, "vocab_size") or 512)
        n_layers = int(getattr(args, "n_layers") or 2)
        d_model = int(getattr(args, "d_model") or 128)
        n_heads = int(getattr(args, "n_heads") or 1)
        dff = int(getattr(args, "dff") or 4 * d_model)
        context_len = int(getattr(args, "context_len") or 64)
        dropout = float(getattr(args, "dropout") or 0.0)
        batch_size = int(getattr(args, "batch_size") or 8)
        epochs = int(getattr(args, "epochs") or 3)
        lr = float(getattr(args, "lr") or 1e-3)
        val_split = float(getattr(args, "val_split") or 0.1)
        clip_grad = float(getattr(args, "clip_grad") or 1.0)
        max_steps = getattr(args, "max_steps", None)

    train_text, val_text = split_corpus(text, val_split)
    tokenizer, train_ids, val_ids = fit_tokenizer_and_encode(
        train_text, val_text, vocab_size
    )
    train_ds = GPTTokenDataset(train_ids, context_len)
    val_ds = GPTTokenDataset(val_ids, context_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    config = validate_gpt_config(
        {
            "vocab_size": tokenizer.vocab_size,
            "n_layers": n_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "dff": dff,
            "context_len": context_len,
            "dropout": dropout,
        }
    )
    model = GPTLanguageModel(**config).to(device)
    model_config = model.config_dict()
    num_params = count_parameters(model)
    expected_params = gpt_parameter_count(
        vocab_size=model_config["vocab_size"],
        n_layers=model_config["n_layers"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        dff=model_config["dff"],
        context_len=model_config["context_len"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, lr, weight_decay=float(getattr(args, "weight_decay") or 0.0))
    training_state = {"step": 0, "optimizer_state_dict": optimizer.state_dict()}
    ckpt_path = Path(getattr(args, "checkpoint") or DEFAULT_CHECKPOINT)

    resumed = False
    start_step = 0
    if getattr(args, "resume", None):
        payload = load_checkpoint(args.resume)
        restored = restore_training_state(payload, model, optimizer, device)
        start_step = int(restored["step"])
        resumed = True

    result = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs,
        clip_grad=clip_grad,
        max_steps=max_steps,
    )
    result["resumed"] = resumed
    result["start_step"] = start_step
    result["steps"] = start_step + int(result.get("steps") or 0)
    result["best_val_loss"] = result.get("best_val_loss")
    result["tokenizer_vocab_size"] = int(tokenizer.vocab_size)
    result["tokenizer_train_tokens"] = len(train_ids)
    result["tokenizer_val_tokens"] = len(val_ids)
    result["tokenizer_train_chars"] = len(train_text)
    result["tokenizer_val_chars"] = len(val_text)
    result["corpus_chars"] = len(text)
    result["data_path"] = data_path
    result["device"] = str(device)
    result["config"] = model_config
    result["num_parameters"] = num_params
    result["expected_parameter_count"] = expected_params
    result["parameter_count_matches"] = int(num_params) == int(expected_params)
    result["estimated_parameter_bytes"] = estimate_parameter_memory_bytes(num_params)
    result["estimated_optimizer_bytes"] = estimate_parameter_memory_bytes(2 * num_params + num_params)

    if result.get("best_state") is not None:
        save_checkpoint(
            ckpt_path,
            model=model,
            config=model_config,
            tokenizer=tokenizer,
            training_state={
                "step": result["steps"],
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
            },
            args={k: v for k, v in vars(args).items() if not callable(v)},
        )
        result["checkpoint"] = str(ckpt_path)
    else:
        result["checkpoint"] = None

    # Final evaluation on the held-out split with no gradients.
    final_val = validate_epoch(model, val_loader, criterion, device)
    result["final_val"] = final_val
    result["final_val_perplexity"] = final_val.get("perplexity")
    return result


def generate_from_checkpoint(args) -> dict:
    device = get_device(getattr(args, "device", None))
    ckpt = Path(getattr(args, "checkpoint") or DEFAULT_CHECKPOINT)
    payload = load_checkpoint(ckpt)
    model = GPTLanguageModel(**{k: v for k, v in dict(payload.get("model_config") or {}).items()})
    restored = restore_training_state(payload, model, None, device)
    tokenizer = restored["tokenizer"]
    output = generate(
        model,
        tokenizer,
        prompt=str(getattr(args, "generate_prompt") or ""),
        max_new_tokens=int(getattr(args, "max_new_tokens") or SMOKE_MAX_NEW_TOKENS),
        temperature=float(getattr(args, "temperature") or SMOKE_TEMPERATURE),
        top_k=getattr(args, "top_k", None),
        device=device,
    )
    return {"output": output, "tokenizer_vocab_size": int(tokenizer.vocab_size), "checkpoint": str(ckpt)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 23 - Scalable GPT training pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to text file (default: local data/pg1342.txt)")
    parser.add_argument("--smoke", action="store_true", help="Use the tiny built-in smoke corpus and tiny hyperparameters")
    parser.add_argument("--vocab_size", type=int, default=None, help="BPE vocabulary target size (smoke default 64)")
    parser.add_argument("--n_layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--d_model", type=int, default=None, help="Model width / embedding dimension")
    parser.add_argument("--n_heads", type=int, default=None, help="Number of attention heads (must divide d_model)")
    parser.add_argument("--dff", type=int, default=None, help="Feed-forward width (defaults to 4*d_model)")
    parser.add_argument("--context_len", type=int, default=None, help="Maximum context length in tokens")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability in [0, 1)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum optimizer steps (overrides epochs)")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--clip_grad", type=float, default=None, help="Gradient clipping max norm")
    parser.add_argument("--val_split", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cpu, cuda, cuda:<idx>")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Best-checkpoint output path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint")
    parser.add_argument("--generate-prompt", type=str, default=None, help="Generate from checkpoint instead of training")
    parser.add_argument("--max_new_tokens", type=int, default=SMOKE_MAX_NEW_TOKENS, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=SMOKE_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Optional top-k sampling")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.generate_prompt is not None:
        out = generate_from_checkpoint(args)
        print(f"Generated from {out['checkpoint']}: {out['output']!r}")
        return 0
    result = run_training(args)
    print(f"Training status: {result['status']}")
    print(f"Device: {result['device']}")
    print(f"Vocabulary size: {result['tokenizer_vocab_size']}")
    print(f"Train tokens: {result['tokenizer_train_tokens']} | Val tokens: {result['tokenizer_val_tokens']}")
    print(f"Corpus chars: {result['corpus_chars']} (train {result['tokenizer_train_chars']}, val {result['tokenizer_val_chars']})")
    print(f"Model params: {result['num_parameters']:,} (expected {result['expected_parameter_count']:,}, match={result['parameter_count_matches']})")
    print(f"Estimated parameter memory: {result['estimated_parameter_bytes'] / 1e6:.4f} MB")
    print(f"Estimated optimizer-state memory: {result['estimated_optimizer_bytes'] / 1e6:.4f} MB")
    for h in result.get("history", []):
        print(
            f"Epoch {h['epoch']} | train {h['train_loss']:.4f} | val {h['val_loss']:.4f} "
            f"| val perplexity {h['val_perplexity']:.4f} | {h['epoch_seconds']:.1f}s"
        )
    print(f"Best val loss {result.get('best_val_loss')} at epoch {result.get('best_epoch')}")
    print(f"Final val loss {result.get('final_val', {}).get('loss')} | perplexity {result.get('final_val_perplexity')}")
    print(f"Checkpoint saved to {result.get('checkpoint')}")
    print(f"Steps run: {result.get('steps')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
