"""
Experiment 08 - BPE subword Transformer language model.

Same decoder-style Transformer LM as experiment 07, but the text is tokenized
with the educational character-level BPE tokenizer from
src/minigrad/bpe_tokenizer.py instead of one-token-per-character.

Key differences from experiment 07:

* The raw corpus is split into train/validation *before* the tokenizer is
  fitted, so the BPE merge rules never see held-out validation text.
* The tokenizer is fitted on the training portion only. Validation text may
  therefore contain characters that were never seen in training; those are
  encoded as the tokenizer's reserved UNK token (never silently dropped), so
  the validation token stream covers every held-out character and the
  next-token targets skip no positions.
* Besides the per-token validation loss (used for checkpoint selection),
  each epoch also reports negative log-likelihood per raw character
  (nats/character) via minigrad.lm_eval.nats_per_character, which weights
  every scored target token by the number of raw characters it covers (UNK
  counts as one character). This makes results comparable to a
  character-level model scored on the same held-out segment.
* The fitted tokenizer (alphabet + merges + UNK) is saved inside the model
  checkpoint, so a checkpoint can never be paired with an incompatible
  tokenizer: rebuild with ``BPETokenizer.from_dict(ckpt["tokenizer"])``.

Equal context lengths are not equal text coverage: a BPE token stands for
several characters, so context_len=64 BPE tokens typically span several times
more underlying text than context_len=64 characters.
"""

import argparse
import importlib.util
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from minigrad.bpe_tokenizer import BPETokenizer
from minigrad.lm_eval import nats_per_character


def _load_transformer_block():
    path = Path(__file__).resolve().parents[1] / "experiments" / "06_transformer_block.py"
    spec = importlib.util.spec_from_file_location("transformer_block_exp06", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TransformerBlockExp = _load_transformer_block()


DEFAULT_DATA_URLS = [
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
]


class TokenDataset(Dataset):
    """Sliding-window next-token dataset over a flat list of token ids."""

    def __init__(self, token_ids, context_len):
        self.token_ids = token_ids  # 1-D torch.long tensor
        self.context_len = context_len
        self.n_windows = len(token_ids) - context_len

    def __len__(self):
        return max(0, self.n_windows)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.context_len]
        y = self.token_ids[idx + 1 : idx + 1 + self.context_len]
        return x, y


class BPETokenLanguageModel(nn.Module):
    def __init__(
        self, vocab_size, d_model=64, n_layers=2, context_len=64, n_heads=1, dropout=0.0
    ):
        super().__init__()
        TransformerBlockExp.SelfAttentionExp.validate_dropout(dropout)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.context_len = context_len
        self.n_heads = n_heads
        self.dropout = dropout

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockExp.TransformerBlock(d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        if T > self.context_len:
            raise ValueError(f"Input length {T} exceeds context_len {self.context_len}")
        pos = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        h = self.token_embed(x) + self.pos_embed(pos)
        h = self.embed_dropout(h)
        for block in self.blocks:
            h = block(h, causal=True)
        h = self.final_norm(h)
        logits = self.lm_head(h)  # [B, T, vocab]
        return logits

    def config_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "context_len": self.context_len,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
        }


def get_device():
    if torch.cuda.is_available():
        # Prefer cuda:0 as specified; never select a higher-index device.
        try:
            return torch.device("cuda:0")
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")


def load_text(args):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        p = Path(args.data)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        return text, str(p)

    for url in DEFAULT_DATA_URLS:
        try:
            import urllib.request

            filename = url.split("/")[-1] or "corpus.txt"
            out = data_dir / filename
            if not out.exists():
                print(f"Downloading corpus from {url}...", file=sys.stderr)
                urllib.request.urlretrieve(url, out)
            text = out.read_text(encoding="utf-8", errors="replace")
            return text, str(out)
        except Exception as e:
            print(f"Failed to download {url}: {e}", file=sys.stderr)
            continue

    raise RuntimeError(
        "No local corpus provided and download failed. Supply --data <path> to a text file."
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)  # [B, T, V]
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(1, len(dataloader.dataset))


def bpe_char_weights(tokenizer):
    """Raw-character weight per token id, for the shared nats/char metric."""
    return [tokenizer.token_char_length(i) for i in range(tokenizer.vocab_size)]


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
        total_loss += loss.item() * x.size(0)
    return total_loss / max(1, len(dataloader.dataset))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if device is None:
        device = get_device()
    model.eval()
    # Characters the tokenizer never saw become UNK tokens (public API:
    # tokenizer.unk_id); they are never silently stripped from the prompt.
    ids = tokenizer.encode(prompt)
    if not ids:
        # Empty prompt: seed generation with the UNK token.
        ids = [tokenizer.unk_id]
    ids = torch.tensor(ids, dtype=torch.long)
    if ids.numel() > model.context_len:
        ids = ids[-model.context_len :]
    ids = ids.unsqueeze(0).to(device)  # [1, T]
    generated = 0
    while generated < max_new_tokens:
        ctx = ids[:, -model.context_len :] if ids.size(1) > model.context_len else ids
        logits = model(ctx)
        next_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
        generated += 1
    return tokenizer.decode(ids[0].cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    parser.add_argument("--vocab_size", type=int, default=512, help="Target BPE vocabulary size")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=1, help="Number of attention heads (must divide d_model)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability in [0, 1)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "08_bpe_best.pt"),
        help="Where to save the best checkpoint (model + tokenizer)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.n_heads < 1 or args.d_model % args.n_heads != 0:
        raise ValueError(
            f"n_heads ({args.n_heads}) must be at least 1 and divide d_model ({args.d_model})"
        )
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {args.dropout}")
    if not 0.0 < args.val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {args.val_split}")
    if args.vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {args.vocab_size}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    text, path = load_text(args)
    print(f"Corpus path: {path}")
    print(f"Corpus chars: {len(text):,}")

    # Split the RAW corpus in sequence order before fitting the tokenizer, so
    # merge rules are never learned from held-out validation text.
    split = int(len(text) * (1 - args.val_split))
    train_text = text[:split]
    val_text = text[split:]
    print(f"Raw split: train {len(train_text):,} chars | val {len(val_text):,} chars "
          f"(first {split:,} / last {len(text) - split:,})")

    print(f"\nFitting BPE tokenizer on the {len(train_text):,}-char training portion...")
    t0 = time.time()
    tokenizer = BPETokenizer(vocab_size=args.vocab_size).fit(train_text)
    print(f"Tokenizer fit in {time.time() - t0:.1f}s")
    print(
        f"Vocab size: {tokenizer.vocab_size} "
        f"(alphabet {tokenizer.alphabet_size} + merges {len(tokenizer.merges)} + UNK)"
    )

    t0 = time.time()
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    # Validation text may contain characters absent from the training
    # alphabet; those become UNK tokens (never dropped), so the validation
    # stream covers every held-out character and next-token targets skip no
    # positions.
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    print(f"Encoded train+val in {time.time() - t0:.1f}s")
    val_only_chars = sorted(set(val_text) - set(train_text))
    n_unk_val = int((val_ids == tokenizer.unk_id).sum().item())
    if val_only_chars:
        print(
            f"Validation text: {n_unk_val} UNK token(s) for character(s) outside the "
            f"training alphabet: {val_only_chars!r}"
        )
    # Sanity check: token char lengths must cover the raw text exactly.
    val_chars_covered = sum(tokenizer.token_char_length(int(t)) for t in val_ids.tolist())
    train_chars_covered = sum(tokenizer.token_char_length(int(t)) for t in train_ids.tolist())
    assert val_chars_covered == len(val_text), (val_chars_covered, len(val_text))
    assert train_chars_covered == len(train_text), (train_chars_covered, len(train_text))
    compression_train = len(train_text) / max(1, len(train_ids))
    compression_val = len(val_text) / max(1, len(val_ids))
    print(
        f"Train tokens: {len(train_ids):,} ({compression_train:.2f} chars/token) | "
        f"val tokens: {len(val_ids):,} ({compression_val:.2f} chars/token)"
    )

    train_ds = TokenDataset(train_ids, args.context_len)
    val_ds = TokenDataset(val_ids, args.context_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Config: vocab_size={tokenizer.vocab_size}, d_model={args.d_model}, "
        f"n_layers={args.n_layers}, n_heads={args.n_heads}, dropout={args.dropout}, "
        f"context_len={args.context_len}, batch_size={args.batch_size}, "
        f"lr={args.lr}, seed={args.seed}, val_split={args.val_split}"
    )
    model = BPETokenLanguageModel(
        tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        context_len=args.context_len,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    try:
        sample_untrained = generate(
            model, tokenizer, prompt="To ", max_new_tokens=40,
            temperature=args.temperature, device=device,
        )
        print("\n=== Untrained sample ===")
        print(sample_untrained)
    except Exception as e:
        print(f"Generation error (untrained): {e}")

    print("\n=== Training ===")
    char_weights = bpe_char_weights(tokenizer)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_dl, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_dl, criterion, device)
        # Shared per-character metric (see minigrad.lm_eval); per-token
        # val_loss above remains the checkpoint-selection criterion.
        val_nats = nats_per_character(model, val_ids, char_weights, device=device)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"Epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| val {val_nats['nats_per_char']:.3f} nats/char "
            f"| {time.time() - epoch_start:.1f}s"
        )
    elapsed = time.time() - start_time
    print(f"Best val {best_val:.4f} at epoch {best_epoch} (checkpoint selected)")
    print(f"Training time: {elapsed:.1f}s")

    # Load the best state and score it on the shared per-character metric.
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_nats = nats_per_character(model, val_ids, char_weights, device=device)
    print(
        f"Best model: val {best_val:.4f} per token | "
        f"{best_nats['nats_per_char']:.3f} nats/char "
        f"(sum NLL {best_nats['sum_nll']:.1f} nats over {best_nats['sum_chars']:.0f} chars)"
    )

    # Save best checkpoint together with the tokenizer so the pairing is
    # always explicit and reloadable.
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model.config_dict(),
            "tokenizer": tokenizer.to_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "best_val_nats_per_char": best_nats["nats_per_char"],
            "args": {k: v for k, v in vars(args).items()},
        },
        ckpt_path,
    )
    print(f"Saved best checkpoint + tokenizer to {ckpt_path}")

    # Reload the checkpoint, rebuild the tokenizer and model from it, and
    # generate again: this verifies checkpoint/tokenizer compatibility.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    restored_tokenizer = BPETokenizer.from_dict(ckpt["tokenizer"])
    restored_model = BPETokenLanguageModel(**ckpt["model_config"]).to(device)
    restored_model.load_state_dict(ckpt["model_state_dict"])
    assert restored_tokenizer.encode("To ") == tokenizer.encode("To ")

    try:
        sample_trained = generate(
            restored_model, restored_tokenizer, prompt="To ",
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device,
        )
        print("\n=== Trained sample (loaded from checkpoint) ===")
        print(sample_trained)
    except Exception as e:
        print(f"Generation error (trained): {e}")


if __name__ == "__main__":
    main()
