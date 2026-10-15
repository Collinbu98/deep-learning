"""
Experiment 07 - Character-level Transformer language model.

Trains a small decoder-style Transformer LM on a real (or supplied) text
corpus. Reuses TransformerBlock from experiments/06_transformer_block.py.
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


class CharTokenizer:
    def __init__(self, text):
        # Build deterministic vocab from text
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


class CharDataset(Dataset):
    def __init__(self, token_ids, context_len):
        self.token_ids = token_ids
        self.context_len = context_len
        # Number of possible windows
        self.n_windows = max(0, len(token_ids) - context_len - 0)  # sliding
        # count as (len - context_len) for input of len context_len, target shifted
        self.n_windows = len(token_ids) - context_len

    def __len__(self):
        return max(0, self.n_windows)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.context_len]
        y = self.token_ids[idx + 1 : idx + 1 + self.context_len]
        return x, y


class CharLanguageModel(nn.Module):
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


def get_device():
    if torch.cuda.is_available():
        # Prefer cuda:0 as specified
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

    # Try to download
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
    return total_loss / max(1, len(dataloader.dataset)) if hasattr(dataloader.dataset, "__len__") else total_loss


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
        total_loss += loss.item() * x.size(0)
    return total_loss / max(1, len(dataloader.dataset)) if hasattr(dataloader.dataset, "__len__") else total_loss


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if device is None:
        device = get_device()
    model.eval()
    ids = tokenizer.encode(prompt)
    if ids.numel() == 0:
        ids = torch.tensor([tokenizer.stoi.get(" ", 0)], dtype=torch.long)
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    text, path = load_text(args)
    print(f"Corpus path: {path}")
    print(f"Corpus chars: {len(text):,}")
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    token_ids = tokenizer.encode(text)
    n = len(token_ids)
    split = int(n * (1 - args.val_split))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    train_ds = CharDataset(train_ids, args.context_len)
    val_ds = CharDataset(val_ids, args.context_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print(f"Device: {device}")
    if args.n_heads < 1 or args.d_model % args.n_heads != 0:
        raise ValueError(
            f"n_heads ({args.n_heads}) must be at least 1 and divide d_model ({args.d_model})"
        )
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {args.dropout}")
    print(
        f"Config: d_model={args.d_model}, n_layers={args.n_layers}, "
        f"n_heads={args.n_heads}, dropout={args.dropout}, "
        f"context_len={args.context_len}, "
        f"batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}"
    )
    model = CharLanguageModel(
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

    # Print untrained sample
    try:
        sample_untrained = generate(model, tokenizer, prompt="To ", max_new_tokens=40, temperature=0.8, device=device)
        print("\n=== Untrained sample ===")
        print(sample_untrained)
    except Exception as e:
        print(f"Generation error (untrained): {e}")

    print("\n=== Training ===")
    best_val = float("inf")
    best_epoch = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dl, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_dl, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            # Save best
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")
    elapsed = time.time() - start_time
    print(f"Best val {best_val:.4f} at epoch {best_epoch} (checkpoint selected)")
    print(f"Training time: {elapsed:.1f}s")
    # Load best
    if 'best_state' in locals():
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    try:
        sample_trained = generate(model, tokenizer, prompt="To ", max_new_tokens=80, temperature=0.8, device=device)
        print("\n=== Trained sample ===")
        print(sample_trained)
    except Exception as e:
        print(f"Generation error (trained): {e}")


if __name__ == "__main__":
    main()
