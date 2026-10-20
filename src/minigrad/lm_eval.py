"""Shared language-model evaluation: exact-sum NLL and nats per character.

``nats_per_character`` scores a held-out token stream with a trained causal
LM and reports the summed next-token negative log-likelihood (in nats)
together with the number of *raw characters* those scored targets cover, so
that models with different tokenizations (character-level vs BPE) can be
compared on the same raw-text segment.

Method
------
The stream is walked in non-overlapping windows of ``context_len`` tokens.
For a window starting at position ``s`` the model sees ``ids[s : s+T]`` and
is scored on the targets ``ids[s+1 : s+1+T]``. Summing over all windows,
every position of the stream except position 0 is scored *exactly once*
(position 0 is only ever context, never a target). The character weight of
each scored target token is ``chars_per_token(id)``:

* character-level models: always 1;
* BPE models: 1 for character and UNK tokens, the concatenated length for
  merged tokens (see ``BPETokenizer.token_char_length``). A UNK target
  therefore counts as exactly one raw character - the unseen character it
  stands for.

``nats_per_char = sum_nll / sum_chars`` is thus a genuine per-character
normalization of the *summed* NLL, not an average-token-loss divided by an
average-chars-per-token ratio.

Documented approximations
-------------------------
1. *Truncated history at window boundaries.* Each target is scored with at
   most ``context_len - 1`` preceding tokens; targets at the start of a
   window do not see the tokens before the window (the causal model cannot
   attend past the fed context). A sliding-window evaluation that re-feeds
   every prefix would avoid this but costs ``context_len``x more compute.
2. *The first token of the stream is never scored*, so its characters are
   not counted in the denominator for any model (the same convention is
   applied to every model, so comparisons stay consistent).
3. NLL is accumulated in float32 per window and summed in float64; rounding
   error is negligible for streams up to millions of tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["nats_per_character"]


def _as_char_weight_fn(chars_per_token, vocab_size):
    if callable(chars_per_token):
        fn = chars_per_token
    else:
        seq = list(chars_per_token)
        if len(seq) != vocab_size:
            raise ValueError(
                f"chars_per_token sequence has {len(seq)} entries, expected vocab_size {vocab_size}"
            )
        fn = seq.__getitem__

    def weight(token_id):
        w = fn(token_id)
        if w <= 0:
            raise ValueError(f"chars_per_token({token_id}) = {w}; must be positive")
        return w

    return weight


@torch.no_grad()
def nats_per_character(model, token_ids, chars_per_token, device=None, batch_size=32):
    """Score ``token_ids`` and return summed NLL plus raw-character counts.

    Args:
        model: causal LM exposing ``vocab_size`` and ``context_len`` and
            accepting a ``[B, T]`` LongTensor of token ids.
        token_ids: 1-D sequence of int token ids (held-out stream). Needs at
            least 2 tokens; nothing is scored otherwise.
        chars_per_token: callable ``id -> positive int`` or a sequence of
            length ``model.vocab_size`` giving the raw characters each token
            represents (use 1 per id for character-level models).
        device: device to run the model on (defaults to the model's device).
        batch_size: windows are batched up to this size for scoring.

    Returns:
        dict with keys ``sum_nll`` (float, nats), ``sum_chars`` (float),
        ``n_tokens_scored`` (int), ``nats_per_char`` (float, nan if nothing
        scored), ``n_windows`` (int).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    T = model.context_len
    V = model.vocab_size
    if not torch.is_tensor(token_ids):
        token_ids = torch.tensor(list(token_ids), dtype=torch.long)
    token_ids = token_ids.long().flatten().to(device)
    weight = _as_char_weight_fn(chars_per_token, V)
    weight_t = torch.tensor([weight(i) for i in range(V)], dtype=torch.float64, device=device)

    # Window starts: s, with targets ids[s+1 : s+1+x_len], x_len <= T.
    starts = [s for s in range(0, token_ids.numel() - 1, T)]
    if not starts:
        return {
            "sum_nll": 0.0,
            "sum_chars": 0.0,
            "n_tokens_scored": 0,
            "nats_per_char": float("nan"),
            "n_windows": 0,
        }

    sum_nll = 0.0
    sum_chars = 0.0
    n_scored = 0
    n_windows = 0
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        xs, ys = [], []
        for s in batch_starts:
            x_len = min(T, token_ids.numel() - s - 1)
            xs.append(token_ids[s : s + x_len])
            ys.append(token_ids[s + 1 : s + 1 + x_len])
        # Pad to the longest window in the batch (only the final window can
        # be shorter than T).
        max_len = max(x.numel() for x in xs)
        x_batch = torch.zeros(len(xs), max_len, dtype=torch.long, device=device)
        y_batch = torch.zeros(len(ys), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(xs), max_len, dtype=torch.bool, device=device)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x_batch[i, : x.numel()] = x
            y_batch[i, : y.numel()] = y
            mask[i, : y.numel()] = True
        logits = model(x_batch)  # [B, T, V]
        nll = F.cross_entropy(
            logits.reshape(-1, V).float(), y_batch.reshape(-1), reduction="none"
        )  # [B * T]
        nll = nll.reshape(len(xs), max_len)
        masked_nll = nll[mask]  # scored tokens only
        sum_nll += masked_nll.sum().item()
        w = weight_t[y_batch][mask]
        sum_chars += w.sum().item()
        n_scored += int(mask.sum().item())
        n_windows += len(xs)

    return {
        "sum_nll": sum_nll,
        "sum_chars": sum_chars,
        "n_tokens_scored": n_scored,
        "nats_per_char": sum_nll / sum_chars if sum_chars > 0 else float("nan"),
        "n_windows": n_windows,
    }
