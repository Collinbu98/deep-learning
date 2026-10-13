"""
Experiment 05 - single-head self-attention from plain tensor operations.

What we are learning
    Attention lets every position in a sequence look at every other position and
    mix their information. This experiment builds one attention head by hand
    with ordinary Linear layers, a matrix multiply and a softmax - no
    nn.MultiheadAttention, no scaled_dot_product_attention, no Transformer - so
    the mechanism and the tensor shapes are visible.

The mechanism
    Given batch-first inputs X of shape [B, T, D] (B examples, T tokens each,
    each token a D-dimensional embedding):

        Q = Wq(X)          [B, T, D]   queries: "what am I looking for?"
        K = Wk(X)          [B, T, D]   keys:    "what do I offer?"
        V = Wv(X)          [B, T, D]   values:  "what do I pass along?"

    Every token's query is compared against every token's key with a dot
    product, giving a [B, T, T] score matrix: entry (i, j) says how much token i
    should attend to token j. Scores are divided by sqrt(D) so they do not grow
    with the embedding size and swamp the softmax. Softmax over the last axis
    turns each row into a set of nonnegative weights that sum to 1. The output
    is the weighted average of the value vectors:

        scores  = Q @ K.transpose(-2, -1) / sqrt(D)   [B, T, T]
        weights = softmax(scores, dim=-1)             [B, T, T]
        output  = weights @ V                         [B, T, D]

    So the output keeps the same sequence length and embedding dimension as the
    input, but each token now contains a blend of the whole sequence.

Causal mode
    In causal (a.k.a. masked) attention a token may look at itself and earlier
    tokens but not at future ones - the constraint a language model needs when
    predicting the next token. We build a lower-triangular boolean mask of shape
    [T, T] where (i, j) is allowed iff j <= i, and set the disallowed
    future-position scores to -inf before the softmax. Because softmax sends
    -inf to weight 0, each row is still a valid distribution over the tokens
    seen so far. Non-causal attention stays the default; pass causal=True to
    turn the mask on.

About the demonstration
    The embeddings below are RANDOM synthetic vectors. Nothing here is trained
    on language and the model has learned no meaningful relationships; the point
    is only to show the shapes, the row-normalized weights, and how information
    flows forward (causal) versus in both directions (non-causal).

Run with:
    .venv/bin/python experiments/05_self_attention.py
"""

import math

import torch
from torch import nn

SEED = 0
BATCH_SIZE = 2
SEQ_LEN = 4
EMBED_DIM = 8


class SelfAttention(nn.Module):
    """Single-head self-attention over batch-first sequences [B, T, D].

    The three projections are learned nn.Linear layers, each mapping D -> D, so
    queries, keys and values all have the same shape as the input. The module
    can return the attention weights as well as the output so the rows can be
    inspected.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_weights=False, causal=False):
        # x: [B, T, D]
        queries = self.query(x)  # [B, T, D]
        keys = self.key(x)  # [B, T, D]
        values = self.value(x)  # [B, T, D]

        # Dot each query with every key. transpose(-2, -1) swaps the last two
        # axes: [B, D, T], so the batched matmul yields [B, T, T]. Scaling by
        # sqrt(D) keeps the scores in a range where softmax stays well-behaved.
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.embed_dim)  # [B, T, T]

        if causal:
            scores = self._apply_causal_mask(scores)

        # Each row of a [T, T] score matrix becomes a distribution over the T
        # tokens: nonnegative, summing to 1.
        weights = torch.softmax(scores, dim=-1)  # [B, T, T]

        # Weighted average of the value vectors. weights [B, T, T] @ V [B, T, D]
        # mixes all tokens and returns [B, T, D] - same shape as the input.
        output = weights @ values  # [B, T, D]

        if return_weights:
            return output, weights
        return output

    @staticmethod
    def _apply_causal_mask(scores):
        """Block attention to future positions by masking scores with -inf.

        For a sequence of length T we allow the score at (i, j) only when
        j <= i (token i may attend to itself and earlier tokens). triu with
        diagonal=1 is True strictly above the main diagonal, i.e. exactly the
        future entries to forbid.
        """
        seq_len = scores.size(-1)
        future = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        # -inf survives as weight 0 after softmax, so each row still sums to 1.
        return scores.masked_fill(future, float("-inf"))


def make_input(seed=SEED):
    """A tiny fixed synthetic sequence: [B, T, D] of random embeddings.

    These numbers have no meaning; they are only a controlled input for studying
    the attention mechanism.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, generator=generator)


def main():
    torch.manual_seed(SEED)

    model = SelfAttention(EMBED_DIM)
    model.eval()

    x = make_input()

    with torch.no_grad():
        output, weights = model(x, return_weights=True)

        # The individual projections, shown here only so every intermediate
        # shape is visible.
        queries = model.query(x)
        keys = model.key(x)
        values = model.value(x)

    print("Single-head self-attention, built from Linear + matmul + softmax")
    print(f"torch {torch.__version__}")
    print("inputs are random synthetic embeddings (no language, no training)")
    print()
    print(f"  X        {tuple(x.shape)}   batch, sequence length, embedding dim")
    print(f"  Q        {tuple(queries.shape)}")
    print(f"  K        {tuple(keys.shape)}")
    print(f"  V        {tuple(values.shape)}")
    print(f"  scores   {tuple(weights.shape)}   one T x T matrix per example")
    print(f"  weights  {tuple(weights.shape)}")
    print(f"  output   {tuple(output.shape)}   same [B, T, D] as the input")

    print()
    print("attention weights for example 0 (each row is a distribution over tokens):")
    row_sums = weights[0].sum(dim=-1)
    for i, row in enumerate(weights[0]):
        cells = "  ".join(f"{v:6.3f}" for v in row.tolist())
        print(f"  token {i}:  {cells}   (sum {row_sums[i]:.4f})")

    max_error = (weights.sum(dim=-1) - 1.0).abs().max().item()
    print()
    print(f"max |row sum - 1| over the whole batch: {max_error:.2e}")

    # Show that attention mixes information: overwrite one input token and the
    # outputs of *every* token can move, because each output is a weighted blend
    # of all value vectors.
    changed = x.clone()
    changed[0, 0] = 3.0
    with torch.no_grad():
        changed_output = model(changed)

    per_token_change = (changed_output[0] - output[0]).abs().max(dim=-1).values

    print()
    print("changing input token 0 of example 0 to [3, 3, ..., 3]:")
    for i, delta in enumerate(per_token_change.tolist()):
        print(f"  output token {i} changed by up to {delta:.4f}")

    # ------------------------------------------------------------------
    # Causal attention. Same inputs and weights, but the mask forbids each
    # token from looking at future positions.
    # ------------------------------------------------------------------
    with torch.no_grad():
        causal_output, causal_weights = model(x, return_weights=True, causal=True)

    print()
    print("=" * 64)
    print("causal attention (token i may only attend to tokens j <= i)")
    print()
    print("causal attention weights for example 0 (zeros above the diagonal):")
    for i, row in enumerate(causal_weights[0]):
        cells = "  ".join(f"{v:6.3f}" for v in row.tolist())
        print(f"  token {i}:  {cells}")

    above_diagonal = torch.triu(
        torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1
    )
    max_above = causal_weights[0][above_diagonal].abs().max().item()
    causal_max_error = (causal_weights.sum(dim=-1) - 1.0).abs().max().item()
    print()
    print(f"max |weight| strictly above the diagonal: {max_above:.2e}  (expected 0)")
    print(f"max |row sum - 1| over the whole batch:   {causal_max_error:.2e}")

    # A future-token perturbation test. Change the token at position k and see
    # which outputs move.
    k = 2
    perturbed = x.clone()
    perturbed[0, k] = 3.0

    with torch.no_grad():
        causal_perturbed = model(perturbed, causal=True)
        noncausal_output = model(x)
        noncausal_perturbed = model(perturbed)

    causal_change = (causal_perturbed[0] - causal_output[0]).abs().max(dim=-1).values
    noncausal_change = (noncausal_perturbed[0] - noncausal_output[0]).abs().max(dim=-1).values

    print()
    print(f"changing input token {k} of example 0 to [3, 3, ..., 3]:")
    print(f"  {'position':>8}  {'causal change':>13}  {'non-causal change':>17}")
    for i in range(SEQ_LEN):
        note = "  (before k: must not change)" if i < k else ""
        print(
            f"  {i:>8}  {causal_change[i].item():>13.6f}"
            f"  {noncausal_change[i].item():>17.6f}{note}"
        )

    print()
    print("causal outputs before position k are unchanged; non-causal outputs")
    print("can change, because there every token also sees future tokens.")


if __name__ == "__main__":
    main()
