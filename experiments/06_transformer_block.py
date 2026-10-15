"""
Experiment 06 - decoder-style Transformer block.

This experiment assembles existing pieces into a small decoder-style
Transformer block: pre-normalization (LayerNorm), causal self-attention
(reusing the SelfAttention implementation from experiment 05), residual
connections, a second LayerNorm, and a position-wise feed-forward network
(Linear(D, 4D) -> GELU -> Linear(4D, D)).

The structure is:

1. LayerNorm(D) on input x
2. Apply causal self-attention to the normalized x
3. Add result to original x (residual)
4. LayerNorm(D) on the result
5. Apply feed-forward: Linear(D,4D)->GELU->Linear(4D,D)
6. Add result to input of step 5 (residual)

Conceptually: x = x + attention(layer_norm_1(x)), x = x + ff(layer_norm_2(x)).
Sequence length and embedding dimension are preserved. Attention is causal.

The block supports configurable multi-head attention via n_heads (default 1, so
the original single-head behavior is unchanged). d_model must be divisible by
n_heads. Dropout is configurable via the dropout probability (default 0.0, i.e.
disabled) and is applied to the attention weights (inside the attention
module), after the attention output projection (multi-head only), between the
two feed-forward linear layers and after the feed-forward output projection. It
follows the module's train/eval mode.

Inputs are RANDOM synthetic embeddings (no language, no training). The
point is to show how pieces fit together, shapes, parameter counts, and
gradient flow/causality behavior.
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn


def _load_self_attention():
    path = Path(__file__).resolve().parents[1] / "experiments" / "05_self_attention.py"
    spec = importlib.util.spec_from_file_location("self_attention_exp05", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SelfAttentionExp = _load_self_attention()


SEED = 0
BATCH_SIZE = 2
SEQ_LEN = 8
EMBED_DIM = 32


class TransformerBlock(nn.Module):
    """A single decoder-style Transformer block (pre-normalization).

    Args:
        d_model: Embedding dimension D.
        n_heads: Number of attention heads. Defaults to 1 (single-head
            self-attention). Must divide d_model exactly.
        dropout: Dropout probability in [0, 1). Defaults to 0.0 (disabled).

    Input:  x of shape [B, T, D] (batch-first)
    Output: y of shape [B, T, D] (batch-first)
    """

    def __init__(self, d_model, n_heads=1, dropout=0.0):
        super().__init__()
        SelfAttentionExp.validate_dropout(dropout)
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        # Pre-normalization layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        if n_heads == 1:
            self.attention = SelfAttentionExp.SelfAttention(d_model, dropout=dropout)
        else:
            self.attention = SelfAttentionExp.MultiHeadSelfAttention(
                d_model, n_heads, dropout=dropout
            )

        # Feed-forward network. The Sequential keeps the original
        # (Linear, GELU, Linear) layout so existing layer indexing still works;
        # dropout is applied between and after those layers in forward().
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_hidden_dropout = nn.Dropout(dropout)
        self.ffn_output_dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        # x: [B, T, D]
        # 1) Normalize, attend, residual
        norm_x1 = self.layer_norm1(x)
        attn_out = self.attention(norm_x1, causal=causal)
        x = x + attn_out

        # 2) Normalize, FFN (with dropout between and after the linears), residual
        norm_x2 = self.layer_norm2(x)
        ffn_out = self.ffn[0](norm_x2)
        ffn_out = self.ffn[1](ffn_out)
        ffn_out = self.ffn_hidden_dropout(ffn_out)
        ffn_out = self.ffn[2](ffn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        x = x + ffn_out

        return x


def make_input(seed=SEED):
    """Create fixed synthetic input [B, T, D]."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, generator=generator)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torch.manual_seed(SEED)

    model = TransformerBlock(EMBED_DIM)
    model.eval()

    x = make_input()

    with torch.no_grad():
        output = model(x, causal=True)

    print("Decoder-style Transformer block (pre-normalization)")
    print(f"torch {torch.__version__}")
    print("inputs are random synthetic embeddings (no language, no training)")
    print()
    print(f"  D={EMBED_DIM}, B={BATCH_SIZE}, T={SEQ_LEN}")
    print(f"  input shape  : {tuple(x.shape)}  [B, T, D]")
    print(f"  output shape : {tuple(output.shape)}  [B, T, D]")
    print(f"  parameter count (trainable): {count_parameters(model):,}")

    # Check finite values
    all_finite = torch.isfinite(output).all().item()
    print(f"  output has only finite values: {all_finite}")

    # Causality test: perturb a future token; earlier output positions must not change
    k = 2  # future index
    perturbed = x.clone()
    perturbed[0, k] = 3.0

    with torch.no_grad():
        output_perturbed = model(perturbed, causal=True)

    change = (output_perturbed[0] - output[0]).abs().max(dim=-1).values
    earlier_unchanged = (change[:k] < 1e-6).all().item()

    print()
    print(f"causality test: perturb token k={k} of example 0")
    print(f"  earlier positions (0..{k-1}) unchanged within 1e-6: {earlier_unchanged}")
    for i in range(SEQ_LEN):
        note = "  (before k: must not change)" if i < k else ""
        print(f"    pos {i}: max change {change[i].item():.6f}{note}")

    print()
    print("Note: LayerNorm and FFN operate positionwise; causal self-attention")
    print("enforces the information-flow constraint.")


if __name__ == "__main__":
    main()
