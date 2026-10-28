"""Low-Rank Adaptation (LoRA): train a small low-rank update instead of the weights.

Idea
----
A pretrained linear layer computes ``y = W x + b``. Full fine-tuning learns a
new weight matrix ``W'`` of the same shape, which means every one of its
entries can move - expensive in parameters and easy to forget the original
task.

LoRA keeps the pretrained matrix ``W`` frozen and learns a *low-rank* update
``dW`` with ``rank(dW) = r << min(in_features, out_features)``:

    W' = W + (alpha / r) * B @ A
    y  = W x + b + (alpha / r) * B @ A x
       = base_linear(x) + (alpha / rank) * lora_B(lora_A(x))

with ``A`` of shape ``[r, in_features]`` and ``B`` of shape
``[out_features, r]``. Only ``A`` and ``B`` are trained; ``W`` and ``b`` stay
exactly where the pretrained checkpoint put them.

Dimensions (for a layer mapping ``in_features -> out_features``)::

    x                [B, T, in_features]   (any leading batch dims work)
    lora_A           [r, in_features]
    lora_A(x)        [B, T, r]
    lora_B           [out_features, r]
    lora_B(lora_A(x))[B, T, out_features]
    base_linear(x)   [B, T, out_features]

Initialization
--------------
``lora_A`` is initialized with the same Kaiming-uniform scheme ``nn.Linear``
uses (nonzero, scale-matched to the base layer) and ``lora_B`` is initialized
to **zeros**. Because ``lora_B`` is zero, the low-rank branch contributes
nothing at step 0, so a freshly wrapped layer reproduces the original layer's
output exactly (up to floating-point noise). This is the "identity at
initialization" property the LoRA paper relies on: adaptation starts from the
pretrained function and moves away from it only as the adapter is trained.

Scaling
-------
The update is multiplied by ``alpha / rank``. With ``alpha = rank`` the scale
is 1; with ``alpha = 2 * rank`` it is 2, which lets a larger effective step
size be expressed without changing the learning rate. ``alpha`` defaults to
``rank`` (scale 1).

What this module deliberately does NOT do
----------------------------------------
* It does not touch embeddings, layer norms, or the LM head: only the
  ``nn.Linear`` modules the caller explicitly asks for are wrapped.
* It does not add dropout by default; an optional input dropout on the LoRA
  branch is available but defaults to 0.0 (disabled).
* It does not merge the update back into ``W``: the adapter stays separate,
  which is what makes the parameter saving visible.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "LoRALinear",
    "apply_lora",
    "freeze_module_parameters",
    "lora_parameters",
    "lora_state_dict",
    "load_lora_state_dict",
    "mark_only_lora_trainable",
]


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with a trainable low-rank adapter.

    Args:
        base: the pretrained ``nn.Linear`` to wrap. Its weight and bias are
            frozen immediately (they keep their pretrained values forever).
        rank: rank ``r`` of the update. Must be an ``int >= 1``.
        alpha: scaling numerator; the update is scaled by ``alpha / rank``.
            Defaults to ``rank`` (scale 1). Must be a positive number.
        dropout: optional dropout probability applied to the *input* of the
            LoRA branch (the base branch is untouched). Must be in ``[0, 1)``.

    Forward pass::

        out = base(x) + (alpha / rank) * lora_B(lora_A(dropout(x)))
    """

    def __init__(self, base, rank, alpha=None, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be an nn.Linear, got {type(base).__name__}")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank must be an int, got {type(rank).__name__}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if alpha is None:
            alpha = float(rank)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"alpha must be a positive finite number, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base = base
        # The pretrained weights (and bias) are frozen: only A and B learn.
        freeze_module_parameters(self.base)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Adapter parameters live on the same device/dtype as the frozen
        # weight they modify, so a wrapped layer follows `.to(device)` /
        # `.to(dtype)` calls and never mixes dtypes.
        weight = base.weight
        dtype = weight.dtype
        device = weight.device

        # lora_A: [rank, in_features] -> maps x [..., in] to [..., rank]
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, dtype=dtype, device=device)
        )
        # lora_B: [out_features, rank] -> maps [..., rank] to [..., out]
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, dtype=dtype, device=device)
        )
        # Nonzero A (same scheme as nn.Linear), zero B => identity at init.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        # x: [..., in_features]
        base_out = self.base(x)  # [..., out_features]
        self._check_adapter_matches_base()
        h = self.lora_dropout(x)  # [..., in_features]
        h = F.linear(h, self.lora_A)  # [..., rank]
        h = F.linear(h, self.lora_B)  # [..., out_features]
        return base_out + self.scaling * h

    def _check_adapter_matches_base(self):
        """Fail loudly if the adapter was left behind by a device/dtype move."""
        weight = self.base.weight
        if self.lora_A.device != weight.device or self.lora_B.device != weight.device:
            raise RuntimeError(
                f"LoRA adapter and base layer are on different devices "
                f"({self.lora_A.device} vs {weight.device}); move the whole "
                f"module with .to(device)"
            )
        if self.lora_A.dtype != weight.dtype:
            raise RuntimeError(
                f"LoRA adapter dtype {self.lora_A.dtype} does not match the base "
                f"weight dtype {weight.dtype}; move the whole module with .to(dtype)"
            )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def delta_weight(self):
        """The effective weight update ``(alpha / rank) * B @ A``.

        Shape ``[out_features, in_features]`` - the same shape as the frozen
        base weight. Useful for reporting how far the adapted layer moved.
        """
        return self.scaling * (self.lora_B @ self.lora_A)

    def delta_weight_norm(self):
        """Frobenius norm of :meth:`delta_weight` (0.0 before any update)."""
        return float(self.delta_weight().detach().norm().item())

    def base_weight_norm(self):
        """Frobenius norm of the frozen base weight."""
        return float(self.base.weight.detach().norm().item())

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}"
        )


def freeze_module_parameters(module):
    """Freeze every parameter of ``module`` in place; return the module."""
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def apply_lora(model, target_layers, rank, alpha=None, dropout=0.0):
    """Wrap the ``nn.Linear`` modules of ``model`` whose names match a pattern.

    Args:
        model: any ``nn.Module``.
        target_layers: iterable of module-name patterns. A module matches when
            its qualified name equals the pattern or ends with ``"." + pattern``
            (so ``"attention.query"`` matches ``"blocks.0.attention.query"``).
        rank: LoRA rank for every wrapped layer.
        alpha: LoRA alpha (defaults to ``rank``).
        dropout: optional LoRA-branch input dropout.

    Returns:
        The list of qualified names of the wrapped modules, in the order they
        were replaced. Raises ``ValueError`` if no module matches, so a typo in
        a target name fails loudly instead of silently training nothing.
    """
    if target_layers is None:
        raise ValueError("target_layers must be a non-empty iterable of names")
    patterns = [str(p) for p in target_layers]
    if not patterns:
        raise ValueError("target_layers must be a non-empty iterable of names")

    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(_name_matches(name, pattern) for pattern in patterns):
            continue
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(
            f"no nn.Linear module matched target_layers={patterns!r}; "
            f"available linears: {[n for n, m in model.named_modules() if isinstance(m, nn.Linear)]}"
        )
    return replaced


def _name_matches(name, pattern):
    """True if ``name`` equals ``pattern`` or ends with ``"." + pattern``."""
    return name == pattern or name.endswith("." + pattern)


def mark_only_lora_trainable(model):
    """Freeze everything in ``model`` except the LoRA adapter parameters.

    Returns the number of parameters left trainable.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    n_trainable = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
            n_trainable += module.lora_A.numel() + module.lora_B.numel()
    return n_trainable


def lora_parameters(model):
    """Yield the trainable LoRA parameters of ``model`` (A and B tensors)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def lora_state_dict(model):
    """Adapter-only state dict: only the ``lora_A``/``lora_B`` tensors."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if name.endswith("lora_A") or name.endswith("lora_B")}


def load_lora_state_dict(model, state_dict):
    """Load an adapter-only state dict into ``model`` (strict about the adapter).

    Raises ``RuntimeError`` if the adapter tensors do not match the model's
    wrapped layers - a mismatched adapter is a silent correctness bug
    otherwise.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if k.endswith("lora_A") or k.endswith("lora_B")]
    bad_unexpected = [k for k in unexpected if k.endswith("lora_A") or k.endswith("lora_B")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"LoRA state dict mismatch: missing={bad_missing} unexpected={bad_unexpected}"
        )
    return model
