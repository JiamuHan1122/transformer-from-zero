"""Day 15 exercises: assemble a complete Transformer Encoder Block.

Architecture:
    Multi-Head Self-Attention + Residual + LayerNorm
    Position-wise FFN + Residual + LayerNorm

Input:
    x: (B, T, D)

Attention parameters (bias-free, reusing Day 08):
    w_q, w_k, w_v, w_o: (D, D)
    num_heads: positive integer dividing D

First LayerNorm:
    gamma_attn, beta_attn: (D,)

FFN parameters (reusing Day 14):
    w1: (D, D_ff)
    b1: (D_ff,)
    w2: (D_ff, D)
    b2: (D,)

Second LayerNorm:
    gamma_ffn, beta_ffn: (D,)

Return:
    output: (B, T, D)
    attention_weights: (B, num_heads, T, T)

Rules:
- Core implementation must use NumPy only.
- Reuse Day 08 multi_head_attention_numpy.
- Reuse Day 12 layer_norm_forward_numpy.
- Reuse Day 13 residual_add_forward_numpy.
- Reuse Day 14 ffn_forward_numpy.
- Support both Pre-LN (norm_first=True) and Post-LN (norm_first=False).
- Use float64 for calculations and return values.
- Do not modify caller-owned arrays.
- This Day 08 attention implementation is unmasked and has no projection biases.
- Dropout and complete block backward are intentionally out of scope today.

Run:
    python -m pytest -q -m exercise tests/test_week15_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week08 import multi_head_attention_numpy
from exercises.week12 import layer_norm_forward_numpy
from exercises.week13 import residual_add_forward_numpy
from exercises.week14 import ffn_forward_numpy


def _validate_encoder_block_inputs(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    gamma_attn: np.ndarray,
    beta_attn: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    gamma_ffn: np.ndarray,
    beta_ffn: np.ndarray,
    eps: float,
    norm_first: bool,
) -> None:
    """Provided helper: validate all arrays and architecture dimensions."""
    arrays = [
        ("x", x, 3),
        ("w_q", w_q, 2),
        ("w_k", w_k, 2),
        ("w_v", w_v, 2),
        ("w_o", w_o, 2),
        ("gamma_attn", gamma_attn, 1),
        ("beta_attn", beta_attn, 1),
        ("w1", w1, 2),
        ("b1", b1, 1),
        ("w2", w2, 2),
        ("b2", b2, 1),
        ("gamma_ffn", gamma_ffn, 1),
        ("beta_ffn", beta_ffn, 1),
    ]

    for name, value, ndim in arrays:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array.")
        if value.ndim != ndim:
            raise ValueError(
                f"{name} must be {ndim}D, got shape {value.shape}."
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(
                f"{name} must have a floating-point dtype."
            )
        if any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must not have an empty dimension.")

    batch_size, seq_len, d_model = x.shape

    if (
        not isinstance(num_heads, (int, np.integer))
        or isinstance(num_heads, (bool, np.bool_))
        or num_heads <= 0
    ):
        raise ValueError("num_heads must be a positive integer.")

    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
        )

    for name, weight in (
        ("w_q", w_q),
        ("w_k", w_k),
        ("w_v", w_v),
        ("w_o", w_o),
    ):
        if weight.shape != (d_model, d_model):
            raise ValueError(
                f"{name} must have shape ({d_model}, {d_model}), "
                f"got {weight.shape}."
            )

    if gamma_attn.shape != (d_model,) or beta_attn.shape != (d_model,):
        raise ValueError("gamma_attn and beta_attn must have shape (D,).")

    if gamma_ffn.shape != (d_model,) or beta_ffn.shape != (d_model,):
        raise ValueError("gamma_ffn and beta_ffn must have shape (D,).")

    d_ff = w1.shape[1]
    if w1.shape[0] != d_model:
        raise ValueError("w1 must have shape (D, D_ff).")
    if b1.shape != (d_ff,):
        raise ValueError(f"b1 must have shape ({d_ff},).")
    if w2.shape != (d_ff, d_model):
        raise ValueError(
            f"w2 must have shape ({d_ff}, {d_model}), got {w2.shape}."
        )
    if b2.shape != (d_model,):
        raise ValueError(f"b2 must have shape ({d_model},).")

    if (
        not isinstance(eps, (int, float, np.integer, np.floating))
        or isinstance(eps, (bool, np.bool_))
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("eps must be a finite positive real number.")

    if not isinstance(norm_first, (bool, np.bool_)):
        raise ValueError("norm_first must be a Boolean.")


def transformer_encoder_block_forward_numpy(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    gamma_attn: np.ndarray,
    beta_attn: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    gamma_ffn: np.ndarray,
    beta_ffn: np.ndarray,
    eps: float = 1e-5,
    norm_first: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Task A: compose one Pre-LN or Post-LN Transformer Encoder Block.

    Pre-LN (norm_first=True):
        attn_input = LN1(x)
        attn_output, weights = MHA(attn_input)
        h = x + attn_output
        ffn_input = LN2(h)
        ffn_output = FFN(ffn_input)
        output = h + ffn_output

    Post-LN (norm_first=False):
        attn_output, weights = MHA(x)
        h = LN1(x + attn_output)
        ffn_output = FFN(h)
        output = LN2(h + ffn_output)

    Returns:
        output: (B, T, D), float64
        attention_weights: (B, num_heads, T, T), float64

    Do not add positional encoding, causal masking, padding masking,
    dropout, or a final stack-level LayerNorm here.
    Do not modify caller-owned arrays.
    """
    _validate_encoder_block_inputs(
        x=x,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=num_heads,
        gamma_attn=gamma_attn,
        beta_attn=beta_attn,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        gamma_ffn=gamma_ffn,
        beta_ffn=beta_ffn,
        eps=eps,
        norm_first=norm_first,
    )

    # Task A: implement the two residual sublayers in the selected order.
    # Reuse the Day 08, Day 12, Day 13 and Day 14 functions imported above.
    raise NotImplementedError(
        "Task A: assemble the Pre-LN / Post-LN Transformer Encoder Block."
    )
