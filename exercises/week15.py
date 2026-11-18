
"""Day 15 exercises: Transformer Encoder Block.

Architecture:
    Multi-Head Self-Attention
    Residual Connection
    Layer Normalization
    Position-wise Feed-Forward Network

Input:
    x: (B, T, D)

Attention parameters (bias-free):
    w_q, w_k, w_v, w_o: (D, D)
    num_heads: positive integer dividing D

First LayerNorm:
    gamma_attn, beta_attn: (D,)

FFN parameters:
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
- Use NumPy only for core implementation.
- Reuse previous Day 08, 12, 13 and 14 modules.
- Support both Pre-LN and Post-LN.
- Use float64 for all calculations.
- Do not modify caller-owned arrays.
- Attention is unmasked and bias-free.
- Dropout and full backward are out of scope.

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
    """Validate Encoder Block inputs and dimensions."""

    # Step 1: Collect input arrays
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

    # Step 2: Validate types, ranks and dtypes
    for name, value, ndim in arrays:

        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"{name} must be a NumPy array."
            )

        if value.ndim != ndim:
            raise ValueError(
                f"{name} must be {ndim}D, "
                f"got shape {value.shape}."
            )

        if not np.issubdtype(
            value.dtype, np.floating
        ):
            raise ValueError(
                f"{name} must have a floating-point dtype."
            )

        if any(size == 0 for size in value.shape):
            raise ValueError(
                f"{name} must not have an empty dimension."
            )

    # Step 3: Extract input dimensions
    batch_size, seq_len, d_model = x.shape

    # Step 4: Validate number of heads
    if (
        not isinstance(
            num_heads, (int, np.integer)
        )
        or isinstance(
            num_heads, (bool, np.bool_)
        )
        or num_heads <= 0
    ):
        raise ValueError(
            "num_heads must be a positive integer."
        )

    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible "
            f"by num_heads ({num_heads})."
        )

    # Step 5: Validate Attention projections
    for name, weight in (
        ("w_q", w_q),
        ("w_k", w_k),
        ("w_v", w_v),
        ("w_o", w_o),
    ):
        if weight.shape != (d_model, d_model):
            raise ValueError(
                f"{name} must have shape "
                f"({d_model}, {d_model}), "
                f"got {weight.shape}."
            )

    # Step 6: Validate first LayerNorm
    if (
        gamma_attn.shape != (d_model,)
        or beta_attn.shape != (d_model,)
    ):
        raise ValueError(
            "gamma_attn and beta_attn "
            "must have shape (D,)."
        )

    # Step 7: Validate second LayerNorm
    if (
        gamma_ffn.shape != (d_model,)
        or beta_ffn.shape != (d_model,)
    ):
        raise ValueError(
            "gamma_ffn and beta_ffn "
            "must have shape (D,)."
        )

    # Step 8: Validate FFN dimensions
    d_ff = w1.shape[1]

    if w1.shape[0] != d_model:
        raise ValueError(
            "w1 must have shape (D, D_ff)."
        )

    if b1.shape != (d_ff,):
        raise ValueError(
            f"b1 must have shape ({d_ff},)."
        )

    if w2.shape != (d_ff, d_model):
        raise ValueError(
            f"w2 must have shape "
            f"({d_ff}, {d_model}), "
            f"got {w2.shape}."
        )

    if b2.shape != (d_model,):
        raise ValueError(
            f"b2 must have shape ({d_model},)."
        )

    # Step 9: Validate epsilon
    if (
        not isinstance(
            eps,
            (int, float, np.integer, np.floating)
        )
        or isinstance(eps, (bool, np.bool_))
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError(
            "eps must be a finite positive real number."
        )

    # Step 10: Validate normalization mode
    if not isinstance(
        norm_first, (bool, np.bool_)
    ):
        raise ValueError(
            "norm_first must be a Boolean."
        )


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
    """Compute a complete Transformer Encoder Block.

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
        output:
            Shape (B, T, D), dtype float64.

        attention_weights:
            Shape (B, num_heads, T, T),
            dtype float64.

    Notes:
        No positional encoding, masks, dropout
        or final stack-level LayerNorm are added.

        All operations reuse previously
        implemented modules.
    """

    # ==========================================
    # Step 1: Validate all inputs
    # ==========================================

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

    # ==========================================
    # Step 2: Convert arrays to float64
    # ==========================================

    x = np.asarray(x, dtype=np.float64)

    w_q = np.asarray(w_q, dtype=np.float64)
    w_k = np.asarray(w_k, dtype=np.float64)
    w_v = np.asarray(w_v, dtype=np.float64)
    w_o = np.asarray(w_o, dtype=np.float64)

    gamma_attn = np.asarray(
        gamma_attn, dtype=np.float64
    )
    beta_attn = np.asarray(
        beta_attn, dtype=np.float64
    )

    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    gamma_ffn = np.asarray(
        gamma_ffn, dtype=np.float64
    )
    beta_ffn = np.asarray(
        beta_ffn, dtype=np.float64
    )

    # ==========================================
    # Step 3: Compute Pre-LN or Post-LN
    # ==========================================

    if norm_first:

        # ======================================
        # Pre-LN Encoder Block
        # ======================================

        # 3.1: Normalize input before Attention
        # (B, T, D) -> (B, T, D)

        attn_input = layer_norm_forward_numpy(
            x,
            gamma_attn,
            beta_attn,
            eps,
        )

        # 3.2: Multi-Head Self-Attention
        # output: (B, T, D)
        # weights: (B, H, T, T)

        attn_output, attention_weights = (
            multi_head_attention_numpy(
                attn_input,
                w_q,
                w_k,
                w_v,
                w_o,
                num_heads,
            )
        )

        # 3.3: First Residual Connection
        # H = X + MHA(LN1(X))

        h = residual_add_forward_numpy(
            x,
            attn_output,
        )

        # 3.4: Normalize H before FFN
        # (B, T, D) -> (B, T, D)

        ffn_input = layer_norm_forward_numpy(
            h,
            gamma_ffn,
            beta_ffn,
            eps,
        )

        # 3.5: Position-wise FFN
        # (B, T, D) -> (B, T, D_ff)
        #             -> (B, T, D)

        ffn_output = ffn_forward_numpy(
            ffn_input,
            w1,
            b1,
            w2,
            b2,
        )

        # 3.6: Second Residual Connection
        # Y = H + FFN(LN2(H))

        output = residual_add_forward_numpy(
            h,
            ffn_output,
        )

    else:

        # ======================================
        # Post-LN Encoder Block
        # ======================================

        # 3.1: Multi-Head Self-Attention
        # (B, T, D) -> (B, T, D)

        attn_output, attention_weights = (
            multi_head_attention_numpy(
                x,
                w_q,
                w_k,
                w_v,
                w_o,
                num_heads,
            )
        )

        # 3.2: First Residual Connection
        # H_raw = X + MHA(X)

        h_raw = residual_add_forward_numpy(
            x,
            attn_output,
        )

        # 3.3: First LayerNorm
        # H = LN1(H_raw)

        h = layer_norm_forward_numpy(
            h_raw,
            gamma_attn,
            beta_attn,
            eps,
        )

        # 3.4: Position-wise FFN
        # FFN receives the normalized H.

        ffn_output = ffn_forward_numpy(
            h,
            w1,
            b1,
            w2,
            b2,
        )

        # 3.5: Second Residual Connection
        # Y_raw = H + FFN(H)
        #
        # Important:
        # Residual must use normalized H,
        # not H_raw.

        y_raw = residual_add_forward_numpy(
            h,
            ffn_output,
        )

        # 3.6: Second LayerNorm
        # Y = LN2(Y_raw)

        output = layer_norm_forward_numpy(
            y_raw,
            gamma_ffn,
            beta_ffn,
            eps,
        )

    # ==========================================
    # Step 4: Return outputs
    # ==========================================

    output = np.asarray(
        output,
        dtype=np.float64,
    )

    attention_weights = np.asarray(
        attention_weights,
        dtype=np.float64,
    )

    return output, attention_weights
