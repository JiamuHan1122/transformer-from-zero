"""Day 18 exercises: complete Transformer Encoder Block backward.

This lesson composes already-verified NumPy backward functions:
    Day 17: Multi-Head Self-Attention
    Day 12: LayerNorm
    Day 13: Residual addition
    Day 14: FFN

Match the exact unmasked, bias-free Day 15 Encoder Block, supporting:
    norm_first=True  -> Pre-LN
    norm_first=False -> Post-LN

No torch or autograd in core code. Return float64, do not modify inputs.
No masks, dropout, positional encoding, final stack norm, or stack backward.

Run:
    python -m pytest -q -m exercise tests/test_week18_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week08 import multi_head_attention_numpy
from exercises.week12 import (
    layer_norm_forward_numpy,
    layer_norm_backward_numpy,
)
from exercises.week13 import (
    residual_add_forward_numpy,
    residual_add_backward_numpy,
)
from exercises.week14 import (
    ffn_forward_numpy,
    ffn_backward_numpy,
)
from exercises.week15 import _validate_encoder_block_inputs
from exercises.week17 import multi_head_attention_backward_numpy


def _validate_encoder_block_backward_inputs(
    grad_output: np.ndarray,
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
    """Reuse Day 15 validation and check upstream gradient G_Y."""
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

    if not isinstance(grad_output, np.ndarray):
        raise ValueError("grad_output must be a NumPy array.")
    if grad_output.shape != x.shape:
        raise ValueError("grad_output must have shape (B, T, D), same as x.")
    if not np.issubdtype(grad_output.dtype, np.floating):
        raise ValueError("grad_output must have a floating-point dtype.")


def transformer_encoder_block_backward_numpy(
    grad_output: np.ndarray,
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
) -> tuple[np.ndarray, ...]:
    """Manually backpropagate through one Day 15 Encoder Block.

    grad_output: dL/dY, shape (B, T, D).

    Returns 13 float64 gradients, in this exact order:
        grad_x,
        grad_w_q, grad_w_k, grad_w_v, grad_w_o,
        grad_gamma_attn, grad_beta_attn,
        grad_w1, grad_b1, grad_w2, grad_b2,
        grad_gamma_ffn, grad_beta_ffn.

    Pre-LN forward:
        n1 = LN1(x)
        a = MHA(n1)
        h = x + a
        n2 = LN2(h)
        f = FFN(n2)
        y = h + f

    Post-LN forward:
        a = MHA(x)
        h_raw = x + a
        h = LN1(h_raw)
        f = FFN(h)
        y_raw = h + f
        y = LN2(y_raw)

    Each residual split sends the upstream gradient into both paths.
    Contributions meeting at the same variable must be summed.
    """
    # ------------------------------------------------------------
    # 1. Validate all inputs before arithmetic.
    # ------------------------------------------------------------
    _validate_encoder_block_backward_inputs(
        grad_output=grad_output,
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

    # ------------------------------------------------------------
    # 2. Promote all arrays to float64; do not mutate caller inputs.
    # ------------------------------------------------------------
    grad_output = np.asarray(grad_output, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    w_q = np.asarray(w_q, dtype=np.float64)
    w_k = np.asarray(w_k, dtype=np.float64)
    w_v = np.asarray(w_v, dtype=np.float64)
    w_o = np.asarray(w_o, dtype=np.float64)

    gamma_attn = np.asarray(gamma_attn, dtype=np.float64)
    beta_attn = np.asarray(beta_attn, dtype=np.float64)
    gamma_ffn = np.asarray(gamma_ffn, dtype=np.float64)
    beta_ffn = np.asarray(beta_ffn, dtype=np.float64)

    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    # ------------------------------------------------------------
    # 3. Pre-LN backward: reverse the exact Pre-LN forward graph.
    # ------------------------------------------------------------
    if norm_first:
        # Forward recomputation (for the exact inputs of backward helpers).
        n1 = layer_norm_forward_numpy(x, gamma_attn, beta_attn, eps)
        a, _ = multi_head_attention_numpy(
            n1, w_q, w_k, w_v, w_o, num_heads
        )
        h = residual_add_forward_numpy(x, a)
        n2 = layer_norm_forward_numpy(h, gamma_ffn, beta_ffn, eps)
        f = ffn_forward_numpy(n2, w1, b1, w2, b2)
        # Final forward: y = h + f. We already have dL/dy.

        # Backward through residual #2: y = h + f.
        grad_h_skip, grad_f = residual_add_backward_numpy(grad_output)

        # Backward through the FFN branch: f = FFN(n2).
        (
            grad_n2,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
        ) = ffn_backward_numpy(grad_f, n2, w1, b1, w2)

        # Backward through LayerNorm #2: n2 = LN2(h).
        (
            grad_h_from_ln2,
            grad_gamma_ffn,
            grad_beta_ffn,
        ) = layer_norm_backward_numpy(
            grad_n2, h, gamma_ffn, eps
        )

        # First gradient MERGE: both paths contribute to h.
        grad_h = grad_h_skip + grad_h_from_ln2

        # Backward through residual #1: h = x + a.
        grad_x_skip, grad_a = residual_add_backward_numpy(grad_h)

        # Backward through Attention: a = MHA(n1).
        (
            grad_n1,
            grad_w_q,
            grad_w_k,
            grad_w_v,
            grad_w_o,
        ) = multi_head_attention_backward_numpy(
            grad_a, n1, w_q, w_k, w_v, w_o, num_heads
        )

        # Backward through LayerNorm #1: n1 = LN1(x).
        (
            grad_x_from_ln1,
            grad_gamma_attn,
            grad_beta_attn,
        ) = layer_norm_backward_numpy(
            grad_n1, x, gamma_attn, eps
        )

        # Second gradient MERGE: both paths contribute to x.
        grad_x = grad_x_skip + grad_x_from_ln1

    # ------------------------------------------------------------
    # 4. Post-LN backward: reverse the exact Post-LN forward graph.
    # ------------------------------------------------------------
    else:
        # Forward recomputation.
        a, _ = multi_head_attention_numpy(
            x, w_q, w_k, w_v, w_o, num_heads
        )
        h_raw = residual_add_forward_numpy(x, a)
        h = layer_norm_forward_numpy(h_raw, gamma_attn, beta_attn, eps)
        f = ffn_forward_numpy(h, w1, b1, w2, b2)
        y_raw = residual_add_forward_numpy(h, f)
        # Final forward: y = LN2(y_raw); dL/dy is grad_output.

        # In Post-LN, LN2 is the FIRST operation in backward.
        (
            grad_y_raw,
            grad_gamma_ffn,
            grad_beta_ffn,
        ) = layer_norm_backward_numpy(
            grad_output, y_raw, gamma_ffn, eps
        )

        # Backward through residual #2: y_raw = h + f.
        grad_h_skip, grad_f = residual_add_backward_numpy(grad_y_raw)

        # Backward through the FFN branch: f = FFN(h).
        (
            grad_h_from_ffn,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
        ) = ffn_backward_numpy(grad_f, h, w1, b1, w2)

        # First gradient MERGE: both paths contribute to h.
        grad_h = grad_h_skip + grad_h_from_ffn

        # Backward through LayerNorm #1: h = LN1(h_raw).
        (
            grad_h_raw,
            grad_gamma_attn,
            grad_beta_attn,
        ) = layer_norm_backward_numpy(
            grad_h, h_raw, gamma_attn, eps
        )

        # Backward through residual #1: h_raw = x + a.
        grad_x_skip, grad_a = residual_add_backward_numpy(grad_h_raw)

        # Backward through Attention: a = MHA(x), NOT MHA(LN1(x)).
        (
            grad_x_from_attn,
            grad_w_q,
            grad_w_k,
            grad_w_v,
            grad_w_o,
        ) = multi_head_attention_backward_numpy(
            grad_a, x, w_q, w_k, w_v, w_o, num_heads
        )

        # Second gradient MERGE: both paths contribute to x.
        grad_x = grad_x_skip + grad_x_from_attn

    # ------------------------------------------------------------
    # 5. Return exactly 13 gradients in Day 15 parameter order.
    # ------------------------------------------------------------
    return (
        grad_x,
        grad_w_q,
        grad_w_k,
        grad_w_v,
        grad_w_o,
        grad_gamma_attn,
        grad_beta_attn,
        grad_w1,
        grad_b1,
        grad_w2,
        grad_b2,
        grad_gamma_ffn,
        grad_beta_ffn,
    )
