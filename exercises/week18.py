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
    """Provided validator: reuse Day 15 contract, then check upstream G_Y."""
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
    """Task A: manually backpropagate through one Day 15 Encoder Block.

    Upstream gradient:
        grad_output = dL/dY, shape (B,T,D)

    Return EXACTLY 13 gradients, all float64, in this order:
        grad_x,
        grad_w_q, grad_w_k, grad_w_v, grad_w_o,
        grad_gamma_attn, grad_beta_attn,
        grad_w1, grad_b1, grad_w2, grad_b2,
        grad_gamma_ffn, grad_beta_ffn.

    Recompute forward intermediate values using the existing components.

    Pre-LN forward:
        n1 = LN1(x)
        a  = MHA(n1)
        h  = x + a
        n2 = LN2(h)
        f  = FFN(n2)
        y  = h + f

    Pre-LN backward in REVERSE:
        - residual y=h+f sends grad_output to both h and f.
        - backpropagate f through FFN, then LN2 to get G_h_from_ffn.
        - add G_h_from_ffn to the identity gradient G_h_skip.
        - backpropagate the combined G_h through MHA(n1), then LN1(x).
        - add the attention-path input gradient to the x identity path.

    Post-LN forward:
        a     = MHA(x)
        h_raw = x + a
        h     = LN1(h_raw)
        f     = FFN(h)
        y_raw = h + f
        y     = LN2(y_raw)

    Post-LN backward in REVERSE:
        - first backpropagate grad_output through LN2(y_raw).
        - residual y_raw=h+f sends G_y_raw to both h and f.
        - backpropagate FFN(h), add resulting G_h to identity path.
        - backpropagate combined G_h through LN1(h_raw).
        - residual h_raw=x+a sends the gradient to x and MHA(x).
        - sum the x identity gradient with MHA's input gradient.

    Existing function signatures:
        layer_norm_backward_numpy(G, input, gamma, eps)
            -> (grad_input, grad_gamma, grad_beta)

        residual_add_backward_numpy(G)
            -> (grad_first_input, grad_second_input)

        ffn_backward_numpy(G, input, w1, b1, w2)
            -> (grad_input, grad_w1, grad_b1, grad_w2, grad_b2)

        multi_head_attention_backward_numpy(
            G, input, w_q, w_k, w_v, w_o, num_heads
        ) -> (grad_input, grad_w_q, grad_w_k, grad_w_v, grad_w_o)

    Important:
    - Day 15 uses w_q/w_k/w_v/w_o all of shape (D,D); keep that rule.
    - Do NOT average parameter gradients over batch/tokens.
    - Never reuse a Pre-LN forward intermediate in the Post-LN branch.
    - Do NOT modify caller-owned arrays, and use float64 throughout.
    - Full encoder STACK backward is intentionally deferred.
    """
    _validate_encoder_block_backward_inputs(
        grad_output=grad_output,
        x=x,
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o,
        num_heads=num_heads,
        gamma_attn=gamma_attn, beta_attn=beta_attn,
        w1=w1, b1=b1, w2=w2, b2=b2,
        gamma_ffn=gamma_ffn, beta_ffn=beta_ffn,
        eps=eps,
        norm_first=norm_first,
    )

    # TODO 1: convert all 14 arrays (G, x, 12 parameters) to float64.
    # TODO 2: recompute the exact Day 15 Pre-LN or Post-LN intermediates.
    # TODO 3: reverse the graph, reusing all four backward helpers above.
    # TODO 4: add gradients at BOTH residual nodes, preserving identity paths.
    # TODO 5: return the 13 gradients in the documented parameter order.
    raise NotImplementedError(
        "Task A: implement complete Encoder Block backward."
    )
