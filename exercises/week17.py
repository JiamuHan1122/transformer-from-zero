"""Day 17: hand-written backward for unmasked Multi-Head Self-Attention.

Reuse the Day 08 forward implementation:
    output, attention_weights = multi_head_attention_numpy(
        x, w_q, w_k, w_v, w_o, num_heads
    )

Today implements:
    Task A: softmax_backward_numpy(weights, grad_weights)
    Task B: multi_head_attention_backward_numpy(
        grad_output, x, w_q, w_k, w_v, w_o, num_heads
    )

Shapes:
    x:           (B, T, D)
    w_q, w_k:    (D, H*dk)    [same dk]
    w_v:         (D, H*dv)    [dv may differ from dk]
    w_o:         (H*dv, D_out)
    grad_output: (B, T, D_out)

Returned gradients, in exact order:
    grad_x, grad_w_q, grad_w_k, grad_w_v, grad_w_o

Rules:
- Use NumPy only in the two core functions; no torch or autograd.
- Reuse Day 07 softmax_numpy and Day 08 split/merge helpers.
- Recompute forward intermediates; no saved cache is required.
- All calculations and outputs must use float64.
- Do not modify any caller-owned arrays.
- This lesson is unmasked and bias-free, matching Day 08.
- Gradients are for a scalar loss with incoming upstream gradient.
- No residual, LayerNorm, FFN or full Encoder Block backward yet.

Test:
    python -m pytest -q -m exercise tests/test_week17_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week07 import softmax_numpy
from exercises.week08 import split_heads_numpy, merge_heads_numpy


def _validate_softmax_backward_inputs(
    weights: np.ndarray,
    grad_weights: np.ndarray,
) -> None:
    """Provided input checks for Task A."""
    for name, value in (
        ("weights", weights),
        ("grad_weights", grad_weights),
    ):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array.")
        if value.ndim < 1 or any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must have a non-empty shape.")
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must have a floating-point dtype.")
    if weights.shape != grad_weights.shape:
        raise ValueError(
            "weights and grad_weights must have identical shapes."
        )


def _validate_attention_backward_inputs(
    grad_output: np.ndarray,
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
) -> None:
    """Provided input checks; shape conventions match Day 08."""
    arrays = (
        ("grad_output", grad_output, 3),
        ("x", x, 3),
        ("w_q", w_q, 2),
        ("w_k", w_k, 2),
        ("w_v", w_v, 2),
        ("w_o", w_o, 2),
    )
    for name, value, rank in arrays:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array.")
        if value.ndim != rank or any(dim == 0 for dim in value.shape):
            raise ValueError(
                f"{name} must have non-empty rank {rank}, got {value.shape}."
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must have a floating-point dtype.")

    if (
        not isinstance(num_heads, (int, np.integer))
        or isinstance(num_heads, (bool, np.bool_))
        or num_heads <= 0
    ):
        raise ValueError("num_heads must be a positive integer.")

    b, t, d = x.shape
    if w_q.shape[0] != d or w_k.shape[0] != d or w_v.shape[0] != d:
        raise ValueError("Q, K, and V projections must start with D.")
    if w_q.shape != w_k.shape:
        raise ValueError("Q and K projections must have the same shape.")

    q_width, v_width = w_q.shape[1], w_v.shape[1]
    if q_width % num_heads or v_width % num_heads:
        raise ValueError(
            "Q/K and V projected widths must be divisible by num_heads."
        )
    if w_o.shape[0] != v_width:
        raise ValueError(
            "Output projection must start with H*dv (V projected width)."
        )
    if grad_output.shape != (b, t, w_o.shape[1]):
        raise ValueError(
            "grad_output must have shape (B, T, D_out)."
        )


def softmax_backward_numpy(
    weights: np.ndarray,
    grad_weights: np.ndarray,
) -> np.ndarray:
    """Task A: Jacobian-vector product for last-axis Softmax.

    Assume weights = softmax(scores, axis=-1), and incoming
    grad_weights = dL/dweights with the same shape.

    For every row:
        row_dot = sum(weights * grad_weights, axis=-1, keepdims=True)
        grad_scores = weights * (grad_weights - row_dot)

    Return grad_scores with the same shape, dtype float64.

    Key invariant:
        sum(grad_scores, axis=-1) approximately equals zero,
        because Softmax is invariant under adding a constant
        to the entire score row.
    """
    _validate_softmax_backward_inputs(weights, grad_weights)

    # TODO 1: convert weights and grad_weights to float64.
    # TODO 2: compute the weighted row dot product.
    # TODO 3: compute and return grad_scores using the formula above.
    raise NotImplementedError("Task A: implement Softmax backward.")


def multi_head_attention_backward_numpy(
    grad_output: np.ndarray,
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Task B: full backward through Day 08 unmasked Multi-Head Attention.

    Recompute the forward graph:
        q, k, v = x@w_q, x@w_k, x@w_v
        qh, kh, vh = split_heads_numpy(q/k/v, num_heads)
        scores = qh @ kh.transpose(0,1,3,2) / sqrt(dk)
        weights = softmax_numpy(scores, axis=-1)
        context = weights @ vh
        merged = merge_heads_numpy(context)
        output = merged @ w_o

    Reverse graph with incoming G = grad_output:

        G_merged = G @ w_o.T
        G_w_o = sum over b,t of outer(merged[b,t], G[b,t])

        G_context = split_heads_numpy(G_merged, num_heads)
        G_weights = G_context @ vh.transpose(0,1,3,2)
        G_vh = weights.transpose(0,1,3,2) @ G_context

        G_scores = softmax_backward_numpy(weights, G_weights)
        G_qh = G_scores @ kh / sqrt(dk)
        G_kh = G_scores.transpose(0,1,3,2) @ qh / sqrt(dk)

        G_q = merge_heads_numpy(G_qh)
        G_k = merge_heads_numpy(G_kh)
        G_v = merge_heads_numpy(G_vh)

        G_w_q = sum over b,t of outer(x[b,t], G_q[b,t])
        G_w_k = sum over b,t of outer(x[b,t], G_k[b,t])
        G_w_v = sum over b,t of outer(x[b,t], G_v[b,t])

        G_x = G_q @ w_q.T + G_k @ w_k.T + G_v @ w_v.T

    The last line is CRITICAL: Q, K, V are three branches from
    the SAME x, so all three paths contribute to grad_x.

    Return in exact order:
        (grad_x, grad_w_q, grad_w_k, grad_w_v, grad_w_o)

    Do not average over B or T; follow the sum-reduction convention
    for the scalar objective sum(output * grad_output).
    """
    _validate_attention_backward_inputs(
        grad_output, x, w_q, w_k, w_v, w_o, num_heads
    )

    # TODO 1: convert all six arrays to float64 BEFORE computing.
    # TODO 2: recompute q, k, v, split heads, scores, weights,
    #         per-head context and merged context.
    # TODO 3: backpropagate through the final output projection.
    # TODO 4: reshape G_merged into G_context for each head.
    # TODO 5: backpropagate through context = weights @ vh.
    # TODO 6: call Task A to backpropagate through Softmax.
    # TODO 7: compute G_qh and G_kh through scaled QK^T.
    # TODO 8: merge all head gradients back to projected dimensions.
    # TODO 9: calculate four weight gradients (Wq, Wk, Wv, Wo).
    # TODO 10: SUM all three branches into G_x.
    # TODO 11: return five float64 arrays in the order above.
    raise NotImplementedError(
        "Task B: implement Multi-Head Self-Attention backward."
    )
