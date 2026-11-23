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
- All calculations and outputs use float64.
- Do not modify caller-owned arrays.
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
    """Validate inputs for the Softmax backward operation."""
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
    """Validate shapes and dtypes; conventions match Day 08."""
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

    batch_size, seq_len, d_model = x.shape

    if (
        w_q.shape[0] != d_model
        or w_k.shape[0] != d_model
        or w_v.shape[0] != d_model
    ):
        raise ValueError("Q, K, and V projections must start with D.")

    if w_q.shape != w_k.shape:
        raise ValueError("Q and K projections must have the same shape.")

    q_width = w_q.shape[1]
    v_width = w_v.shape[1]

    if q_width % num_heads or v_width % num_heads:
        raise ValueError(
            "Q/K and V projected widths must be divisible by num_heads."
        )

    if w_o.shape[0] != v_width:
        raise ValueError(
            "Output projection must start with H*dv (V projected width)."
        )

    if grad_output.shape != (batch_size, seq_len, w_o.shape[1]):
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

    Return grad_scores with the same shape and dtype float64.

    Invariant:
        sum(grad_scores, axis=-1) is approximately zero because
        Softmax is invariant to adding a constant to an entire score row.
    """
    _validate_softmax_backward_inputs(weights, grad_weights)

    weights64 = np.asarray(weights, dtype=np.float64)
    grad_weights64 = np.asarray(grad_weights, dtype=np.float64)

    row_dot = np.sum(
        weights64 * grad_weights64,
        axis=-1,
        keepdims=True,
    )
    grad_scores = weights64 * (grad_weights64 - row_dot)

    return grad_scores


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

    Forward graph:
        q, k, v = x@w_q, x@w_k, x@w_v
        qh, kh, vh = split_heads_numpy(q/k/v, num_heads)
        scores = qh @ kh.transpose(0,1,3,2) / sqrt(dk)
        weights = softmax_numpy(scores, axis=-1)
        context = weights @ vh
        merged = merge_heads_numpy(context)
        output = merged @ w_o

    Reverse graph, with G = grad_output:
        G_merged = G @ w_o.T
        G_w_o = sum over b,t of outer(merged[b,t], G[b,t])

        G_context = split_heads_numpy(G_merged, num_heads)
        G_weights = G_context @ vh.transpose(0,1,3,2)
        G_vh = weights.transpose(0,1,3,2) @ G_context

        G_scores = softmax_backward_numpy(weights, G_weights)
        G_qh = G_scores @ kh / sqrt(dk)
        G_kh = G_scores.transpose(0,1,3,2) @ qh / sqrt(dk)

        G_q, G_k, G_v = merge_heads_numpy(G_qh, G_kh, G_vh)

        G_w_q = sum over b,t of outer(x[b,t], G_q[b,t])
        G_w_k = sum over b,t of outer(x[b,t], G_k[b,t])
        G_w_v = sum over b,t of outer(x[b,t], G_v[b,t])

        G_x = G_q @ w_q.T + G_k @ w_k.T + G_v @ w_v.T

    Q, K, and V are three branches from the SAME input x, so all
    three paths must contribute to grad_x.

    Return in exact order:
        (grad_x, grad_w_q, grad_w_k, grad_w_v, grad_w_o)

    Do not average parameter gradients over B or T. This follows the
    sum-reduction convention for the scalar objective
        sum(output * grad_output).

    Use NumPy only. Do not mutate caller-owned arrays.
    """
    _validate_attention_backward_inputs(
        grad_output,
        x,
        w_q,
        w_k,
        w_v,
        w_o,
        num_heads,
    )

    # Step 1: convert every input to float64 without mutating the originals.
    grad_output64 = np.asarray(grad_output, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    w_q64 = np.asarray(w_q, dtype=np.float64)
    w_k64 = np.asarray(w_k, dtype=np.float64)
    w_v64 = np.asarray(w_v, dtype=np.float64)
    w_o64 = np.asarray(w_o, dtype=np.float64)

    # Step 2: recompute the Day 08 forward intermediates.
    q = x64 @ w_q64
    k = x64 @ w_k64
    v = x64 @ w_v64

    q_heads = split_heads_numpy(q, num_heads)
    k_heads = split_heads_numpy(k, num_heads)
    v_heads = split_heads_numpy(v, num_heads)

    d_k = q_heads.shape[-1]
    scale = np.sqrt(d_k)

    scores = q_heads @ k_heads.transpose(0, 1, 3, 2)
    scores = scores / scale
    weights = softmax_numpy(scores, axis=-1)

    context = weights @ v_heads
    merged = merge_heads_numpy(context)

    # Step 3: backpropagate through the output projection.
    grad_w_o = np.einsum(
        "bti,bto->io",
        merged,
        grad_output64,
    )
    grad_merged = grad_output64 @ w_o64.T

    # Step 4: split the merged context gradient back into heads.
    grad_context = split_heads_numpy(grad_merged, num_heads)

    # Step 5: backpropagate through context = weights @ values.
    grad_weights = grad_context @ v_heads.transpose(0, 1, 3, 2)
    grad_v_heads = weights.transpose(0, 1, 3, 2) @ grad_context

    # Step 6: backpropagate through the Softmax.
    grad_scores = softmax_backward_numpy(weights, grad_weights)

    # Step 7: backpropagate through scaled QK^T.
    grad_q_heads = (grad_scores @ k_heads) / scale
    grad_k_heads = (
        grad_scores.transpose(0, 1, 3, 2) @ q_heads
    ) / scale

    # Step 8: merge the head gradients into projected feature dimensions.
    grad_q = merge_heads_numpy(grad_q_heads)
    grad_k = merge_heads_numpy(grad_k_heads)
    grad_v = merge_heads_numpy(grad_v_heads)

    # Step 9: accumulate parameter gradients over batch and token axes.
    grad_w_q = np.einsum("btd,bte->de", x64, grad_q)
    grad_w_k = np.einsum("btd,bte->de", x64, grad_k)
    grad_w_v = np.einsum("btd,bte->de", x64, grad_v)

    # Step 10: sum all three paths back to the shared input x.
    grad_x = (
        grad_q @ w_q64.T
        + grad_k @ w_k64.T
        + grad_v @ w_v64.T
    )

    return grad_x, grad_w_q, grad_w_k, grad_w_v, grad_w_o
