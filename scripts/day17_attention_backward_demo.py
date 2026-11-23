"""Day 17 demo: check gradients of unmasked Multi-Head Attention.

Run:
    python -m scripts.day17_attention_backward_demo

Requires the student to finish both Day 17 NumPy backward functions.
Saves numerical diagnostics and plots under results/day17/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from exercises.week08 import multi_head_attention_numpy
from exercises.week17 import (
    softmax_backward_numpy,
    multi_head_attention_backward_numpy,
)


def _make_case(seed: int = 1717):
    rng = np.random.default_rng(seed)
    batch, tokens, d_model = 2, 5, 8
    num_heads, d_k, d_v, d_out = 2, 4, 3, 8
    scale = 0.25
    x = rng.normal(scale=0.7, size=(batch, tokens, d_model))
    wq = rng.normal(scale=scale, size=(d_model, num_heads * d_k))
    wk = rng.normal(scale=scale, size=(d_model, num_heads * d_k))
    wv = rng.normal(scale=scale, size=(d_model, num_heads * d_v))
    wo = rng.normal(scale=scale, size=(num_heads * d_v, d_out))
    upstream = rng.normal(size=(batch, tokens, d_out))
    return upstream, x, wq, wk, wv, wo, num_heads


def _torch_autograd_grads(args):
    upstream, x, wq, wk, wv, wo, heads = args
    tensors = [
        torch.tensor(a, dtype=torch.float64, requires_grad=True)
        for a in (x, wq, wk, wv, wo)
    ]
    tx, tq, tk, tv, to = tensors
    batch, seq_len, _ = x.shape
    dk = wq.shape[1] // heads
    dv = wv.shape[1] // heads

    def split(z, width):
        return z.reshape(batch, seq_len, heads, width).transpose(1, 2)

    q = split(tx @ tq, dk)
    k = split(tx @ tk, dk)
    v = split(tx @ tv, dv)
    weights = torch.softmax(
        (q @ k.transpose(-2, -1)) / np.sqrt(dk), dim=-1
    )
    context = weights @ v
    merged = context.transpose(1, 2).contiguous().reshape(
        batch, seq_len, heads * dv
    )
    output = merged @ to
    loss = torch.sum(output * torch.from_numpy(upstream))
    loss.backward()
    return (
        [tensor.grad.detach().numpy() for tensor in tensors],
        float(loss.detach().item()),
    )


def main() -> None:
    args = _make_case()
    upstream, x, wq, wk, wv, wo, heads = args

    gradients = multi_head_attention_backward_numpy(*args)
    reference, objective = _torch_autograd_grads(args)
    names = ["x", "w_q", "w_k", "w_v", "w_o"]
    max_abs_errors = {
        name: float(np.max(np.abs(actual - expected)))
        for name, actual, expected in zip(names, gradients, reference)
    }
    relative_l2_errors = {
        name: float(
            np.linalg.norm(actual - expected) /
            max(np.linalg.norm(expected), 1e-12)
        )
        for name, actual, expected in zip(names, gradients, reference)
    }
    gradient_l2_norms = {
        name: float(np.linalg.norm(value))
        for name, value in zip(names, gradients)
    }

    _, attention_weights = multi_head_attention_numpy(
        x, wq, wk, wv, wo, heads
    )
    rng = np.random.default_rng(1718)
    scores = rng.normal(size=(2, 3, 5))
    score_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    score_weights /= score_weights.sum(axis=-1, keepdims=True)
    upstream_softmax = rng.normal(size=score_weights.shape)
    grad_scores = softmax_backward_numpy(score_weights, upstream_softmax)
    softmax_row_sum_error = float(
        np.max(np.abs(grad_scores.sum(axis=-1)))
    )

    # Select a few input coordinates for central finite-difference checks.
    epsilon = 1e-6
    coordinates = [(0, 0, 0), (0, 2, 3), (1, 4, 7)]
    finite_difference_comparisons = []
    for coordinate in coordinates:
        xp = x.copy()
        xm = x.copy()
        xp[coordinate] += epsilon
        xm[coordinate] -= epsilon
        yp, _ = multi_head_attention_numpy(
            xp, wq, wk, wv, wo, heads
        )
        ym, _ = multi_head_attention_numpy(
            xm, wq, wk, wv, wo, heads
        )
        numeric = float(
            (np.sum(yp * upstream) - np.sum(ym * upstream)) /
            (2 * epsilon)
        )
        analytic = float(gradients[0][coordinate])
        finite_difference_comparisons.append({
            "input_coordinate": list(coordinate),
            "numeric_gradient": numeric,
            "analytic_gradient": analytic,
            "absolute_error": abs(analytic - numeric),
        })

    metrics = {
        "input_shape": list(x.shape),
        "num_heads": heads,
        "d_k": wq.shape[1] // heads,
        "d_v": wv.shape[1] // heads,
        "output_shape": list(upstream.shape),
        "gradient_shapes": [list(value.shape) for value in gradients],
        "scalar_objective": objective,
        "gradient_l2_norms": gradient_l2_norms,
        "max_abs_grad_error_vs_autograd": max_abs_errors,
        "relative_l2_grad_error_vs_autograd": relative_l2_errors,
        "softmax_score_gradient_row_sum_max_error":
            softmax_row_sum_error,
        "attention_row_sum_max_error": float(
            np.max(np.abs(attention_weights.sum(axis=-1) - 1.0))
        ),
        "input_finite_difference_checks": finite_difference_comparisons,
        "all_gradients_finite": all(
            np.isfinite(value).all() for value in gradients
        ),
    }

    output_dir = Path("results/day17")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x_positions = np.arange(len(names))
    error_values = [
        max(value, 1e-18) for value in max_abs_errors.values()
    ]
    ax.bar(x_positions, error_values)
    ax.set_xticks(x_positions, names)
    ax.set_yscale("log")
    ax.set_xlabel("Gradient")
    ax.set_ylabel("Maximum absolute error (log scale)")
    ax.set_title("Manual Attention backward vs PyTorch Autograd")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "gradient_error_by_tensor.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    first = axes[0].imshow(gradients[0][0], aspect="auto")
    axes[0].set_title("Manual dL/dX | batch 0")
    axes[0].set_xlabel("Feature dimension")
    axes[0].set_ylabel("Token position")
    fig.colorbar(first, ax=axes[0])

    second = axes[1].imshow(
        gradients[0][0] - reference[0][0], aspect="auto"
    )
    axes[1].set_title("Manual minus Autograd | batch 0")
    axes[1].set_xlabel("Feature dimension")
    axes[1].set_ylabel("Token position")
    fig.colorbar(second, ax=axes[1])
    fig.savefig(output_dir / "input_gradient_heatmaps.png", dpi=180)
    plt.close(fig)

    print("=" * 78)
    print("Transformer From Zero | Day 17 | Attention Backward Demo")
    print("=" * 78)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    assert all(value < 1e-10 for value in max_abs_errors.values())
    assert softmax_row_sum_error < 1e-12
    assert all(
        row["absolute_error"] < 1e-6
        for row in finite_difference_comparisons
    )
    assert metrics["all_gradients_finite"]
    print("\nSUCCESS: Day 17 Attention Backward checks passed.")
    print("Saved: results/day17/metrics.json")
    print("Saved: results/day17/gradient_error_by_tensor.png")
    print("Saved: results/day17/input_gradient_heatmaps.png")


if __name__ == "__main__":
    main()
