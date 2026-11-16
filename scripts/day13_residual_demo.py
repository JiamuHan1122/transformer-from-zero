"""Day 13 demo: visualize the identity path in residual connections.

Run:
    python -m scripts.day13_residual_demo

Local outputs:
    results/day13/metrics.json
    results/day13/gradient_flow_comparison.png

The scalar-Jacobian comparison is illustrative, not a universal guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercises.week13 import (
    residual_add_backward_numpy,
    residual_add_forward_numpy,
    residual_linear_backward_numpy,
    residual_linear_forward_numpy,
)


def main() -> None:
    rng = np.random.default_rng(1313)
    batch_size, seq_len, d_model = 2, 4, 6
    x = rng.normal(size=(batch_size, seq_len, d_model)).astype(np.float64)
    branch = rng.normal(scale=0.2, size=x.shape).astype(np.float64)
    x0, branch0 = x.copy(), branch.copy()

    output = residual_add_forward_numpy(x, branch)
    upstream = rng.normal(size=x.shape).astype(np.float64)
    grad_skip, grad_branch = residual_add_backward_numpy(upstream)

    # Illustrative scalar Jacobians: plain=-0.1, residual=1-0.1=0.9.
    depths = np.arange(21)
    plain_factor, residual_factor = -0.1, 0.9
    base_norm = float(np.linalg.norm(upstream))
    plain_norms = base_norm * np.abs(plain_factor) ** depths
    residual_norms = base_norm * np.abs(residual_factor) ** depths

    weight = -0.1 * np.eye(d_model, dtype=np.float64)
    bias = np.zeros(d_model, dtype=np.float64)
    linear_output = residual_linear_forward_numpy(x, weight, bias)
    grad_x, grad_weight, grad_bias = residual_linear_backward_numpy(
        upstream, x, weight
    )

    metrics = {
        "input_shape": list(x.shape),
        "residual_add_output_shape": list(output.shape),
        "grad_skip_shape": list(grad_skip.shape),
        "grad_branch_shape": list(grad_branch.shape),
        "linear_residual_output_shape": list(linear_output.shape),
        "linear_residual_grad_x_shape": list(grad_x.shape),
        "linear_residual_grad_weight_shape": list(grad_weight.shape),
        "linear_residual_grad_bias_shape": list(grad_bias.shape),
        "addition_forward_matches_reference": bool(np.allclose(output, x + branch)),
        "both_branches_receive_same_upstream_gradient": bool(
            np.array_equal(grad_skip, upstream) and np.array_equal(grad_branch, upstream)
        ),
        "linear_residual_gradient_error": float(np.max(np.abs(grad_x - residual_factor * upstream))),
        "illustrative_plain_local_gradient_factor": plain_factor,
        "illustrative_residual_local_gradient_factor": residual_factor,
        "illustrative_depth_20_plain_gradient_norm": float(plain_norms[-1]),
        "illustrative_depth_20_residual_gradient_norm": float(residual_norms[-1]),
        "all_outputs_and_gradients_finite": bool(
            np.isfinite(output).all() and np.isfinite(grad_skip).all()
            and np.isfinite(grad_branch).all() and np.isfinite(linear_output).all()
            and np.isfinite(grad_x).all() and np.isfinite(grad_weight).all()
            and np.isfinite(grad_bias).all()
        ),
        "inputs_unchanged": bool(
            np.array_equal(x, x0) and np.array_equal(branch, branch0)
        ),
    }

    result_dir = Path("results/day13")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("Transformer From Zero | Day 13 | Residual Connection Demo")
    print("=" * 76)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.semilogy(depths, plain_norms, marker="o", label="Plain path")
    ax.semilogy(depths, residual_norms, marker="s", label="Residual path")
    ax.set_title("Illustrative Gradient Norm Across Repeated Scalar Mappings")
    ax.set_xlabel("Number of repeated mappings")
    ax.set_ylabel("Gradient norm (log scale)")
    ax.set_xticks(np.arange(0, 21, 2))
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(result_dir / "gradient_flow_comparison.png", dpi=180)
    plt.close(fig)

    assert output.shape == x.shape and linear_output.shape == x.shape
    assert np.array_equal(grad_skip, upstream)
    assert np.array_equal(grad_branch, upstream)
    assert metrics["linear_residual_gradient_error"] < 1e-12
    assert metrics["all_outputs_and_gradients_finite"]
    assert metrics["inputs_unchanged"]
    print("\nSaved: results/day13/metrics.json")
    print("Saved: results/day13/gradient_flow_comparison.png")
    print("\nSUCCESS: Day 13 Residual Connection checks passed.")


if __name__ == "__main__":
    main()
