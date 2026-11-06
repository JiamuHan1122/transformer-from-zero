"""Day 03 milestone demo: train y = 3x + 2 using YOUR NumPy exercise code.

From repository root:

    conda activate tfz
    python -m pip install matplotlib   # only needed once for PNG charts
    python scripts/day03_training_demo.py

This script deliberately imports your Week 03 functions rather than replacing
them with a pre-written training implementation.

Generated output (ignored by Git):
    results/day03/metrics.json
    results/day03/loss_curve.png              (if matplotlib is available)
    results/day03/linear_fit_comparison.png   (if matplotlib is available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Running "python scripts/..." can put only the scripts folder on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exercises.week03 import (  # noqa: E402
    gradient_check_linear_mse,
    train_linear_regression_numpy,
)


def main() -> None:
    output_dir = REPO_ROOT / "results" / "day03"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Same model and loss convention as your Week 02/03 exercise tests.
    x = np.linspace(-2.0, 2.0, 41, dtype=np.float64).reshape(-1, 1)
    targets = 3.0 * x + 2.0
    initial_weight = np.array([[-0.5]], dtype=np.float64)
    initial_bias = np.array([0.1], dtype=np.float64)
    learning_rate = 0.1
    steps = 200

    print("=" * 64)
    print("Transformer From Zero | Day 03 | First NumPy Training Demo")
    print("=" * 64)
    print("Ground truth: y = 3x + 2")
    print("Start model:  y = -0.5x + 0.1")
    print(f"Samples: {len(x)} | Learning rate: {learning_rate} | Steps: {steps}")

    # Check that the analytical gradients agree with finite differences.
    rel_w, rel_b = gradient_check_linear_mse(
        x, initial_weight, initial_bias, targets, epsilon=1e-5
    )
    print(f"\nGradient check: weight={rel_w:.3e}, bias={rel_b:.3e}")
    if not (np.isfinite(rel_w) and np.isfinite(rel_b) and
            rel_w < 1e-7 and rel_b < 1e-7):
        raise RuntimeError("Gradient checking failed: inspect your Week 03 implementation.")

    # Train through YOUR Task C function, not torch.optim or an imported model.
    trained_weight, trained_bias, loss_history = train_linear_regression_numpy(
        x, targets, initial_weight, initial_bias,
        learning_rate=learning_rate, steps=steps,
    )

    loss_history = np.asarray(loss_history, dtype=np.float64)
    if loss_history.shape != (steps + 1,) or not np.all(np.isfinite(loss_history)):
        raise RuntimeError(
            f"Expected {steps + 1} finite loss entries, got {loss_history.shape}."
        )
    if trained_weight.shape != initial_weight.shape or trained_bias.shape != initial_bias.shape:
        raise RuntimeError("Trained weight/bias shapes do not match the input shapes.")

    print("\nTraining history:")
    print(f"{'Step':>8}  {'Loss':>18}")
    print("-" * 30)
    checkpoints = [0, 1, 2, 5, 10, 20, 50, 100, steps]
    for step in checkpoints:
        print(f"{step:8d}  {loss_history[step]:18.10e}")

    print("\nBefore / after:")
    print(f"  weight: {float(initial_weight[0, 0]): .8f} -> {float(trained_weight[0, 0]): .8f} (target 3)")
    print(f"  bias:   {float(initial_bias[0]): .8f} -> {float(trained_bias[0]): .8f} (target 2)")
    print(f"  loss:   {float(loss_history[0]):.10e} -> {float(loss_history[-1]):.10e}")

    initial_prediction = x @ initial_weight + initial_bias
    trained_prediction = x @ trained_weight + trained_bias
    probe_x = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    print("\nPredictions:")
    print(f"{'x':>8} {'true y':>12} {'initial':>12} {'trained':>12}")
    for value in probe_x:
        yy = 3.0 * value + 2.0
        yp0 = float(value * initial_weight[0, 0] + initial_bias[0])
        yp1 = float(value * trained_weight[0, 0] + trained_bias[0])
        print(f"{value:8.2f} {yy:12.6f} {yp0:12.6f} {yp1:12.6f}")

    metrics = {
        "project": "Transformer From Zero - Day 03",
        "target_equation": "y = 3x + 2",
        "model": "NumPy linear regression, full-batch gradient descent",
        "sample_count": len(x),
        "learning_rate": learning_rate,
        "steps": steps,
        "initial_weight": float(initial_weight[0, 0]),
        "initial_bias": float(initial_bias[0]),
        "final_weight": float(trained_weight[0, 0]),
        "final_bias": float(trained_bias[0]),
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "gradient_check_relative_error_weight": float(rel_w),
        "gradient_check_relative_error_bias": float(rel_b),
        "loss_checkpoints": {str(i): float(loss_history[i]) for i in checkpoints},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {metrics_path.relative_to(REPO_ROOT)}")

    # Visualization is optional: plain NumPy training and JSON still work
    # if matplotlib is not yet installed.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[INFO] Install matplotlib to generate two PNG charts:")
        print("       python -m pip install matplotlib")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(
            np.arange(steps + 1),
            np.maximum(loss_history, 1e-20),
            linewidth=2,
        )
        ax.set(title="Day 03 | Full-Batch Gradient Descent",
               xlabel="Training step", ylabel="Loss (log scale)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x[:, 0], targets[:, 0], s=20, label="Training samples")
        ax.plot(x[:, 0], initial_prediction[:, 0], "--", linewidth=2,
                label="Before training")
        ax.plot(x[:, 0], trained_prediction[:, 0], linewidth=2,
                label="After training")
        ax.set(title="Day 03 | Learn y = 3x + 2 From Scratch",
               xlabel="Input x", ylabel="Prediction y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "linear_fit_comparison.png", dpi=180)
        plt.close(fig)

        print(f"Saved: {(output_dir / 'loss_curve.png').relative_to(REPO_ROOT)}")
        print(f"Saved: {(output_dir / 'linear_fit_comparison.png').relative_to(REPO_ROOT)}")

    if abs(float(trained_weight[0, 0]) - 3.0) < 1e-3 and \
       abs(float(trained_bias[0]) - 2.0) < 1e-3 and \
       float(loss_history[-1]) < 1e-8:
        print("\nSUCCESS: your NumPy model learned y = 3x + 2.")
    else:
        print("\nWARNING: training completed, but convergence did not meet the demo target.")


if __name__ == "__main__":
    main()
