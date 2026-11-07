"""Day 04: nonlinear fitting showdown — linear model vs hand-built ReLU MLP.

Run from the repository root:
    conda activate tfz
    python scripts/day04_nonlinear_mlp_demo.py

Uses YOUR Day 02, 03 and 04 exercise implementations:
- batch_linear_mse_gradients
- finite_difference_gradient
- two_layer_mlp_numpy
- two_layer_mlp_mse_gradients

No torch autograd, torch optimizers, or replacement MLP training code.

Outputs are local and ignored by Git:
- results/day04/metrics.json
- results/day04/loss_curves.png
- results/day04/nonlinear_fit_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exercises.week02 import batch_linear_mse_gradients  # noqa: E402
from exercises.week03 import finite_difference_gradient  # noqa: E402
from exercises.week04 import (  # noqa: E402
    two_layer_mlp_numpy,
    two_layer_mlp_mse_gradients,
)


def half_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Same loss convention as Weeks 02–04: 0.5 * sum(square(error)) / N."""
    error = predictions - targets
    return float(0.5 * np.sum(error**2) / len(targets))


def mlp_gradient_check(
    x: np.ndarray,
    targets: np.ndarray,
    parameters: list[np.ndarray],
    epsilon: float = 1e-5,
) -> dict[str, float]:
    """Independent central-difference check of ALL four parameter groups."""
    names = ("w1", "b1", "w2", "b2")
    _, *analytical = two_layer_mlp_mse_gradients(x, *parameters, targets)
    errors: dict[str, float] = {}

    for index, (name, analytic, parameter) in enumerate(
        zip(names, analytical, parameters)
    ):
        def loss_for_changed_parameter(value: np.ndarray) -> float:
            candidate = list(parameters)
            candidate[index] = value
            prediction = two_layer_mlp_numpy(x, *candidate)
            return half_mse(prediction, targets)

        numeric = finite_difference_gradient(
            loss_for_changed_parameter, parameter, epsilon=epsilon
        )
        relative_error = np.linalg.norm(analytic - numeric) / max(
            1e-12, np.linalg.norm(analytic) + np.linalg.norm(numeric)
        )
        errors[name] = float(relative_error)

    return errors


def main() -> None:
    # Deterministic, noiseless nonlinear task. Validation points lie strictly
    # between training points; this checks interpolation, not extrapolation.
    x_train = np.linspace(-2.0, 2.0, 81, dtype=np.float64).reshape(-1, 1)
    y_train = x_train**2
    x_valid = (x_train[:-1] + x_train[1:]) / 2.0
    y_valid = x_valid**2

    learning_rate = 0.05
    steps = 3000
    hidden_units = 32
    checkpoints = [0, 1, 2, 10, 50, 100, 500, 1000, 2000, 3000]

    # Same initial baseline for every run.
    linear_weight = np.zeros((1, 1), dtype=np.float64)
    linear_bias = np.zeros(1, dtype=np.float64)

    # Random initial biases create ReLU breakpoints across the x domain.
    rng = np.random.default_rng(2026)
    w1 = rng.normal(0.0, 0.65, size=(1, hidden_units))
    b1 = rng.uniform(-1.0, 1.0, size=(hidden_units,))
    w2 = rng.normal(0.0, 0.1, size=(hidden_units, 1))
    b2 = np.zeros(1, dtype=np.float64)
    params = [w1, b1, w2, b2]

    print("=" * 73)
    print("Transformer From Zero | Day 04 | Nonlinear Fit Comparison")
    print("=" * 73)
    print("Target: y = x^2; interval [-2, 2], no added noise")
    print(f"Training: {len(x_train)} points | Validation: {len(x_valid)} midpoints")
    print(f"Learning rate: {learning_rate} | Updates: {steps} | MLP hidden: {hidden_units}")

    print("\nIndependent MLP gradient checking (your Day 03 finite differences):")
    relative_errors = mlp_gradient_check(x_train, y_train, params)
    for key, error in relative_errors.items():
        print(f"    {key:>2}: {error:.3e}")
    if any(not np.isfinite(err) or err >= 1e-6 for err in relative_errors.values()):
        raise RuntimeError(
            "MLP gradient check failed. Inspect Week 04 backward implementation."
        )

    linear_history: list[float] = []
    mlp_history: list[float] = []

    print("\nTraining loss, same 0.5 * mean squared-error convention:")
    print(f"{'Step':>6} | {'Linear Loss':>17} | {'ReLU MLP Loss':>17}")
    print("-" * 47)

    for step in range(steps + 1):
        # Both models use hand-derived, full-batch gradients and the same LR.
        linear_loss, linear_dw, linear_db = batch_linear_mse_gradients(
            x_train, linear_weight, linear_bias, y_train
        )
        mlp_loss, dw1, db1, dw2, db2 = two_layer_mlp_mse_gradients(
            x_train, *params, y_train
        )
        linear_history.append(float(linear_loss))
        mlp_history.append(float(mlp_loss))

        if step in checkpoints:
            print(f"{step:6d} | {linear_loss:17.9e} | {mlp_loss:17.9e}")

        if step == steps:
            break

        linear_weight -= learning_rate * linear_dw
        linear_bias -= learning_rate * linear_db

        # Simultaneous parameter update, using all gradients computed BEFORE
        # changing parameters. This is NOT an in-place sequential backprop.
        for param, gradient in zip(params, (dw1, db1, dw2, db2)):
            param -= learning_rate * gradient

    train_linear = half_mse(x_train @ linear_weight + linear_bias, y_train)
    valid_linear = half_mse(x_valid @ linear_weight + linear_bias, y_valid)
    train_mlp = half_mse(two_layer_mlp_numpy(x_train, *params), y_train)
    valid_mlp = half_mse(two_layer_mlp_numpy(x_valid, *params), y_valid)

    print("\nFinal result:")
    print(f"  Linear: weight={linear_weight[0,0]:.8f}, bias={linear_bias[0]:.8f}")
    print(f"  Linear: train_loss={train_linear:.9e}, val_loss={valid_linear:.9e}")
    print(f"  MLP:    train_loss={train_mlp:.9e}, val_loss={valid_mlp:.9e}")
    print(f"  Train loss reduction vs linear: {(1 - train_mlp/train_linear)*100:.2f}%")

    save_dir = ROOT / "results" / "day04"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment": "Day 04: linear vs ReLU MLP on y=x^2",
        "seed": 2026,
        "training_points": len(x_train),
        "validation_midpoints": len(x_valid),
        "learning_rate": learning_rate,
        "steps": steps,
        "hidden_units": hidden_units,
        "loss_definition": "0.5 * sum((prediction-target)^2) / N",
        "gradient_check_relative_errors": relative_errors,
        "linear": {
            "weight": float(linear_weight[0, 0]),
            "bias": float(linear_bias[0]),
            "train_loss": train_linear,
            "validation_loss": valid_linear,
        },
        "relu_mlp": {
            "train_loss": train_mlp,
            "validation_loss": valid_mlp,
        },
        "training_loss_checkpoints": {
            str(step): {
                "linear": linear_history[step],
                "mlp": mlp_history[step],
            }
            for step in checkpoints
        },
    }
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {metrics_path.relative_to(ROOT)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed: install with python -m pip install matplotlib")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(np.arange(steps+1), np.maximum(linear_history, 1e-16),
                    label="Linear model")
        ax.semilogy(np.arange(steps+1), np.maximum(mlp_history, 1e-16),
                    label="2-layer ReLU MLP")
        ax.set(xlabel="Full-batch gradient updates", ylabel="Training loss (log scale)",
               title="Day 04 | Linear vs MLP: Training Loss")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "loss_curves.png", dpi=180)
        plt.close(fig)

        x_plot = np.linspace(-2.0, 2.0, 401).reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x_train[:, 0], y_train[:, 0], s=17, alpha=0.4,
                   label="Training samples")
        ax.plot(x_plot[:, 0], x_plot[:, 0]**2, linewidth=2.0,
                label="Ground truth: x²")
        ax.plot(x_plot[:, 0], (x_plot @ linear_weight + linear_bias)[:, 0],
                linestyle="--", linewidth=2.0, label="Linear model")
        ax.plot(x_plot[:, 0], two_layer_mlp_numpy(x_plot, *params)[:, 0],
                linewidth=2.0, label="2-layer ReLU MLP")
        ax.set(xlabel="Input x", ylabel="Prediction y",
               title="Day 04 | Fit y = x² With and Without ReLU")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / "nonlinear_fit_comparison.png", dpi=180)
        plt.close(fig)

        print(f"Saved: {(save_dir / 'loss_curves.png').relative_to(ROOT)}")
        print(f"Saved: {(save_dir / 'nonlinear_fit_comparison.png').relative_to(ROOT)}")

    if train_mlp < 0.01 and valid_mlp < 0.01 and train_linear > 0.5:
        print("\nSUCCESS: the handwritten ReLU MLP fits this nonlinear task much better.")
    else:
        print("\nWARNING: inspect convergence and results before drawing conclusions.")


if __name__ == "__main__":
    main()
