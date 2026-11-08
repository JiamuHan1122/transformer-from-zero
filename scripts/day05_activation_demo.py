"""Day 05 final demo: compare activation functions and their derivatives.

Run from the repository root:
    conda activate tfz
    python -m pip install matplotlib   # once, if needed
    python scripts/day05_activation_demo.py

This demo imports YOUR NumPy implementations from Week 04 and Week 05.
It does not use PyTorch to calculate the plotted curves.

Generated local outputs (ignored by Git):
    results/day05/metrics.json
    results/day05/activation_functions.png
    results/day05/activation_gradients.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exercises.week04 import relu_numpy  # noqa: E402
from exercises.week05 import (  # noqa: E402
    activation_gradient_profiles,
    gelu_tanh_numpy,
    sigmoid_numpy,
    tanh_numpy,
)


def main() -> None:
    output_dir = ROOT / "results" / "day05"
    output_dir.mkdir(parents=True, exist_ok=True)

    # One shared input grid makes the four activation functions comparable.
    x = np.linspace(-6.0, 6.0, 1201, dtype=np.float64)

    activations = {
        "Sigmoid": sigmoid_numpy(x),
        "Tanh": tanh_numpy(x),
        "ReLU": relu_numpy(x),
        "GELU (tanh approximation)": gelu_tanh_numpy(x),
    }
    gradients = activation_gradient_profiles(x)

    expected_keys = {"sigmoid", "tanh", "relu", "gelu"}
    if set(gradients) != expected_keys:
        raise RuntimeError(
            f"Expected gradient keys {expected_keys}, got {set(gradients)}"
        )

    for name, values in activations.items():
        if values.shape != x.shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} produced invalid output or shape.")

    for name, values in gradients.items():
        if values.shape != x.shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} produced invalid gradients or shape.")

    # Sanity-check the mathematically important slopes at x = 0.
    zero_index = int(np.argmin(np.abs(x)))
    zero_slopes = {
        "sigmoid": float(gradients["sigmoid"][zero_index]),
        "tanh": float(gradients["tanh"][zero_index]),
        "relu": float(gradients["relu"][zero_index]),
        "gelu": float(gradients["gelu"][zero_index]),
    }
    expected_zero_slopes = {
        "sigmoid": 0.25,
        "tanh": 1.0,
        "relu": 0.0,
        "gelu": 0.5,
    }
    for name, expected in expected_zero_slopes.items():
        if not np.isclose(zero_slopes[name], expected, atol=1e-10, rtol=0):
            raise RuntimeError(
                f"{name} slope at zero should be {expected}, "
                f"but got {zero_slopes[name]}."
            )

    sample_points = [-4.0, -2.0, 0.0, 2.0, 4.0]
    sample_indices = [int(np.argmin(np.abs(x - value))) for value in sample_points]

    print("=" * 76)
    print("Transformer From Zero | Day 05 | Activation Functions & Gradient Demo")
    print("=" * 76)
    print("Input domain: [-6, 6] | Samples: 1201 | dtype: float64")
    print("\nActivation outputs at selected inputs:")
    print(f"{'x':>7} | {'Sigmoid':>11} | {'Tanh':>11} | {'ReLU':>11} | {'GELU':>11}")
    print("-" * 64)
    for value, index in zip(sample_points, sample_indices):
        print(
            f"{value:7.2f} | "
            f"{activations['Sigmoid'][index]:11.6f} | "
            f"{activations['Tanh'][index]:11.6f} | "
            f"{activations['ReLU'][index]:11.6f} | "
            f"{activations['GELU (tanh approximation)'][index]:11.6f}"
        )

    print("\nActivation derivatives at selected inputs:")
    print(f"{'x':>7} | {'Sigmoid':>11} | {'Tanh':>11} | {'ReLU':>11} | {'GELU':>11}")
    print("-" * 64)
    for value, index in zip(sample_points, sample_indices):
        print(
            f"{value:7.2f} | "
            f"{gradients['sigmoid'][index]:11.6f} | "
            f"{gradients['tanh'][index]:11.6f} | "
            f"{gradients['relu'][index]:11.6f} | "
            f"{gradients['gelu'][index]:11.6f}"
        )

    metrics = {
        "project": "Transformer From Zero - Day 05",
        "experiment": "Activation functions and gradient profiles",
        "input_domain": [-6.0, 6.0],
        "grid_points": int(x.size),
        "dtype": str(x.dtype),
        "gelu_variant": "tanh approximation",
        "derivatives_at_zero": zero_slopes,
        "sample_points": {
            str(value): {
                "outputs": {
                    "sigmoid": float(activations["Sigmoid"][index]),
                    "tanh": float(activations["Tanh"][index]),
                    "relu": float(activations["ReLU"][index]),
                    "gelu_tanh": float(
                        activations["GELU (tanh approximation)"][index]
                    ),
                },
                "derivatives": {
                    name: float(values[index])
                    for name, values in gradients.items()
                },
            }
            for value, index in zip(sample_points, sample_indices)
        },
        "gradient_min_max_on_domain": {
            name: {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in gradients.items()
        },
    }
    metrics_path = output_dir / "metrics.json"
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
        print("\nMatplotlib is not installed. Install it with:")
        print("  python -m pip install matplotlib")
        print("Metrics were saved; PNG figures were not generated.")
        return

    # Figure 1: forward activation functions.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, activations["Sigmoid"], label="Sigmoid", linewidth=2.2)
    ax.plot(x, activations["Tanh"], label="Tanh", linewidth=2.2)
    ax.plot(x, activations["ReLU"], label="ReLU", linewidth=2.2)
    ax.plot(
        x,
        activations["GELU (tanh approximation)"],
        label="GELU (tanh approximation)",
        linewidth=2.2,
    )
    ax.axhline(0.0, linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, linewidth=0.8, alpha=0.5)
    ax.set(
        title="Day 05 | Four Activation Functions",
        xlabel="Input x",
        ylabel="Activation output",
        xlim=(-6, 6),
    )
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    activation_path = output_dir / "activation_functions.png"
    fig.savefig(activation_path, dpi=190)
    plt.close(fig)

    # Figure 2: derivatives on the same scale. Keep negative GELU slopes visible.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, gradients["sigmoid"], label="Sigmoid derivative", linewidth=2.2)
    ax.plot(x, gradients["tanh"], label="Tanh derivative", linewidth=2.2)
    ax.plot(x, gradients["relu"], label="ReLU derivative", linewidth=2.2)
    ax.plot(x, gradients["gelu"], label="GELU derivative", linewidth=2.2)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, linewidth=0.8, alpha=0.5)
    ax.scatter(
        [0.0, 0.0, 0.0, 0.0],
        [
            zero_slopes["sigmoid"],
            zero_slopes["tanh"],
            zero_slopes["relu"],
            zero_slopes["gelu"],
        ],
        s=35,
        zorder=5,
        label="Derivatives at x = 0",
    )
    ax.set(
        title="Day 05 | Activation Gradient Profiles",
        xlabel="Input x",
        ylabel="Local derivative",
        xlim=(-6, 6),
    )
    ax.grid(True, alpha=0.22)
    ax.legend(ncol=2)
    fig.tight_layout()
    gradient_path = output_dir / "activation_gradients.png"
    fig.savefig(gradient_path, dpi=190)
    plt.close(fig)

    print(f"Saved: {activation_path.relative_to(ROOT)}")
    print(f"Saved: {gradient_path.relative_to(ROOT)}")
    print("\nVerified derivatives at x = 0:")
    for name, value in zero_slopes.items():
        print(f"  {name:>7}: {value:.6f}")
    print("\nSUCCESS: all four activation curves and gradient profiles are valid.")


if __name__ == "__main__":
    main()
