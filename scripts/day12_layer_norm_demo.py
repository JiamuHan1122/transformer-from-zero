"""Day 12 demo: inspect LayerNorm statistics and manual gradients.

Run from the repository root:
    python -m scripts.day12_layer_norm_demo

Generated local outputs (ignored by Git):
    results/day12/metrics.json
    results/day12/layer_norm_heatmap.png
    results/day12/layer_norm_statistics.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercises.week12 import (
    layer_norm_backward_numpy,
    layer_norm_forward_numpy,
)


def main() -> None:
    rng = np.random.default_rng(1212)
    batch_size, seq_len, d_model = 2, 5, 8
    eps = 1e-5

    x = rng.normal(loc=1.5, scale=2.0, size=(batch_size, seq_len, d_model))
    x_before = x.copy()
    gamma = np.linspace(0.8, 1.2, d_model, dtype=np.float64)
    beta = np.linspace(-0.2, 0.2, d_model, dtype=np.float64)
    upstream = rng.normal(size=x.shape)

    output = layer_norm_forward_numpy(x, gamma, beta, eps=eps)
    normalized = layer_norm_forward_numpy(
        x,
        np.ones(d_model, dtype=np.float64),
        np.zeros(d_model, dtype=np.float64),
        eps=eps,
    )
    grad_x, grad_gamma, grad_beta = layer_norm_backward_numpy(
        upstream, x, gamma, eps=eps
    )

    normalized_means = np.mean(normalized, axis=-1)
    normalized_vars = np.var(normalized, axis=-1)

    # A token is normalized independently: changing one token cannot affect others.
    changed_x = x.copy()
    changed_x[0, 2, :] += np.linspace(0.5, 2.0, d_model)
    changed_output = layer_norm_forward_numpy(changed_x, gamma, beta, eps=eps)
    other_token_mask = np.ones((batch_size, seq_len), dtype=bool)
    other_token_mask[0, 2] = False
    token_isolation_change = float(
        np.max(np.abs(changed_output - output)[other_token_mask])
    )

    metrics = {
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "normalized_shape": list(normalized.shape),
        "gamma_shape": list(gamma.shape),
        "beta_shape": list(beta.shape),
        "grad_x_shape": list(grad_x.shape),
        "grad_gamma_shape": list(grad_gamma.shape),
        "grad_beta_shape": list(grad_beta.shape),
        "maximum_absolute_normalized_token_mean": float(
            np.max(np.abs(normalized_means))
        ),
        "maximum_normalized_token_variance_error": float(
            np.max(np.abs(normalized_vars - 1.0))
        ),
        "maximum_other_token_change_after_single_token_perturbation": (
            token_isolation_change
        ),
        "all_outputs_and_gradients_finite": bool(
            np.isfinite(output).all()
            and np.isfinite(grad_x).all()
            and np.isfinite(grad_gamma).all()
            and np.isfinite(grad_beta).all()
        ),
        "input_unchanged": bool(np.array_equal(x, x_before)),
    }

    result_dir = Path("results/day12")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("Transformer From Zero | Day 12 | Layer Normalization Demo")
    print("=" * 76)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    input_image = axes[0].imshow(x[0], aspect="auto", cmap="coolwarm")
    axes[0].set_title("Input features (Batch 0)")
    axes[0].set_xlabel("Feature dimension")
    axes[0].set_ylabel("Token position")
    fig.colorbar(input_image, ax=axes[0], label="Value")

    norm_image = axes[1].imshow(normalized[0], aspect="auto", cmap="coolwarm")
    axes[1].set_title("After per-token LayerNorm (Batch 0)")
    axes[1].set_xlabel("Feature dimension")
    axes[1].set_ylabel("Token position")
    fig.colorbar(norm_image, ax=axes[1], label="Normalized value")
    fig.savefig(result_dir / "layer_norm_heatmap.png", dpi=180)
    plt.close(fig)

    token_positions = np.arange(seq_len)
    input_means = np.mean(x[0], axis=-1)
    input_stds = np.std(x[0], axis=-1)
    norm_means = np.mean(normalized[0], axis=-1)
    norm_stds = np.std(normalized[0], axis=-1)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(token_positions, input_means, marker="o", label="Input mean")
    ax.plot(token_positions, input_stds, marker="o", label="Input std")
    ax.plot(token_positions, norm_means, marker="s", label="Normalized mean")
    ax.plot(token_positions, norm_stds, marker="s", label="Normalized std")
    ax.axhline(0.0, linewidth=0.8, alpha=0.5)
    ax.axhline(1.0, linewidth=0.8, alpha=0.5)
    ax.set_title("Day 12 | Per-token Statistics Before and After LayerNorm")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Statistic")
    ax.set_xticks(token_positions)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(result_dir / "layer_norm_statistics.png", dpi=180)
    plt.close(fig)

    assert output.shape == (batch_size, seq_len, d_model)
    assert normalized.shape == x.shape
    assert grad_x.shape == x.shape
    assert grad_gamma.shape == (d_model,)
    assert grad_beta.shape == (d_model,)
    assert metrics["maximum_absolute_normalized_token_mean"] < 1e-12
    assert metrics["maximum_normalized_token_variance_error"] < 1e-3
    assert token_isolation_change == 0.0
    assert metrics["all_outputs_and_gradients_finite"]
    assert metrics["input_unchanged"]
    print("\nSaved: results/day12/metrics.json")
    print("Saved: results/day12/layer_norm_heatmap.png")
    print("Saved: results/day12/layer_norm_statistics.png")
    print("\nSUCCESS: Day 12 LayerNorm checks passed.")


if __name__ == "__main__":
    main()
