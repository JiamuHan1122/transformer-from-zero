
"""Day 08 demonstration: visualize Multi-Head Self-Attention.

Run from the repository root:
    conda activate tfz
    python scripts/day08_multi_head_attention_demo.py

Generated local outputs (ignored by Git):
    results/day08/metrics.json
    results/day08/attention_heatmaps.png
    results/day08/head_difference.png
    results/day08/perturbation_impact.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exercises.week08 import multi_head_attention_numpy  # noqa: E402


def main() -> None:
    output_dir = ROOT / "results" / "day08"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Step 1: Create a reproducible input and projections
    # --------------------------------------------------
    rng = np.random.default_rng(808)

    batch_size = 2
    seq_len = 5
    model_dim = 8
    num_heads = 2
    dk = 4
    dv = 4
    output_dim = 8

    x = rng.normal(size=(batch_size, seq_len, model_dim))

    w_q = rng.normal(scale=0.5, size=(model_dim, num_heads * dk))
    w_k = rng.normal(scale=0.5, size=(model_dim, num_heads * dk))
    w_v = rng.normal(scale=0.5, size=(model_dim, num_heads * dv))
    w_o = rng.normal(scale=0.5, size=(num_heads * dv, output_dim))

    # --------------------------------------------------
    # Step 2: Run YOUR NumPy Multi-Head Attention
    # --------------------------------------------------
    output, weights = multi_head_attention_numpy(
        x, w_q, w_k, w_v, w_o, num_heads
    )

    row_sum_error = float(
        np.max(np.abs(weights.sum(axis=-1) - 1.0))
    )

    head_difference = np.abs(weights[0, 0] - weights[0, 1])
    head_mean_difference = float(np.mean(head_difference))
    head_max_difference = float(np.max(head_difference))

    # --------------------------------------------------
    # Step 3: Perturb one input token
    # --------------------------------------------------
    perturbed_token = 2

    changed_x = x.copy()
    perturbation = np.array(
        [1.0, -0.5, 0.75, 0.25, -0.4, 0.6, -0.2, 0.9],
        dtype=np.float64,
    )
    changed_x[0, perturbed_token, :] += perturbation

    changed_output, changed_weights = multi_head_attention_numpy(
        changed_x, w_q, w_k, w_v, w_o, num_heads
    )

    # Measure output changes at every query-token position.
    output_change_by_token = np.linalg.norm(
        changed_output[0] - output[0], axis=-1
    )

    # Exclude the directly perturbed token to inspect cross-token effects.
    other_tokens = np.arange(seq_len) != perturbed_token
    cross_token_changes = output_change_by_token[other_tokens]

    cross_token_mean_change = float(np.mean(cross_token_changes))
    cross_token_max_change = float(np.max(cross_token_changes))

    # --------------------------------------------------
    # Step 4: Print experiment results
    # --------------------------------------------------
    print("=" * 76)
    print(
        "Transformer From Zero | Day 08 | "
        "Multi-Head Self-Attention Demo"
    )
    print("=" * 76)

    print("\nConfiguration:")
    print(f"Input X:             {x.shape}")
    print(f"Number of heads:     {num_heads}")
    print(f"Key dimension/head:  {dk}")
    print(f"Value dimension/head:{dv}")
    print(f"Output:              {output.shape}")
    print(f"Attention weights:   {weights.shape}")

    print("\nAttention normalization:")
    print(f"Maximum row-sum error: {row_sum_error:.3e}")

    print("\nHead diversity (Batch 0):")
    print(f"Mean absolute difference: {head_mean_difference:.6f}")
    print(f"Maximum absolute difference: {head_max_difference:.6f}")

    print("\nAttention weights for Batch 0:")
    for head_idx in range(num_heads):
        print(f"\nHead {head_idx}:")
        print(
            np.array2string(
                weights[0, head_idx],
                precision=4,
                suppress_small=True,
            )
        )

    print("\nInput perturbation experiment:")
    print(f"Perturbed batch: {0}")
    print(f"Perturbed input token: {perturbed_token}")
    print(f"Perturbation: {perturbation}")

    print("\nOutput L2 change at each query token:")
    for token_idx, change in enumerate(output_change_by_token):
        marker = "  <-- directly perturbed" if token_idx == perturbed_token else ""
        print(f"Token {token_idx}: {change:.6e}{marker}")

    print(
        "\nMean output change at OTHER tokens:",
        f"{cross_token_mean_change:.6e}",
    )
    print(
        "Maximum output change at OTHER tokens:",
        f"{cross_token_max_change:.6e}",
    )

    # --------------------------------------------------
    # Step 5: Save numerical metrics
    # --------------------------------------------------
    metrics = {
        "project": "Transformer From Zero - Day 08",
        "experiment": "Multi-Head Self-Attention visualization",
        "seed": 808,
        "input_shape": list(x.shape),
        "num_heads": num_heads,
        "key_dim_per_head": dk,
        "value_dim_per_head": dv,
        "output_shape": list(output.shape),
        "attention_weights_shape": list(weights.shape),
        "max_attention_row_sum_error": row_sum_error,
        "head_mean_absolute_difference_batch0": head_mean_difference,
        "head_max_absolute_difference_batch0": head_max_difference,
        "perturbed_batch": 0,
        "perturbed_token": perturbed_token,
        "perturbation": perturbation.tolist(),
        "output_l2_change_by_token": output_change_by_token.tolist(),
        "mean_cross_token_output_change": cross_token_mean_change,
        "max_cross_token_output_change": cross_token_max_change,
        "attention_weights_batch0": weights[0].tolist(),
        "attention_weights_after_perturbation_batch0": (
            changed_weights[0].tolist()
        ),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {metrics_path.relative_to(ROOT)}")

    # --------------------------------------------------
    # Step 6: Generate visualizations
    # --------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nMatplotlib is not installed. "
            "Install it with: pip install matplotlib"
        )
        plt = None

    if plt is not None:
        token_positions = np.arange(seq_len)

        # Figure 1: Four attention heatmaps.
        fig, axes = plt.subplots(
            batch_size, num_heads,
            figsize=(10, 8),
            squeeze=False,
        )

        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                ax = axes[batch_idx, head_idx]
                im = ax.imshow(
                    weights[batch_idx, head_idx],
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="equal",
                )
                ax.set_title(
                    f"Batch {batch_idx} | Head {head_idx}"
                )
                ax.set_xlabel("Key token")
                ax.set_ylabel("Query token")
                ax.set_xticks(token_positions)
                ax.set_yticks(token_positions)

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            "Day 08 | Multi-Head Attention Weights",
            fontsize=14,
        )
        fig.tight_layout()

        heatmap_path = output_dir / "attention_heatmaps.png"
        fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {heatmap_path.relative_to(ROOT)}")

        # Figure 2: Difference between two heads in Batch 0.
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            head_difference,
            cmap="magma",
            aspect="equal",
        )
        ax.set_title("Absolute Attention Difference: Head 0 vs Head 1")
        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")
        ax.set_xticks(token_positions)
        ax.set_yticks(token_positions)
        fig.colorbar(im, ax=ax, label="Absolute weight difference")
        fig.tight_layout()

        difference_path = output_dir / "head_difference.png"
        fig.savefig(difference_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {difference_path.relative_to(ROOT)}")

        # Figure 3: Output changes after perturbing one input token.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(token_positions, output_change_by_token)
        ax.axvline(
            perturbed_token,
            linestyle="--",
            label=f"Perturbed input token {perturbed_token}",
        )
        ax.set_title("Cross-Token Output Changes")
        ax.set_xlabel("Query token position")
        ax.set_ylabel("L2 change in output")
        ax.set_xticks(token_positions)
        ax.legend()
        fig.tight_layout()

        perturbation_path = output_dir / "perturbation_impact.png"
        fig.savefig(perturbation_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {perturbation_path.relative_to(ROOT)}")

    # --------------------------------------------------
    # Step 7: Final checks
    # --------------------------------------------------
    valid = (
        output.shape == (batch_size, seq_len, output_dim)
        and weights.shape == (batch_size, num_heads, seq_len, seq_len)
        and np.all(np.isfinite(output))
        and np.all(np.isfinite(weights))
        and np.allclose(weights.sum(axis=-1), 1.0, atol=1e-12)
        and head_mean_difference > 1e-12
        and cross_token_mean_change > 1e-12
    )

    if not valid:
        raise RuntimeError(
            "Day 08 demo checks failed. Inspect the implementation and metrics."
        )

    print("\nSUCCESS: Multi-Head Attention visualization checks passed.")


if __name__ == "__main__":
    main()
