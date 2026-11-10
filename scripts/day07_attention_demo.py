"""Day 07 demonstration: inspect single-head self-attention.

Run from the repository root after completing exercises/week07.py:
    conda activate tfz
    python scripts/day07_attention_demo.py

This is a forward-mechanism demonstration, not a training run.
Generated local outputs (ignored by Git):
    results/day07/metrics.json
    results/day07/attention_weights.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exercises.week07 import self_attention_numpy  # noqa: E402


def main() -> None:
    output_dir = ROOT / "results" / "day07"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(707)
    batch_size, sequence_length, input_dim = 2, 5, 4
    key_dim, value_dim = 3, 4

    x = rng.normal(size=(batch_size, sequence_length, input_dim))
    w_q = rng.normal(scale=0.5, size=(input_dim, key_dim))
    w_k = rng.normal(scale=0.5, size=(input_dim, key_dim))
    w_v = rng.normal(scale=0.5, size=(input_dim, value_dim))

    output, weights = self_attention_numpy(x, w_q, w_k, w_v)

    print("=" * 76)
    print("Transformer From Zero | Day 07 | Single-Head Self-Attention Demo")
    print("=" * 76)
    print(f"Input X:       {x.shape}")
    print(f"Query:         {(batch_size, sequence_length, key_dim)}")
    print(f"Key:           {(batch_size, sequence_length, key_dim)}")
    print(f"Value:         {(batch_size, sequence_length, value_dim)}")
    print(f"Attention:     {weights.shape}")
    print(f"Output:        {output.shape}")
    print(
        "Maximum row-sum error:",
        f"{np.max(np.abs(weights.sum(axis=-1) - 1.0)):.3e}",
    )

    print("\nAttention weights for sequence 0:")
    print(np.array2string(weights[0], precision=4, suppress_small=True))

    # Perturb one token and inspect whether another query's output changes.
    changed_x = x.copy()
    changed_x[0, 2, :] += np.array([1.0, -0.5, 0.75, 0.25])
    changed_output, _ = self_attention_numpy(changed_x, w_q, w_k, w_v)
    cross_token_change = float(
        np.linalg.norm(changed_output[0, 0] - output[0, 0])
    )
    print(
        "\nChange in output at token 0 after perturbing input token 2:",
        f"{cross_token_change:.6e}",
    )

    metrics = {
        "project": "Transformer From Zero - Day 07",
        "experiment": "Single-head unmasked self-attention forward pass",
        "seed": 707,
        "input_shape": list(x.shape),
        "query_key_dim": key_dim,
        "value_dim": value_dim,
        "attention_shape": list(weights.shape),
        "output_shape": list(output.shape),
        "max_attention_row_sum_error": float(
            np.max(np.abs(weights.sum(axis=-1) - 1.0))
        ),
        "cross_token_output_change": cross_token_change,
        "attention_weights_sequence_0": weights[0].tolist(),
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
        print("Matplotlib is not installed; install it to generate the heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(weights[0], aspect="auto", interpolation="nearest")
        ax.set(
            title="Day 07 | Self-Attention Weights (Sequence 0)",
            xlabel="Key token position",
            ylabel="Query token position",
            xticks=np.arange(sequence_length),
            yticks=np.arange(sequence_length),
        )
        fig.colorbar(image, ax=ax, label="Attention weight")
        fig.tight_layout()
        figure_path = output_dir / "attention_weights.png"
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {figure_path.relative_to(ROOT)}")

    if (
        np.all(np.isfinite(output))
        and np.all(np.isfinite(weights))
        and np.allclose(weights.sum(axis=-1), 1.0, atol=1e-12)
        and output.shape == (batch_size, sequence_length, value_dim)
        and weights.shape == (batch_size, sequence_length, sequence_length)
    ):
        print("\nSUCCESS: single-head self-attention forward pass is valid.")
    else:
        raise RuntimeError("Attention demo checks failed. Inspect Task A-C.")


if __name__ == "__main__":
    main()
