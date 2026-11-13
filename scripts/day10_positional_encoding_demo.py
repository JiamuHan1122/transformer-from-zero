"""Day 10 demo: visualize sinusoidal positional encoding."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercises.week10 import (
    add_positional_encoding_numpy,
    sinusoidal_positional_encoding_numpy,
)


def main() -> None:
    rng = np.random.default_rng(1010)
    batch_size, seq_len, d_model = 2, 32, 16

    token_embeddings = rng.normal(size=(batch_size, seq_len, d_model))
    positional_encoding = sinusoidal_positional_encoding_numpy(seq_len, d_model)
    combined = add_positional_encoding_numpy(
        token_embeddings,
        positional_encoding,
    )

    result_dir = Path("results/day10")
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    image = ax.imshow(
        positional_encoding.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_title("Day 10 | Sinusoidal Positional Encoding")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Embedding dimension")
    fig.colorbar(image, ax=ax, label="Encoding value")
    fig.savefig(result_dir / "positional_encoding_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for dim in (0, 1, 2, 3, 6, 7, 14, 15):
        ax.plot(
            np.arange(seq_len),
            positional_encoding[:, dim],
            label=f"dimension {dim}",
        )
    ax.set_title("Day 10 | Positional Encoding by Dimension")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Encoding value")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.savefig(result_dir / "positional_encoding_channels.png", dpi=180)
    plt.close(fig)

    metrics = {
        "token_embeddings_shape": list(token_embeddings.shape),
        "positional_encoding_shape": list(positional_encoding.shape),
        "combined_input_shape": list(combined.shape),
        "positional_encoding_dtype": str(positional_encoding.dtype),
        "position_zero_max_error": float(
            np.max(
                np.abs(
                    positional_encoding[0]
                    - np.tile([0.0, 1.0], d_model // 2)
                )
            )
        ),
        "all_encoding_values_finite": bool(np.isfinite(positional_encoding).all()),
        "mean_absolute_encoding_value": float(np.mean(np.abs(positional_encoding))),
    }

    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))

    assert positional_encoding.shape == (seq_len, d_model)
    assert combined.shape == (batch_size, seq_len, d_model)
    assert positional_encoding.dtype == np.float64
    assert np.isfinite(combined).all()
    assert metrics["position_zero_max_error"] < 1e-12
    print("SUCCESS: Day 10 positional encoding visualization checks passed.")


if __name__ == "__main__":
    main()
