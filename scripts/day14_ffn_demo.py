"""Day 14 demo: verify FFN token independence and feature transformation.

Run from repository root:
    python -m scripts.day14_ffn_demo

Local outputs (not automatically committed):
    results/day14/metrics.json
    results/day14/ffn_feature_heatmaps.png
    results/day14/token_perturbation_impact.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercises.week14 import ffn_forward_numpy, ffn_backward_numpy


def main() -> None:
    rng = np.random.default_rng(1414)
    batch_size, seq_len, d_model, d_ff = 2, 5, 8, 32
    x = rng.normal(size=(batch_size, seq_len, d_model))
    w1 = rng.normal(scale=0.25, size=(d_model, d_ff))
    b1 = rng.normal(scale=0.1, size=(d_ff,))
    w2 = rng.normal(scale=0.2, size=(d_ff, d_model))
    b2 = rng.normal(scale=0.1, size=(d_model,))
    upstream = rng.normal(size=x.shape)

    output = ffn_forward_numpy(x, w1, b1, w2, b2)
    grads = ffn_backward_numpy(upstream, x, w1, b1, w2)

    changed_x = x.copy()
    changed_x[0, 2, :] += np.linspace(0.5, 1.0, d_model)
    changed_output = ffn_forward_numpy(changed_x, w1, b1, w2, b2)
    token_delta = np.linalg.norm(changed_output[0] - output[0], axis=-1)
    unaffected_positions = np.arange(seq_len) != 2

    metrics = {
        "input_shape": list(x.shape),
        "expanded_shape": [batch_size, seq_len, d_ff],
        "output_shape": list(output.shape),
        "gradient_shapes": [list(g.shape) for g in grads],
        "parameter_count": int(w1.size + b1.size + w2.size + b2.size),
        "perturbed_token_index": 2,
        "perturbed_token_output_change": float(token_delta[2]),
        "maximum_other_token_output_change": float(np.max(token_delta[unaffected_positions])),
        "all_values_finite": bool(np.isfinite(output).all() and all(np.isfinite(g).all() for g in grads)),
    }

    result_dir = Path("results/day14")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    a = axes[0].imshow(x[0], aspect="auto")
    axes[0].set_title("Input token features")
    fig.colorbar(a, ax=axes[0])
    b = axes[1].imshow(output[0], aspect="auto")
    axes[1].set_title("Output token features after FFN")
    fig.colorbar(b, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("Feature dimension")
        ax.set_ylabel("Token position")
    fig.savefig(result_dir / "ffn_feature_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(np.arange(seq_len), token_delta)
    ax.set_title("Output change after modifying input token 2")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Output L2 change")
    ax.set_xticks(np.arange(seq_len))
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(result_dir / "token_perturbation_impact.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    assert output.shape == x.shape
    assert metrics["maximum_other_token_output_change"] == 0.0
    assert metrics["perturbed_token_output_change"] > 0.0
    assert metrics["all_values_finite"]
    print("SUCCESS: Day 14 FFN checks passed.")


if __name__ == "__main__":
    main()
