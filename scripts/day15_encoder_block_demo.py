"""Day 15 demo: assemble and compare Pre-LN / Post-LN Encoder Blocks.

Run from repository root:
    python -m scripts.day15_encoder_block_demo

Local outputs (not automatically committed):
    results/day15/metrics.json
    results/day15/attention_weights.png
    results/day15/pre_post_ln_outputs.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercises.week15 import transformer_encoder_block_forward_numpy


def main() -> None:
    rng = np.random.default_rng(1515)
    batch_size, seq_len, d_model = 2, 5, 8
    d_ff, num_heads = 32, 2
    scale = 1.0 / np.sqrt(d_model)

    x = rng.normal(size=(batch_size, seq_len, d_model)).astype(np.float64)
    inputs_before = x.copy()

    w_q = rng.normal(scale=scale, size=(d_model, d_model))
    w_k = rng.normal(scale=scale, size=(d_model, d_model))
    w_v = rng.normal(scale=scale, size=(d_model, d_model))
    w_o = rng.normal(scale=scale, size=(d_model, d_model))

    gamma_attn = np.ones(d_model, dtype=np.float64)
    beta_attn = np.zeros(d_model, dtype=np.float64)
    gamma_ffn = np.ones(d_model, dtype=np.float64)
    beta_ffn = np.zeros(d_model, dtype=np.float64)

    w1 = rng.normal(scale=0.2, size=(d_model, d_ff))
    b1 = rng.normal(scale=0.1, size=d_ff)
    w2 = rng.normal(scale=0.2, size=(d_ff, d_model))
    b2 = rng.normal(scale=0.1, size=d_model)

    params = (
        x, w_q, w_k, w_v, w_o, num_heads,
        gamma_attn, beta_attn,
        w1, b1, w2, b2,
        gamma_ffn, beta_ffn,
    )

    pre_output, pre_weights = transformer_encoder_block_forward_numpy(
        *params, norm_first=True
    )
    post_output, post_weights = transformer_encoder_block_forward_numpy(
        *params, norm_first=False
    )

    changed_x = x.copy()
    changed_x[0, 2, :] += np.linspace(0.4, 1.1, d_model)
    changed_params = (changed_x,) + params[1:]
    changed_output, _ = transformer_encoder_block_forward_numpy(
        *changed_params, norm_first=True
    )
    other_positions = np.arange(seq_len) != 2
    other_token_change = float(
        np.max(np.abs(changed_output[0, other_positions] - pre_output[0, other_positions]))
    )

    attention_row_error = float(
        max(
            np.max(np.abs(pre_weights.sum(axis=-1) - 1.0)),
            np.max(np.abs(post_weights.sum(axis=-1) - 1.0)),
        )
    )
    attention_entropy = -np.sum(
        pre_weights * np.log(np.maximum(pre_weights, 1e-15)),
        axis=-1,
    )

    parameter_count = (
        4 * d_model * d_model
        + 2 * d_model * d_ff + d_ff + d_model
        + 4 * d_model
    )

    metrics = {
        "input_shape": list(x.shape),
        "pre_ln_output_shape": list(pre_output.shape),
        "post_ln_output_shape": list(post_output.shape),
        "attention_weights_shape": list(pre_weights.shape),
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "head_dim": d_model // num_heads,
        "parameter_count": int(parameter_count),
        "pre_post_output_l2_difference": float(np.linalg.norm(pre_output - post_output)),
        "maximum_attention_row_sum_error": attention_row_error,
        "mean_attention_entropy_nats": float(np.mean(attention_entropy)),
        "other_token_max_change_after_perturbing_token_2": other_token_change,
        "all_outputs_finite": bool(
            np.isfinite(pre_output).all()
            and np.isfinite(post_output).all()
            and np.isfinite(pre_weights).all()
            and np.isfinite(post_weights).all()
        ),
        "input_unchanged": bool(np.array_equal(x, inputs_before)),
    }

    result_dir = Path("results/day15")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        1, num_heads, figsize=(5 * num_heads, 4.5), constrained_layout=True
    )
    if num_heads == 1:
        axes = [axes]
    for head, ax in enumerate(axes):
        image = ax.imshow(pre_weights[0, head], vmin=0.0, vmax=1.0)
        ax.set_title(f"Pre-LN attention | head {head}")
        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        fig.colorbar(image, ax=ax)
    fig.savefig(result_dir / "attention_weights.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    left = axes[0].imshow(pre_output[0], aspect="auto")
    axes[0].set_title("Pre-LN Encoder Block output")
    fig.colorbar(left, ax=axes[0])
    right = axes[1].imshow(post_output[0], aspect="auto")
    axes[1].set_title("Post-LN Encoder Block output")
    fig.colorbar(right, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("Feature dimension")
        ax.set_ylabel("Token position")
    fig.savefig(result_dir / "pre_post_ln_outputs.png", dpi=180)
    plt.close(fig)

    print("=" * 78)
    print("Transformer From Zero | Day 15 | Encoder Block Demo")
    print("=" * 78)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    assert pre_output.shape == post_output.shape == x.shape
    assert pre_weights.shape == post_weights.shape == (
        batch_size, num_heads, seq_len, seq_len
    )
    assert attention_row_error < 1e-12
    assert metrics["pre_post_output_l2_difference"] > 1e-8
    assert other_token_change > 1e-10
    assert metrics["all_outputs_finite"]
    assert metrics["input_unchanged"]
    print("\nSaved: results/day15/metrics.json")
    print("Saved: results/day15/attention_weights.png")
    print("Saved: results/day15/pre_post_ln_outputs.png")
    print("\nSUCCESS: Day 15 Encoder Block checks passed.")


if __name__ == "__main__":
    main()
