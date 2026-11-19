"""Day 16 demo: depth-wise diagnostics for a multi-layer Encoder.

Run from the repository root:
    python -m scripts.day16_encoder_stack_demo

Output files:
    results/day16/metrics.json
    results/day16/representation_norms.png
    results/day16/attention_entropy_by_layer.png
    results/day16/token_perturbation_by_depth.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercises.week16 import encoder_stack_forward_numpy


def build_layer(rng: np.random.Generator, d_model: int, d_ff: int):
    """Generate one independent, reproducible parameter dictionary."""
    scale = 1.0 / np.sqrt(d_model)
    return {
        "w_q": rng.normal(scale=scale, size=(d_model, d_model)),
        "w_k": rng.normal(scale=scale, size=(d_model, d_model)),
        "w_v": rng.normal(scale=scale, size=(d_model, d_model)),
        "w_o": rng.normal(scale=scale, size=(d_model, d_model)),
        "gamma_attn": np.ones(d_model, dtype=np.float64),
        "beta_attn": np.zeros(d_model, dtype=np.float64),
        "w1": rng.normal(scale=0.2, size=(d_model, d_ff)),
        "b1": rng.normal(scale=0.05, size=(d_ff,)),
        "w2": rng.normal(scale=0.2, size=(d_ff, d_model)),
        "b2": rng.normal(scale=0.05, size=(d_model,)),
        "gamma_ffn": np.ones(d_model, dtype=np.float64),
        "beta_ffn": np.zeros(d_model, dtype=np.float64),
    }


def main() -> None:
    rng = np.random.default_rng(1616)
    batch_size, seq_len, d_model, d_ff = 2, 5, 8, 32
    depth, num_heads = 3, 2

    x = rng.normal(size=(batch_size, seq_len, d_model))
    original_input = x.copy()
    layers = [build_layer(rng, d_model, d_ff) for _ in range(depth)]

    pre_out, pre_attn, pre_states = encoder_stack_forward_numpy(
        x, layers, num_heads, norm_first=True
    )
    post_out, post_attn, post_states = encoder_stack_forward_numpy(
        x, layers, num_heads, norm_first=False
    )

    # Per-layer representation magnitudes and update sizes.
    pre_token_norms = np.linalg.norm(pre_states, axis=-1)
    post_token_norms = np.linalg.norm(post_states, axis=-1)
    pre_mean_norm_by_depth = pre_token_norms.mean(axis=(1, 2))
    post_mean_norm_by_depth = post_token_norms.mean(axis=(1, 2))

    pre_update_norm = np.linalg.norm(
        pre_states[1:] - pre_states[:-1], axis=-1
    ).mean(axis=(1, 2))
    post_update_norm = np.linalg.norm(
        post_states[1:] - post_states[:-1], axis=-1
    ).mean(axis=(1, 2))

    # Attention entropy at every layer and head, averaged across B and T.
    def entropy_by_head(weights: np.ndarray) -> np.ndarray:
        entropy = -np.sum(
            weights * np.log(np.maximum(weights, 1e-15)), axis=-1
        )  # (L,B,H,T)
        return entropy.mean(axis=(1, 3))  # (L,H)

    pre_entropy = entropy_by_head(pre_attn)
    post_entropy = entropy_by_head(post_attn)
    pre_row_error = float(np.max(np.abs(pre_attn.sum(axis=-1) - 1.0)))
    post_row_error = float(np.max(np.abs(post_attn.sum(axis=-1) - 1.0)))

    # Probe the response to a controlled change at input token 2.
    changed_input = x.copy()
    changed_input[0, 2, :] += np.linspace(0.4, 1.1, d_model)
    _, _, changed_states = encoder_stack_forward_numpy(
        changed_input, layers, num_heads, norm_first=True
    )
    change_by_depth_and_token = np.linalg.norm(
        changed_states[:, 0] - pre_states[:, 0], axis=-1
    )  # (L+1,T)

    other_token_positions = np.arange(seq_len) != 2
    other_token_max_change_at_output = float(
        np.max(
            np.abs(
                changed_states[-1, 0, other_token_positions]
                - pre_states[-1, 0, other_token_positions]
            )
        )
    )

    unique_parameter_count = sum(
        value.size for layer in layers for value in layer.values()
    )
    metrics = {
        "input_shape": list(x.shape),
        "num_layers": depth,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "final_output_shape": list(pre_out.shape),
        "attention_history_shape": list(pre_attn.shape),
        "hidden_state_history_shape": list(pre_states.shape),
        "total_unique_parameters": int(unique_parameter_count),
        "mean_token_norm_pre_by_depth": pre_mean_norm_by_depth.tolist(),
        "mean_token_norm_post_by_depth": post_mean_norm_by_depth.tolist(),
        "mean_update_norm_pre_by_layer": pre_update_norm.tolist(),
        "mean_update_norm_post_by_layer": post_update_norm.tolist(),
        "attention_entropy_pre_by_layer_head": pre_entropy.tolist(),
        "attention_entropy_post_by_layer_head": post_entropy.tolist(),
        "maximum_attention_row_sum_error_pre": pre_row_error,
        "maximum_attention_row_sum_error_post": post_row_error,
        "pre_post_final_output_l2_difference": float(
            np.linalg.norm(pre_out - post_out)
        ),
        "perturbed_token_index": 2,
        "token_response_l2_by_depth": change_by_depth_and_token.tolist(),
        "other_token_max_absolute_change_at_output":
            other_token_max_change_at_output,
        "all_values_finite": bool(
            all(np.isfinite(v).all() for v in (
                pre_out, post_out, pre_attn, post_attn,
                pre_states, post_states,
                change_by_depth_and_token,
            ))
        ),
        "input_unchanged": bool(np.array_equal(x, original_input)),
    }

    result_dir = Path("results/day16")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    depth_axis = np.arange(depth + 1)
    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    ax.plot(depth_axis, pre_mean_norm_by_depth, marker="o", label="Pre-LN")
    ax.plot(depth_axis, post_mean_norm_by_depth, marker="o", label="Post-LN")
    ax.set_xlabel("Hidden-state depth (0 = input)")
    ax.set_ylabel("Mean token L2 norm")
    ax.set_title("Representation magnitude across Encoder layers")
    ax.set_xticks(depth_axis)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(result_dir / "representation_norms.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    for head in range(num_heads):
        ax.plot(
            np.arange(1, depth + 1),
            pre_entropy[:, head],
            marker="o", label=f"Head {head}",
        )
    ax.axhline(np.log(seq_len), linestyle="--", label="Uniform maximum")
    ax.set_xticks(np.arange(1, depth + 1))
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("Mean attention entropy (nats)")
    ax.set_title("Pre-LN attention entropy at each layer")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(result_dir / "attention_entropy_by_layer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    image = ax.imshow(
        change_by_depth_and_token, aspect="auto", interpolation="nearest"
    )
    ax.set_xlabel("Token position")
    ax.set_ylabel("Depth (0 = input)")
    ax.set_title("Token 2 perturbation response through the stack")
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(depth_axis)
    fig.colorbar(image, ax=ax, label="Output L2 change per token")
    fig.savefig(result_dir / "token_perturbation_by_depth.png", dpi=180)
    plt.close(fig)

    print("=" * 78)
    print("Transformer From Zero | Day 16 | Deep Encoder Demo")
    print("=" * 78)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    assert pre_out.shape == post_out.shape == x.shape
    assert pre_attn.shape == post_attn.shape == (
        depth, batch_size, num_heads, seq_len, seq_len
    )
    assert pre_states.shape == post_states.shape == (
        depth + 1, batch_size, seq_len, d_model
    )
    assert pre_row_error < 1e-12 and post_row_error < 1e-12
    assert other_token_max_change_at_output > 1e-12
    assert metrics["all_values_finite"] and metrics["input_unchanged"]
    assert metrics["pre_post_final_output_l2_difference"] > 1e-12
    print("\nSUCCESS: Day 16 Encoder Stack checks passed.")
    print("Saved outputs to results/day16/")


if __name__ == "__main__":
    main()
