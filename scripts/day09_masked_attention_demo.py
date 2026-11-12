"""Day 09 demo: visualize causal masking and masked multi-head attention."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercises.week09 import (
    causal_mask_numpy,
    masked_softmax_numpy,
    multi_head_attention_with_mask_numpy,
    padding_mask_numpy,
)


def main() -> None:
    rng = np.random.default_rng(909)
    batch_size, seq_len, model_dim, num_heads = 2, 5, 8, 2
    x = rng.normal(size=(batch_size, seq_len, model_dim))
    w_q = rng.normal(size=(model_dim, model_dim))
    w_k = rng.normal(size=(model_dim, model_dim))
    w_v = rng.normal(size=(model_dim, model_dim))
    w_o = rng.normal(size=(model_dim, model_dim))

    causal = causal_mask_numpy(seq_len)[None, None, :, :]
    token_ids = np.array([[11, 12, 13, 14, 15], [21, 22, 23, 0, 0]])
    padding = padding_mask_numpy(token_ids, pad_token_id=0)[:, None, None, :]
    combined = causal | padding

    output, weights = multi_head_attention_with_mask_numpy(
        x, w_q, w_k, w_v, w_o, num_heads, combined
    )
    row_sum_error = float(np.max(np.abs(weights.sum(axis=-1)[..., :3] - 1.0)))
    blocked_max = float(np.max(np.abs(weights[np.broadcast_to(combined, weights.shape)])))

    # Causality diagnostic: perturb the final token; earlier outputs must not move.
    changed = x.copy()
    changed[:, -1, :] += np.array([1, -0.5, 0.75, 0.25, -0.4, 0.6, -0.2, 0.9])
    changed_output, _ = multi_head_attention_with_mask_numpy(
        changed, w_q, w_k, w_v, w_o, num_heads, combined
    )
    past_delta = float(np.max(np.abs(output[:, :-1] - changed_output[:, :-1])))

    softmax_scores = np.array([[10000.0, 9999.0, 10001.0]])
    softmax_mask = np.array([[False, False, True]])
    softmax_demo = masked_softmax_numpy(softmax_scores, softmax_mask)

    result_dir = Path("results/day09")
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].imshow(causal[0, 0], cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("Causal blocked positions")
    axes[0].set_xlabel("Key token")
    axes[0].set_ylabel("Query token")

    axes[1].imshow(weights[0, 0], cmap="viridis", vmin=0)
    axes[1].set_title("Batch 0 / Head 0 weights")
    axes[1].set_xlabel("Key token")
    axes[1].set_ylabel("Query token")

    axes[2].imshow(combined[1, 0], cmap="Greys", vmin=0, vmax=1)
    axes[2].set_title("Batch 1: causal + padding")
    axes[2].set_xlabel("Key token")
    axes[2].set_ylabel("Query token")

    fig.suptitle("Day 09 | Masked Multi-Head Attention")
    fig.savefig(result_dir / "masked_attention.png", dpi=180)
    plt.close(fig)

    metrics = {
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "attention_weights_shape": list(weights.shape),
        "maximum_blocked_attention_weight": blocked_max,
        "maximum_visible_row_sum_error_for_non_padding_queries": row_sum_error,
        "maximum_past_output_change_after_future_token_perturbation": past_delta,
        "stable_masked_softmax": softmax_demo.tolist(),
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    assert blocked_max == 0.0
    assert past_delta < 1e-10
    assert np.isfinite(output).all()
    print("SUCCESS: Day 09 masked attention visualization checks passed.")


if __name__ == "__main__":
    main()
