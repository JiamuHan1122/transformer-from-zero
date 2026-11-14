"""Day 11 demo: Token IDs -> embeddings + positions -> masked attention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exercises.week11 import prepare_decoder_input_numpy
from exercises.week09 import multi_head_attention_with_mask_numpy


def main() -> None:
    rng = np.random.default_rng(1111)
    batch_size, seq_len, vocab_size, d_model, num_heads = 2, 5, 16, 8, 2
    pad_token_id = 0

    token_ids = np.array(
        [[4, 7, 3, 9, 2], [6, 5, 8, 0, 0]],
        dtype=np.int64,
    )
    embedding_table = rng.normal(size=(vocab_size, d_model))
    positional_encoding = rng.normal(size=(seq_len, d_model))

    x, mask = prepare_decoder_input_numpy(
        token_ids, embedding_table, positional_encoding, pad_token_id
    )

    # Bias-free fused projections; each head has dk=dv=d_model/num_heads.
    w_q = rng.normal(scale=0.2, size=(d_model, d_model))
    w_k = rng.normal(scale=0.2, size=(d_model, d_model))
    w_v = rng.normal(scale=0.2, size=(d_model, d_model))
    w_o = rng.normal(scale=0.2, size=(d_model, d_model))
    output, weights = multi_head_attention_with_mask_numpy(
        x, w_q, w_k, w_v, w_o, num_heads, mask
    )

    blocked_weights = np.where(mask, weights, 0.0)
    row_sums = weights.sum(axis=-1)

    # Mask has one head dimension; broadcast valid-row flags to all heads.
    visible_rows = np.broadcast_to(
        ~np.all(mask, axis=-1),
        row_sums.shape,
    )
    visible_row_error = float(
        np.max(np.abs(row_sums[visible_rows] - 1.0))
    )

    metrics = {
        "token_ids_shape": list(token_ids.shape),
        "embedding_table_shape": list(embedding_table.shape),
        "input_embeddings_shape": list(x.shape),
        "attention_mask_shape": list(mask.shape),
        "attention_weights_shape": list(weights.shape),
        "attention_output_shape": list(output.shape),
        "maximum_blocked_attention_weight": float(blocked_weights.max()),
        "maximum_visible_row_sum_error": visible_row_error,
        "all_outputs_finite": bool(np.isfinite(output).all()),
    }

    result_dir = Path("results/day11")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    assert x.shape == (batch_size, seq_len, d_model)
    assert mask.shape == (batch_size, 1, seq_len, seq_len)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert output.shape == (batch_size, seq_len, d_model)
    assert metrics["maximum_blocked_attention_weight"] == 0.0
    assert visible_row_error < 1e-12
    assert metrics["all_outputs_finite"]
    print("SUCCESS: Day 11 decoder input pipeline checks passed.")


if __name__ == "__main__":
    main()
