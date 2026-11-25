"""Day 19 demo: Encoder Stack backward diagnostics for Pre-LN/Post-LN.

Requires completed exercises/week19.py. Saves results/day19/*.png/json.
Run: python -m scripts.day19_encoder_stack_backward_demo
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from exercises.week16 import _LAYER_KEYS, encoder_stack_forward_numpy
from exercises.week19 import (
    encoder_stack_backward_numpy,
    aggregate_tied_parameter_gradients_numpy,
)


def _case(seed=1919, depth=3, dim=8, tokens=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, tokens, dim))
    upstream = rng.normal(size=x.shape)
    layers = []
    for i in range(depth):
        width = [12, 16, 20][i]
        s = 1 / np.sqrt(dim)
        layers.append({
            "w_q": rng.normal(scale=s, size=(dim, dim)),
            "w_k": rng.normal(scale=s, size=(dim, dim)),
            "w_v": rng.normal(scale=s, size=(dim, dim)),
            "w_o": rng.normal(scale=s, size=(dim, dim)),
            "gamma_attn": rng.uniform(.85, 1.15, size=dim),
            "beta_attn": rng.normal(scale=.06, size=dim),
            "w1": rng.normal(scale=.17, size=(dim, width)),
            "b1": rng.normal(scale=.06, size=width),
            "w2": rng.normal(scale=.17, size=(width, dim)),
            "b2": rng.normal(scale=.06, size=dim),
            "gamma_ffn": rng.uniform(.85, 1.15, size=dim),
            "beta_ffn": rng.normal(scale=.06, size=dim),
        })
    return x, layers, upstream


def _reference(x, layers, upstream, norm_first):
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    b, t, d = x.shape
    heads, dh = 2, d // 2
    tensor_layers = [
        {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
         for k, v in params.items()}
        for params in layers
    ]
    hidden = [tx]
    current = tx

    for p in tensor_layers:
        def ln(z, gamma_key, beta_key):
            return F.layer_norm(z, (d,), p[gamma_key], p[beta_key], eps=1e-5)

        def attn(z):
            q = (z @ p["w_q"]).reshape(b, t, heads, dh).transpose(1, 2)
            k = (z @ p["w_k"]).reshape(b, t, heads, dh).transpose(1, 2)
            v = (z @ p["w_v"]).reshape(b, t, heads, dh).transpose(1, 2)
            probs = torch.softmax((q @ k.transpose(-1, -2)) / np.sqrt(dh), dim=-1)
            merged = (probs @ v).transpose(1, 2).contiguous().reshape(b, t, d)
            return merged @ p["w_o"]

        def ffn(z):
            return F.gelu(z @ p["w1"] + p["b1"], approximate="tanh") @ p["w2"] + p["b2"]

        if norm_first:
            h = current + attn(ln(current, "gamma_attn", "beta_attn"))
            current = h + ffn(ln(h, "gamma_ffn", "beta_ffn"))
        else:
            h = ln(current + attn(current), "gamma_attn", "beta_attn")
            current = ln(h + ffn(h), "gamma_ffn", "beta_ffn")
        current.retain_grad()
        hidden.append(current)

    loss = (current * torch.from_numpy(upstream)).sum()
    loss.backward()
    return (
        tx.grad.detach().numpy(),
        [
            {key: p[key].grad.detach().numpy() for key in _LAYER_KEYS}
            for p in tensor_layers
        ],
        np.stack([h.grad.detach().numpy() for h in hidden]),
    )


def main() -> None:
    x, layers, upstream = _case()
    before = [x.copy(), upstream.copy()] + [
        v.copy() for layer in layers for v in layer.values()
    ]
    results = {}
    mode_grad_history = {}

    for mode, norm_first in (("pre_ln", True), ("post_ln", False)):
        gx, per, hidden = encoder_stack_backward_numpy(
            upstream, x, layers, 2, norm_first=norm_first
        )
        ref_x, ref_per, ref_hidden = _reference(x, layers, upstream, norm_first)
        max_error = {
            f"layer_{i+1}_{key}": float(np.max(np.abs(per[i][key] - ref_per[i][key])))
            for i in range(len(layers)) for key in _LAYER_KEYS
        }
        max_error["X"] = float(np.max(np.abs(gx - ref_x)))
        history_error = float(np.max(np.abs(hidden - ref_hidden)))
        mode_grad_history[mode] = [
            float(np.linalg.norm(g)) for g in hidden
        ]
        results[mode] = {
            "max_absolute_error_vs_autograd": max_error,
            "max_hidden_gradient_error": history_error,
            "gradient_norm_by_hidden_state": mode_grad_history[mode],
            "all_gradients_finite": bool(
                np.isfinite(gx).all()
                and np.isfinite(hidden).all()
                and all(np.isfinite(g).all() for p in per for g in p.values())
            ),
        }

    # Extra check of actual tied parameters: same array objects reused.
    tied_layers = [layers[0], layers[0], layers[0]]
    _, tied_per, _ = encoder_stack_backward_numpy(upstream, x, tied_layers, 2)
    total = aggregate_tied_parameter_gradients_numpy(tied_layers, tied_per)
    tied_error = max(
        float(np.max(np.abs(total[0][key] - sum(p[key] for p in tied_per))))
        for key in _LAYER_KEYS
    )

    originals = [x, upstream] + [
        v for layer in layers for v in layer.values()
    ]
    unchanged = all(
        np.array_equal(current, old)
        for current, old in zip(originals, before)
    )

    metrics = {
        "depth": len(layers),
        "input_shape": list(x.shape),
        "num_heads": 2,
        "parameter_keys": list(_LAYER_KEYS),
        "pre_ln": results["pre_ln"],
        "post_ln": results["post_ln"],
        "tied_parameter_aggregation_max_error": tied_error,
        "input_arrays_unchanged": bool(unchanged),
    }

    directory = Path("results/day19")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    positions = np.arange(len(layers) + 1)
    for mode in ("pre_ln", "post_ln"):
        ax.plot(positions, mode_grad_history[mode], marker="o", label=mode)
    ax.set_xlabel("Hidden state index (0=input, L=output)")
    ax.set_ylabel("L2 norm of dL/dX^(l)")
    ax.set_title("Day 19 | Gradient magnitude across encoder depth")
    ax.set_xticks(positions)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(directory / "gradient_norm_across_depth.png", dpi=180)
    plt.close(fig)

    names = ["X"] + [f"L{i+1}:Wq" for i in range(len(layers))]
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    width = .35
    for j, mode in enumerate(("pre_ln", "post_ln")):
        info = results[mode]["max_absolute_error_vs_autograd"]
        vals = [max(info["X"], 1e-18)] + [
            max(info[f"layer_{i+1}_w_q"], 1e-18) for i in range(len(layers))
        ]
        ax.bar(np.arange(len(names)) + j * width, vals, width, label=mode)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(names)) + width / 2, names)
    ax.set_ylabel("Maximum absolute error (log scale)")
    ax.set_title("Day 19 | Manual backward vs PyTorch Autograd")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.savefig(directory / "gradient_error_by_layer.png", dpi=180)
    plt.close(fig)

    print("=" * 76)
    print("Transformer From Zero | Day 19 | Encoder Stack Backward Demo")
    print("=" * 76)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    assert all(
        v < 1e-9
        for mode in ("pre_ln", "post_ln")
        for v in results[mode]["max_absolute_error_vs_autograd"].values()
    )
    assert all(results[mode]["max_hidden_gradient_error"] < 1e-9
               for mode in ("pre_ln", "post_ln"))
    assert all(results[mode]["all_gradients_finite"]
               for mode in ("pre_ln", "post_ln"))
    assert tied_error < 1e-12
    assert unchanged
    print("\nSUCCESS: Day 19 Encoder Stack Backward checks passed.")
    print("Saved: results/day19/metrics.json and diagnostic PNGs")


if __name__ == "__main__":
    main()
