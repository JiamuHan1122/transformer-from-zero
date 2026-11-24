"""Day 18 demo: verify full Encoder Block manual backward.

Run from the repository root:
    python -m scripts.day18_encoder_block_backward_demo

This demo runs only AFTER the student completes exercises/week18.py.
It checks both Pre-LN and Post-LN with an independent PyTorch graph,
and writes local diagnostics to results/day18/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from exercises.week15 import transformer_encoder_block_forward_numpy
from exercises.week18 import transformer_encoder_block_backward_numpy


ARRAY_INDICES = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13)
NAMES = (
    "X", "Wq", "Wk", "Wv", "Wo",
    "gamma_attn", "beta_attn",
    "W1", "b1", "W2", "b2",
    "gamma_ffn", "beta_ffn",
)


def _make_inputs(seed=1818):
    rng = np.random.default_rng(seed)
    b, t, d, h, dff = 2, 5, 8, 2, 32
    x = rng.normal(size=(b, t, d))
    wq, wk, wv, wo = [
        rng.normal(scale=1 / np.sqrt(d), size=(d, d))
        for _ in range(4)
    ]
    ga = rng.uniform(0.8, 1.2, size=d)
    ba = rng.normal(scale=0.08, size=d)
    w1 = rng.normal(scale=0.18, size=(d, dff))
    b1 = rng.normal(scale=0.05, size=dff)
    w2 = rng.normal(scale=0.18, size=(dff, d))
    b2 = rng.normal(scale=0.05, size=d)
    gf = rng.uniform(0.8, 1.2, size=d)
    bf = rng.normal(scale=0.07, size=d)
    grad_output = rng.normal(size=(b, t, d))
    args = (x, wq, wk, wv, wo, h, ga, ba, w1, b1, w2, b2, gf, bf)
    return grad_output, args


def _torch_gradients(grad_output, args, norm_first):
    torch_arrays = [
        torch.tensor(args[idx], dtype=torch.float64, requires_grad=True)
        for idx in ARRAY_INDICES
    ]
    (
        x, wq, wk, wv, wo, ga, ba,
        w1, b1, w2, b2, gf, bf
    ) = torch_arrays
    heads = args[5]
    b, t, d = x.shape
    dh = d // heads

    def attn(inp):
        q = (inp @ wq).reshape(b, t, heads, dh).transpose(1, 2)
        k = (inp @ wk).reshape(b, t, heads, dh).transpose(1, 2)
        v = (inp @ wv).reshape(b, t, heads, dh).transpose(1, 2)
        probs = torch.softmax(
            (q @ k.transpose(-2, -1)) / np.sqrt(dh), dim=-1
        )
        context = (probs @ v).transpose(1, 2).contiguous()
        return context.reshape(b, t, d) @ wo

    def norm(inp, gamma, beta):
        return F.layer_norm(inp, (d,), gamma, beta, eps=1e-5)

    def ffn(inp):
        return F.gelu(inp @ w1 + b1, approximate="tanh") @ w2 + b2

    if norm_first:
        hidden = x + attn(norm(x, ga, ba))
        output = hidden + ffn(norm(hidden, gf, bf))
    else:
        hidden = norm(x + attn(x), ga, ba)
        output = norm(hidden + ffn(hidden), gf, bf)

    loss = torch.sum(output * torch.from_numpy(grad_output))
    loss.backward()
    return [a.grad.detach().numpy() for a in torch_arrays], float(
        loss.detach().item()
    )


def _scalar_objective(upstream, args, norm_first):
    out, _ = transformer_encoder_block_forward_numpy(
        *args, norm_first=norm_first
    )
    return float(np.sum(out * upstream))


def main() -> None:
    upstream, args = _make_inputs()
    preserved = [value.copy() if isinstance(value, np.ndarray) else value for value in args]
    preserved_upstream = upstream.copy()
    modes = {"pre_ln": True, "post_ln": False}
    gradients = {}
    references = {}
    details = {}

    for key, norm_first in modes.items():
        manual = transformer_encoder_block_backward_numpy(
            upstream, *args, norm_first=norm_first
        )
        reference, scalar_loss = _torch_gradients(
            upstream, args, norm_first
        )
        gradients[key] = manual
        references[key] = reference
        max_error = {
            n: float(np.max(np.abs(g - r)))
            for n, g, r in zip(NAMES, manual, reference)
        }
        relative_l2_error = {
            n: float(np.linalg.norm(g - r) / max(np.linalg.norm(r), 1e-12))
            for n, g, r in zip(NAMES, manual, reference)
        }
        norm_per_parameter = {
            n: float(np.linalg.norm(g))
            for n, g in zip(NAMES, manual)
        }

        # One input and one LayerNorm parameter finite-difference check.
        checks = []
        for grad_index, coord in ((0, (0, 2, 3)), (5, (1,))):
            source_pos = ARRAY_INDICES[grad_index]
            numerical_eps = 1e-6

            def loss_at(delta):
                new_args = list(args)
                new_args[source_pos] = args[source_pos].copy()
                new_args[source_pos][coord] += delta
                return _scalar_objective(upstream, new_args, norm_first)

            numeric = (loss_at(numerical_eps) - loss_at(-numerical_eps)) / (
                2 * numerical_eps
            )
            analytic = float(manual[grad_index][coord])
            checks.append({
                "parameter": NAMES[grad_index],
                "coordinate": list(coord),
                "numeric": float(numeric),
                "analytic": analytic,
                "absolute_error": float(abs(numeric - analytic)),
            })

        details[key] = {
            "scalar_objective": scalar_loss,
            "max_absolute_error_vs_autograd": max_error,
            "relative_l2_error_vs_autograd": relative_l2_error,
            "gradient_l2_norms": norm_per_parameter,
            "finite_difference_checks": checks,
            "all_grads_finite": bool(
                all(np.isfinite(g).all() for g in manual)
            ),
        }

    # Verify that the Pre-LN identity path survives when both branches
    # contribute zero gradients to X.
    zeroed = list(args)
    for idx in (3, 4, 10, 11):
        zeroed[idx] = np.zeros_like(zeroed[idx])
    zero_pre = transformer_encoder_block_backward_numpy(
        upstream, *zeroed, norm_first=True
    )[0]
    pre_identity_error = float(np.max(np.abs(zero_pre - upstream)))

    metrics = {
        "input_shape": list(args[0].shape),
        "num_heads": args[5],
        "num_gradients": len(NAMES),
        "gradient_names": list(NAMES),
        "gradient_shapes": [
            list(args[idx].shape) for idx in ARRAY_INDICES
        ],
        "pre_ln": details["pre_ln"],
        "post_ln": details["post_ln"],
        "pre_ln_zero_branches_identity_gradient_error": pre_identity_error,
        "input_arrays_unchanged_by_backward": bool(
            np.array_equal(upstream, preserved_upstream)
            and all(
                np.array_equal(value, original)
                if isinstance(value, np.ndarray) else value == original
                for value, original in zip(args, preserved)
            )
        ),
    }

    directory = Path("results/day18")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    x_labels = list(NAMES)
    xs = np.arange(len(x_labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12.4, 5.3), constrained_layout=True)
    pre = [
        max(details["pre_ln"]["max_absolute_error_vs_autograd"][n], 1e-18)
        for n in NAMES
    ]
    post = [
        max(details["post_ln"]["max_absolute_error_vs_autograd"][n], 1e-18)
        for n in NAMES
    ]
    ax.bar(xs - width / 2, pre, width, label="Pre-LN")
    ax.bar(xs + width / 2, post, width, label="Post-LN")
    ax.set_yscale("log")
    ax.set_xticks(xs, x_labels, rotation=50, ha="right")
    ax.set_ylabel("Max absolute gradient error (log scale)")
    ax.set_title("Day 18 | Manual gradients vs PyTorch Autograd")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(directory / "gradient_error_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.4, 5.3), constrained_layout=True)
    pre_norm = [details["pre_ln"]["gradient_l2_norms"][n] for n in NAMES]
    post_norm = [details["post_ln"]["gradient_l2_norms"][n] for n in NAMES]
    ax.bar(xs - width / 2, pre_norm, width, label="Pre-LN")
    ax.bar(xs + width / 2, post_norm, width, label="Post-LN")
    ax.set_xticks(xs, x_labels, rotation=50, ha="right")
    ax.set_ylabel("L2 gradient norm")
    ax.set_title("Day 18 | Pre-LN and Post-LN gradient magnitudes")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(directory / "pre_post_gradient_norms.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, mode in zip(axes, ("pre_ln", "post_ln")):
        im = ax.imshow(gradients[mode][0][0], aspect="auto")
        ax.set_title(mode + " | dL/dX | batch 0")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Token")
        fig.colorbar(im, ax=ax)
    fig.savefig(directory / "input_gradient_pre_post.png", dpi=180)
    plt.close(fig)

    print("=" * 78)
    print("Transformer From Zero | Day 18 | Encoder Block Backward Demo")
    print("=" * 78)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    assert all(
        val < 1e-9
        for mode in modes
        for val in details[mode]["max_absolute_error_vs_autograd"].values()
    )
    assert all(
        row["absolute_error"] < 1e-6
        for mode in modes
        for row in details[mode]["finite_difference_checks"]
    )
    assert all(details[mode]["all_grads_finite"] for mode in modes)
    assert pre_identity_error < 1e-12
    assert metrics["input_arrays_unchanged_by_backward"]
    print("\nSUCCESS: Day 18 Encoder Block Backward checks passed.")
    print("Saved outputs to results/day18/")


if __name__ == "__main__":
    main()
