"""Day 16 exercise: stack several Transformer Encoder Blocks with NumPy.

What today adds to Day 15:
    x^(0) = x
    x^(l+1), attention^(l) = EncoderBlock(x^(l); theta_l)

Input:
    x: (B, T, D), a NumPy floating-point array
    layers: non-empty list or tuple of per-layer parameter dictionaries
    num_heads: shared head count, positive and dividing D
    norm_first: True (Pre-LN) or False (Post-LN)

Every layer dictionary contains exactly the Day 15 parameter names:
    w_q, w_k, w_v, w_o: (D, D)
    gamma_attn, beta_attn: (D,)
    w1: (D, D_ff_l)
    b1: (D_ff_l,)
    w2: (D_ff_l, D)
    b2: (D,)
    gamma_ffn, beta_ffn: (D,)

Return:
    output: (B, T, D)
    all_attentions: (L, B, num_heads, T, T)
    all_hidden_states: (L + 1, B, T, D), includes initial input x^(0)

Rules:
- Use NumPy for the core; no Torch or Autograd.
- Reuse the complete Day 15 Transformer Encoder Block.
- No dropout, masks, positional encoding, extra final normalization,
  classification head, or complete backward pass today.
- Use float64 and do not mutate caller-owned arrays.
- Pass separate dictionaries for independently parameterized layers.
  Reusing the same dictionary explicitly models tied parameters.

Run:
    python -m pytest -q -m exercise tests/test_week16_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week15 import transformer_encoder_block_forward_numpy


_LAYER_KEYS = (
    "w_q", "w_k", "w_v", "w_o",
    "gamma_attn", "beta_attn",
    "w1", "b1", "w2", "b2",
    "gamma_ffn", "beta_ffn",
)


def _validate_encoder_stack_inputs(
    x: np.ndarray,
    layers: list[dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], ...],
    num_heads: int,
    norm_first: bool,
    eps: float,
) -> None:
    """Provided helper: validate the stack interface and required keys.

    The existing Day 15 block validates the detailed parameter shapes
    and dtypes for every individual layer.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a NumPy array.")
    if x.ndim != 3 or any(dim == 0 for dim in x.shape):
        raise ValueError("x must have non-empty shape (B, T, D).")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("x must have a floating-point dtype.")

    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("layers must be a non-empty list or tuple.")

    for layer_idx, params in enumerate(layers):
        if not isinstance(params, dict):
            raise ValueError(
                f"layers[{layer_idx}] must be a parameter dictionary."
            )
        missing = [name for name in _LAYER_KEYS if name not in params]
        if missing:
            raise ValueError(
                f"layers[{layer_idx}] is missing parameters: {missing}."
            )

    if (
        not isinstance(num_heads, (int, np.integer))
        or isinstance(num_heads, (bool, np.bool_))
        or num_heads <= 0
        or x.shape[-1] % num_heads != 0
    ):
        raise ValueError(
            "num_heads must be positive and divide the model dimension."
        )

    if not isinstance(norm_first, (bool, np.bool_)):
        raise ValueError("norm_first must be a Boolean.")

    if (
        not isinstance(eps, (int, float, np.integer, np.floating))
        or isinstance(eps, (bool, np.bool_))
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("eps must be a positive finite real number.")


def encoder_stack_forward_numpy(
    x: np.ndarray,
    layers: list[dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], ...],
    num_heads: int,
    norm_first: bool = True,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Task A: compose L independently parameterized Encoder Blocks.

    Implement the recurrence:
        x^(0) = float64(x)
        for layer_index in range(L):
            x^(layer_index + 1), attn = transformer_encoder_block_forward_numpy(
                x^(layer_index),
                w_q=..., w_k=..., w_v=..., w_o=...,
                num_heads=num_heads,
                gamma_attn=..., beta_attn=...,
                w1=..., b1=..., w2=..., b2=...,
                gamma_ffn=..., beta_ffn=...,
                norm_first=norm_first, eps=eps,
            )

    Required return ordering:
        final_output: shape (B, T, D), float64
        all_attentions: np.stack of L weight tensors at axis=0
        all_hidden_states: np.stack of [input, layer_1, ..., layer_L]
            at axis=0; shape (L+1, B, T, D)

    Notes:
    - Never average the attention maps across different layers here:
      retaining their depth dimension is the point of today's exercise.
    - Do not accidentally reuse the original x for every layer.
    - Each layer's own weight dictionary must be used in its block call.
    - Do not append attention matrices to the hidden-state history.
    """
    _validate_encoder_stack_inputs(
        x=x,
        layers=layers,
        num_heads=num_heads,
        norm_first=norm_first,
        eps=eps,
    )

    # TODO 1: initialize the current hidden state from x as float64.
    # TODO 2: initialize two lists: hidden_states (starting with input)
    #         and attentions (initially empty).
    # TODO 3: loop over layers, calling Day 15 with this layer's parameters.
    # TODO 4: feed each layer's output into the NEXT layer.
    # TODO 5: stack histories along NEW depth axis=0.
    # TODO 6: return final output, all attentions, all hidden states.
    raise NotImplementedError(
        "Task A: implement multi-layer Transformer Encoder forward."
    )
