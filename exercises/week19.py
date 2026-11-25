"""Day 19 exercises: multi-layer Transformer Encoder Stack backward.

Reuse Day 16 forward and Day 18 full Encoder Block backward.

Forward:
    x^(0) = x
    x^(l+1) = EncoderBlock(x^(l), params[l]) for l = 0..L-1

Backward:
    grad^(L) = grad_output
    grad^(l) = EncoderBlockBackward(grad^(l+1), x^(l), params[l])

Return:
    grad_x: shape (B, T, D), float64
    per_layer_grads: list of L dictionaries, in FORWARD layer order
    hidden_grads: shape (L+1, B, T, D), in FORWARD state order

The 12 keys in each gradient dictionary match Day 16 _LAYER_KEYS.

Task B aggregates contributions for parameters whose NumPy arrays are the
SAME OBJECT (identity-based weight tying). Equal-valued independent arrays
are not tied. This lesson does not implement an optimizer or SGD step.

No Torch/Autograd in the core; no masks, dropout, or final stack norm.

Run:
    python -m pytest -q -m exercise tests/test_week19_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week16 import (
    _LAYER_KEYS,
    _validate_encoder_stack_inputs,
    encoder_stack_forward_numpy,
)
from exercises.week18 import transformer_encoder_block_backward_numpy


def _validate_stack_backward_inputs(
    grad_output: np.ndarray,
    x: np.ndarray,
    layers: list[dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], ...],
    num_heads: int,
    norm_first: bool,
    eps: float,
) -> None:
    """Validate upstream gradient and reuse Day 16 stack contract."""
    _validate_encoder_stack_inputs(x, layers, num_heads, norm_first, eps)
    if not isinstance(grad_output, np.ndarray):
        raise ValueError("grad_output must be a NumPy array.")
    if grad_output.shape != x.shape:
        raise ValueError("grad_output must have shape (B, T, D), same as x.")
    if not np.issubdtype(grad_output.dtype, np.floating):
        raise ValueError("grad_output must have a floating-point dtype.")


def encoder_stack_backward_numpy(
    grad_output: np.ndarray,
    x: np.ndarray,
    layers: list[dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], ...],
    num_heads: int,
    norm_first: bool = True,
    eps: float = 1e-5,
) -> tuple[
    np.ndarray,
    list[dict[str, np.ndarray]],
    np.ndarray,
]:
    """Task A: backpropagate through a complete L-layer Encoder Stack.

    The output/parameter contract must match Day 16 exactly.

    1. Validate inputs using _validate_stack_backward_inputs().
    2. Call encoder_stack_forward_numpy() to obtain all_hidden_states.
       Its shape is (L+1, B, T, D); index l is input to layer l.
    3. Convert upstream gradient to float64.
    4. Traverse layer indices L-1, ..., 0.
       Call transformer_encoder_block_backward_numpy(
           grad_current,
           all_hidden_states[l],
           num_heads=num_heads, norm_first=norm_first, eps=eps,
           **layers[l],
       ).
       It returns grad_previous and 12 parameter gradients.
    5. Store each layer's 12 gradients as a dictionary keyed by
       _LAYER_KEYS, IN THE ORIGINAL FORWARD ORDER.
    6. Store G^(L), ..., G^(0) in a stack ordered as G^(0), ..., G^(L).
    7. Return (grad_x, per_layer_grads, hidden_grads).

    DO NOT add G^(l) across different layers; each derivative belongs to
    a different intermediate variable.

    DO NOT average parameter gradients over B or T.

    The per-layer gradients are contributions for each USAGE of a
    parameter. For tied parameters they must subsequently be summed by
    aggregate_tied_parameter_gradients_numpy(). Do not silently sum them
    here because they have different layer provenance.
    """
    _validate_stack_backward_inputs(
        grad_output, x, layers, num_heads, norm_first, eps
    )

    # TODO 1: Recompute forward history using Day 16.
    # TODO 2: Initialize upstream G^(L), per-layer gradient containers,
    #         and hidden-state gradient history, all with correct order.
    # TODO 3: Iterate through layers in reverse depth order.
    # TODO 4: Call Day 18 Block Backward with the correct x^(l).
    # TODO 5: Store 12 parameter gradient contributions for layer l.
    # TODO 6: Save each G^(l), and return the three documented outputs.
    raise NotImplementedError("Task A: implement Encoder Stack backward.")


def aggregate_tied_parameter_gradients_numpy(
    layers: list[dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], ...],
    per_layer_grads: list[dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], ...],
) -> list[dict[str, np.ndarray]]:
    """Task B: sum gradient contributions for shared NumPy parameter objects.

    Example:
        shared = make_layer()
        layers = [shared, shared]
        per_layer_grads = [G0, G1]

    Then returned gradients at BOTH layer indices for a shared field
    must equal G0[field] + G1[field].

    Rules:
    - Object IDENTITY, not equal array values, determines tying.
    - Also handle sharing a single NumPy array across different dicts.
    - Return L dictionaries with _LAYER_KEYS and float64 gradient arrays.
    - All aliases of a shared parameter get equal-valued copies of its
      TOTAL gradient for inspection.
    - An optimizer must update each distinct shared ndarray ONCE, not
      once per repeated dictionary entry.
    - Do not mutate layers or per_layer_grads.
    - Validate lengths, keys, and gradient shapes/dtypes.
    """
    # TODO 7: Validate layers and per_layer_grads.
    # TODO 8: Group all gradient contributions by id(parameter_array).
    # TODO 9: Sum contributions for each shared parameter.
    # TODO 10: Return dictionaries in original layer order. Use copies.
    raise NotImplementedError("Task B: aggregate tied parameter gradients.")
