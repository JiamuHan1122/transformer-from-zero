
"""Day 19 exercises: multi-layer Transformer Encoder Stack backward.

Reuse Day 16 forward and Day 18 full Encoder Block backward.

Forward:
    x^(0) = x
    x^(l+1) = EncoderBlock(x^(l), params[l])

Backward:
    grad^(L) = grad_output
    grad^(l) = EncoderBlockBackward(
        grad^(l+1), x^(l), params[l]
    )

Task A: full stack backward.
Task B: aggregate gradients for identity-tied parameters.

No Torch or Autograd in the core.
"""

from __future__ import annotations

import numpy as np

from exercises.week16 import (
    _LAYER_KEYS,
    _validate_encoder_stack_inputs,
    encoder_stack_forward_numpy,
)

from exercises.week18 import (
    transformer_encoder_block_backward_numpy,
)


def _validate_stack_backward_inputs(
    grad_output: np.ndarray,
    x: np.ndarray,
    layers: list[dict[str, np.ndarray]]
    | tuple[dict[str, np.ndarray], ...],
    num_heads: int,
    norm_first: bool,
    eps: float,
) -> None:
    """Validate upstream gradient and reuse Day 16 stack contract."""

    _validate_encoder_stack_inputs(
        x, layers, num_heads, norm_first, eps
    )

    if not isinstance(grad_output, np.ndarray):
        raise ValueError("grad_output must be a NumPy array.")

    if grad_output.shape != x.shape:
        raise ValueError(
            "grad_output must have shape (B, T, D), same as x."
        )

    if not np.issubdtype(grad_output.dtype, np.floating):
        raise ValueError(
            "grad_output must have a floating-point dtype."
        )


def encoder_stack_backward_numpy(
    grad_output: np.ndarray,
    x: np.ndarray,
    layers: list[dict[str, np.ndarray]]
    | tuple[dict[str, np.ndarray], ...],
    num_heads: int,
    norm_first: bool = True,
    eps: float = 1e-5,
) -> tuple[
    np.ndarray,
    list[dict[str, np.ndarray]],
    np.ndarray,
]:
    """Task A: backpropagate through a complete L-layer Encoder Stack.

    Returns:
        grad_x:
            (B, T, D)

        per_layer_grads:
            L dictionaries in forward layer order.

        hidden_grads:
            (L+1, B, T, D), ordered as G^(0), ..., G^(L).
    """

    # ============================================================
    # Step 1: Validate inputs
    # ============================================================

    _validate_stack_backward_inputs(
        grad_output,
        x,
        layers,
        num_heads,
        norm_first,
        eps,
    )

    # ============================================================
    # Step 2: Recompute forward hidden states
    # ============================================================

    _, _, all_hidden_states = encoder_stack_forward_numpy(
        x,
        layers,
        num_heads,
        norm_first=norm_first,
        eps=eps,
    )

    depth = len(layers)

    # ============================================================
    # Step 3: Initialize backward variables
    # ============================================================

    grad_current = np.asarray(
        grad_output,
        dtype=np.float64,
    ).copy()

    per_layer_grads = [None] * depth

    hidden_grads = [None] * (depth + 1)

    # The gradient of the final hidden state is the upstream gradient.
    hidden_grads[depth] = grad_current.copy()

    # ============================================================
    # Step 4: Traverse Encoder layers in reverse order
    # ============================================================

    for layer_idx in reversed(range(depth)):

        # Parameters of the current layer.
        params = layers[layer_idx]

        # Input to this layer during the forward pass.
        layer_input = all_hidden_states[layer_idx]

        # Keep only the 12 documented parameter names.
        layer_params = {
            key: params[key]
            for key in _LAYER_KEYS
        }

        # Backpropagate through one complete Encoder Block.
        block_grads = transformer_encoder_block_backward_numpy(
            grad_current,
            layer_input,
            num_heads=num_heads,
            norm_first=norm_first,
            eps=eps,
            **layer_params,
        )

        # The first returned gradient belongs to the layer input.
        grad_previous = block_grads[0]

        # The remaining 12 gradients belong to layer parameters.
        parameter_gradients = block_grads[1:]

        # Save parameter gradients using their original names.
        per_layer_grads[layer_idx] = {
            key: np.asarray(
                grad,
                dtype=np.float64,
            ).copy()
            for key, grad in zip(
                _LAYER_KEYS,
                parameter_gradients,
            )
        }

        # Pass the input gradient to the preceding layer.
        grad_current = np.asarray(
            grad_previous,
            dtype=np.float64,
        )

        # Store the gradient of the corresponding hidden state.
        hidden_grads[layer_idx] = grad_current.copy()

    # ============================================================
    # Step 5: Return results in forward order
    # ============================================================

    return (
        grad_current.copy(),
        per_layer_grads,
        np.stack(hidden_grads, axis=0),
    )


def aggregate_tied_parameter_gradients_numpy(
    layers: list[dict[str, np.ndarray]]
    | tuple[dict[str, np.ndarray], ...],

    per_layer_grads: list[dict[str, np.ndarray]]
    | tuple[dict[str, np.ndarray], ...],
) -> list[dict[str, np.ndarray]]:
    """Task B: aggregate gradients for identity-tied parameters.

    Object identity determines whether parameters are shared.

    Equal-valued but independently allocated arrays are NOT tied.

    Returns:
        L dictionaries containing the total gradient for each
        parameter object, with independent float64 copies.
    """

    # ============================================================
    # Step 1: Validate collection interfaces
    # ============================================================

    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError(
            "layers must be a non-empty list or tuple."
        )

    if not isinstance(per_layer_grads, (list, tuple)):
        raise ValueError(
            "per_layer_grads must be a list or tuple."
        )

    if len(layers) != len(per_layer_grads):
        raise ValueError(
            "layers and per_layer_grads must have equal lengths."
        )

    # ============================================================
    # Step 2: Validate parameter dictionaries and gradients
    # ============================================================

    for layer_idx, (params, grads) in enumerate(
        zip(layers, per_layer_grads)
    ):

        if not isinstance(params, dict):
            raise ValueError(
                f"layers[{layer_idx}] must be a dictionary."
            )

        if not isinstance(grads, dict):
            raise ValueError(
                f"per_layer_grads[{layer_idx}] must be a dictionary."
            )

        missing_params = [
            key for key in _LAYER_KEYS
            if key not in params
        ]

        missing_grads = [
            key for key in _LAYER_KEYS
            if key not in grads
        ]

        if missing_params:
            raise ValueError(
                f"layers[{layer_idx}] is missing keys: "
                f"{missing_params}."
            )

        if missing_grads:
            raise ValueError(
                f"per_layer_grads[{layer_idx}] is missing keys: "
                f"{missing_grads}."
            )

        extra_grads = [
            key for key in grads
            if key not in _LAYER_KEYS
        ]

        if extra_grads:
            raise ValueError(
                f"per_layer_grads[{layer_idx}] has unexpected keys: "
                f"{extra_grads}."
            )

        for key in _LAYER_KEYS:

            param = params[key]
            grad = grads[key]

            if not isinstance(param, np.ndarray):
                raise ValueError(
                    f"layers[{layer_idx}][{key}] must be a NumPy array."
                )

            if not isinstance(grad, np.ndarray):
                raise ValueError(
                    f"per_layer_grads[{layer_idx}][{key}] "
                    "must be a NumPy array."
                )

            if not np.issubdtype(param.dtype, np.floating):
                raise ValueError(
                    f"Parameter {key} must have a floating-point dtype."
                )

            if not np.issubdtype(grad.dtype, np.floating):
                raise ValueError(
                    f"Gradient {key} must have a floating-point dtype."
                )

            if any(dim == 0 for dim in param.shape):
                raise ValueError(
                    f"Parameter {key} must not have empty dimensions."
                )

            if grad.shape != param.shape:
                raise ValueError(
                    f"Gradient shape mismatch at layer {layer_idx}, "
                    f"parameter {key}: expected {param.shape}, "
                    f"got {grad.shape}."
                )

    # ============================================================
    # Step 3: Group gradients by parameter object identity
    # ============================================================

    gradient_totals = {}

    for params, grads in zip(layers, per_layer_grads):

        for key in _LAYER_KEYS:

            param = params[key]
            grad = grads[key]

            # Python object identity, not numerical equality.
            param_id = id(param)

            grad_float64 = np.asarray(
                grad,
                dtype=np.float64,
            )

            if param_id not in gradient_totals:

                # First contribution: create an independent copy.
                gradient_totals[param_id] = grad_float64.copy()

            else:

                # Shared parameter: accumulate another contribution.
                gradient_totals[param_id] += grad_float64

    # ============================================================
    # Step 4: Distribute total gradients to all parameter aliases
    # ============================================================

    aggregated_grads = []

    for params in layers:

        layer_grads = {}

        for key in _LAYER_KEYS:

            param_id = id(params[key])

            layer_grads[key] = (
                gradient_totals[param_id].copy()
            )

        aggregated_grads.append(layer_grads)

    # ============================================================
    # Step 5: Return aggregated gradients
    # ============================================================

    return aggregated_grads
