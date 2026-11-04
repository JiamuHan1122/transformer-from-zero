"""Small, reusable helpers. No Transformer implementation is supplied yet."""

from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set the common Python/NumPy/PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable scalar parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
