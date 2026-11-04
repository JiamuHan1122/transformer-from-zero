"""These should pass immediately after installing the project."""

import numpy as np
import torch

from tfz import __version__
from tfz.utils import count_parameters, set_seed


def test_package_import():
    assert __version__ == "0.1.0"


def test_torch_tensor_dimensions():
    x = torch.ones(2, 3)
    w = torch.ones(3, 4)
    assert tuple((x @ w).shape) == (2, 4)
    assert torch.allclose(x @ w, torch.full((2, 4), 3.0))


def test_seed_is_reproducible():
    set_seed(123)
    first = np.random.randn(3)
    first_torch = torch.rand(3)
    set_seed(123)
    assert np.array_equal(first, np.random.randn(3))
    assert torch.equal(first_torch, torch.rand(3))


def test_trainable_parameter_count():
    model = torch.nn.Linear(3, 2)
    assert count_parameters(model) == 8  # 3*2 weights + 2 bias
