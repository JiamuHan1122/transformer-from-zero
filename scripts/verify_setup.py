"""A safe smoke test for CPU and CUDA environments."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    print("== Transformer From Zero: environment check ==")
    print("Python:", sys.version.split()[0])
    print("Python executable:", sys.executable)
    print("Platform:", platform.platform())
    print("NumPy:", np.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.eye(2)
    assert torch.allclose(a @ b, a)
    print("Matrix multiplication sanity check: PASS")
    print("Project folder:", Path(__file__).resolve().parents[1])
    print("Next: python -m pytest -q")
    print("Then: python -m pytest -q -m exercise (expected to fail until completed)")


if __name__ == "__main__":
    main()
