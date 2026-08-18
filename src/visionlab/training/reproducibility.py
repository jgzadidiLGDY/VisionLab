"""Small reproducibility helpers for Phase 3 training."""

from __future__ import annotations

import platform
import random
from typing import Any

import torch


def apply_reproducibility(seed: int) -> None:
    """Apply the CPU reproducibility controls used by Phase 3 smoke runs."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def environment_summary(device: str = "cpu") -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
