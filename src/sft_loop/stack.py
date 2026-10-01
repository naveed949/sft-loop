"""Probe for the pinned Unsloth + TRL train stack.

A missing GPU or a missing library is a block. This module does not train and
does not score.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def cuda_is_available() -> bool:
    if not module_available("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def stack_block_reason(
    cuda_available: Callable[[], bool] | None = None,
) -> str | None:
    """Return ``gpu_or_train_deps_unavailable`` or ``None`` when a train may start.

    ``None`` means torch CUDA, Unsloth, and TRL all look importable. It does
    not mean a train step has already run.
    """

    probe = cuda_is_available if cuda_available is None else cuda_available
    try:
        has_cuda = bool(probe())
    except Exception:
        has_cuda = False
    if not has_cuda or not module_available("unsloth") or not module_available("trl"):
        return "gpu_or_train_deps_unavailable"
    return None
