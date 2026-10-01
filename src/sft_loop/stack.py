"""Probe for the scored train stack.

The default scored path is TRL + PEFT LoRA SFT on MPS. It does not require
Unsloth. Unsloth is an optional CUDA-only backend. A missing device or a
missing library is a block. This module does not train and does not score.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
BACKEND_TRL_PEFT = "trl-peft"
BACKEND_UNSLOTH = "unsloth"
DEVICES = (DEVICE_MPS, DEVICE_CUDA)

_PEFT_MODULES = ("torch", "transformers", "peft", "trl", "datasets", "accelerate")
_UNSLOTH_MODULES = ("unsloth", "trl")


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


def mps_is_available() -> bool:
    if not module_available("torch"):
        return False
    try:
        import torch

        backend = getattr(torch.backends, "mps", None)
        if backend is None or not hasattr(backend, "is_available"):
            return False
        return bool(backend.is_available())
    except Exception:
        return False


def _probe(fn: Callable[[], bool]) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


def _modules_ready(names: tuple[str, ...]) -> bool:
    return all(module_available(name) for name in names)


def resolve_train_target(
    device: str | None = None,
    *,
    cuda_available: Callable[[], bool] | None = None,
    mps_available: Callable[[], bool] | None = None,
) -> tuple[str, str] | None:
    """Return ``(device, backend)`` when a train may start.

    ``mps`` selects TRL + PEFT and does not require Unsloth. ``cuda`` selects
    the optional Unsloth backend. ``None`` prefers MPS when that stack is
    ready, otherwise CUDA Unsloth. ``None`` as the return value means the
    requested device or its libraries are unavailable.
    """

    if device is not None and device not in DEVICES:
        raise ValueError("device must be mps or cuda")
    cuda_probe = cuda_is_available if cuda_available is None else cuda_available
    mps_probe = mps_is_available if mps_available is None else mps_available
    has_mps = _probe(mps_probe)
    has_cuda = _probe(cuda_probe)
    peft_ready = has_mps and _modules_ready(_PEFT_MODULES)
    unsloth_ready = has_cuda and _modules_ready(_UNSLOTH_MODULES)
    if device == DEVICE_MPS:
        if peft_ready:
            return (DEVICE_MPS, BACKEND_TRL_PEFT)
        return None
    if device == DEVICE_CUDA:
        if unsloth_ready:
            return (DEVICE_CUDA, BACKEND_UNSLOTH)
        return None
    if peft_ready:
        return (DEVICE_MPS, BACKEND_TRL_PEFT)
    if unsloth_ready:
        return (DEVICE_CUDA, BACKEND_UNSLOTH)
    return None


def stack_block_reason(
    cuda_available: Callable[[], bool] | None = None,
    device: str | None = None,
    *,
    mps_available: Callable[[], bool] | None = None,
) -> str | None:
    """Return ``gpu_or_train_deps_unavailable`` or ``None`` when a train may start.

    ``None`` means the requested device and its train libraries look usable.
    It does not mean a train step has already run. Unsloth is required only
    for the CUDA backend.
    """

    target = resolve_train_target(
        device,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
    if target is None:
        return "gpu_or_train_deps_unavailable"
    return None
