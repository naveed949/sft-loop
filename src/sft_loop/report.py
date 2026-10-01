"""Fail-closed unsupported reports.

This module does not train, does not score, and does not claim a weight change.
The only report it builds has status ``unsupported``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
UNSLOTH_VERSION = "2026.9.12"
TRL_VERSION = "0.24.0"
SCHEMA_VERSION = "1"
REFUSAL_SENTENCE = "Soft-PASS is unused and is refused."

_REASONS = ("gpu_or_train_deps_unavailable", "train_not_run")

_DETAILS = {
    "gpu_or_train_deps_unavailable": (
        "GPU or train dependencies are unavailable. "
        "No train step ran. No before/after metrics were produced. "
        "Status is unsupported. "
        + REFUSAL_SENTENCE
    ),
    "train_not_run": (
        "This scaffold does not run a train step. "
        "No before/after metrics were produced. "
        "Status is unsupported. "
        + REFUSAL_SENTENCE
    ),
}


def schema_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "schema" / "run_report.schema.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("schema/run_report.schema.json")


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def train_block_reason(cuda_available: Callable[[], bool] | None = None) -> str:
    """Why this scaffold must not emit metrics.

    A missing GPU or train dependency yields ``gpu_or_train_deps_unavailable``.
    A usable CUDA device still yields ``train_not_run`` because no train step
    exists in this package.
    """

    probe = _cuda_available if cuda_available is None else cuda_available
    try:
        available = bool(probe())
    except Exception:
        return "gpu_or_train_deps_unavailable"
    if not available:
        return "gpu_or_train_deps_unavailable"
    return "train_not_run"


def unsupported_report(
    *,
    run_id: str = "scaffold-unsupported",
    reason: str | None = None,
    cuda_available: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    chosen = reason if reason is not None else train_block_reason(cuda_available)
    if chosen not in _REASONS:
        raise ValueError(f"unknown unsupported reason: {chosen}")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "stack": {
            "unsloth": UNSLOTH_VERSION,
            "trl": TRL_VERSION,
            "method": "sft-lora",
        },
        "status": "unsupported",
        "metric_role": "task_score_not_safety",
        "weight_change_claimed": False,
        "unsupported": {
            "reason": chosen,
            "detail": _DETAILS[chosen],
        },
    }


def validate_report(report: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda err: [str(part) for part in err.path],
    )
    if errors:
        raise ValueError("; ".join(err.message for err in errors))
