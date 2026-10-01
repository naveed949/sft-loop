"""Scaffold for a schema-locked <=3B SFT loop. This package does not train."""

from sft_loop.report import (
    MODEL_ID,
    REFUSAL_SENTENCE,
    TRL_VERSION,
    UNSLOTH_VERSION,
    unsupported_report,
)

__all__ = [
    "MODEL_ID",
    "REFUSAL_SENTENCE",
    "TRL_VERSION",
    "UNSLOTH_VERSION",
    "unsupported_report",
]
