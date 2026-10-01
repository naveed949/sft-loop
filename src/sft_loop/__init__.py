"""Same-seed <=3B Unsloth + TRL PEFT SFT loop.

Training runs only when CUDA and the pinned stack import. Otherwise report
builders emit status ``unsupported`` and do not invent task scores.
"""

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
