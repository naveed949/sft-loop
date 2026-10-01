"""Same-seed <=3B TRL + PEFT SFT loop.

The scored path trains with TRL and PEFT on MPS and does not require Unsloth.
Unsloth is an optional CUDA-only backend. Otherwise report builders emit
status ``unsupported`` and do not invent task scores.
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
