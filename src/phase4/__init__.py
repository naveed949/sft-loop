"""Closed sibling for Phase 4 10-K LoRA. Not part of sft_loop."""

from phase4.assemble import assemble, paired_bootstrap
from phase4.build import build, build_default
from phase4.constants import TASK_ID
from phase4.grade import canonicalize, grade, render_prompt
from phase4.paths import BuildStopped, Phase3PathRefused, Phase4Unsupported, RefusedCeiling
from phase4.run import local_scaler_probe, require_gpu_scaler

__all__ = [
    "TASK_ID",
    "BuildStopped",
    "Phase3PathRefused",
    "Phase4Unsupported",
    "RefusedCeiling",
    "assemble",
    "build",
    "build_default",
    "canonicalize",
    "grade",
    "local_scaler_probe",
    "paired_bootstrap",
    "render_prompt",
    "require_gpu_scaler",
]
