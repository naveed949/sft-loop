"""Refuse any path that would overwrite the Phase 3 portfolio."""

from __future__ import annotations

from pathlib import Path


class Phase3PathRefused(ValueError):
    """A Phase 4 writer was pointed at runs/phase3 or adapters/phase3."""


class BuildStopped(ValueError):
    """Builder refused to emit a dataset. Destination files were not replaced."""


class RefusedCeiling(ValueError):
    """Baseline exact match is at or above the headroom cutoff."""


class Phase4Unsupported(ValueError):
    """GPU path cannot run. No adapter and no after-eval."""


def refuse_phase3(path: Path | str) -> Path:
    raw = Path(path)
    text = raw.as_posix().replace("\\", "/")
    lowered = text.lower()
    if "/runs/phase3" in lowered or "/adapters/phase3" in lowered:
        raise Phase3PathRefused(f"phase 4 refuses path under phase3: {raw}")
    try:
        resolved = raw.expanduser().resolve()
    except OSError:
        resolved = raw
    parts = [part.lower() for part in resolved.parts]
    for index, part in enumerate(parts):
        if part == "phase3" and index > 0 and parts[index - 1] in {"runs", "adapters"}:
            raise Phase3PathRefused(f"phase 4 refuses path under phase3: {resolved}")
    return resolved
