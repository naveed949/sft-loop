"""Local Phase 4 helpers. The Kaggle kernel does not import this module."""

from __future__ import annotations

import json
from pathlib import Path

from phase4.constants import (
    CEILING,
    SMOKE_MODEL_ID,
    SMOKE_MODEL_REVISION,
    SMOKE_STEPS,
    SPLIT_SEED,
)
from phase4.grade import grade, render_prompt
from phase4.paths import Phase4Unsupported, RefusedCeiling, refuse_phase3


def require_gpu_scaler():
    try:
        import torch
    except ImportError as exc:
        raise Phase4Unsupported("gpu_or_train_deps_unavailable: torch missing") from exc
    if getattr(torch.version, "cuda", None) is None:
        raise Phase4Unsupported("gpu_or_train_deps_unavailable: cpu torch wheel")
    if not torch.cuda.is_available():
        raise Phase4Unsupported("gpu_or_train_deps_unavailable: cuda unavailable")
    scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_cls is None:
        scaler_cls = getattr(getattr(torch.cuda, "amp", None), "GradScaler", None)
    if scaler_cls is None:
        raise Phase4Unsupported("gpu_or_train_deps_unavailable: GradScaler missing")
    caps = []
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        caps.append((major, minor))
    bf16 = "unsupported"
    bf16_reason = "Tesla T4 compute capability 7.5 has no bf16; fp16 only"
    if caps and all(major >= 8 for major, _minor in caps):
        bf16 = "available"
        bf16_reason = "compute capability >= 8.0"
    return scaler_cls, bf16, bf16_reason


def refuse_ceiling(score: float) -> None:
    if score >= CEILING:
        raise RefusedCeiling(f"baseline exact match {score} is at or above {CEILING}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_rows(rows: list[dict], predictions: list[str]) -> dict:
    completions = [row["completion"] for row in rows]
    hits = []
    items = []
    for row, prediction in zip(rows, predictions):
        item_hit = int(grade([prediction], [row["completion"]]) == 1.0)
        hits.append(item_hit)
        items.append(
            {
                "item_id": row["item_id"],
                "doc_id": row["doc_id"],
                "question": row["question"],
                "completion": row["completion"],
                "prediction": prediction,
                "hit": item_hit,
            }
        )
    return {
        "metric_name": "exact_match",
        "task_score": grade(predictions, completions),
        "n": len(rows),
        "items": items,
        "seed": SPLIT_SEED,
    }


def smoke_prompts(rows: list[dict]) -> list[str]:
    return [row.get("prompt") or render_prompt(row["chunk"], row["question"]) for row in rows]


def write_unsupported(dest: Path, reason: str, detail: str) -> dict:
    refuse_phase3(dest)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": "phase4-1",
        "document_kind": "phase4-tenk-lora",
        "status": "unsupported",
        "unsupported": {"reason": reason, "detail": detail},
        "adapter_written": False,
    }
    dest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def local_scaler_probe(dest: Path) -> dict:
    refuse_phase3(dest)
    try:
        require_gpu_scaler()
    except Phase4Unsupported as exc:
        return write_unsupported(dest, "gpu_or_train_deps_unavailable", str(exc))
    raise Phase4Unsupported("local probe must not train")


def smoke_model_pins() -> tuple[str, str]:
    return SMOKE_MODEL_ID, SMOKE_STEPS and SMOKE_MODEL_REVISION
