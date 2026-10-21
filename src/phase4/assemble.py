"""Phase 2: pair before/after item scores and write the sibling report."""

from __future__ import annotations

import json
import random
from pathlib import Path

from phase4.constants import BOOTSTRAP_ROUNDS, CEILING, MIN_EVAL, TASK_ID
from phase4.grade import METRIC_NAME, canonicalize, grade
from phase4.paths import BuildStopped, RefusedCeiling, refuse_phase3


def paired_bootstrap(
    before_hits: list[int],
    after_hits: list[int],
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 0,
) -> dict[str, float | int | str]:
    if len(before_hits) != len(after_hits) or len(before_hits) < MIN_EVAL:
        raise BuildStopped("paired bootstrap needs the same n >= 100 item ids")
    deltas = [after - before for before, after in zip(before_hits, after_hits)]
    mean = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    samples: list[float] = []
    for _ in range(rounds):
        draw = [deltas[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo = samples[int(0.025 * (rounds - 1))]
    hi = samples[int(0.975 * (rounds - 1))]
    verdict = "no uplift" if lo <= 0.0 <= hi else ("uplift" if mean > 0 else "drop")
    return {
        "mean_delta": mean,
        "low": lo,
        "high": hi,
        "n": n,
        "rounds": rounds,
        "seed": seed,
        "verdict": verdict,
    }


def _hits(rows: list[dict]) -> list[int]:
    return [int(row.get("hit", canonicalize(row.get("prediction", "")) == canonicalize(row["completion"]))) for row in rows]


def assemble(
    before_path: Path,
    after_path: Path,
    dest: Path,
    *,
    extra: dict | None = None,
) -> dict:
    refuse_phase3(dest)
    refuse_phase3(before_path)
    refuse_phase3(after_path)
    before_doc = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after_doc = json.loads(Path(after_path).read_text(encoding="utf-8"))
    before_items = before_doc["items"]
    after_items = after_doc["items"]
    if [row["item_id"] for row in before_items] != [row["item_id"] for row in after_items]:
        raise BuildStopped("before and after item ids must match and stay in the same order")
    if len(before_items) < MIN_EVAL:
        raise BuildStopped(f"eval n {len(before_items)} is below {MIN_EVAL}")
    before_hits = _hits(before_items)
    after_hits = _hits(after_items)
    before_score = grade([row.get("prediction", "") for row in before_items], [row["completion"] for row in before_items])
    after_score = grade([row.get("prediction", "") for row in after_items], [row["completion"] for row in after_items])
    if before_score >= CEILING:
        raise RefusedCeiling(f"baseline exact match {before_score} is at or above {CEILING}")
    ci = paired_bootstrap(before_hits, after_hits, seed=int(before_doc.get("seed", 0)))
    report = {
        "schema_version": "phase4-1",
        "document_kind": "phase4-tenk-lora",
        "task_id": TASK_ID,
        "metric_name": METRIC_NAME,
        "status": "scored",
        "n": len(before_items),
        "seed": before_doc.get("seed", 0),
        "before": {"metric_name": METRIC_NAME, "task_score": before_score, "n": len(before_items)},
        "after": {"metric_name": METRIC_NAME, "task_score": after_score, "n": len(after_items)},
        "paired_ci": ci,
        "uplift": ci["verdict"],
    }
    if extra:
        report.update(extra)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
