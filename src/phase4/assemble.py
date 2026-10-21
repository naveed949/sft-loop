"""Phase 2: pair before/after item scores and write the sibling report."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from phase4.constants import (
    BACKEND,
    BOOTSTRAP_ROUNDS,
    CEILING,
    DDP_UNSUPPORTED_REASON,
    MIN_EVAL,
    TASK_ID,
    VERDICT_DETAIL_NO_FILING_UPLIFT,
    issuer_family,
)
from phase4.grade import METRIC_NAME, format_equivalent, grade, grade_one
from phase4.paths import BuildStopped, RefusedCeiling, refuse_phase3


def _verdict(mean: float, lo: float, hi: float) -> str:
    if lo <= 0.0 <= hi:
        return "no uplift"
    return "uplift" if mean > 0 else "drop"


def _ci_bounds(samples: list[float], rounds: int) -> tuple[float, float]:
    samples = sorted(samples)
    return samples[int(0.025 * (rounds - 1))], samples[int(0.975 * (rounds - 1))]


def paired_bootstrap(
    before_hits: list[int],
    after_hits: list[int],
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 0,
    min_n: int = MIN_EVAL,
) -> dict[str, float | int | str]:
    if len(before_hits) != len(after_hits) or len(before_hits) < min_n:
        raise BuildStopped("paired bootstrap needs matching item ids")
    deltas = [after - before for before, after in zip(before_hits, after_hits)]
    mean = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    samples: list[float] = []
    for _ in range(rounds):
        draw = [deltas[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    lo, hi = _ci_bounds(samples, rounds)
    return {
        "mean_delta": mean,
        "low": lo,
        "high": hi,
        "n": n,
        "rounds": rounds,
        "seed": seed,
        "verdict": _verdict(mean, lo, hi),
        "grouping": "item",
    }


def cluster_bootstrap(
    labels: list[str],
    before_hits: list[int],
    after_hits: list[int],
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 0,
) -> dict[str, float | int | str]:
    if len(labels) != len(before_hits) or len(before_hits) != len(after_hits):
        raise BuildStopped("cluster bootstrap needs aligned labels and hits")
    if not labels:
        raise BuildStopped("cluster bootstrap needs at least one filing")
    deltas = [after - before for before, after in zip(before_hits, after_hits)]
    groups: dict[str, list[int]] = defaultdict(list)
    for label, delta in zip(labels, deltas):
        groups[label].append(delta)
    keys = sorted(groups)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(rounds):
        drawn = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        sample: list[int] = []
        for key in drawn:
            sample.extend(groups[key])
        samples.append(sum(sample) / len(sample))
    lo, hi = _ci_bounds(samples, rounds)
    mean = sum(deltas) / len(deltas)
    return {
        "mean_delta": mean,
        "low": lo,
        "high": hi,
        "n": len(deltas),
        "n_clusters": len(keys),
        "rounds": rounds,
        "seed": seed,
        "verdict": _verdict(mean, lo, hi),
        "grouping": "filing",
    }


def leave_one_filing_out(labels: list[str], deltas: list[int]) -> list[dict]:
    rows = []
    for leave in sorted(set(labels)):
        keep = [delta for delta, label in zip(deltas, labels) if label != leave]
        rows.append(
            {
                "left_out": leave,
                "n": len(keep),
                "mean_delta": (sum(keep) / len(keep)) if keep else None,
            }
        )
    return rows


def _hits(rows: list[dict]) -> list[int]:
    return [grade_one(row.get("prediction", ""), row["completion"]) for row in rows]


def _format_breakdown(
    before_items: list[dict],
    after_items: list[dict],
    before_hits: list[int],
    after_hits: list[int],
    seed: int,
) -> dict:
    format_after = []
    substantive_after = []
    format_gains = 0
    substantive_gains = 0
    losses = 0
    for before_row, after_row, before_hit, after_hit in zip(
        before_items, after_items, before_hits, after_hits
    ):
        is_format_gain = (
            after_hit == 1
            and before_hit == 0
            and format_equivalent(before_row.get("prediction", ""), before_row["completion"])
        )
        if after_hit == 1 and before_hit == 0:
            if is_format_gain:
                format_gains += 1
            else:
                substantive_gains += 1
        if after_hit == 0 and before_hit == 1:
            losses += 1
        format_after.append(1 if is_format_gain else before_hit)
        substantive_after.append(before_hit if is_format_gain else after_hit)
    return {
        "format_gains": format_gains,
        "substantive_gains": substantive_gains,
        "losses": losses,
        "net_substantive": substantive_gains - losses,
        "format_ci": paired_bootstrap(before_hits, format_after, seed=seed, min_n=1),
        "substantive_ci": paired_bootstrap(before_hits, substantive_after, seed=seed, min_n=1),
    }


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
    stored_mismatches = sum(
        1
        for row, hit in zip(before_items, before_hits)
        if "hit" in row and int(row["hit"]) != hit
    ) + sum(
        1
        for row, hit in zip(after_items, after_hits)
        if "hit" in row and int(row["hit"]) != hit
    )
    before_score = grade([row.get("prediction", "") for row in before_items], [row["completion"] for row in before_items])
    after_score = grade([row.get("prediction", "") for row in after_items], [row["completion"] for row in after_items])
    if before_score >= CEILING:
        raise RefusedCeiling(f"baseline exact match {before_score} is at or above {CEILING}")
    seed = int(before_doc.get("seed", 0))
    item_ci = paired_bootstrap(before_hits, after_hits, seed=seed)
    filings = [str(row.get("doc_id") or row["item_id"]) for row in before_items]
    cluster_ci = cluster_bootstrap(filings, before_hits, after_hits, seed=seed)
    deltas = [after - before for before, after in zip(before_hits, after_hits)]
    families = [
        issuer_family(str(row.get("issuer") or ""), str(row.get("doc_id") or ""))
        for row in before_items
    ]
    keep = [index for index, family in enumerate(families) if family != "JPM"]
    sensitivity = {
        "excluded_family": "JPM",
        "excluded_n": len(before_items) - len(keep),
        "n": len(keep),
        "before_hits": sum(before_hits[index] for index in keep),
        "after_hits": sum(after_hits[index] for index in keep),
        "item_ci": paired_bootstrap(
            [before_hits[index] for index in keep],
            [after_hits[index] for index in keep],
            seed=seed,
            min_n=1,
        ),
        "cluster_ci": cluster_bootstrap(
            [filings[index] for index in keep],
            [before_hits[index] for index in keep],
            [after_hits[index] for index in keep],
            seed=seed,
        ),
    }
    report = {
        "schema_version": "phase4-1",
        "document_kind": "phase4-tenk-lora",
        "task_id": TASK_ID,
        "metric_name": METRIC_NAME,
        "status": "scored",
        "n": len(before_items),
        "seed": seed,
        "before": {
            "metric_name": METRIC_NAME,
            "task_score": before_score,
            "n": len(before_items),
            "hits": sum(before_hits),
        },
        "after": {
            "metric_name": METRIC_NAME,
            "task_score": after_score,
            "n": len(after_items),
            "hits": sum(after_hits),
        },
        "paired_ci": item_ci,
        "cluster_ci": cluster_ci,
        "leave_one_filing_out": leave_one_filing_out(filings, deltas),
        "format_breakdown": _format_breakdown(before_items, after_items, before_hits, after_hits, seed),
        "jpm_sensitivity": sensitivity,
        "stored_hit_mismatches": stored_mismatches,
        "uplift": cluster_ci["verdict"],
        "verdict_detail": (
            VERDICT_DETAIL_NO_FILING_UPLIFT
            if cluster_ci["verdict"] == "no uplift" and item_ci["mean_delta"] > 0
            else cluster_ci["verdict"]
        ),
    }
    if extra:
        report.update(extra)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _source_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def extra_from_raw(raw_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    env = json.loads((raw_dir / "env.json").read_text(encoding="utf-8"))
    history = json.loads((raw_dir / "trainer_history.json").read_text(encoding="utf-8"))
    fp16 = json.loads((raw_dir / "fp16.json").read_text(encoding="utf-8"))
    before = json.loads((raw_dir / "eval_before.json").read_text(encoding="utf-8"))
    source = _source_fields(raw_dir / "SOURCE.txt")
    adapter_line = (raw_dir / "adapter.sha256").read_text(encoding="utf-8").strip()
    digest, name = adapter_line.split()
    hist = history.get("history") or []
    interpolated = next((row["step"] for row in hist if float(row.get("loss") or 1.0) < 0.01), None)
    nll = dict(before.get("baseline_nll") or {})
    nll.setdefault("scope", "first_eval_row")
    nll.setdefault("n_rows", 1)
    gpu0 = (env.get("gpus") or [{}])[0]
    return {
        "adapter_written": True,
        "backend": BACKEND,
        "bf16": "unsupported",
        "ddp": "unsupported",
        "device": "cuda",
        "hardware": {
            "cuda": env.get("cuda"),
            "gpu_count": env.get("gpu_count"),
            "major": gpu0.get("major"),
            "minor": gpu0.get("minor"),
            "mixed_precision": env.get("mixed_precision"),
            "name": gpu0.get("name"),
            "torch": env.get("torch"),
            "total_memory": gpu0.get("total_memory"),
            "weight_dtype": env.get("weight_dtype"),
        },
        "model_id": env.get("model_id"),
        "model_revision": env.get("model_revision"),
        "train_proof": {
            "adapter_bytes": int(source["adapter_model.safetensors_bytes"]),
            "adapter_sha256": {name: digest},
            "backend": BACKEND,
            "baseline_nll": nll,
            "ddp": {
                "status": "unsupported",
                "reason": DDP_UNSUPPORTED_REASON,
            },
            "eval_at_step": history.get("global_step"),
            "first_overflow_module": ((fp16.get("overflow_modules") or [None])[0]),
            "fp16": fp16.get("status"),
            "fp16_reason": fp16.get("reason"),
            "global_step": history.get("global_step"),
            "grad_scaler": history.get("grad_scaler"),
            "kaggle_kernel": source.get("kaggle_kernel"),
            "kaggle_version": int(source["kaggle_version"]) if source.get("kaggle_version") else None,
            "kernel_sha": source.get("kernel_sha"),
            "mixed_precision": history.get("mixed_precision"),
            "nll_scope": "first_eval_row",
            "overfitting_risk": (
                "train loss interpolated by step 7; eval ran only at step 200"
            ),
            "tarball_sha256": source.get("tarball_sha256"),
            "train_loss_interpolated_by_step": interpolated,
            "wall_s": int(source["wall_s"]) if source.get("wall_s") else None,
            "weight_dtype": history.get("weight_dtype") or env.get("weight_dtype"),
        },
    }
