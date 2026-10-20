"""Quant comparison. The comparison owns protocol, pairing, and difference.

Arm JSON files are projections. Do not construct Comparison except through
assemble_comparison or parse_comparison. This module does not assign
weight_change_claimed and does not call build_scored_report.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sft_loop.report import MODEL_ID, REFUSAL_SENTENCE
from sft_loop.task import prompt_for, target_for

PINNED_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SPLIT_ID = "quant-heldout-v1"
MIN_ITEMS = 100
BOOTSTRAP_DRAWS = 1000
WARMUP_ITERS = 2
REPEATS_PER_ITEM = 5
THREADS = 1
DOCUMENT_KIND = "quant-comparison"
PROJECTION_KIND = "quant-arm-projection"
METRIC_ROLE = "task_score_not_safety"
ARM_BUDGET_S = 3600
NF4_CUDA_DETAIL = (
    "NF4 needs CUDA. torch.cuda.is_available is false on this host. "
    "No NF4 numbers were produced. " + REFUSAL_SENTENCE
)
NF4_DEP_DETAIL = (
    "NF4 needs torch or bitsandbytes. The import failed. "
    "No NF4 numbers were produced. " + REFUSAL_SENTENCE
)

__all__ = [
    "assemble_comparison",
    "parse_comparison",
    "validate_comparison",
    "project_tree",
    "render_table",
    "run_comparison",
    "probe_nf4",
    "tracked_relpaths",
    "assert_tracked_quant_file",
    "paired_difference",
]


class Precision(str, enum.Enum):
    FP32 = "fp32"
    INT8 = "int8"
    ONNX_INT8 = "onnx-int8"
    NF4 = "nf4"


class UnsupportedReason(str, enum.Enum):
    OOM = "oom"
    MISSING_CUDA = "missing_cuda"
    EXPORT_TIMEOUT = "export_timeout"
    DEPENDENCY_MISSING = "dependency_missing"
    PROBED_FAILURE = "probed_failure"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "schema" / "run_report.schema.json").is_file():
            return parent
    raise FileNotFoundError("schema/run_report.schema.json")


def quant_schema_path() -> Path:
    return repo_root() / "schema" / "quant_comparison.schema.json"


def heldout_path() -> Path:
    return repo_root() / "evals" / "quant-heldout-v1.txt"


def canonical_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_item_ids() -> tuple[str, ...]:
    from sft_loop.task import EVAL_WORDS, TRAIN_WORDS

    words = tuple(
        line.strip()
        for line in heldout_path().read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(words) < MIN_ITEMS:
        raise ValueError("held-out split is shorter than 100 items")
    if len(set(words)) != len(words):
        raise ValueError("held-out split has duplicate item ids")
    overlap = set(words) & (set(TRAIN_WORDS) | set(EVAL_WORDS))
    if overlap:
        raise ValueError("held-out split overlaps TRAIN_WORDS or EVAL_WORDS")
    return words


def load_protocol(*, revision: str = PINNED_REVISION, seed: int = 0) -> dict[str, Any]:
    if not isinstance(revision, str) or len(revision) != 40 or any(
        ch not in "0123456789abcdef" for ch in revision
    ):
        raise ValueError("revision must be 40 lowercase hex characters")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be an integer >= 0")
    item_ids = load_item_ids()
    return {
        "model_tier": "pinned",
        "model_id": MODEL_ID,
        "revision": revision,
        "seed": seed,
        "split_id": SPLIT_ID,
        "split_sha256": sha256_text(heldout_path().read_text(encoding="utf-8")),
        "item_ids": list(item_ids),
        "grader": "exact_match",
        "decode": {"do_sample": False, "max_new_tokens": 8},
        "threads": THREADS,
        "warmup_iters": WARMUP_ITERS,
        "repeats_per_item": REPEATS_PER_ITEM,
        "metric_role": METRIC_ROLE,
    }


def _require_detail(text: str) -> str:
    if REFUSAL_SENTENCE not in text:
        raise ValueError("unsupported detail must include the refusal sentence")
    if not 1 <= len(text) <= 500:
        raise ValueError("unsupported detail length")
    return text


def _round(value: float) -> float:
    return float(format(float(value), ".8g"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _se(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    if all(value == values[0] for value in values):
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _percentile_nearest(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty list")
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def _bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    low_i = int(0.025 * (BOOTSTRAP_DRAWS - 1))
    high_i = int(0.975 * (BOOTSTRAP_DRAWS - 1))
    return _round(means[low_i]), _round(means[high_i])


def _summarize_items(items: list[dict[str, Any]]) -> dict[str, float]:
    correct = [1.0 if row["correct"] else 0.0 for row in items]
    latencies = [float(row["latency_ms"]) for row in items]
    tokens = sum(int(row["generated_tokens"]) for row in items)
    seconds = sum(float(row["latency_ms"]) for row in items) / 1000.0
    return {
        "task_score": _round(_mean(correct)),
        "task_score_se": _round(_se(correct)),
        "latency_p50_ms": _round(_percentile_nearest(latencies, 0.5)),
        "latency_p95_ms": _round(_percentile_nearest(latencies, 0.95)),
        "tokens_per_s": _round(tokens / seconds if seconds else 0.0),
    }


def _check_item_alignment(protocol: dict[str, Any], items: list[dict[str, Any]]) -> None:
    expected = list(protocol["item_ids"])
    got = [row["item_id"] for row in items]
    if got != expected:
        raise ValueError("item rows must equal protocol.item_ids in order")


def measured_arm(
    *,
    precision: Precision,
    items: list[dict[str, Any]],
    on_disk_bytes: int,
    peak_rss_bytes: int,
    libraries: dict[str, str],
    limiter: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if precision is Precision.NF4:
        raise ValueError("nf4 cannot be a measured arm")
    _check_item_alignment(protocol, items)
    summary = _summarize_items(items)
    raw = canonical_dumps({"items": items})
    body = {
        "status": "measured",
        "precision": precision.value,
        "on_disk_bytes": int(on_disk_bytes),
        "peak_rss_bytes": int(peak_rss_bytes),
        "libraries": libraries,
        "limiter": limiter,
        "items": items,
        "raw_sha256": sha256_text(raw),
        **summary,
    }
    return body


def unsupported_arm(
    *,
    precision: Precision,
    reason: UnsupportedReason,
    detail: str,
    probe: dict[str, Any],
) -> dict[str, Any]:
    if precision is Precision.NF4 and reason not in {
        UnsupportedReason.MISSING_CUDA,
        UnsupportedReason.DEPENDENCY_MISSING,
    }:
        raise ValueError("nf4 reason must be missing_cuda or dependency_missing")
    return {
        "status": "unsupported",
        "precision": precision.value,
        "reason": reason.value,
        "detail": _require_detail(detail),
        "probe": probe,
    }


def paired_difference(
    protocol: dict[str, Any],
    fp32: dict[str, Any],
    int8: dict[str, Any],
) -> dict[str, Any]:
    if fp32.get("status") != "measured" or int8.get("status") != "measured":
        raise ValueError("paired_difference needs two measured arms")
    _check_item_alignment(protocol, fp32["items"])
    _check_item_alignment(protocol, int8["items"])
    score_vals = []
    latency_vals = []
    nll_vals = []
    have_nll = True
    for left, right in zip(fp32["items"], int8["items"]):
        if left["item_id"] != right["item_id"]:
            raise ValueError("paired item ids drifted")
        score_vals.append(float(right["correct"]) - float(left["correct"]))
        latency_vals.append(float(right["latency_ms"]) - float(left["latency_ms"]))
        if "nll" not in left or "nll" not in right:
            have_nll = False
        else:
            nll_vals.append(float(right["nll"]) - float(left["nll"]))
    fp32_tps = float(_summarize_items(fp32["items"])["tokens_per_s"])
    int8_tps = float(_summarize_items(int8["items"])["tokens_per_s"])
    low, high = _bootstrap_ci(score_vals, int(protocol["seed"]))
    outcome = {
        "kind": "paired",
        "n": len(score_vals),
        "score_delta": _round(_mean(score_vals)),
        "score_delta_se": _round(_se(score_vals)),
        "score_delta_ci95_low": low,
        "score_delta_ci95_high": high,
        "latency_ms_delta": _round(_mean(latency_vals)),
        "latency_ms_delta_se": _round(_se(latency_vals)),
        "nll_delta": _round(_mean(nll_vals)) if have_nll else None,
        "nll_delta_se": _round(_se(nll_vals)) if have_nll else None,
        "tokens_per_s_ratio": _round(int8_tps / fp32_tps if fp32_tps else 0.0),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": int(protocol["seed"]),
        "method": "paired-item-mean",
    }
    return outcome


def assemble_comparison(
    protocol: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    expected = {p.value for p in Precision}
    if set(arms) != expected:
        raise ValueError("arms must contain fp32, int8, onnx-int8, and nf4")
    nf4 = arms[Precision.NF4.value]
    if nf4.get("status") != "unsupported":
        raise ValueError("nf4 must be unsupported")
    if nf4.get("precision") != Precision.NF4.value:
        raise ValueError("nf4 slot precision mismatch")
    if nf4.get("reason") not in {
        UnsupportedReason.MISSING_CUDA.value,
        UnsupportedReason.DEPENDENCY_MISSING.value,
    }:
        raise ValueError("nf4 reason must be missing_cuda or dependency_missing")
    _require_detail(str(nf4.get("detail", "")))
    for key in ("fp32", "int8", "onnx-int8"):
        arm = arms[key]
        if arm.get("precision") != key:
            raise ValueError(f"{key} slot precision mismatch")
        if arm.get("status") == "measured":
            _check_item_alignment(protocol, arm["items"])
            summary = _summarize_items(arm["items"])
            raw = canonical_dumps({"items": arm["items"]})
            if arm.get("raw_sha256") != sha256_text(raw):
                raise ValueError(f"{key} raw_sha256 does not match items")
            for field, value in summary.items():
                if _round(float(arm[field])) != value:
                    raise ValueError(f"{key} {field} does not match items")
        elif arm.get("status") == "unsupported":
            for banned in (
                "items",
                "on_disk_bytes",
                "peak_rss_bytes",
                "task_score",
                "latency_p50_ms",
                "latency_p95_ms",
                "tokens_per_s",
            ):
                if banned in arm:
                    raise ValueError(f"unsupported {key} has {banned}")
            _require_detail(str(arm.get("detail", "")))
        else:
            raise ValueError(f"{key} has unknown status")
    fp32 = arms["fp32"]
    int8 = arms["int8"]
    if fp32.get("status") == "measured" and int8.get("status") == "measured":
        outcome = paired_difference(protocol, fp32, int8)
    elif fp32.get("status") != "measured":
        outcome = {"kind": "unavailable", "reason": "fp32_unmeasured"}
    else:
        outcome = {"kind": "unavailable", "reason": "int8_unmeasured"}
    document = {
        "schema_version": "1",
        "document_kind": DOCUMENT_KIND,
        "model_id": MODEL_ID,
        "metric_role": METRIC_ROLE,
        "protocol": protocol,
        "hardware": hardware,
        "arms": arms,
        "outcome": outcome,
    }
    _validate_schema(document)
    return document


def _validate_schema(document: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads(quant_schema_path().read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda err: [str(part) for part in err.path],
    )
    if errors:
        raise ValueError("; ".join(err.message for err in errors))


def parse_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_schema(payload)
    stored = payload.get("outcome")
    rebuilt = assemble_comparison(payload["protocol"], payload["arms"], payload["hardware"])
    if rebuilt["outcome"] != stored:
        raise ValueError("stored outcome does not match recomputed paired difference")
    return rebuilt


def tracked_relpaths() -> frozenset[str]:
    names = [f"runs/quant/{p.value}.json" for p in Precision]
    raw = [f"runs/quant/raw/{p.value}.json" for p in Precision]
    return frozenset(["runs/quant/comparison.json", *names, *raw])


def project_tree(comparison: dict[str, Any]) -> dict[str, str]:
    parse_comparison(comparison)
    comparison_text = canonical_dumps(comparison)
    comparison_sha = sha256_text(comparison_text)
    protocol_sha = sha256_text(canonical_dumps(comparison["protocol"]))
    tree: dict[str, str] = {}
    for precision in Precision:
        body = comparison["arms"][precision.value]
        projection = {
            "document_kind": PROJECTION_KIND,
            "schema_version": "1",
            "comparison_sha256": comparison_sha,
            "protocol_sha256": protocol_sha,
            "precision": precision.value,
            "body": body,
        }
        tree[f"runs/quant/{precision.value}.json"] = canonical_dumps(projection)
        if body.get("status") == "measured":
            raw_body = {"items": body["items"]}
        else:
            raw_body = {"probe": body["probe"], "reason": body["reason"]}
        tree[f"runs/quant/raw/{precision.value}.json"] = canonical_dumps(raw_body)
    return tree


def validate_comparison(path: Path, *, rewrite_projections: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    comparison = parse_comparison(payload)
    root = repo_root()
    tree = project_tree(comparison)
    for rel, text in tree.items():
        dest = root / rel
        if rewrite_projections:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, dest)
            continue
        if not dest.is_file():
            raise ValueError(f"missing projection {rel}")
        if dest.read_text(encoding="utf-8") != text:
            raise ValueError(f"projection drift at {rel}")
    extra = []
    quant_dir = root / "runs" / "quant"
    if quant_dir.is_dir():
        for found in quant_dir.rglob("*.json"):
            rel = found.resolve().relative_to(root.resolve()).as_posix()
            if rel.endswith(".tmp"):
                continue
            if rel not in tracked_relpaths():
                extra.append(rel)
    if extra:
        raise ValueError("untracked JSON under runs/quant: " + ", ".join(sorted(extra)))
    return comparison


def assert_tracked_quant_file(root: Path, rel: str, document: dict[str, Any]) -> None:
    if rel not in tracked_relpaths():
        raise AssertionError(f"{rel} is not a tracked quant path")
    if "weight_change_claimed" in document:
        raise AssertionError(f"{rel} must not set weight_change_claimed")
    if "adaptiveSandboxQualified" in json.dumps(document):
        raise AssertionError(f"{rel} must not set adaptiveSandboxQualified")
    from sft_loop.report import validate_report

    try:
        validate_report(document)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{rel} must not validate as a run report")
    comparison_path = root / "runs" / "quant" / "comparison.json"
    if rel == "runs/quant/comparison.json":
        parse_comparison(document)
        return
    if not comparison_path.is_file():
        raise AssertionError("comparison.json is required to check projections")
    comparison = parse_comparison(json.loads(comparison_path.read_text(encoding="utf-8")))
    tree = project_tree(comparison)
    expected = tree.get(rel)
    if expected is None:
        raise AssertionError(f"{rel} is not a projection")
    if canonical_dumps(document) != expected:
        raise AssertionError(f"{rel} does not match project_tree")


def render_table(comparison: dict[str, Any]) -> str:
    parse_comparison(comparison)
    protocol = comparison["protocol"]
    lines = [
        f"Fixture `{protocol['split_id']}` sha256 `{protocol['split_sha256']}` "
        f"revision `{protocol['revision']}` seed `{protocol['seed']}` "
        f"items `{len(protocol['item_ids'])}` threads `{protocol['threads']}` "
        f"warmup `{protocol['warmup_iters']}` repeats `{protocol['repeats_per_item']}`.",
        "",
        "| precision | status | on-disk bytes | peak rss bytes | p50 ms | p95 ms | tokens/s | exact match | n | limiter |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for precision in Precision:
        arm = comparison["arms"][precision.value]
        if arm["status"] == "unsupported":
            lines.append(
                f"| {precision.value} | unsupported | unsupported | unsupported | "
                f"unsupported | unsupported | unsupported | unsupported | unsupported | "
                f"{arm['reason']} |"
            )
            continue
        lines.append(
            f"| {precision.value} | measured | {arm['on_disk_bytes']} | {arm['peak_rss_bytes']} | "
            f"{arm['latency_p50_ms']} | {arm['latency_p95_ms']} | {arm['tokens_per_s']} | "
            f"{arm['task_score']} | {len(arm['items'])} | {arm['limiter']['kind']} |"
        )
    lines.extend(
        [
            "",
            "| comparison | status | mean item delta | standard error | bootstrap low | bootstrap high | n |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    outcome = comparison["outcome"]
    if outcome["kind"] == "unavailable":
        lines.append(
            f"| int8 minus fp32 | unsupported | unsupported | unsupported | "
            f"unsupported | unsupported | {outcome['reason']} |"
        )
    else:
        lines.append(
            f"| int8 minus fp32 | measured | {outcome['score_delta']} | "
            f"{outcome['score_delta_se']} | {outcome['score_delta_ci95_low']} | "
            f"{outcome['score_delta_ci95_high']} | {outcome['n']} |"
        )
        lines.append("")
        lines.append(
            f"Paired method `{outcome['method']}` on the same {outcome['n']} items. "
            "A score delta whose interval covers 0 is no measurable difference."
        )
    return "\n".join(lines) + "\n"


def _marker_block(text: str) -> str:
    start = "<!-- quant-table:start -->"
    end = "<!-- quant-table:end -->"
    if start not in text or end not in text:
        raise ValueError("README is missing the quant table markers")
    before, rest = text.split(start, 1)
    body, after = rest.split(end, 1)
    del before, after
    return body.strip("\n") + "\n"


def splice_table(readme: Path, markdown: str) -> None:
    text = readme.read_text(encoding="utf-8")
    start = "<!-- quant-table:start -->"
    end = "<!-- quant-table:end -->"
    if start not in text or end not in text:
        raise ValueError("README is missing the quant table markers")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    readme.write_text(before + start + "\n" + markdown + end + after, encoding="utf-8")


def probe_hardware() -> dict[str, Any]:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "cpu_model": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count() or 1,
        "ram_bytes": int(ram),
        "cuda": False,
        "mps": False,
    }


def probe_nf4() -> dict[str, Any]:
    cuda_available = None
    try:
        import torch
    except Exception as exc:
        return unsupported_arm(
            precision=Precision.NF4,
            reason=UnsupportedReason.DEPENDENCY_MISSING,
            detail=NF4_DEP_DETAIL,
            probe={
                "exception_type": type(exc).__name__,
                "message": str(exc)[:200] or "torch import failed",
                "cuda_available": None,
            },
        )
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return unsupported_arm(
            precision=Precision.NF4,
            reason=UnsupportedReason.DEPENDENCY_MISSING,
            detail=NF4_DEP_DETAIL,
            probe={
                "exception_type": type(exc).__name__,
                "message": str(exc)[:200] or "cuda probe failed",
                "cuda_available": None,
            },
        )
    if not cuda_available:
        return unsupported_arm(
            precision=Precision.NF4,
            reason=UnsupportedReason.MISSING_CUDA,
            detail=NF4_CUDA_DETAIL,
            probe={
                "exception_type": "CUDAUnavailable",
                "message": "torch.cuda.is_available() is False",
                "cuda_available": False,
            },
        )
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        return unsupported_arm(
            precision=Precision.NF4,
            reason=UnsupportedReason.DEPENDENCY_MISSING,
            detail=NF4_DEP_DETAIL,
            probe={
                "exception_type": type(exc).__name__,
                "message": str(exc)[:200] or "bitsandbytes import failed",
                "cuda_available": True,
            },
        )
    return unsupported_arm(
        precision=Precision.NF4,
        reason=UnsupportedReason.MISSING_CUDA,
        detail=NF4_CUDA_DETAIL,
        probe={
            "exception_type": "CUDAUnavailable",
            "message": "this comparison does not publish NF4 numbers",
            "cuda_available": True,
        },
    )


def _commit(out_dir: Path, comparison: dict[str, Any]) -> None:
    root = repo_root()
    tree = project_tree(comparison)
    tree["runs/quant/comparison.json"] = canonical_dumps(comparison)
    for rel, text in tree.items():
        dest = (out_dir / Path(rel).name) if out_dir.resolve() != (root / "runs" / "quant").resolve() else root / rel
        if out_dir.resolve() == (root / "runs" / "quant").resolve():
            dest = root / rel
        else:
            if rel.endswith("comparison.json"):
                dest = out_dir / "comparison.json"
            elif "/raw/" in rel:
                dest = out_dir / "raw" / Path(rel).name
            else:
                dest = out_dir / Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dest)


def run_comparison(*, out_dir: Path | None = None) -> dict[str, Any]:
    dest = out_dir if out_dir is not None else repo_root() / "runs" / "quant"
    protocol = load_protocol()
    hardware = probe_hardware()
    arms = {
        Precision.FP32.value: _run_child(Precision.FP32, protocol),
        Precision.INT8.value: _run_child(Precision.INT8, protocol),
        Precision.ONNX_INT8.value: _run_child(Precision.ONNX_INT8, protocol),
        Precision.NF4.value: probe_nf4(),
    }
    comparison = assemble_comparison(protocol, arms, hardware)
    _commit(dest, comparison)
    return comparison


def _run_child(precision: Precision, protocol: dict[str, Any]) -> dict[str, Any]:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sft-quant-") as tmp:
        work = Path(tmp)
        protocol_path = work / "protocol.json"
        result_path = work / "result.json"
        protocol_path.write_text(canonical_dumps(protocol), encoding="utf-8")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(THREADS)
        env["MKL_NUM_THREADS"] = str(THREADS)
        env["OPENBLAS_NUM_THREADS"] = str(THREADS)
        env["NUMEXPR_NUM_THREADS"] = str(THREADS)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("HF_HOME", str(repo_root() / "hf_cache"))
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = [
            sys.executable,
            "-m",
            "sft_loop.quant",
            "_child",
            "--precision",
            precision.value,
            "--protocol",
            str(protocol_path),
            "--out",
            str(result_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(repo_root()),
                env=env,
                capture_output=True,
                text=True,
                timeout=ARM_BUDGET_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return unsupported_arm(
                precision=precision,
                reason=UnsupportedReason.EXPORT_TIMEOUT,
                detail=(
                    f"{precision.value} exceeded the {ARM_BUDGET_S}s arm budget. "
                    "No metrics were produced. " + REFUSAL_SENTENCE
                ),
                probe={
                    "exception_type": "TimeoutExpired",
                    "message": f"arm budget {ARM_BUDGET_S}s",
                    "cuda_available": False,
                },
            )
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return payload
        reason = UnsupportedReason.OOM if completed.returncode in {-9, 137} else UnsupportedReason.PROBED_FAILURE
        message = (completed.stderr or completed.stdout or "child left no result").strip()[:200] or "child left no result"
        return unsupported_arm(
            precision=precision,
            reason=reason,
            detail=(
                f"{precision.value} child exited {completed.returncode}. "
                "No metrics were produced. " + REFUSAL_SENTENCE
            ),
            probe={
                "exception_type": "ChildExit",
                "message": message,
                "cuda_available": False,
            },
        )


def _child_main(precision: str, protocol_path: Path, out_path: Path) -> int:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    kind = Precision(precision)
    try:
        if kind is Precision.FP32:
            body = _measure_fp32(protocol)
        elif kind is Precision.INT8:
            body = _measure_int8(protocol)
        elif kind is Precision.ONNX_INT8:
            body = _measure_onnx(protocol)
        else:
            body = probe_nf4()
    except MemoryError as exc:
        body = unsupported_arm(
            precision=kind,
            reason=UnsupportedReason.OOM,
            detail=f"{precision} hit MemoryError. No metrics were produced. " + REFUSAL_SENTENCE,
            probe={
                "exception_type": "MemoryError",
                "message": str(exc)[:200] or "MemoryError",
                "cuda_available": False,
            },
        )
    except Exception as exc:
        text = str(exc)
        if isinstance(exc, ModuleNotFoundError):
            reason = UnsupportedReason.DEPENDENCY_MISSING
        elif "out of memory" in text.lower():
            reason = UnsupportedReason.OOM
        else:
            reason = UnsupportedReason.PROBED_FAILURE
        body = unsupported_arm(
            precision=kind,
            reason=reason,
            detail=f"{precision} failed: {type(exc).__name__}. No metrics were produced. " + REFUSAL_SENTENCE,
            probe={
                "exception_type": type(exc).__name__,
                "message": text[:200] or type(exc).__name__,
                "cuda_available": False,
            },
        )
    out_path.write_text(canonical_dumps(body), encoding="utf-8")
    return 0


def _measure_prepared(protocol: dict[str, Any], generate, nll_fn, *, precision: Precision, on_disk_bytes: int, libraries: dict[str, str]) -> dict[str, Any]:
    import resource
    import time

    start_ram = _available_ram()
    start = time.perf_counter()
    cpu0 = time.process_time()
    items = []
    peak = 0
    for item_id in protocol["item_ids"]:
        prompt = prompt_for(item_id)
        target = target_for(item_id)
        latencies = []
        prediction = None
        tokens = 0
        for step in range(protocol["warmup_iters"] + protocol["repeats_per_item"]):
            text, n_tokens, elapsed_ms = generate(prompt)
            if step < protocol["warmup_iters"]:
                continue
            latencies.append(elapsed_ms)
            if prediction is None:
                prediction = text
                tokens = n_tokens
            elif text != prediction:
                raise RuntimeError("greedy prediction drifted across repeats")
        from sft_loop.task import prediction_token

        token = prediction_token(prediction or "")
        row: dict[str, Any] = {
            "item_id": item_id,
            "correct": token == target,
            "latency_ms": _round(_mean(latencies)),
            "generated_tokens": int(tokens),
        }
        if nll_fn is not None:
            row["nll"] = _round(float(nll_fn(prompt, target)))
        items.append(row)
        peak = max(peak, _rss_bytes())
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu0
    rss = max(peak, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    limiter_kind = "rss" if rss > start_ram * 0.8 else "cpu"
    return measured_arm(
        precision=precision,
        items=items,
        on_disk_bytes=on_disk_bytes,
        peak_rss_bytes=max(rss, 1),
        libraries=libraries,
        limiter={
            "kind": limiter_kind,
            "wall_s": _round(wall),
            "cpu_time_s": _round(cpu),
            "peak_rss_bytes": max(rss, 1),
            "available_ram_bytes_at_start": start_ram,
        },
        protocol=protocol,
    )


def _available_ram() -> int:
    try:
        return int(Path("/proc/meminfo").read_text(encoding="utf-8").split("MemAvailable:")[1].split("kB")[0]) * 1024
    except Exception:
        return 1


def _rss_bytes() -> int:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return 1
    return 1


def _load_causal_lm(protocol: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(protocol["threads"]))
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model_id"],
        revision=protocol["revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model_id"],
        revision=protocol["revision"],
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("cpu")
    model.eval()
    return torch, tokenizer, model


def _generate_fn(torch, tokenizer, model):
    def generate(prompt: str) -> tuple[str, int, float]:
        import time

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded, int(new_tokens.shape[0]), elapsed

    return generate


def _nll_fn(torch, tokenizer, model):
    def nll(prompt: str, target: str) -> float:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + target
        encoded = tokenizer(text, return_tensors="pt")
        labels = encoded["input_ids"].clone()
        with torch.no_grad():
            loss = model(input_ids=encoded["input_ids"], labels=labels).loss
        return float(loss.item())

    return nll


def _snapshot_bytes(protocol: dict[str, Any]) -> int:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            protocol["model_id"],
            revision=protocol["revision"],
            allow_patterns=["*.safetensors", "*.bin"],
        )
    )
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _measure_fp32(protocol: dict[str, Any]) -> dict[str, Any]:
    import torch
    from importlib.metadata import version

    torch_mod, tokenizer, model = _load_causal_lm(protocol)
    return _measure_prepared(
        protocol,
        _generate_fn(torch_mod, tokenizer, model),
        _nll_fn(torch_mod, tokenizer, model),
        precision=Precision.FP32,
        on_disk_bytes=max(_snapshot_bytes(protocol), 1),
        libraries={
            "kind": "torch",
            "torch": torch.__version__,
            "transformers": version("transformers"),
        },
    )


def _measure_int8(protocol: dict[str, Any]) -> dict[str, Any]:
    import torch
    from importlib.metadata import version

    torch_mod, tokenizer, model = _load_causal_lm(protocol)
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "sft-quant-int8.pt"
    torch.save(model.state_dict(), work)
    size = max(work.stat().st_size, 1)
    work.unlink(missing_ok=True)
    return _measure_prepared(
        protocol,
        _generate_fn(torch_mod, tokenizer, model),
        _nll_fn(torch_mod, tokenizer, model),
        precision=Precision.INT8,
        on_disk_bytes=size,
        libraries={
            "kind": "torch",
            "torch": torch.__version__,
            "transformers": version("transformers"),
        },
    )


def _measure_onnx(protocol: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx  # noqa: F401
        import onnxruntime as ort
        import torch
        from importlib.metadata import version
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:
        return unsupported_arm(
            precision=Precision.ONNX_INT8,
            reason=UnsupportedReason.DEPENDENCY_MISSING,
            detail="onnx or onnxruntime is missing. No metrics were produced. " + REFUSAL_SENTENCE,
            probe={
                "exception_type": type(exc).__name__,
                "message": str(exc)[:200] or "onnx import failed",
                "cuda_available": False,
            },
        )

    import gc
    import tempfile
    import time

    _, tokenizer, model = _load_causal_lm(protocol)

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

    wrapper = _LogitsOnly(model)
    wrapper.eval()
    sample = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(protocol["item_ids"][0])}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        return_tensors="pt",
    )
    work = Path(tempfile.mkdtemp(prefix="sft-quant-onnx-"))
    fp_path = work / "model.onnx"
    int8_path = work / "model.int8.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    del model, wrapper
    gc.collect()
    quantize_dynamic(str(fp_path), str(int8_path), weight_type=QuantType.QInt8)
    fp_path.unlink(missing_ok=True)
    size = max(int8_path.stat().st_size, 1)
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    def generate(prompt: str) -> tuple[str, int, float]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(text, return_tensors="np")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        start = time.perf_counter()
        new_tokens: list[int] = []
        for _ in range(int(protocol["decode"]["max_new_tokens"])):
            logits = sess.run(
                ["logits"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )[0]
            next_id = int(np.argmax(logits[0, -1]))
            new_tokens.append(next_id)
            input_ids = np.concatenate([input_ids, [[next_id]]], axis=1)
            attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
            if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
        elapsed = (time.perf_counter() - start) * 1000.0
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded, len(new_tokens), elapsed

    def nll(prompt: str, target: str) -> float:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_len = tokenizer(prefix, return_tensors="np")["input_ids"].shape[1]
        full = tokenizer(prefix + target, return_tensors="np")
        logits = sess.run(
            ["logits"],
            {
                "input_ids": full["input_ids"],
                "attention_mask": full["attention_mask"],
            },
        )[0]
        target_ids = full["input_ids"][0, prefix_len:]
        if target_ids.size == 0:
            return 0.0
        pred = logits[0, prefix_len - 1 : prefix_len - 1 + target_ids.shape[0]]
        shifted = pred - pred.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return float(-log_probs[np.arange(target_ids.shape[0]), target_ids].mean())

    try:
        return _measure_prepared(
            protocol,
            generate,
            nll,
            precision=Precision.ONNX_INT8,
            on_disk_bytes=size,
            libraries={
                "kind": "onnx",
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "onnx": version("onnx"),
                "onnxruntime": version("onnxruntime"),
            },
        )
    finally:
        int8_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinned-model quantization comparison.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, default=None)
    validate = sub.add_parser("validate")
    validate.add_argument("path", nargs="?", type=Path)
    validate.add_argument("--rewrite-projections", action="store_true")
    table = sub.add_parser("table")
    table.add_argument("--check", type=Path, default=None)
    table.add_argument("--write", type=Path, default=None)
    probe = sub.add_parser("probe-nf4")
    probe.add_argument("--dest", type=Path, default=None)
    child = sub.add_parser("_child")
    child.add_argument("--precision", required=True)
    child.add_argument("--protocol", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_comparison(out_dir=args.out)
            return 0
        if args.command == "validate":
            path = args.path or repo_root() / "runs" / "quant" / "comparison.json"
            validate_comparison(path, rewrite_projections=args.rewrite_projections)
            return 0
        if args.command == "table":
            comparison = parse_comparison(
                json.loads((repo_root() / "runs" / "quant" / "comparison.json").read_text(encoding="utf-8"))
            )
            markdown = render_table(comparison)
            target = args.write or args.check
            if args.write is not None:
                splice_table(args.write, markdown)
                return 0
            if args.check is not None:
                if _marker_block(target.read_text(encoding="utf-8")) != markdown:
                    return 2
                return 0
            sys.stdout.write(markdown)
            return 0
        if args.command == "probe-nf4":
            body = probe_nf4()
            text = canonical_dumps(body)
            if args.dest is not None:
                args.dest.parent.mkdir(parents=True, exist_ok=True)
                args.dest.write_text(text, encoding="utf-8")
            else:
                sys.stdout.write(text)
            return 0
        if args.command == "_child":
            return _child_main(args.precision, args.protocol, args.out)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
