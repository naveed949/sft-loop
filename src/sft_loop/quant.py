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
from sft_loop.task import prediction_token, prompt_for, target_for

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
ARM_BUDGET_S = 7200
DECODE_NEW_TOKENS = 8
IGNORE_INDEX = -100
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
    "assert_prompt_and_generation_parity",
    "assert_kernel_arm_parity",
    "check_kernel_scored_parity",
    "load_kernel_scored_parity",
    "paired_difference",
    "apply_prompt_mask",
    "ingest_kernel_arm",
    "merge_ingested_arm",
    "cross_hardware_view",
    "is_kernel_evidence",
    "PAIRING_SPECS",
    "KERNEL_EVIDENCE_PREFIX",
    "additive_causal_mask",
]


class Precision(str, enum.Enum):
    FP32 = "fp32"
    INT8 = "int8"
    ONNX_INT8 = "onnx-int8"
    T4_FP32 = "t4-fp32"
    FP16 = "fp16"
    NF4 = "nf4"
    NF4_FP16 = "nf4-fp16"


class UnsupportedReason(str, enum.Enum):
    OOM = "oom"
    MISSING_CUDA = "missing_cuda"
    EXPORT_TIMEOUT = "export_timeout"
    DEPENDENCY_MISSING = "dependency_missing"
    PROBED_FAILURE = "probed_failure"
    KERNEL_OUTPUT_MISSING = "kernel_output_missing"


@dataclass(frozen=True)
class PairingSpec:
    key: str
    minus: str
    baseline: str
    require: Literal["same_cpu_host", "same_accelerator"]


PAIRING_SPECS: tuple[PairingSpec, ...] = (
    PairingSpec("int8-minus-fp32", "int8", "fp32", "same_cpu_host"),
    PairingSpec("nf4-minus-t4-fp32", "nf4", "t4-fp32", "same_accelerator"),
    PairingSpec("nf4-fp16-minus-t4-fp32", "nf4-fp16", "t4-fp32", "same_accelerator"),
)

GPU_PRECISIONS = frozenset({Precision.T4_FP32, Precision.FP16, Precision.NF4, Precision.NF4_FP16})
BASELINE_PRECISIONS = frozenset({Precision.T4_FP32})
KERNEL_DIR_NF4 = "kaggle/quant_nf4"
KERNEL_DIR_ONNX = "kaggle/quant_onnx_int8"
KERNEL_OUTPUT_NF4 = "runs/quant/kernel/quant_nf4"
KERNEL_EVIDENCE_PREFIX = "runs/quant/kernel/"
PUBLISHED_CPU_FP32_TARGET_NLL_MEAN = 0.79975264
ONNX_NLL_NEAR_FP32_MAX = 2.0
FP16_OVERFLOW_DETAIL = (
    "fp16 overflow on T4, 0/129. Soft-PASS is unused and is refused."
)
KERNEL_STUB_FILES = ("env.json", "progress.json", "log.txt")
KERNEL_ARM_FILES = {
    Precision.T4_FP32: ("t4-fp32/items.json", "t4-fp32/metrics.json"),
    Precision.FP16: ("fp16/items.json", "fp16/metrics.json"),
    Precision.NF4: ("nf4/items.json", "nf4/metrics.json"),
    Precision.NF4_FP16: ("nf4-fp16/items.json", "nf4-fp16/metrics.json"),
    Precision.ONNX_INT8: ("onnx-int8/items.json", "onnx-int8/metrics.json"),
}
GPU_COMPUTE_DTYPE = {
    Precision.T4_FP32: "float32",
    Precision.FP16: "float16",
    Precision.NF4: "float32",
    Precision.NF4_FP16: "float16",
}
NF4_QUANT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float32",
}
NF4_QUANT_CONFIGS = {
    Precision.NF4: NF4_QUANT_CONFIG,
    Precision.NF4_FP16: {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    },
}
KERNEL_MISSING_DETAIL = (
    "Kaggle kernel output is missing. No GPU or ONNX numbers were produced. "
    + REFUSAL_SENTENCE
)


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
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def is_kernel_evidence(rel: str) -> bool:
    return rel.startswith(KERNEL_EVIDENCE_PREFIX)


ITEM_ROW_KEYS = frozenset(
    {
        "item_id",
        "correct",
        "latency_ms",
        "generated_tokens",
        "decode_latency_ms",
        "decode_generated_tokens",
        "nll",
        "sequence_nll",
        "case_folded",
    }
)


def _sanitize_kernel_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in items:
        item = {key: row[key] for key in ITEM_ROW_KEYS if key in row}
        for key in ("nll", "sequence_nll"):
            if key not in item:
                continue
            try:
                number = float(item[key])
            except (TypeError, ValueError):
                del item[key]
                continue
            if not math.isfinite(number):
                del item[key]
        cleaned.append(item)
    return cleaned


def _require_grader_case_sensitive(metrics: dict[str, Any]) -> None:
    if metrics.get("grader_case_sensitive") is not True:
        raise ValueError(
            "kernel metrics must declare grader_case_sensitive=true; "
            "the first-token grader compares prediction_token to item_id.upper()"
        )


def _attach_case_folded(
    items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row.get("item_id")): row for row in predictions}
    attached: list[dict[str, Any]] = []
    for item in items:
        pred = by_id.get(item["item_id"])
        if pred is None:
            raise ValueError(f"predictions.json is missing {item['item_id']}")
        token = pred.get("prediction_token")
        if not isinstance(token, str):
            token = prediction_token(str(pred.get("prediction") or ""))
        row = dict(item)
        row["case_folded"] = token.casefold() == item["item_id"].casefold()
        attached.append(row)
    return attached


def assert_prompt_and_generation_parity(left: dict[str, Any], right: dict[str, Any]) -> None:
    if left.get("prompt_hashes") != right.get("prompt_hashes"):
        raise ValueError("prompt hashes drifted between arms")
    if left.get("generation_config") != right.get("generation_config"):
        raise ValueError("generation config drifted between arms")


def load_kernel_scored_parity(source: Path, precision: str) -> dict[str, Any]:
    arm = source / precision
    hashes_path = arm / "prompt_hashes.json"
    if not hashes_path.is_file():
        hashes_path = source / "prompt_hashes.json"
    config_path = arm / "generation_config.json"
    if not hashes_path.is_file():
        raise ValueError(f"{precision} prompt hashes missing")
    if not config_path.is_file():
        raise ValueError(f"{precision} generation config missing")
    return {
        "prompt_hashes": json.loads(hashes_path.read_text(encoding="utf-8")),
        "generation_config": json.loads(config_path.read_text(encoding="utf-8")),
    }


def assert_kernel_arm_parity(source: Path, left: str, right: str) -> None:
    assert_prompt_and_generation_parity(
        load_kernel_scored_parity(source, left),
        load_kernel_scored_parity(source, right),
    )


def check_kernel_scored_parity(source: Path) -> None:
    names = [
        precision.value
        for precision in GPU_PRECISIONS
        if (source / precision.value / "generation_config.json").is_file()
    ]
    if len(names) < 2:
        return
    first = names[0]
    left = load_kernel_scored_parity(source, first)
    for name in names[1:]:
        assert_prompt_and_generation_parity(left, load_kernel_scored_parity(source, name))


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
        "grader_case_sensitive": True,
        "decode": {"do_sample": False, "max_new_tokens": DECODE_NEW_TOKENS},
        "throughput_decode": {
            "do_sample": False,
            "min_new_tokens": DECODE_NEW_TOKENS,
            "max_new_tokens": DECODE_NEW_TOKENS,
        },
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


def _detail_from_external(text: object) -> str:
    cleaned = str(text).strip() if text is not None else ""
    if not cleaned:
        return KERNEL_MISSING_DETAIL
    if REFUSAL_SENTENCE not in cleaned:
        suffix = " " + REFUSAL_SENTENCE
        room = 500 - len(suffix)
        cleaned = cleaned[:room] + suffix
    return _require_detail(cleaned)


def _probe_from_external(probe: object, fallback: str) -> dict[str, Any]:
    body = probe if isinstance(probe, dict) else {}
    exception_type = str(body.get("exception_type") or "KernelProgress")[:80] or "KernelProgress"
    message = str(body.get("message") or fallback)[:400] or "kernel probe"
    cuda = body.get("cuda_available")
    if cuda is not None:
        cuda = bool(cuda)
    return {
        "exception_type": exception_type,
        "message": message,
        "cuda_available": cuda,
    }


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


def additive_causal_mask(attention_mask, dtype):
    import torch

    batch, seqlen = attention_mask.shape
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=attention_mask.device))
    keep = attention_mask.to(torch.bool)[:, None, None, :] & causal
    fill = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, seqlen, seqlen), dtype=dtype, device=attention_mask.device)
    return mask.masked_fill(~keep, fill)


def apply_prompt_mask(labels: list[int], prefix_len: int) -> list[int]:
    if prefix_len < 0 or prefix_len > len(labels):
        raise ValueError("prefix_len is outside the label sequence")
    return [IGNORE_INDEX] * prefix_len + list(labels[prefix_len:])


def _summarize_items(items: list[dict[str, Any]]) -> dict[str, float]:
    correct = [1.0 if row["correct"] else 0.0 for row in items]
    latencies = [float(row["latency_ms"]) for row in items]
    scored_tokens = [float(row["generated_tokens"]) for row in items]
    decode_tokens = sum(int(row["decode_generated_tokens"]) for row in items)
    decode_seconds = sum(float(row["decode_latency_ms"]) for row in items) / 1000.0
    summary = {
        "task_score": _round(_mean(correct)),
        "task_score_se": _round(_se(correct)),
        "latency_p50_ms": _round(_percentile_nearest(latencies, 0.5)),
        "latency_p95_ms": _round(_percentile_nearest(latencies, 0.95)),
        "mean_generated_tokens": _round(_mean(scored_tokens)),
        "decode_tokens_per_s": _round(decode_tokens / decode_seconds if decode_seconds else 0.0),
    }
    if items and all("case_folded" in row for row in items):
        folded = [1.0 if row["case_folded"] else 0.0 for row in items]
        summary["case_folded_match"] = _round(_mean(folded))
    return summary


def _check_item_alignment(protocol: dict[str, Any], items: list[dict[str, Any]]) -> None:
    expected = list(protocol["item_ids"])
    got = [row["item_id"] for row in items]
    if got != expected:
        raise ValueError("item rows must equal protocol.item_ids in order")
    expected_decode = int(protocol["throughput_decode"]["max_new_tokens"])
    for row in items:
        if int(row["decode_generated_tokens"]) != expected_decode:
            raise ValueError("decode pass must emit a fixed token count")


def measured_arm(
    *,
    precision: Precision,
    items: list[dict[str, Any]],
    on_disk_bytes: int,
    peak_rss_bytes: int,
    libraries: dict[str, str],
    limiter: dict[str, Any],
    protocol: dict[str, Any],
    hardware: dict[str, Any] | None = None,
    peak_gpu_bytes: int | None = None,
    kernel: dict[str, Any] | None = None,
    compute_dtype: str | None = None,
    quant_config: dict[str, Any] | None = None,
    finding: str | None = None,
) -> dict[str, Any]:
    _check_item_alignment(protocol, items)
    summary = _summarize_items(items)
    raw = canonical_dumps({"items": items})
    body: dict[str, Any] = {
        "status": "measured",
        "precision": precision.value,
        "on_disk_bytes": int(on_disk_bytes),
        "peak_rss_bytes": int(peak_rss_bytes),
        "libraries": libraries,
        "limiter": limiter,
        "items": items,
        "raw_sha256": sha256_text(raw),
        "grader_case_sensitive": True,
        **summary,
    }
    if precision in GPU_PRECISIONS:
        expected_dtype = GPU_COMPUTE_DTYPE[precision]
        if hardware is None or not hardware.get("accelerator"):
            raise ValueError("gpu arm needs hardware.accelerator")
        if peak_gpu_bytes is None:
            raise ValueError("gpu arm needs peak_gpu_bytes")
        if kernel is None:
            raise ValueError("gpu arm needs kernel placeholder")
        if compute_dtype != expected_dtype:
            raise ValueError(f"{precision.value} compute_dtype must be {expected_dtype}")
        body["hardware"] = hardware
        body["peak_gpu_bytes"] = int(peak_gpu_bytes)
        body["kernel"] = kernel
        body["compute_dtype"] = compute_dtype
        if precision in NF4_QUANT_CONFIGS:
            if quant_config != NF4_QUANT_CONFIGS[precision]:
                raise ValueError("nf4 quant_config must be the recorded BitsAndBytes NF4 settings")
            body["quant_config"] = quant_config
        if finding is not None:
            if finding not in {"overflow", "collapse"}:
                raise ValueError("finding must be overflow or collapse")
            body["finding"] = finding
    elif precision is Precision.ONNX_INT8:
        if hardware is not None or peak_gpu_bytes is not None:
            raise ValueError("onnx arm must not carry gpu fields")
        if kernel is not None:
            body["kernel"] = kernel
        if finding is not None:
            if finding not in {"overflow", "collapse"}:
                raise ValueError("finding must be overflow or collapse")
            body["finding"] = finding
    elif hardware is not None or peak_gpu_bytes is not None or kernel is not None:
        raise ValueError("cpu arm must not carry gpu fields")
    return body


def unsupported_arm(
    *,
    precision: Precision,
    reason: UnsupportedReason,
    detail: str,
    probe: dict[str, Any],
    kernel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "status": "unsupported",
        "precision": precision.value,
        "reason": reason.value,
        "detail": _require_detail(detail),
        "probe": probe,
    }
    if kernel is not None:
        body["kernel"] = kernel
    return body


def _accelerator(arm: dict[str, Any]) -> str | None:
    hardware = arm.get("hardware")
    if not isinstance(hardware, dict):
        return None
    value = hardware.get("accelerator")
    return value if isinstance(value, str) and value else None


def paired_difference(
    protocol: dict[str, Any],
    minus_arm: dict[str, Any],
    baseline_arm: dict[str, Any],
    *,
    minus: str,
    baseline: str,
) -> dict[str, Any]:
    if minus_arm.get("status") != "measured" or baseline_arm.get("status") != "measured":
        raise ValueError("paired_difference needs two measured arms")
    spec = next((item for item in PAIRING_SPECS if item.minus == minus and item.baseline == baseline), None)
    if spec is None:
        raise ValueError(f"{minus} versus {baseline} is not a pairing")
    if spec.require == "same_accelerator":
        left = _accelerator(minus_arm)
        right = _accelerator(baseline_arm)
        if not left or left != right:
            raise ValueError("gpu pairing needs the same hardware.accelerator")
    _check_item_alignment(protocol, minus_arm["items"])
    _check_item_alignment(protocol, baseline_arm["items"])
    score_vals = []
    latency_vals = []
    nll_vals = []
    sequence_nll_vals = []
    folded_vals = []
    have_nll = True
    have_sequence = True
    have_folded = True
    for left, right in zip(baseline_arm["items"], minus_arm["items"]):
        if left["item_id"] != right["item_id"]:
            raise ValueError("paired item ids drifted")
        score_vals.append(float(right["correct"]) - float(left["correct"]))
        latency_vals.append(float(right["latency_ms"]) - float(left["latency_ms"]))
        if "nll" not in left or "nll" not in right:
            have_nll = False
        else:
            nll_vals.append(float(right["nll"]) - float(left["nll"]))
        if "sequence_nll" not in left or "sequence_nll" not in right:
            have_sequence = False
        else:
            sequence_nll_vals.append(float(right["sequence_nll"]) - float(left["sequence_nll"]))
        if "case_folded" not in left or "case_folded" not in right:
            have_folded = False
        else:
            folded_vals.append(float(right["case_folded"]) - float(left["case_folded"]))
    low, high = _bootstrap_ci(score_vals, int(protocol["seed"]))
    folded_low, folded_high = _bootstrap_ci(folded_vals, int(protocol["seed"])) if have_folded else (None, None)
    return {
        "kind": "paired",
        "minus": minus,
        "baseline": baseline,
        "n": len(score_vals),
        "score_delta": _round(_mean(score_vals)),
        "score_delta_se": _round(_se(score_vals)),
        "score_delta_ci95_low": low,
        "score_delta_ci95_high": high,
        "latency_ms_delta": _round(_mean(latency_vals)),
        "latency_ms_delta_se": _round(_se(latency_vals)),
        "nll_delta": _round(_mean(nll_vals)) if have_nll else None,
        "nll_delta_se": _round(_se(nll_vals)) if have_nll else None,
        "sequence_nll_delta": _round(_mean(sequence_nll_vals)) if have_sequence else None,
        "sequence_nll_delta_se": _round(_se(sequence_nll_vals)) if have_sequence else None,
        "case_folded_delta": _round(_mean(folded_vals)) if have_folded else None,
        "case_folded_delta_se": _round(_se(folded_vals)) if have_folded else None,
        "case_folded_delta_ci95_low": folded_low,
        "case_folded_delta_ci95_high": folded_high,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": int(protocol["seed"]),
        "method": "paired-item-mean",
    }


def cross_hardware_view(*, minus: str, baseline: str) -> dict[str, Any]:
    return {
        "kind": "not_paired",
        "minus": minus,
        "baseline": baseline,
        "reason": "cross_hardware",
    }


def _outcome_for_spec(
    protocol: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    spec: PairingSpec,
) -> dict[str, Any]:
    minus_arm = arms[spec.minus]
    baseline_arm = arms[spec.baseline]
    if baseline_arm.get("status") != "measured":
        return {
            "kind": "unavailable",
            "minus": spec.minus,
            "baseline": spec.baseline,
            "reason": f"{spec.baseline}_unmeasured",
        }
    if minus_arm.get("status") != "measured":
        return {
            "kind": "unavailable",
            "minus": spec.minus,
            "baseline": spec.baseline,
            "reason": f"{spec.minus}_unmeasured",
        }
    if spec.require == "same_accelerator":
        left = _accelerator(minus_arm)
        right = _accelerator(baseline_arm)
        if not left or left != right:
            return {
                "kind": "not_paired",
                "minus": spec.minus,
                "baseline": spec.baseline,
                "reason": "accelerator_mismatch",
            }
    return paired_difference(
        protocol,
        minus_arm,
        baseline_arm,
        minus=spec.minus,
        baseline=spec.baseline,
    )


def _check_arm_body(protocol: dict[str, Any], key: str, arm: dict[str, Any]) -> None:
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
        if arm.get("grader_case_sensitive") is not True:
            raise ValueError(f"{key} measured arm must declare grader_case_sensitive")
        if key in {p.value for p in GPU_PRECISIONS}:
            if not _accelerator(arm):
                raise ValueError(f"{key} measured arm needs hardware.accelerator")
            if "peak_gpu_bytes" not in arm:
                raise ValueError(f"{key} measured arm needs peak_gpu_bytes")
        elif "hardware" in arm or "peak_gpu_bytes" in arm:
            raise ValueError(f"{key} cpu arm must not carry gpu fields")
        return
    if arm.get("status") == "unsupported":
        for banned in (
            "items",
            "on_disk_bytes",
            "peak_rss_bytes",
            "peak_gpu_bytes",
            "task_score",
            "latency_p50_ms",
            "latency_p95_ms",
            "tokens_per_s",
            "mean_generated_tokens",
            "decode_tokens_per_s",
        ):
            if banned in arm:
                raise ValueError(f"unsupported {key} has {banned}")
        _require_detail(str(arm.get("detail", "")))
        return
    raise ValueError(f"{key} has unknown status")


def _declare_grader_case_sensitive(
    protocol: dict[str, Any],
    arms: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    proto = dict(protocol)
    proto["grader_case_sensitive"] = True
    stamped: dict[str, dict[str, Any]] = {}
    for key, arm in arms.items():
        body = dict(arm)
        if body.get("status") == "measured":
            body["grader_case_sensitive"] = True
        stamped[key] = body
    return proto, stamped


def assemble_comparison(
    protocol: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    expected = {p.value for p in Precision}
    if set(arms) != expected:
        raise ValueError(
            "arms must contain fp32, int8, onnx-int8, t4-fp32, fp16, nf4, and nf4-fp16"
        )
    protocol, arms = _declare_grader_case_sensitive(protocol, arms)
    for key in expected:
        _check_arm_body(protocol, key, arms[key])
    outcomes = {spec.key: _outcome_for_spec(protocol, arms, spec) for spec in PAIRING_SPECS}
    document = {
        "schema_version": "1",
        "document_kind": DOCUMENT_KIND,
        "model_id": MODEL_ID,
        "metric_role": METRIC_ROLE,
        "protocol": protocol,
        "hardware": hardware,
        "arms": arms,
        "outcomes": outcomes,
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
    stored = payload.get("outcomes")
    rebuilt = assemble_comparison(payload["protocol"], payload["arms"], payload["hardware"])
    if rebuilt["outcomes"] != stored:
        raise ValueError("stored outcomes do not match recomputed pairings")
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
            if is_kernel_evidence(rel):
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


def _arm_table_row(precision: Precision, arm: dict[str, Any]) -> str:
    if arm["status"] == "unsupported":
        return (
            f"| {precision.value} | unsupported | unsupported | unsupported | unsupported | "
            f"unsupported | unsupported | unsupported | unsupported | unsupported | "
            f"unsupported | unsupported | unsupported | {arm['reason']} |"
        )
    gpu = arm.get("peak_gpu_bytes", "n/a")
    accelerator = "n/a"
    hardware = arm.get("hardware")
    if isinstance(hardware, dict) and hardware.get("accelerator"):
        accelerator = hardware["accelerator"]
    folded = arm.get("case_folded_match", "n/a")
    return (
        f"| {precision.value} | measured | {arm['on_disk_bytes']} | {arm['peak_rss_bytes']} | "
        f"{gpu} | {arm['latency_p50_ms']} | {arm['latency_p95_ms']} | "
        f"{arm['mean_generated_tokens']} | {arm['decode_tokens_per_s']} | {arm['task_score']} | "
        f"{folded} | {len(arm['items'])} | {accelerator} | {arm['limiter']['kind']} |"
    )


def _outcome_table_row(spec: PairingSpec, outcome: dict[str, Any], *, folded: bool = False) -> str:
    label = f"{spec.minus} minus {spec.baseline}"
    if outcome["kind"] == "paired":
        if folded:
            if outcome.get("case_folded_delta") is None:
                return (
                    f"| {label} | paired | n/a | n/a | n/a | n/a | {outcome['n']} |"
                )
            return (
                f"| {label} | paired | {outcome['case_folded_delta']} | "
                f"{outcome['case_folded_delta_se']} | "
                f"{outcome['case_folded_delta_ci95_low']} | "
                f"{outcome['case_folded_delta_ci95_high']} | {outcome['n']} |"
            )
        return (
            f"| {label} | paired | {outcome['score_delta']} | {outcome['score_delta_se']} | "
            f"{outcome['score_delta_ci95_low']} | {outcome['score_delta_ci95_high']} | "
            f"{outcome['n']} |"
        )
    return (
        f"| {label} | {outcome['kind']} | {outcome['kind']} | {outcome['kind']} | "
        f"{outcome['kind']} | {outcome['kind']} | {outcome['reason']} |"
    )


def render_table(comparison: dict[str, Any]) -> str:
    parse_comparison(comparison)
    protocol = comparison["protocol"]
    lines = [
        f"Fixture `{protocol['split_id']}` sha256 `{protocol['split_sha256']}` "
        f"revision `{protocol['revision']}` seed `{protocol['seed']}` "
        f"items `{len(protocol['item_ids'])}` threads `{protocol['threads']}` "
        f"warmup `{protocol['warmup_iters']}` repeats `{protocol['repeats_per_item']}` "
        f"scored decode greedy `max_new_tokens {protocol['decode']['max_new_tokens']}` "
        f"throughput decode `min_new_tokens={protocol['throughput_decode']['min_new_tokens']}` "
        f"`max_new_tokens={protocol['throughput_decode']['max_new_tokens']}`.",
        "",
        "| precision | status | on-disk bytes | peak rss bytes | peak gpu bytes | scored p50 ms | scored p95 ms | mean generated tokens | decode tokens/s (min=max=8) | exact match | case-folded match | n | accelerator | limiter |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for precision in Precision:
        lines.append(_arm_table_row(precision, comparison["arms"][precision.value]))
    lines.extend(
        [
            "",
            "| comparison | status | mean item delta | standard error | bootstrap low | bootstrap high | n |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    outcomes = comparison["outcomes"]
    for spec in PAIRING_SPECS:
        lines.append(_outcome_table_row(spec, outcomes[spec.key]))
    lines.extend(
        [
            "",
            "| case-folded comparison | status | mean item delta | standard error | bootstrap low | bootstrap high | n |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for spec in PAIRING_SPECS:
        lines.append(_outcome_table_row(spec, outcomes[spec.key], folded=True))
    lines.extend(
        [
            "",
            "Paired rows share one hardware class. NF4 is paired only with fp32 on the same T4. "
            "int8 is paired only with CPU fp32. GPU versus CPU is not paired. "
            "T4 fp16 is not a pairing baseline. "
            "scored p50/p95 allow EOS. decode tokens/s is a separate fixed-length pass. "
            "Exact match is case-sensitive first-token equality with `item_id.upper()`. "
            "Case-folded match is first-token equality after `str.casefold()`. "
            "n/a means the arm has no raw generations to rescore.",
        ]
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


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                name = line.split(":", 1)[1].strip()
                if name:
                    return name[:200]
    except OSError:
        pass
    return (platform.processor() or platform.machine() or "unknown")[:200]


def kernel_placeholder(rel_dir: str) -> dict[str, Any]:
    return {"dir": rel_dir, "url": None, "version": None}


def _unsupported_from_progress(
    precision: Precision,
    kernel: dict[str, Any],
    progress: dict[str, Any],
    arm_state: dict[str, Any],
) -> dict[str, Any]:
    reason_text = str(arm_state.get("reason") or progress.get("reason") or "probed_failure")
    stage = str(progress.get("stage") or "")
    if reason_text == "probed_failure" and stage in {"export", "quantize"}:
        reason_text = "oom"
    try:
        reason = UnsupportedReason(reason_text)
    except ValueError:
        reason = UnsupportedReason.PROBED_FAILURE
    raw_detail = arm_state.get("detail") or progress.get("detail")
    return unsupported_arm(
        precision=precision,
        reason=reason,
        detail=_detail_from_external(raw_detail),
        probe=_probe_from_external(
            arm_state.get("probe") or progress.get("probe"),
            str(raw_detail or reason_text),
        ),
        kernel=kernel,
    )


def attach_kernel_ref(
    arm: dict[str, Any],
    *,
    url: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    if url is None and version is None:
        return arm
    body = dict(arm)
    kernel = dict(body.get("kernel") or {})
    if not kernel.get("dir"):
        kernel["dir"] = (
            KERNEL_DIR_NF4
            if body.get("precision") in {item.value for item in GPU_PRECISIONS}
            else KERNEL_DIR_ONNX
        )
    if url is not None:
        kernel["url"] = url
    if version is not None:
        kernel["version"] = version
    if "url" not in kernel:
        kernel["url"] = None
    if "version" not in kernel:
        kernel["version"] = None
    body["kernel"] = kernel
    return body


def merge_ingested_arm(
    dest: Path,
    precision: Precision,
    arm: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads((dest / "comparison.json").read_text(encoding="utf-8"))
    current = parse_comparison(payload)
    arms = dict(current["arms"])
    arms[precision.value] = arm
    comparison = assemble_comparison(current["protocol"], arms, current["hardware"])
    _commit(dest, comparison)
    return comparison


def kernel_dir_for(precision: Precision) -> Path:
    if precision in GPU_PRECISIONS:
        return repo_root() / KERNEL_DIR_NF4
    if precision is Precision.ONNX_INT8:
        return repo_root() / KERNEL_DIR_ONNX
    raise ValueError(f"{precision.value} is not a kernel arm")


def _items_have_finite_nll(items: list[dict[str, Any]]) -> bool:
    for row in items:
        nll = row.get("nll")
        sequence = row.get("sequence_nll")
        if not isinstance(nll, (int, float)) or not math.isfinite(float(nll)):
            return False
        if not isinstance(sequence, (int, float)) or not math.isfinite(float(sequence)):
            return False
    return True


def _mean_finite_nll(items: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in items:
        nll = row.get("nll")
        if not isinstance(nll, (int, float)) or not math.isfinite(float(nll)):
            return None
        values.append(float(nll))
    if not values:
        return None
    return sum(values) / len(values)


def _onnx_nll_is_collapse(items: list[dict[str, Any]]) -> bool:
    mean_nll = _mean_finite_nll(items)
    return mean_nll is not None and mean_nll > ONNX_NLL_NEAR_FP32_MAX


def _allowed_broken_finding(precision: Precision, finding: object) -> bool:
    if precision in BASELINE_PRECISIONS:
        return finding == "collapse"
    if precision is Precision.FP16:
        return False
    return finding in {"overflow", "collapse"}


def _refuse_broken_kernel_arm(
    precision: Precision,
    items: list[dict[str, Any]],
    finding: object,
    kernel: dict[str, Any],
) -> dict[str, Any] | None:
    summary = _summarize_items(items)
    finite = _items_have_finite_nll(items)
    broken = summary["task_score"] == 0.0 or not finite
    if precision is Precision.FP16 and broken:
        return unsupported_arm(
            precision=precision,
            reason=UnsupportedReason.PROBED_FAILURE,
            detail=FP16_OVERFLOW_DETAIL,
            probe={
                "exception_type": "Fp16Overflow",
                "message": "fp16 overflow on T4, 0/129",
                "cuda_available": True,
            },
            kernel=kernel,
        )
    if precision is Precision.ONNX_INT8 and broken:
        if finite and _onnx_nll_is_collapse(items):
            return None
        mean_nll = _mean_finite_nll(items)
        return unsupported_arm(
            precision=precision,
            reason=UnsupportedReason.PROBED_FAILURE,
            detail=(
                f"{precision.value} scored {summary['task_score']} with "
                f"{'finite' if finite else 'non-finite'} NLL"
                f"{'' if mean_nll is None else f' mean {mean_nll:.4f}'}. "
                "A zero score with teacher NLL near CPU fp32 "
                f"({PUBLISHED_CPU_FP32_TARGET_NLL_MEAN}) is a probable generate-loop "
                "bug, not a quantization result. "
                + REFUSAL_SENTENCE
            ),
            probe={
                "exception_type": "BrokenBaseline",
                "message": f"{precision.value} score={summary['task_score']}",
                "cuda_available": False,
            },
            kernel=kernel,
        )
    if not broken or _allowed_broken_finding(precision, finding):
        return None
    return unsupported_arm(
        precision=precision,
        reason=UnsupportedReason.PROBED_FAILURE,
        detail=(
            f"{precision.value} scored {summary['task_score']} with "
            f"{'finite' if finite else 'non-finite'} NLL. "
            "A zero-score or non-finite baseline is a harness failure unless marked "
            "as a collapse finding. No GPU pairing numbers were produced. "
            + REFUSAL_SENTENCE
        ),
        probe={
            "exception_type": "BrokenBaseline",
            "message": f"{precision.value} score={summary['task_score']}",
            "cuda_available": True,
        },
        kernel=kernel,
    )


def ingest_kernel_arm(source: Path, precision: Precision) -> dict[str, Any]:
    kernel = kernel_placeholder(
        KERNEL_DIR_NF4 if precision in GPU_PRECISIONS else KERNEL_DIR_ONNX
    )
    if not source.is_dir():
        return unsupported_arm(
            precision=precision,
            reason=UnsupportedReason.KERNEL_OUTPUT_MISSING,
            detail=KERNEL_MISSING_DETAIL,
            probe={
                "exception_type": "KernelOutputMissing",
                "message": f"{source} is not a directory",
                "cuda_available": None,
            },
            kernel=kernel,
        )
    progress_path = source / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        arm_state = progress.get("arms", {}).get(precision.value, {})
        status = arm_state.get("status") or progress.get("status")
        if status != "measured":
            return _unsupported_from_progress(precision, kernel, progress, arm_state)
    stub_missing = [name for name in KERNEL_STUB_FILES if not (source / name).is_file()]
    if stub_missing:
        return unsupported_arm(
            precision=precision,
            reason=UnsupportedReason.KERNEL_OUTPUT_MISSING,
            detail=KERNEL_MISSING_DETAIL,
            probe={
                "exception_type": "KernelOutputMissing",
                "message": "missing " + ", ".join(stub_missing),
                "cuda_available": None,
            },
            kernel=kernel,
        )
    arm_missing = [name for name in KERNEL_ARM_FILES[precision] if not (source / name).is_file()]
    if arm_missing:
        return unsupported_arm(
            precision=precision,
            reason=UnsupportedReason.KERNEL_OUTPUT_MISSING,
            detail=KERNEL_MISSING_DETAIL,
            probe={
                "exception_type": "KernelOutputMissing",
                "message": "missing " + ", ".join(arm_missing),
                "cuda_available": None,
            },
            kernel=kernel,
        )
    check_kernel_scored_parity(source)
    items = _sanitize_kernel_items(
        json.loads((source / precision.value / "items.json").read_text(encoding="utf-8"))
    )
    metrics = json.loads((source / precision.value / "metrics.json").read_text(encoding="utf-8"))
    _require_grader_case_sensitive(metrics)
    predictions_path = source / precision.value / "predictions.json"
    if predictions_path.is_file():
        items = _attach_case_folded(
            items,
            json.loads(predictions_path.read_text(encoding="utf-8")),
        )
    env = json.loads((source / "env.json").read_text(encoding="utf-8"))
    protocol = load_protocol()
    kernel_ref = {
        "dir": kernel["dir"],
        "url": env.get("kernel_url"),
        "version": env.get("kernel_version"),
    }
    gpu = precision in GPU_PRECISIONS
    finding = metrics.get("finding")
    refused = _refuse_broken_kernel_arm(precision, items, finding, kernel_ref)
    if refused is not None:
        return refused
    if (
        precision is Precision.ONNX_INT8
        and _summarize_items(items)["task_score"] == 0.0
        and _onnx_nll_is_collapse(items)
    ):
        finding = "collapse"
    return measured_arm(
        precision=precision,
        items=items,
        on_disk_bytes=int(metrics["on_disk_bytes"]),
        peak_rss_bytes=int(metrics["peak_rss_bytes"]),
        libraries=metrics["libraries"],
        limiter=metrics["limiter"],
        protocol=protocol,
        hardware=env.get("hardware") if gpu else None,
        peak_gpu_bytes=metrics.get("peak_gpu_bytes") if gpu else None,
        kernel=kernel_ref if gpu or precision is Precision.ONNX_INT8 else None,
        compute_dtype=metrics.get("compute_dtype") if gpu else None,
        quant_config=metrics.get("quant_config") if precision in NF4_QUANT_CONFIGS else None,
        finding=finding if finding in {"overflow", "collapse"} else None,
    )


def probe_hardware() -> dict[str, Any]:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "cpu_model": _cpu_model(),
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
    import importlib.util

    if importlib.util.find_spec("bitsandbytes") is None:
        return unsupported_arm(
            precision=Precision.NF4,
            reason=UnsupportedReason.DEPENDENCY_MISSING,
            detail=NF4_DEP_DETAIL,
            probe={
                "exception_type": "ModuleNotFoundError",
                "message": "No module named 'bitsandbytes'",
                "cuda_available": True,
            },
        )
    return unsupported_arm(
        precision=Precision.NF4,
        reason=UnsupportedReason.KERNEL_OUTPUT_MISSING,
        detail=KERNEL_MISSING_DETAIL,
        probe={
            "exception_type": "KernelOutputMissing",
            "message": "local CUDA is present but Kaggle kernel output is missing",
            "cuda_available": True,
        },
        kernel=kernel_placeholder(KERNEL_DIR_NF4),
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


def _reuse_arm(dest: Path, precision: Precision) -> dict[str, Any] | None:
    path = dest / "comparison.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = payload["arms"][precision.value]
    except (OSError, KeyError, json.JSONDecodeError, TypeError):
        return None
    if arm.get("precision") == precision.value and arm.get("status") in {"unsupported", "measured"}:
        return arm
    return None


def run_comparison(*, out_dir: Path | None = None, reuse_unsupported: bool = True) -> dict[str, Any]:
    dest = out_dir if out_dir is not None else repo_root() / "runs" / "quant"
    protocol = load_protocol()
    hardware = probe_hardware()
    onnx = _reuse_arm(dest, Precision.ONNX_INT8) if reuse_unsupported else None

    def gpu_arm(precision: Precision) -> dict[str, Any]:
        reused = _reuse_arm(dest, precision)
        if reused is not None:
            return reused
        if precision is Precision.NF4:
            return probe_nf4()
        return ingest_kernel_arm(repo_root() / KERNEL_OUTPUT_NF4, precision)

    arms = {
        Precision.FP32.value: _run_child(Precision.FP32, protocol),
        Precision.INT8.value: _run_child(Precision.INT8, protocol),
        Precision.ONNX_INT8.value: onnx if onnx is not None else _run_child(Precision.ONNX_INT8, protocol),
        Precision.T4_FP32.value: gpu_arm(Precision.T4_FP32),
        Precision.FP16.value: gpu_arm(Precision.FP16),
        Precision.NF4.value: gpu_arm(Precision.NF4),
        Precision.NF4_FP16.value: gpu_arm(Precision.NF4_FP16),
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


def _repeat_generate(generate, prompt: str, *, warmup: int, repeats: int) -> tuple[str, int, float]:
    latencies = []
    prediction = None
    tokens = 0
    for step in range(warmup + repeats):
        text, n_tokens, elapsed_ms = generate(prompt)
        if step < warmup:
            continue
        latencies.append(elapsed_ms)
        if prediction is None:
            prediction = text
            tokens = n_tokens
        elif text != prediction:
            raise RuntimeError("greedy prediction drifted across repeats")
        elif n_tokens != tokens:
            raise RuntimeError("generated token count drifted across repeats")
    return prediction or "", int(tokens), _round(_mean(latencies))


def _measure_prepared(
    protocol: dict[str, Any],
    generate,
    decode_generate,
    nll_fn,
    *,
    precision: Precision,
    on_disk_bytes: int,
    libraries: dict[str, str],
) -> dict[str, Any]:
    import resource
    import time

    start_ram = _available_ram()
    start = time.perf_counter()
    cpu0 = time.process_time()
    items = []
    peak = 0
    warmup = int(protocol["warmup_iters"])
    repeats = int(protocol["repeats_per_item"])
    for item_id in protocol["item_ids"]:
        prompt = prompt_for(item_id)
        target = target_for(item_id)
        prediction, tokens, latency_ms = _repeat_generate(
            generate, prompt, warmup=warmup, repeats=repeats
        )
        _, decode_tokens, decode_latency_ms = _repeat_generate(
            decode_generate, prompt, warmup=warmup, repeats=repeats
        )
        from sft_loop.task import prediction_token

        token = prediction_token(prediction)
        row: dict[str, Any] = {
            "item_id": item_id,
            "correct": token == target,
            "latency_ms": latency_ms,
            "generated_tokens": int(tokens),
            "decode_latency_ms": decode_latency_ms,
            "decode_generated_tokens": int(decode_tokens),
        }
        if nll_fn is not None:
            target_nll, sequence_nll = nll_fn(prompt, target)
            row["nll"] = _round(float(target_nll))
            row["sequence_nll"] = _round(float(sequence_nll))
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


def _generate_fn(torch, tokenizer, model, *, min_new_tokens: int | None = None):
    def generate(prompt: str) -> tuple[str, int, float]:
        import time

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        kwargs = {
            "max_new_tokens": DECODE_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if min_new_tokens is not None:
            kwargs["min_new_tokens"] = min_new_tokens
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**encoded, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded, int(new_tokens.shape[0]), elapsed

    return generate


def _nll_fn(torch, tokenizer, model):
    def nll(prompt: str, target: str) -> tuple[float, float]:
        F = torch.nn.functional

        messages = [{"role": "user", "content": prompt}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_len = tokenizer(prefix, return_tensors="pt")["input_ids"].shape[1]
        encoded = tokenizer(prefix + target, return_tensors="pt")
        labels = encoded["input_ids"].clone()
        target_labels = labels.clone()
        target_labels[:, :prefix_len] = IGNORE_INDEX
        with torch.no_grad():
            logits = model(input_ids=encoded["input_ids"]).logits
        shift_logits = logits[:, :-1, :].contiguous()
        sequence_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=IGNORE_INDEX,
        )
        target_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_labels[:, 1:].contiguous().view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return float(target_loss.item()), float(sequence_loss.item())

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
        _generate_fn(torch_mod, tokenizer, model, min_new_tokens=DECODE_NEW_TOKENS),
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
    model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig},
        dtype=torch.qint8,
    )
    model.eval()
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "sft-quant-int8.pt"
    torch.save(model.state_dict(), work)
    size = max(work.stat().st_size, 1)
    work.unlink(missing_ok=True)
    return _measure_prepared(
        protocol,
        _generate_fn(torch_mod, tokenizer, model),
        _generate_fn(torch_mod, tokenizer, model, min_new_tokens=DECODE_NEW_TOKENS),
        _nll_fn(torch_mod, tokenizer, model),
        precision=Precision.INT8,
        on_disk_bytes=size,
        libraries={
            "kind": "torch",
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "qconfig": "default_dynamic_qconfig",
        },
    )


def _measure_onnx(protocol: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
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
            mask4 = additive_causal_mask(attention_mask, next(self.inner.parameters()).dtype)
            return self.inner(
                input_ids=input_ids,
                attention_mask=mask4,
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
            dynamo=False,
        )
    del model, wrapper
    gc.collect()
    quantize_dynamic(str(fp_path), str(int8_path), weight_type=QuantType.QInt8)
    fp_path.unlink(missing_ok=True)
    size = max(int8_path.stat().st_size, 1)
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    def make_generate(*, stop_on_eos: bool):
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
            for _ in range(DECODE_NEW_TOKENS):
                logits = sess.run(
                    ["logits"],
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                )[0]
                next_id = int(np.argmax(logits[0, -1]))
                new_tokens.append(next_id)
                input_ids = np.concatenate([input_ids, [[next_id]]], axis=1)
                attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
                if (
                    stop_on_eos
                    and tokenizer.eos_token_id is not None
                    and next_id == int(tokenizer.eos_token_id)
                ):
                    break
            elapsed = (time.perf_counter() - start) * 1000.0
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return decoded, len(new_tokens), elapsed

        return generate

    def nll(prompt: str, target: str) -> tuple[float, float]:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(prefix, return_tensors="np")["input_ids"][0].tolist()
        full = tokenizer(prefix + target, return_tensors="np")
        logits = sess.run(
            ["logits"],
            {
                "input_ids": full["input_ids"],
                "attention_mask": full["attention_mask"],
            },
        )[0]
        label_ids = full["input_ids"][0].tolist()
        target_labels = apply_prompt_mask(label_ids, len(prefix_ids))
        pred = logits[0, :-1]
        gold = label_ids[1:]
        masked = target_labels[1:]
        shifted = pred - pred.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        sequence = float(-log_probs[np.arange(len(gold)), gold].mean()) if gold else 0.0
        keep = [i for i, lab in enumerate(masked) if lab != IGNORE_INDEX]
        target_nll = (
            float(-log_probs[keep, np.array(gold)[keep]].mean()) if keep else 0.0
        )
        return target_nll, sequence

    try:
        return _measure_prepared(
            protocol,
            make_generate(stop_on_eos=True),
            make_generate(stop_on_eos=False),
            nll,
            precision=Precision.ONNX_INT8,
            on_disk_bytes=size,
            libraries={
                "kind": "onnx",
                "torch": torch.__version__,
                "transformers": version("transformers"),
                "onnx": onnx.__version__,
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
    run.add_argument("--no-reuse-unsupported", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("path", nargs="?", type=Path)
    validate.add_argument("--rewrite-projections", action="store_true")
    table = sub.add_parser("table")
    table.add_argument("--check", type=Path, default=None)
    table.add_argument("--write", type=Path, default=None)
    probe = sub.add_parser("probe-nf4")
    probe.add_argument("--dest", type=Path, default=None)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--precision", required=True, choices=[p.value for p in Precision])
    ingest.add_argument("--from-dir", type=Path, required=True)
    ingest.add_argument("--write", action="store_true")
    ingest.add_argument("--kernel-url", default=None)
    ingest.add_argument("--kernel-version", default=None)
    child = sub.add_parser("_child")
    child.add_argument("--precision", required=True)
    child.add_argument("--protocol", type=Path, required=True)
    child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_comparison(
                out_dir=args.out,
                reuse_unsupported=not args.no_reuse_unsupported,
            )
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
        if args.command == "ingest":
            body = attach_kernel_ref(
                ingest_kernel_arm(args.from_dir, Precision(args.precision)),
                url=args.kernel_url,
                version=args.kernel_version,
            )
            if args.write:
                dest = repo_root() / "runs" / "quant"
                merge_ingested_arm(dest, Precision(args.precision), body)
            sys.stdout.write(canonical_dumps(body))
            if args.write or body.get("status") == "measured":
                return 0
            return 2
        if args.command == "_child":
            return _child_main(args.precision, args.protocol, args.out)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
