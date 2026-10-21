from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_loop.quant import (
    KERNEL_DIR_NF4,
    KERNEL_EVIDENCE_PREFIX,
    NF4_QUANT_CONFIG,
    PINNED_REVISION,
    Precision,
    UnsupportedReason,
    apply_prompt_mask,
    assemble_comparison,
    canonical_dumps,
    cross_hardware_view,
    ingest_kernel_arm,
    kernel_placeholder,
    load_item_ids,
    load_protocol,
    measured_arm,
    parse_comparison,
    probe_nf4,
    project_tree,
    quant_schema_path,
    render_table,
    splice_table,
    tracked_relpaths,
    unsupported_arm,
    _summarize_items,
)
from sft_loop.report import MODEL_ID, REFUSAL_SENTENCE, validate_report

ROOT = Path(__file__).resolve().parents[1]


def _hardware() -> dict:
    return {
        "cpu_model": "test-cpu",
        "logical_cpus": 4,
        "ram_bytes": 16 * 1024 ** 3,
        "cuda": False,
        "mps": False,
    }


def _limiter() -> dict:
    return {
        "kind": "cpu",
        "wall_s": 1.0,
        "cpu_time_s": 0.8,
        "peak_rss_bytes": 1024,
        "available_ram_bytes_at_start": 8 * 1024 ** 3,
    }


def _libraries() -> dict:
    return {"kind": "torch", "torch": "2.0.0", "transformers": "4.56.1"}


def _items(correct_prefix: int, latency: float = 10.0) -> list[dict]:
    rows = []
    for index, item_id in enumerate(load_item_ids()):
        rows.append(
            {
                "item_id": item_id,
                "correct": index < correct_prefix,
                "latency_ms": latency,
                "generated_tokens": 1,
                "decode_latency_ms": 8.0,
                "decode_generated_tokens": 8,
                "nll": 0.5,
                "sequence_nll": 4.6,
            }
        )
    return rows


def _gpu_hardware(accelerator: str = "NvidiaTeslaT4") -> dict:
    return {
        "cpu_model": "test-cpu",
        "logical_cpus": 4,
        "ram_bytes": 16 * 1024 ** 3,
        "cuda": True,
        "mps": False,
        "accelerator": accelerator,
        "gpu_name": "Tesla T4",
        "gpu_count": 2,
        "gpu_memory_bytes": 15 * 1024 ** 3,
        "compute_capability": "7.5",
        "driver": "580",
    }


def _measured(precision: Precision, correct_prefix: int, latency: float = 10.0) -> dict:
    protocol = load_protocol()
    kwargs = {}
    if precision in {Precision.NF4, Precision.NF4_FP16, Precision.FP16, Precision.T4_FP32}:
        from sft_loop.quant import GPU_COMPUTE_DTYPE, NF4_QUANT_CONFIGS

        kwargs = {
            "hardware": _gpu_hardware(),
            "peak_gpu_bytes": 2 * 1024 ** 3,
            "kernel": kernel_placeholder(KERNEL_DIR_NF4),
            "compute_dtype": GPU_COMPUTE_DTYPE[precision],
        }
        if precision in NF4_QUANT_CONFIGS:
            kwargs["quant_config"] = dict(NF4_QUANT_CONFIGS[precision])
            kwargs["libraries"] = {
                "kind": "bitsandbytes",
                "torch": "2.11.0+cu128",
                "transformers": "5.20.2",
                "bitsandbytes": "0.50.2",
            }
    return measured_arm(
        precision=precision,
        items=_items(correct_prefix, latency),
        on_disk_bytes=2048,
        peak_rss_bytes=4096,
        libraries=kwargs.pop("libraries", _libraries()),
        limiter=_limiter(),
        protocol=protocol,
        **kwargs,
    )


def _unsupported(precision: Precision, reason: UnsupportedReason) -> dict:
    kernel = None
    if precision in {Precision.NF4, Precision.NF4_FP16, Precision.FP16, Precision.T4_FP32, Precision.ONNX_INT8}:
        kernel = kernel_placeholder(
            KERNEL_DIR_NF4 if precision is not Precision.ONNX_INT8 else "kaggle/quant_onnx_int8"
        )
    return unsupported_arm(
        precision=precision,
        reason=reason,
        detail=f"{precision.value} blocked in the unit test. " + REFUSAL_SENTENCE,
        probe={
            "exception_type": "TestProbe",
            "message": "unit test",
            "cuda_available": False,
        },
        kernel=kernel,
    )


def _arms(**overrides: dict) -> dict:
    base = {
        "fp32": _measured(Precision.FP32, 64),
        "int8": _measured(Precision.INT8, 60, latency=8.0),
        "onnx-int8": _unsupported(Precision.ONNX_INT8, UnsupportedReason.OOM),
        "t4-fp32": _unsupported(Precision.T4_FP32, UnsupportedReason.KERNEL_OUTPUT_MISSING),
        "fp16": _unsupported(Precision.FP16, UnsupportedReason.KERNEL_OUTPUT_MISSING),
        "nf4": _unsupported(Precision.NF4, UnsupportedReason.MISSING_CUDA),
        "nf4-fp16": _unsupported(Precision.NF4_FP16, UnsupportedReason.KERNEL_OUTPUT_MISSING),
    }
    base.update(overrides)
    return base


def test_schema_is_closed_and_omits_sft_claim_keys():
    schema = json.loads(quant_schema_path().read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
            yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    names = set()
    for node in walk(schema):
        props = node.get("properties")
        if isinstance(props, dict):
            names.update(props)
    assert "weight_change_claimed" not in names
    assert "adaptiveSandboxQualified" not in names
    assert "before" not in names
    assert "after" not in names
    assert "train_proof" not in names
    assert schema["properties"]["model_id"]["const"] == MODEL_ID
    assert schema["properties"]["document_kind"]["const"] == "quant-comparison"


def test_paired_difference_is_owned_by_the_comparison():
    protocol = load_protocol()
    n = len(protocol["item_ids"])
    comparison = assemble_comparison(protocol, _arms(), _hardware())
    cpu = comparison["outcomes"]["int8-minus-fp32"]
    assert cpu["kind"] == "paired"
    assert cpu["minus"] == "int8"
    assert cpu["baseline"] == "fp32"
    assert cpu["score_delta"] == pytest.approx((60 - 64) / n)
    assert cpu["n"] == n
    assert comparison["outcomes"]["nf4-minus-t4-fp32"]["kind"] == "unavailable"
    assert "paired-item-mean" not in cpu or cpu["method"] == "paired-item-mean"
    assert "tokens_per_s_ratio" not in cpu
    table = render_table(comparison)
    assert "decode tokens/s (min=max=8)" in table
    assert "mean generated tokens" in table
    assert "not paired" in table
    again = assemble_comparison(protocol, _arms(), _hardware())
    assert project_tree(comparison) == project_tree(again)
    fp32 = json.loads(project_tree(comparison)["runs/quant/fp32.json"])
    assert "seed" not in fp32
    assert "item_ids" not in fp32
    assert fp32["document_kind"] == "quant-arm-projection"


def test_nf4_measured_requires_accelerator():
    protocol = load_protocol()
    with pytest.raises(ValueError, match="accelerator"):
        measured_arm(
            precision=Precision.NF4,
            items=_items(1),
            on_disk_bytes=1,
            peak_rss_bytes=1,
            libraries=_libraries(),
            limiter=_limiter(),
            protocol=protocol,
        )


def test_nf4_pairs_with_t4_fp32_not_cpu_fp32():
    protocol = load_protocol()
    n = len(protocol["item_ids"])
    comparison = assemble_comparison(
        protocol,
        _arms(
            nf4=_measured(Precision.NF4, 50),
            **{"t4-fp32": _measured(Precision.T4_FP32, 60)},
        ),
        _hardware(),
    )
    gpu = comparison["outcomes"]["nf4-minus-t4-fp32"]
    cpu = comparison["outcomes"]["int8-minus-fp32"]
    assert gpu["kind"] == "paired"
    assert gpu["minus"] == "nf4"
    assert gpu["baseline"] == "t4-fp32"
    assert gpu["score_delta"] == pytest.approx((50 - 60) / n)
    assert cpu["baseline"] == "fp32"
    assert gpu["score_delta"] != cpu["score_delta"]
    assert comparison["arms"]["nf4"]["hardware"]["accelerator"] == "NvidiaTeslaT4"
    view = cross_hardware_view(minus="nf4", baseline="fp32")
    assert view == {
        "kind": "not_paired",
        "minus": "nf4",
        "baseline": "fp32",
        "reason": "cross_hardware",
    }


def test_accelerator_mismatch_is_not_paired():
    protocol = load_protocol()
    nf4 = _measured(Precision.NF4, 50)
    t4 = _measured(Precision.T4_FP32, 60)
    t4["hardware"] = _gpu_hardware("NvidiaL4")
    comparison = assemble_comparison(protocol, _arms(nf4=nf4, **{"t4-fp32": t4}), _hardware())
    assert comparison["outcomes"]["nf4-minus-t4-fp32"] == {
        "kind": "not_paired",
        "minus": "nf4",
        "baseline": "t4-fp32",
        "reason": "accelerator_mismatch",
    }


def test_ingest_missing_kernel_output_is_unsupported(tmp_path: Path):
    body = ingest_kernel_arm(tmp_path / "absent", Precision.NF4)
    assert body["status"] == "unsupported"
    assert body["reason"] == "kernel_output_missing"
    assert "task_score" not in body
    assert "items" not in body
    assert REFUSAL_SENTENCE in body["detail"]


def test_ingest_kernel_failure_without_refusal_stays_unsupported(tmp_path: Path):
    tmp_path.joinpath("env.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("log.txt").write_text("killed\n", encoding="utf-8")
    tmp_path.joinpath("progress.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "oom",
                "detail": "onnx-int8 hit MemoryError after export",
                "probe": {
                    "exception_type": "MemoryError",
                    "message": "MemoryError",
                    "cuda_available": False,
                },
                "arms": {"onnx-int8": {"status": "unsupported", "reason": "oom"}},
            }
        ),
        encoding="utf-8",
    )
    body = ingest_kernel_arm(tmp_path, Precision.ONNX_INT8)
    assert body["status"] == "unsupported"
    assert body["reason"] == "oom"
    assert REFUSAL_SENTENCE in body["detail"]
    assert "task_score" not in body


def test_ingest_onnx_ir_missing_is_probed_failure(tmp_path: Path):
    tmp_path.joinpath("env.json").write_text(
        json.dumps(
            {
                "logical_cpus": 4,
                "ram_bytes": 33658318848,
                "python": "3.13.15",
            }
        ),
        encoding="utf-8",
    )
    tmp_path.joinpath("log.txt").write_text(
        "ModuleNotFoundError: No module named 'onnx_ir'\n",
        encoding="utf-8",
    )
    tmp_path.joinpath("progress.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "probed_failure",
                "detail": "ModuleNotFoundError: No module named 'onnx_ir'",
                "probe": {
                    "exception_type": "ModuleNotFoundError",
                    "message": "No module named 'onnx_ir'",
                    "cuda_available": False,
                },
                "arms": {"onnx-int8": {"status": "unsupported", "reason": "probed_failure"}},
            }
        ),
        encoding="utf-8",
    )
    body = ingest_kernel_arm(tmp_path, Precision.ONNX_INT8)
    assert body["status"] == "unsupported"
    assert body["reason"] == "probed_failure"
    assert "onnx_ir" in body["detail"]
    assert REFUSAL_SENTENCE in body["detail"]
    assert "task_score" not in body


def test_ingest_progress_failure_without_env_is_probed_failure(tmp_path: Path):
    tmp_path.joinpath("log.txt").write_text("pip failed\n", encoding="utf-8")
    tmp_path.joinpath("progress.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "probed_failure",
                "detail": "CalledProcessError: onnx==1.17.0 has no cp313 wheel",
                "probe": {
                    "exception_type": "CalledProcessError",
                    "message": "pip install onnx==1.17.0 failed",
                    "cuda_available": False,
                },
                "arms": {"onnx-int8": {"status": "unsupported", "reason": "probed_failure"}},
            }
        ),
        encoding="utf-8",
    )
    body = ingest_kernel_arm(tmp_path, Precision.ONNX_INT8)
    assert body["status"] == "unsupported"
    assert body["reason"] == "probed_failure"
    assert REFUSAL_SENTENCE in body["detail"]
    assert "task_score" not in body


def test_ingest_drops_non_finite_nll(tmp_path: Path):
    protocol = load_protocol()
    tmp_path.joinpath("env.json").write_text(
        json.dumps(
            {
                "kernel_url": "https://www.kaggle.com/code/example/nf4",
                "kernel_version": "1",
                "hardware": {
                    "accelerator": "NvidiaTeslaT4",
                    "gpu_name": "Tesla T4",
                    "gpu_count": 1,
                    "gpu_memory_bytes": 15 * 1024 ** 3,
                    "compute_capability": "7.5",
                    "driver": "580",
                    "cuda": True,
                    "mps": False,
                    "logical_cpus": 4,
                    "ram_bytes": 32 * 1024 ** 3,
                    "cpu_model": "x86_64",
                },
            }
        ),
        encoding="utf-8",
    )
    tmp_path.joinpath("log.txt").write_text("ok\n", encoding="utf-8")
    tmp_path.joinpath("progress.json").write_text(
        json.dumps({"status": "measured", "arms": {"fp16": {"status": "measured"}}}),
        encoding="utf-8",
    )
    items = []
    for item_id in protocol["item_ids"]:
        items.append(
            {
                "item_id": item_id,
                "correct": False,
                "latency_ms": 10.0,
                "generated_tokens": 8,
                "decode_latency_ms": 10.0,
                "decode_generated_tokens": 8,
                "nll": float("nan"),
                "sequence_nll": float("nan"),
            }
        )
    (tmp_path / "fp16").mkdir()
    (tmp_path / "fp16" / "items.json").write_text(json.dumps(items), encoding="utf-8")
    (tmp_path / "fp16" / "metrics.json").write_text(
        json.dumps(
            {
                "on_disk_bytes": 1000,
                "peak_rss_bytes": 2000,
                "peak_gpu_bytes": 3000,
                "compute_dtype": "float16",
                "libraries": {"kind": "torch", "torch": "2.11.0", "transformers": "4.56.2"},
                "limiter": {
                    "kind": "gpu",
                    "wall_s": 1.0,
                    "cpu_time_s": 1.0,
                    "peak_rss_bytes": 2000,
                    "available_ram_bytes_at_start": 10000,
                },
            }
        ),
        encoding="utf-8",
    )
    body = ingest_kernel_arm(tmp_path, Precision.FP16)
    assert body["status"] == "unsupported"
    assert body["reason"] == "probed_failure"
    assert "task_score" not in body
    assert REFUSAL_SENTENCE in body["detail"]


def _write_gpu_kernel(tmp_path: Path, precision: Precision, items: list, extra: dict | None = None):
    hardware = {
        "accelerator": "NvidiaTeslaT4",
        "gpu_name": "Tesla T4",
        "gpu_count": 1,
        "gpu_memory_bytes": 15 * 1024 ** 3,
        "compute_capability": "7.5",
        "driver": "580",
        "cuda": True,
        "mps": False,
        "logical_cpus": 4,
        "ram_bytes": 32 * 1024 ** 3,
        "cpu_model": "x86_64",
    }
    tmp_path.joinpath("env.json").write_text(json.dumps({"hardware": hardware}), encoding="utf-8")
    tmp_path.joinpath("log.txt").write_text("ok\n", encoding="utf-8")
    tmp_path.joinpath("progress.json").write_text(
        json.dumps({"status": "measured", "arms": {precision.value: {"status": "measured"}}}),
        encoding="utf-8",
    )
    arm = tmp_path / precision.value
    arm.mkdir(exist_ok=True)
    arm.joinpath("items.json").write_text(json.dumps(items), encoding="utf-8")
    from sft_loop.quant import GPU_COMPUTE_DTYPE, NF4_QUANT_CONFIGS

    metrics = {
        "on_disk_bytes": 1000,
        "peak_rss_bytes": 2000,
        "peak_gpu_bytes": 3000,
        "compute_dtype": GPU_COMPUTE_DTYPE[precision],
        "libraries": {"kind": "torch", "torch": "2.11.0", "transformers": "5.20.2"},
        "limiter": {
            "kind": "gpu",
            "wall_s": 1.0,
            "cpu_time_s": 1.0,
            "peak_rss_bytes": 2000,
            "available_ram_bytes_at_start": 10000,
        },
    }
    if precision in NF4_QUANT_CONFIGS:
        metrics["quant_config"] = dict(NF4_QUANT_CONFIGS[precision])
        metrics["libraries"] = {
            "kind": "bitsandbytes",
            "torch": "2.11.0",
            "transformers": "5.20.2",
            "bitsandbytes": "0.50.2",
        }
    if extra:
        metrics.update(extra)
    arm.joinpath("metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_ingest_refuses_zero_score_baseline_unless_collapse(tmp_path: Path):
    items = _items(0)
    _write_gpu_kernel(tmp_path, Precision.T4_FP32, items)
    refused = ingest_kernel_arm(tmp_path, Precision.T4_FP32)
    assert refused["status"] == "unsupported"
    assert refused["reason"] == "probed_failure"
    assert "task_score" not in refused
    _write_gpu_kernel(tmp_path, Precision.T4_FP32, items, {"finding": "collapse"})
    allowed = ingest_kernel_arm(tmp_path, Precision.T4_FP32)
    assert allowed["status"] == "measured"
    assert allowed["task_score"] == 0.0
    assert allowed["finding"] == "collapse"


def test_ingest_allows_fp16_overflow_finding(tmp_path: Path):
    items = _items(0)
    for row in items:
        row["nll"] = float("nan")
        row["sequence_nll"] = float("nan")
    _write_gpu_kernel(tmp_path, Precision.FP16, items, {"finding": "overflow"})
    body = ingest_kernel_arm(tmp_path, Precision.FP16)
    assert body["status"] == "measured"
    assert body["finding"] == "overflow"
    assert body["task_score"] == 0.0
    assert "nll" not in body["items"][0]


def test_ingest_export_stage_without_items_is_oom(tmp_path: Path):
    tmp_path.joinpath("env.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("log.txt").write_text("export started\n", encoding="utf-8")
    tmp_path.joinpath("progress.json").write_text(
        json.dumps({"status": "running", "stage": "export", "arms": {}}),
        encoding="utf-8",
    )
    body = ingest_kernel_arm(tmp_path, Precision.ONNX_INT8)
    assert body["status"] == "unsupported"
    assert body["reason"] == "oom"
    assert "task_score" not in body


def test_parse_rejects_a_hand_typed_difference():
    protocol = load_protocol()
    comparison = assemble_comparison(protocol, _arms(), _hardware())
    comparison["outcomes"]["int8-minus-fp32"]["score_delta"] = 0.0
    with pytest.raises(ValueError, match="stored outcomes"):
        parse_comparison(comparison)


def test_unavailable_when_fp32_is_unmeasured():
    protocol = load_protocol()
    comparison = assemble_comparison(
        protocol,
        _arms(
            fp32=_unsupported(Precision.FP32, UnsupportedReason.OOM),
            int8=_unsupported(Precision.INT8, UnsupportedReason.OOM),
        ),
        _hardware(),
    )
    assert comparison["outcomes"]["int8-minus-fp32"] == {
        "kind": "unavailable",
        "minus": "int8",
        "baseline": "fp32",
        "reason": "fp32_unmeasured",
    }
    table = render_table(comparison)
    assert "unsupported" in table
    assert "0.0" not in table.split("int8 minus fp32")[1]


def test_quant_document_fails_run_report_validation():
    protocol = load_protocol()
    comparison = assemble_comparison(
        protocol,
        _arms(
            fp32=_unsupported(Precision.FP32, UnsupportedReason.OOM),
            int8=_unsupported(Precision.INT8, UnsupportedReason.OOM),
        ),
        _hardware(),
    )
    with pytest.raises(ValueError):
        validate_report(comparison)
    projection = json.loads(project_tree(comparison)["runs/quant/nf4.json"])
    with pytest.raises(ValueError):
        validate_report(projection)


def test_probe_nf4_is_unsupported_and_has_no_metrics():
    body = probe_nf4()
    assert body["status"] == "unsupported"
    assert body["precision"] == "nf4"
    assert body["reason"] in {"missing_cuda", "dependency_missing"}
    assert REFUSAL_SENTENCE in body["detail"]
    assert "task_score" not in body
    assert "items" not in body
    assert "on_disk_bytes" not in body
    second = probe_nf4()
    assert canonical_dumps(body) == canonical_dumps(second)


def test_tracked_relpaths_are_derived_from_precision():
    paths = tracked_relpaths()
    assert "runs/quant/comparison.json" in paths
    for precision in Precision:
        assert f"runs/quant/{precision.value}.json" in paths
        assert f"runs/quant/raw/{precision.value}.json" in paths
    assert "runs/quant/extra.json" not in paths
    assert not any(path.startswith(KERNEL_EVIDENCE_PREFIX) for path in paths)


def test_source_does_not_assign_weight_change_claimed_true():
    assignments = []
    for path in (ROOT / "src").rglob("*.py"):
        if '"weight_change_claimed": True' in path.read_text(encoding="utf-8"):
            assignments.append(path)
    assert assignments == [ROOT / "src" / "sft_loop" / "report.py"]


def test_revision_is_the_pinned_sha():
    protocol = load_protocol()
    assert protocol["revision"] == PINNED_REVISION
    assert len(load_item_ids()) >= 100


def test_prompt_mask_keeps_only_target_labels():
    assert apply_prompt_mask([11, 12, 13, 14], 2) == [-100, -100, 13, 14]


def test_decode_tokens_per_s_is_not_scored_emitted_tokens():
    items = _items(10)
    summary = _summarize_items(items)
    assert "tokens_per_s" not in summary
    assert summary["mean_generated_tokens"] == 1.0
    assert summary["decode_tokens_per_s"] == 1000.0


def test_heldout_is_disjoint_from_sft_splits():
    from sft_loop.task import EVAL_WORDS, TRAIN_WORDS

    ids = set(load_item_ids())
    assert ids.isdisjoint(TRAIN_WORDS)
    assert ids.isdisjoint(EVAL_WORDS)
    source = (ROOT / "src" / "sft_loop" / "quant.py").read_text(encoding="utf-8")
    assert "ExportNotRun" not in source


def test_table_check_fails_on_a_hand_edit(tmp_path: Path):
    protocol = load_protocol()
    comparison = assemble_comparison(
        protocol,
        _arms(
            fp32=_unsupported(Precision.FP32, UnsupportedReason.OOM),
            int8=_unsupported(Precision.INT8, UnsupportedReason.OOM),
        ),
        _hardware(),
    )
    readme = tmp_path / "README.md"
    readme.write_text("<!-- quant-table:start -->\n<!-- quant-table:end -->\n", encoding="utf-8")
    splice_table(readme, render_table(comparison))
    from sft_loop.quant import _marker_block

    assert _marker_block(readme.read_text(encoding="utf-8")) == render_table(comparison)
    text = readme.read_text(encoding="utf-8").replace("unsupported", "0.0", 1)
    readme.write_text(text, encoding="utf-8")
    assert _marker_block(readme.read_text(encoding="utf-8")) != render_table(comparison)


def test_kernels_embed_the_heldout_split_and_metadata():
    heldout = load_item_ids()
    for rel, gpu in (("kaggle/quant_nf4", True), ("kaggle/quant_onnx_int8", False)):
        meta = json.loads((ROOT / rel / "kernel-metadata.json").read_text(encoding="utf-8"))
        script = (ROOT / rel / "kernel.py").read_text(encoding="utf-8")
        assert meta["id"].startswith("KAGGLE_USER/")
        assert meta["enable_internet"] is True
        assert meta["is_private"] is True
        assert meta["enable_gpu"] is gpu
        if gpu:
            assert meta["machine_shape"] == "NvidiaTeslaT4"
            assert "bitsandbytes==0.50.2" in script
            assert "transformers==5.20.2" in script
            assert "tokenizers==0.23.3" in script
            assert "bnb_4bit_quant_type" in script
            assert "float32" in script
            assert "t4-fp32" in script
            assert "Fail closed" in script
            assert "probe_numerics" in script
        else:
            assert meta.get("enable_gpu") is False
            assert "quantize_dynamic" in script
            assert "onnx==1.18.0" in script
            assert "onnx==1.17.0" not in script
            assert "dynamo=False" in script
            assert "onnx-ir" not in script
            assert "record_export_mem" in script
        for item_id in heldout:
            assert f'"{item_id}"' in script
        assert script.count("ITEM_IDS") >= 1
