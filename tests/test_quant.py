from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_loop.quant import (
    PINNED_REVISION,
    Precision,
    UnsupportedReason,
    assemble_comparison,
    canonical_dumps,
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
            }
        )
    return rows


def _measured(precision: Precision, correct_prefix: int, latency: float = 10.0) -> dict:
    protocol = load_protocol()
    return measured_arm(
        precision=precision,
        items=_items(correct_prefix, latency),
        on_disk_bytes=2048,
        peak_rss_bytes=4096,
        libraries=_libraries(),
        limiter=_limiter(),
        protocol=protocol,
    )


def _unsupported(precision: Precision, reason: UnsupportedReason) -> dict:
    return unsupported_arm(
        precision=precision,
        reason=reason,
        detail=f"{precision.value} blocked in the unit test. " + REFUSAL_SENTENCE,
        probe={
            "exception_type": "TestProbe",
            "message": "unit test",
            "cuda_available": False,
        },
    )


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
    arms = {
        "fp32": _measured(Precision.FP32, 64),
        "int8": _measured(Precision.INT8, 60, latency=8.0),
        "onnx-int8": _unsupported(Precision.ONNX_INT8, UnsupportedReason.OOM),
        "nf4": _unsupported(Precision.NF4, UnsupportedReason.MISSING_CUDA),
    }
    comparison = assemble_comparison(protocol, arms, _hardware())
    assert comparison["outcome"]["kind"] == "paired"
    assert comparison["outcome"]["score_delta"] == pytest.approx((60 - 64) / n)
    assert comparison["outcome"]["n"] == n
    assert "paired-item-mean" in render_table(comparison)
    again = assemble_comparison(protocol, arms, _hardware())
    assert project_tree(comparison) == project_tree(again)
    fp32 = json.loads(project_tree(comparison)["runs/quant/fp32.json"])
    assert "seed" not in fp32
    assert "item_ids" not in fp32
    assert fp32["document_kind"] == "quant-arm-projection"


def test_nf4_cannot_be_measured():
    protocol = load_protocol()
    with pytest.raises(ValueError, match="nf4"):
        measured_arm(
            precision=Precision.NF4,
            items=_items(1),
            on_disk_bytes=1,
            peak_rss_bytes=1,
            libraries=_libraries(),
            limiter=_limiter(),
            protocol=protocol,
        )
    arms = {
        "fp32": _unsupported(Precision.FP32, UnsupportedReason.OOM),
        "int8": _unsupported(Precision.INT8, UnsupportedReason.OOM),
        "onnx-int8": _unsupported(Precision.ONNX_INT8, UnsupportedReason.OOM),
        "nf4": _measured(Precision.FP32, 1),
    }
    arms["nf4"] = {**arms["nf4"], "precision": "nf4"}
    with pytest.raises(ValueError):
        assemble_comparison(protocol, arms, _hardware())


def test_parse_rejects_a_hand_typed_difference():
    protocol = load_protocol()
    arms = {
        "fp32": _measured(Precision.FP32, 64),
        "int8": _measured(Precision.INT8, 60),
        "onnx-int8": _unsupported(Precision.ONNX_INT8, UnsupportedReason.OOM),
        "nf4": _unsupported(Precision.NF4, UnsupportedReason.MISSING_CUDA),
    }
    comparison = assemble_comparison(protocol, arms, _hardware())
    comparison["outcome"]["score_delta"] = 0.0
    with pytest.raises(ValueError, match="stored outcome"):
        parse_comparison(comparison)


def test_unavailable_when_fp32_is_unmeasured():
    protocol = load_protocol()
    arms = {
        "fp32": _unsupported(Precision.FP32, UnsupportedReason.OOM),
        "int8": _unsupported(Precision.INT8, UnsupportedReason.OOM),
        "onnx-int8": _unsupported(Precision.ONNX_INT8, UnsupportedReason.OOM),
        "nf4": _unsupported(Precision.NF4, UnsupportedReason.MISSING_CUDA),
    }
    comparison = assemble_comparison(protocol, arms, _hardware())
    assert comparison["outcome"] == {"kind": "unavailable", "reason": "fp32_unmeasured"}
    table = render_table(comparison)
    assert "unsupported" in table
    assert "0.0" not in table.split("int8 minus fp32")[1]


def test_quant_document_fails_run_report_validation():
    protocol = load_protocol()
    comparison = assemble_comparison(
        protocol,
        {
            "fp32": _unsupported(Precision.FP32, UnsupportedReason.OOM),
            "int8": _unsupported(Precision.INT8, UnsupportedReason.OOM),
            "onnx-int8": _unsupported(Precision.ONNX_INT8, UnsupportedReason.OOM),
            "nf4": _unsupported(Precision.NF4, UnsupportedReason.MISSING_CUDA),
        },
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
        {
            "fp32": _unsupported(Precision.FP32, UnsupportedReason.OOM),
            "int8": _unsupported(Precision.INT8, UnsupportedReason.OOM),
            "onnx-int8": _unsupported(Precision.ONNX_INT8, UnsupportedReason.OOM),
            "nf4": _unsupported(Precision.NF4, UnsupportedReason.MISSING_CUDA),
        },
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
