"""Phase 4 sibling locks: fence, dataset, grader, and no invented scores."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jsonschema import Draft202012Validator

from phase4.assemble import assemble, extra_from_raw, paired_bootstrap
from phase4.build import build, build_default, contamination_report, sha256_file, vendor_root
from phase4.constants import (
    BACKEND,
    CAPITALCITE_COMMIT,
    DDP_UNSUPPORTED_REASON,
    LOCKED_EVAL_SHA256,
    LOCKED_SPLIT_HASH,
    LOCKED_TRAIN_SHA256,
    MIN_EVAL,
    MODEL_REVISION,
    TASK_ID,
    VERDICT_DETAIL_NO_FILING_UPLIFT,
)
from phase4.fence import fence_violations, iter_fenced_files, product_imports_phase4
from phase4.grade import canonicalize, format_equivalent, grade, prompt_template_static, render_prompt
from phase4.paths import BuildStopped, Phase3PathRefused, RefusedCeiling, refuse_phase3
from phase4.run import local_scaler_probe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle" / "phase4_data"
KERNEL = ROOT / "kaggle" / "phase4_train"
PHASE4 = ROOT / "runs" / "phase4"
TARBALL_SHA256 = "421df4ada47a4b8d98c22c8c0f606cc6eef038760bc62dfb15124ae4768cf507"
ADAPTER_SHA256 = "e94fb237398a3e496552014b5ddb8dad6556a2ac56a669d480bbf6098f5e0109"
ADAPTER_BYTES = 8745704


def test_uppercase_fixture_unchanged():
    from sft_loop.task import EVAL_WORDS, TASK_ID as FIXTURE_ID, TRAIN_WORDS

    assert FIXTURE_ID == "uppercase-word-v1"
    assert len(EVAL_WORDS) == 8
    assert len(TRAIN_WORDS) == 16
    assert set(TRAIN_WORDS).isdisjoint(EVAL_WORDS)


def test_product_and_verify_do_not_import_phase4():
    assert product_imports_phase4(ROOT) == []


def test_phase4_ast_fence_bans_sft_loop_and_unsloth():
    files = iter_fenced_files(ROOT)
    assert files
    for path in files:
        assert fence_violations(path) == [], path


def test_builder_emits_eval_n_and_clean_contamination():
    dest = DATA
    if not (dest / "eval.jsonl").is_file():
        build_default(ROOT, dest=dest)
    eval_rows = [json.loads(line) for line in (dest / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line]
    train_rows = [json.loads(line) for line in (dest / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    contamination = json.loads((dest / "contamination.json").read_text(encoding="utf-8"))
    assert len(eval_rows) >= MIN_EVAL
    assert len(train_rows) >= 8
    assert manifest["n_eval"] == len(eval_rows)
    assert manifest["n_train"] == len(train_rows)
    assert manifest["capitalcite_commit"] == CAPITALCITE_COMMIT
    assert manifest["chunks_jsonl"] == "absent"
    assert manifest["model_revision"] == MODEL_REVISION
    assert manifest["task_id"] == TASK_ID
    assert manifest["split_hash"]
    assert len(manifest["split_hash"]) == 64
    assert [row["item_id"] for row in eval_rows] == sorted(row["item_id"] for row in eval_rows)
    assert [row["item_id"] for row in train_rows] == sorted(row["item_id"] for row in train_rows)
    assert contamination["chunk_ok"] is True
    assert contamination["ok"] is False
    assert contamination["status"] == "same-issuer overlap"
    assert contamination["issuer_overlap"][0]["family"] == "JPM"
    assert contamination["issuer_overlap"][0]["n_eval_rows"] == 6
    assert contamination["issuer_overlap"][0]["eval_issuers"] == ["0001628280"]
    assert contamination["exact_chunk_overlap"] == 0
    assert contamination["near_duplicate_pairs"] == []
    assert contamination["eval_in_prompt_template"] == []
    train_docs = {row["doc_id"] for row in train_rows}
    eval_docs = {row["doc_id"] for row in eval_rows}
    assert train_docs.isdisjoint(eval_docs)
    train_issuers = {row["issuer"] for row in train_rows}
    eval_issuers = {row["issuer"] for row in eval_rows}
    assert train_issuers.isdisjoint(eval_issuers)
    assert len(eval_rows) == 123
    assert {row["item_id"] for row in eval_rows if row["issuer"] == "0001628280"}


def test_builder_is_deterministic(tmp_path):
    dest = tmp_path / "data"
    first = build_default(ROOT, dest=dest)
    second = build_default(ROOT, dest=dest)
    assert first["train_sha256"] == second["train_sha256"]
    assert first["eval_sha256"] == second["eval_sha256"]
    assert first["split_issuers_sha256"] == second["split_issuers_sha256"]
    assert first["split_hash"] == second["split_hash"]
    assert sha256_file(DATA / "grade.py") == sha256_file(ROOT / "src" / "phase4" / "grade.py")
    assert sha256_file(DATA / "eval.jsonl") == LOCKED_EVAL_SHA256
    assert sha256_file(DATA / "train.jsonl") == LOCKED_TRAIN_SHA256
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_hash"] == LOCKED_SPLIT_HASH
    assert manifest["eval_sha256"] == LOCKED_EVAL_SHA256
    assert manifest["train_sha256"] == LOCKED_TRAIN_SHA256


def test_builder_refuses_chunks_jsonl_and_leaves_dest(tmp_path):
    source = tmp_path / "capitalcite"
    passages = source / "passages"
    passages.mkdir(parents=True)
    passages.joinpath("doc.txt").write_text("placeholder 1,234 5,678 9,012\n", encoding="utf-8")
    parsed = source / "data" / "parsed" / "x"
    parsed.mkdir(parents=True)
    parsed.joinpath("chunks.jsonl").write_text("{}\n", encoding="utf-8")
    dest = tmp_path / "out"
    dest.mkdir()
    marker = dest / "keep.txt"
    marker.write_text("stay", encoding="utf-8")
    with pytest.raises(BuildStopped, match="chunks.jsonl"):
        build(source, dest)
    assert marker.read_text(encoding="utf-8") == "stay"
    assert not (dest / "eval.jsonl").exists()


def test_builder_leaves_dest_untouched_on_refuse(tmp_path):
    dest = tmp_path / "data"
    dest.mkdir()
    marker = dest / "keep.txt"
    marker.write_text("stay", encoding="utf-8")
    with pytest.raises(BuildStopped):
        build(tmp_path / "missing-source", dest)
    assert marker.read_text(encoding="utf-8") == "stay"
    assert not (dest / "eval.jsonl").exists()


def test_grader_is_canonical_exact_match():
    assert canonicalize("$ 3,813") == "3813"
    assert canonicalize("(157)") == "-157"
    assert canonicalize("$(370)$") == "-370"
    assert canonicalize("$(545)$") == "-545"
    assert canonicalize(" 15.0%.\nextra") == "15.0%"
    assert grade(["$(370)$"], ["-370"]) == 1.0
    assert grade(["$3813", "nope"], ["3813", "2165"]) == 0.5
    assert grade(["3813 is the answer"], ["3813"]) == 0.0
    assert format_equivalent("$410 million", "410") is True
    assert format_equivalent("376", "420") is False
    with pytest.raises(ValueError):
        grade([], [])
    template = prompt_template_static()
    assert "3813" not in template
    assert "15.0%" not in template
    prompt = render_prompt("chunk text", "What was X?")
    assert prompt.startswith("Read the 10-K excerpt.")
    assert "Answer:" in prompt


def test_refuse_phase3_paths(tmp_path):
    with pytest.raises(Phase3PathRefused):
        refuse_phase3(ROOT / "runs" / "phase3" / "scored.json")
    with pytest.raises(Phase3PathRefused):
        refuse_phase3(ROOT / "adapters" / "phase3" / "adapter")
    ok = tmp_path / "out.json"
    assert refuse_phase3(ok) == ok.resolve()


def test_local_scaler_probe_is_unsupported_on_cpu(tmp_path):
    dest = tmp_path / "probe.json"
    body = local_scaler_probe(dest)
    assert body["status"] == "unsupported"
    assert body["adapter_written"] is False
    assert "weight_change_claimed" not in body
    assert dest.is_file()


def test_paired_bootstrap_no_uplift_when_ci_covers_zero():
    before = [1, 0] * 50
    after = [1, 0] * 50
    ci = paired_bootstrap(before, after, rounds=500, seed=0)
    assert ci["n"] == 100
    assert ci["verdict"] == "no uplift"
    assert ci["low"] <= 0.0 <= ci["high"]


def test_assemble_refuses_ceiling_and_phase3(tmp_path):
    items = [
        {"item_id": f"i{i}", "completion": "1", "prediction": "1", "hit": 1}
        for i in range(100)
    ]
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps({"seed": 0, "items": items}) + "\n", encoding="utf-8")
    after.write_text(json.dumps({"seed": 0, "items": items}) + "\n", encoding="utf-8")
    with pytest.raises(RefusedCeiling):
        assemble(before, after, tmp_path / "out.json")
    with pytest.raises(Phase3PathRefused):
        assemble(before, after, ROOT / "runs" / "phase3" / "scored.json")


def test_phase4_scored_report_is_assembled_from_raw(tmp_path):
    scored_path = PHASE4 / "scored.json"
    raw = PHASE4 / "raw"
    assert scored_path.is_file()
    scored = json.loads(scored_path.read_text(encoding="utf-8"))
    rebuilt = assemble(
        raw / "eval_before.json",
        raw / "eval_after.json",
        tmp_path / "scored.json",
        extra=extra_from_raw(raw),
    )
    assert rebuilt["before"] == scored["before"]
    assert rebuilt["after"] == scored["after"]
    assert rebuilt["paired_ci"] == scored["paired_ci"]
    assert rebuilt["cluster_ci"] == scored["cluster_ci"]
    assert rebuilt["format_breakdown"] == scored["format_breakdown"]
    assert rebuilt["jpm_sensitivity"] == scored["jpm_sensitivity"]
    assert rebuilt["n"] == 123
    assert rebuilt["uplift"] == scored["uplift"] == "no uplift"
    assert rebuilt["verdict_detail"] == scored["verdict_detail"] == VERDICT_DETAIL_NO_FILING_UPLIFT
    assert scored["before"]["hits"] == 81
    assert scored["after"]["hits"] == 90
    assert scored["cluster_ci"]["low"] <= 0.0 <= scored["cluster_ci"]["high"]
    assert scored["paired_ci"]["grouping"] == "item"
    assert scored["cluster_ci"]["grouping"] == "filing"
    assert scored["format_breakdown"]["format_gains"] == 5
    assert scored["format_breakdown"]["substantive_gains"] == 10
    assert scored["format_breakdown"]["losses"] == 6
    assert scored["format_breakdown"]["net_substantive"] == 4
    assert scored["jpm_sensitivity"]["excluded_n"] == 6
    assert scored["jpm_sensitivity"]["n"] == 117
    assert scored["stored_hit_mismatches"] == 2
    schema = json.loads((ROOT / "schema" / "phase4_run.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(scored)
    assert scored["status"] == "scored"
    assert "weight_change_claimed" not in scored
    assert scored["backend"] == BACKEND
    assert scored["train_proof"]["mixed_precision"] == "fp32"
    assert scored["train_proof"]["fp16"] == "unsupported"
    assert scored["train_proof"]["backend"] == BACKEND
    assert scored["train_proof"]["ddp"]["reason"] == DDP_UNSUPPORTED_REASON
    assert scored["train_proof"]["nll_scope"] == "first_eval_row"
    assert scored["train_proof"]["overfitting_risk"]


def test_kernel_metadata_and_script_are_self_contained():
    meta = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
    data_meta = json.loads((DATA / "dataset-metadata.json").read_text(encoding="utf-8"))
    script = (KERNEL / "kernel.py").read_text(encoding="utf-8")
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["is_private"] is True
    assert meta["id"] == "KAGGLE_USER/phase4-tenk-lora"
    assert meta["dataset_sources"] == ["KAGGLE_USER/phase4-tenk-data"]
    assert data_meta["id"] == "KAGGLE_USER/phase4-tenk-data"
    assert fence_violations(KERNEL / "kernel.py") == []
    assert "GradScaler" in script
    assert "autocast" in script
    assert "global_step" in script
    assert "200" in script
    assert 'BACKEND = "manual-peft"' in script
    assert "backend={BACKEND}" in script
    assert "backend=trl-peft" not in script
    assert "Run 2" not in script
    assert DDP_UNSUPPORTED_REASON in script


def test_vendor_records_capitalcite_pin():
    source = vendor_root(ROOT) / "SOURCE.txt"
    text = source.read_text(encoding="utf-8")
    assert CAPITALCITE_COMMIT in text
    assert "chunks_jsonl: absent" in text
    assert (vendor_root(ROOT) / "passages").is_dir()
    assert len(list((vendor_root(ROOT) / "passages").glob("*.txt"))) == 30


def test_phase4_cli_fence_exits_zero():
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "phase4", "fence"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _load_kernel():
    spec = importlib.util.spec_from_file_location("phase4_kernel", KERNEL / "kernel.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_pins_committed_manifest_hash():
    kernel = _load_kernel()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert kernel.EXPECTED_SPLIT_HASH == manifest["split_hash"]
    assert kernel.EXPECTED_EVAL_SHA256 == manifest["eval_sha256"]
    assert kernel.EXPECTED_TRAIN_SHA256 == manifest["train_sha256"]


def test_kernel_finds_nested_kaggle_input(tmp_path, monkeypatch):
    kernel = _load_kernel()
    monkeypatch.delenv("PHASE4_DATA_DIR", raising=False)
    nested = tmp_path / "input" / "datasets" / "inaveed101" / "phase4-tenk-data"
    nested.mkdir(parents=True)
    for name in ("manifest.json", "eval.jsonl", "train.jsonl", "grade.py"):
        shutil.copyfile(DATA / name, nested / name)
    decoy = tmp_path / "input" / "datasets" / "other" / "wrong-set"
    decoy.mkdir(parents=True)
    (decoy / "eval.jsonl").write_text('{"item_id": "x"}\n', encoding="utf-8")
    (decoy / "train.jsonl").write_text('{"item_id": "y"}\n', encoding="utf-8")
    (decoy / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": "tenk-extract-v1",
                "capitalcite_commit": "34b42ee0b387738725705a0af682dd3f030a9944",
                "split_hash": "0" * 64,
                "eval_sha256": "0" * 64,
                "train_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    found = kernel.find_data(tmp_path / "input")
    assert found.resolve() == nested.resolve()


def test_kernel_missing_data_writes_unsupported_json(tmp_path, monkeypatch):
    kernel = _load_kernel()
    monkeypatch.delenv("PHASE4_DATA_DIR", raising=False)
    kernel.WORK = tmp_path / "work"
    kernel.WORK.mkdir()
    with pytest.raises(kernel.DatasetNotFound, match="manifest.json"):
        kernel.find_data(tmp_path / "empty-input")
    try:
        kernel.find_data(tmp_path / "empty-input")
    except kernel.DatasetNotFound as exc:
        code = kernel.write_unsupported(exc.reason, str(exc), exit_code=1)
    assert code == 1
    body = json.loads((kernel.WORK / "unsupported.json").read_text(encoding="utf-8"))
    assert body["status"] == "unsupported"
    assert body["adapter_written"] is False
    assert body["unsupported"]["reason"] == "dataset_not_found"
    assert "before" not in body
    assert "after" not in body


def test_kernel_rejects_nested_dataset_with_wrong_hash(tmp_path, monkeypatch):
    kernel = _load_kernel()
    monkeypatch.delenv("PHASE4_DATA_DIR", raising=False)
    nested = tmp_path / "input" / "datasets" / "inaveed101" / "phase4-tenk-data"
    nested.mkdir(parents=True)
    (nested / "eval.jsonl").write_text('{"item_id": "x"}\n', encoding="utf-8")
    (nested / "train.jsonl").write_text('{"item_id": "y"}\n', encoding="utf-8")
    (nested / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": "tenk-extract-v1",
                "capitalcite_commit": "34b42ee0b387738725705a0af682dd3f030a9944",
                "split_hash": "0" * 64,
                "eval_sha256": "0" * 64,
                "train_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(kernel.DatasetHashMismatch, match="split_hash"):
        kernel.find_data(tmp_path / "input")


def test_generations_sane_rejects_run2_bang_collapse():
    kernel = _load_kernel()
    ok, reason = kernel.generations_sane(["!!!!!!!!!!!!!!!!"] * 123)
    assert ok is False
    assert "punctuation" in reason or "identical" in reason
    ok_empty, reason_empty = kernel.generations_sane(["", "420"])
    assert ok_empty is False
    assert "empty" in reason_empty
    ok_good, reason_good = kernel.generations_sane(["420", "376", "-157"])
    assert ok_good is True
    assert reason_good == "ok"


def test_kernel_loads_fp32_masters_not_fp16_weights():
    script = (KERNEL / "kernel.py").read_text(encoding="utf-8")
    assert "torch_dtype=torch.float32" in script
    assert "torch_dtype=torch.float16" not in script
    assert "FP16_MAX = 65504" in script
    assert "fp16_probe.json" in script
    assert "hidden_maxabs_probe" in script
    assert "baseline_invalid" in script
    kernel = _load_kernel()
    assert kernel.FP16_MAX == 65504


def test_kernel_writes_baseline_invalid_json(tmp_path):
    kernel = _load_kernel()
    kernel.WORK = tmp_path / "work"
    kernel.WORK.mkdir()
    code = kernel.write_unsupported(
        "baseline_invalid",
        "eval_before generations are not a usable baseline: punctuation-only",
        exit_code=1,
    )
    assert code == 1
    body = json.loads((kernel.WORK / "unsupported.json").read_text(encoding="utf-8"))
    assert body["status"] == "unsupported"
    assert body["adapter_written"] is False
    assert body["unsupported"]["reason"] == "baseline_invalid"
    assert "before" not in body
    assert "after" not in body


def test_worst_overflow_ranks_by_max_abs():
    kernel = _load_kernel()
    probe = {
        "overflow_modules": ["model.layers.0.mlp", "lm_head"],
        "modules": {
            "model.layers.0.mlp": {"max_abs": 1200.0, "finite": True, "dtype": "torch.float16"},
            "lm_head": {"max_abs": 200000.0, "finite": False, "dtype": "torch.float16"},
        },
    }
    worst = kernel.worst_overflow(probe)
    assert worst["name"] == "lm_head"
    assert worst["max_abs"] == 200000.0


def test_try_ddp_cannot_succeed():
    kernel = _load_kernel()
    body = kernel.try_ddp(None)
    assert body == {"status": "unsupported", "reason": DDP_UNSUPPORTED_REASON}
    assert kernel.BACKEND == BACKEND
    assert kernel.DDP_UNSUPPORTED_REASON == DDP_UNSUPPORTED_REASON


def test_jpm_rows_stay_in_eval():
    eval_rows = [json.loads(line) for line in (DATA / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line]
    jpm = [row for row in eval_rows if row["issuer"] == "0001628280"]
    assert len(jpm) == 6
    assert all(row["doc_id"] == "0001628280-26-008131" for row in jpm)
    train_rows = [json.loads(line) for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    report = contamination_report(train_rows, eval_rows)
    assert report["status"] == "same-issuer overlap"
    assert len(eval_rows) == 123


def test_ingest_hashes_and_baseline_are_sane():
    kernel = _load_kernel()
    raw = PHASE4 / "raw"
    source = (raw / "SOURCE.txt").read_text(encoding="utf-8")
    assert TARBALL_SHA256 in source
    assert ADAPTER_SHA256 in source
    assert str(ADAPTER_BYTES) in source
    sha_line = (raw / "adapter.sha256").read_text(encoding="utf-8").strip()
    assert sha_line.startswith(ADAPTER_SHA256)
    assert sha_line.endswith("adapter_model.safetensors")
    proof_sha = (PHASE4 / "proof" / "adapter.sha256").read_text(encoding="utf-8").strip()
    assert proof_sha == sha_line
    before = json.loads((raw / "eval_before.json").read_text(encoding="utf-8"))
    after = json.loads((raw / "eval_after.json").read_text(encoding="utf-8"))
    preds = [item["prediction"] for item in before["items"]]
    sane, reason = kernel.generations_sane(preds)
    assert sane is True, reason
    assert before["baseline_nll"]["finite"] is True
    assert before["n"] == after["n"] == 123
    assert [item["item_id"] for item in before["items"]] == [item["item_id"] for item in after["items"]]
    manifest = json.loads((raw / "split_manifest.json").read_text(encoding="utf-8"))
    committed = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_hash"] == committed["split_hash"]
    assert manifest["eval_sha256"] == committed["eval_sha256"]
    assert manifest["train_sha256"] == committed["train_sha256"]
    contamination = json.loads((DATA / "contamination.json").read_text(encoding="utf-8"))
    assert contamination["ok"] is False
    assert contamination["status"] == "same-issuer overlap"
    assert contamination["chunk_ok"] is True
    fp16 = json.loads((raw / "fp16.json").read_text(encoding="utf-8"))
    env = json.loads((raw / "env.json").read_text(encoding="utf-8"))
    assert fp16["status"] == "unsupported"
    assert fp16["reason"] == "fp16_amp_overflow"
    assert env["mixed_precision"] == "fp32"
    assert env["weight_dtype"] == "float32"
    assert env["grad_scaler"] is False
    history = json.loads((raw / "trainer_history.json").read_text(encoding="utf-8"))
    assert history["mixed_precision"] == "fp32"
    assert history["global_step"] == 200
    assert not (PHASE4 / "proof" / "adapter_model.safetensors").exists()
