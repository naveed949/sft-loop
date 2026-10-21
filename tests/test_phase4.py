"""Phase 4 sibling locks: fence, dataset, grader, and no invented scores."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phase4.assemble import assemble, paired_bootstrap
from phase4.build import build, build_default, sha256_file, vendor_root
from phase4.constants import CAPITALCITE_COMMIT, MIN_EVAL, MODEL_REVISION, TASK_ID
from phase4.fence import fence_violations, iter_fenced_files, product_imports_phase4
from phase4.grade import canonicalize, grade, prompt_template_static, render_prompt
from phase4.paths import BuildStopped, Phase3PathRefused, RefusedCeiling, refuse_phase3
from phase4.run import local_scaler_probe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle" / "phase4_data"
KERNEL = ROOT / "kaggle" / "phase4_train"


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
    assert contamination["ok"] is True
    assert contamination["exact_chunk_overlap"] == 0
    assert contamination["near_duplicate_pairs"] == []
    assert contamination["eval_in_prompt_template"] == []
    train_docs = {row["doc_id"] for row in train_rows}
    eval_docs = {row["doc_id"] for row in eval_rows}
    assert train_docs.isdisjoint(eval_docs)
    train_issuers = {row["issuer"] for row in train_rows}
    eval_issuers = {row["issuer"] for row in eval_rows}
    assert train_issuers.isdisjoint(eval_issuers)


def test_builder_is_deterministic():
    first = build_default(ROOT, dest=DATA)
    second = build_default(ROOT, dest=DATA)
    assert first["train_sha256"] == second["train_sha256"]
    assert first["eval_sha256"] == second["eval_sha256"]
    assert first["split_issuers_sha256"] == second["split_issuers_sha256"]
    assert sha256_file(DATA / "grade.py") == sha256_file(ROOT / "src" / "phase4" / "grade.py")


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
    assert canonicalize(" 15.0%.\nextra") == "15.0%"
    assert grade(["$3813", "nope"], ["3813", "2165"]) == 0.5
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


def test_phase4_has_no_invented_scored_report():
    assert not (ROOT / "runs" / "phase4" / "scored.json").exists()


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
    assert "import sft_loop" not in script
    assert "import unsloth" not in script
    assert "GradScaler" in script
    assert "autocast" in script
    assert "global_step" in script
    assert "200" in script
    assert fence_violations(KERNEL / "kernel.py") == []


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
