"""Schema, fixture, and honesty locks for the sft-loop scaffold."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from sft_loop.report import (
    MODEL_ID,
    REFUSAL_SENTENCE,
    TRL_VERSION,
    UNSLOTH_VERSION,
    load_schema,
    stack_block_reason,
    unsupported_report,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "runs" / "fixtures"
README = ROOT / "README.md"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "tests"}
BANNED_STATUSES = {
    "soft-pass",
    "soft_pass",
    "softpass",
    "pass",
    "fail",
    "qualified",
}


def _schema() -> dict:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    return schema


def _validator() -> Draft202012Validator:
    return Draft202012Validator(_schema())


def _walk_objects(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_objects(value)


def _numbers(node):
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield node
        return
    if isinstance(node, dict):
        for value in node.values():
            yield from _numbers(value)
    elif isinstance(node, list):
        for value in node:
            yield from _numbers(value)


def _repo_texts():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in {".py", ".json", ".md", ".toml"}:
            continue
        yield path, path.read_text(encoding="utf-8")


def test_schema_locks_objects_and_pins():
    schema = _schema()
    object_nodes = [
        node
        for node in _walk_objects(schema)
        if node.get("type") == "object"
    ]
    assert object_nodes
    for node in object_nodes:
        assert node["additionalProperties"] is False

    assert schema["properties"]["status"]["enum"] == [
        "unsupported",
        "unscored",
        "scored",
    ]
    assert schema["properties"]["model_id"]["const"] == MODEL_ID
    assert schema["properties"]["metric_role"]["const"] == "task_score_not_safety"
    assert schema["properties"]["weight_change_claimed"]["type"] == "boolean"
    assert "const" not in schema["properties"]["weight_change_claimed"]
    stack = schema["$defs"]["stack"]["properties"]
    assert stack["unsloth"]["const"] == UNSLOTH_VERSION
    assert stack["trl"]["const"] == TRL_VERSION
    assert stack["method"]["const"] == "sft-lora"
    assert set(schema["properties"]) >= {"before", "after", "unsupported"}

    property_names = set()
    for node in _walk_objects(schema):
        properties = node.get("properties")
        if isinstance(properties, dict):
            property_names.update(properties)
    assert "adaptiveSandboxQualified" not in property_names
    assert "safety" not in property_names
    assert property_names.isdisjoint(BANNED_STATUSES)


def test_fixtures_are_minimal_and_valid():
    validator = _validator()
    paths = sorted(FIXTURES.glob("*.json"))
    assert [path.name for path in paths] == ["unscored.json", "unsupported.json"]
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        validator.validate(report)
        assert report["status"] in {"unsupported", "unscored"}
        assert report["weight_change_claimed"] is False
        assert "before" not in report
        assert "after" not in report
        assert "task_score" not in report
        assert list(_numbers(report)) == []
        assert report["model_id"] == MODEL_ID

    unsupported = json.loads((FIXTURES / "unsupported.json").read_text(encoding="utf-8"))
    assert unsupported["status"] == "unsupported"
    assert unsupported["unsupported"]["reason"] == "gpu_or_train_deps_unavailable"
    assert REFUSAL_SENTENCE in unsupported["unsupported"]["detail"]
    assert unsupported == unsupported_report(
        run_id="fixture-unsupported-no-gpu",
        reason="gpu_or_train_deps_unavailable",
    )

    unscored = json.loads((FIXTURES / "unscored.json").read_text(encoding="utf-8"))
    assert unscored["status"] == "unscored"
    assert "unsupported" not in unscored


def test_schema_rejects_closed_failures():
    validator = _validator()
    base = unsupported_report(run_id="fixture-unsupported-no-gpu")

    extra = dict(base)
    extra["adaptiveSandboxQualified"] = True
    assert list(validator.iter_errors(extra))

    soft = dict(base)
    soft["status"] = "soft-pass"
    assert list(validator.iter_errors(soft))

    with_metrics = dict(base)
    with_metrics["before"] = {}
    assert list(validator.iter_errors(with_metrics))

    scored = dict(base)
    scored["status"] = "scored"
    scored.pop("unsupported")
    assert list(validator.iter_errors(scored))

    claimed = dict(base)
    claimed["weight_change_claimed"] = True
    assert list(validator.iter_errors(claimed))


def test_cuda_present_without_stack_still_unsupported():
    report = unsupported_report(cuda_available=lambda: True)
    validate_report(report)
    assert stack_block_reason(lambda: True) == "gpu_or_train_deps_unavailable"
    assert report["status"] == "unsupported"
    assert report["unsupported"]["reason"] == "gpu_or_train_deps_unavailable"
    assert report["weight_change_claimed"] is False
    assert "before" not in report
    assert "after" not in report
    assert "train_proof" not in report
    assert REFUSAL_SENTENCE in report["unsupported"]["detail"]


def test_explicit_train_not_run_has_no_metrics():
    report = unsupported_report(run_id="train-not-run", reason="train_not_run")
    validate_report(report)
    assert report["status"] == "unsupported"
    assert report["weight_change_claimed"] is False
    assert "before" not in report
    assert "after" not in report
    assert "train_proof" not in report
    assert REFUSAL_SENTENCE in report["unsupported"]["detail"]


def test_default_reason_without_cuda():
    def unavailable() -> bool:
        raise RuntimeError("no gpu")

    assert stack_block_reason(lambda: False) == "gpu_or_train_deps_unavailable"
    assert stack_block_reason(unavailable) == "gpu_or_train_deps_unavailable"


def test_cli_prints_unsupported_report():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "sft_loop"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    validate_report(report)
    assert report["status"] == "unsupported"
    assert report["weight_change_claimed"] is False
    assert "before" not in report
    assert "after" not in report
    assert REFUSAL_SENTENCE in report["unsupported"]["detail"]

    rejected = subprocess.run(
        [sys.executable, "-m", "sft_loop", "--run-id", "Not Valid"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""


def test_readme_pins_and_honesty():
    text = README.read_text(encoding="utf-8")
    assert "Unsloth" in text
    assert "TRL" in text
    assert f"unsloth=={UNSLOTH_VERSION}" in text
    assert f"trl=={TRL_VERSION}" in text
    assert MODEL_ID in text
    assert "≤3B" in text
    assert "1.54B" in text
    assert REFUSAL_SENTENCE in text
    assert "adaptiveSandboxQualified" in text
    assert "score ≠ safety" in text
    assert "not frontier RL" in text
    assert "not full pretraining" in text
    assert "No invented uplift" in text
    assert "runs/fixtures/" in text
    assert "schema/run_report.schema.json" in text
    assert "gpu_or_train_deps_unavailable" in text
    assert "sft_loop.train" in text
    assert "sft_loop.eval" in text


def test_soft_pass_token_only_inside_refusal_sentence():
    hits = []
    for path, text in _repo_texts():
        for line in text.splitlines():
            lowered = line.lower()
            if "soft-pass" in lowered or "soft_pass" in lowered or "softpass" in lowered:
                hits.append((path.relative_to(ROOT), line))
    assert hits
    for path, line in hits:
        assert REFUSAL_SENTENCE in line, (path, line)


def test_json_files_omit_forbidden_keys_and_scored_status():
    for path, text in _repo_texts():
        if path.suffix != ".json":
            continue
        assert "adaptiveSandboxQualified" not in text
        document = json.loads(text)
        for node in _walk_objects(document):
            assert "adaptiveSandboxQualified" not in node
            if "status" in node and path.parts[-2] == "fixtures":
                assert node["status"] in {"unsupported", "unscored"}
                assert node["status"] not in BANNED_STATUSES


def test_train_and_eval_modules_are_importable():
    assert importlib.util.find_spec("sft_loop.train") is not None
    assert importlib.util.find_spec("sft_loop.eval") is not None
    assert importlib.util.find_spec("sft_loop.loop") is not None


def test_gitignore_carve_out_for_fixtures():
    def excluded(path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", "--", path],
            cwd=ROOT,
            check=False,
        )
        return result.returncode == 0

    assert not excluded("runs/fixtures/unsupported.json")
    assert not excluded("runs/fixtures/unscored.json")
    assert not excluded("runs/fixtures/nested/later.json")
    assert not excluded("runs/phase3/unsupported.json")
    assert not excluded("runs/phase3/proof/adapter_config.json")
    assert not excluded("runs/phase3/proof/train_log.txt")
    assert excluded("runs/scratch/unsupported.json")
    assert excluded("runs/phase3/scratch.json")
    assert excluded("runs/phase3/proof/adapter_model.safetensors")
    assert excluded("checkpoints/adapter.safetensors")
    assert excluded("adapters/run/adapter_model.safetensors")
