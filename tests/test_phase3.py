"""Phase 3 honesty locks: same seed, no invented uplift, proof required to claim."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sft_loop import stack as stack_mod
from sft_loop.eval import build_report as build_eval_report
from sft_loop.eval import main as eval_main
from sft_loop.loop import PHASE3_RUN_ID, PHASE3_SEED, main as loop_main, run_loop
from sft_loop.report import (
    MODEL_ID,
    REFUSAL_SENTENCE,
    build_scored_report,
    render_train_log,
    sha256_bytes,
    unsupported_report,
    validate_report,
)
from sft_loop.task import (
    EVAL_WORDS,
    METRIC_NAME,
    TASK_ID,
    TRAIN_WORDS,
    eval_examples,
    grade,
    prediction_token,
    train_examples,
)
from sft_loop.train import execute_train, main as train_main

ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "runs" / "phase3" / "unsupported.json"
SRC = ROOT / "src"


def _has_key(node, key: str) -> bool:
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_has_key(value, key) for value in node.values())
    if isinstance(node, list):
        return any(_has_key(value, key) for value in node)
    return False


def _slice(seed: int, score: float) -> dict:
    return {
        "seed": seed,
        "metric_name": METRIC_NAME,
        "task_score": score,
        "n": len(EVAL_WORDS),
    }


def _write_proof(directory: Path, *, seed: int = 0, history: list | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": MODEL_ID,
        "r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    config_bytes = (json.dumps(config, indent=2) + "\n").encode("utf-8")
    log_history = history if history is not None else [{"loss": 1.25, "step": 2}]
    log_text = render_train_log(
        seed=seed,
        global_step=2,
        train_loss=1.25,
        max_steps=2,
        lora_r=8,
        log_history=log_history,
    )
    (directory / "adapter_config.json").write_bytes(config_bytes)
    (directory / "train_log.txt").write_text(log_text, encoding="utf-8")
    return {
        "trainer": "trl.SFTTrainer",
        "global_step": 2,
        "train_loss": 1.25,
        "lora_r": 8,
        "max_steps": 2,
        "adapter_config_sha256": sha256_bytes(config_bytes),
        "train_log_sha256": sha256_bytes(log_text.encode("utf-8")),
        "adapter_config_path": "adapter_config.json",
        "train_log_path": "train_log.txt",
    }


def test_task_sets_are_disjoint_and_grader_is_exact():
    assert set(TRAIN_WORDS).isdisjoint(EVAL_WORDS)
    assert train_examples()[0]["completion"] == "APPLE"
    assert eval_examples()[0]["completion"] == "AMBER"
    assert grade(["AMBER", "nope"], ["AMBER", "BRIDGE"]) == 0.5
    assert prediction_token("  FALCON\nextra") == "FALCON"
    with pytest.raises(ValueError):
        grade([], [])


def test_checked_in_phase3_report_matches_live_probe_and_has_no_uplift():
    assert PHASE3.is_file()
    checked = json.loads(PHASE3.read_text(encoding="utf-8"))
    live = run_loop(seed=PHASE3_SEED, model_id=MODEL_ID, run_id=PHASE3_RUN_ID)
    assert checked == live
    validate_report(checked)
    assert checked["status"] == "unsupported"
    assert checked["weight_change_claimed"] is False
    assert checked["seed"] == PHASE3_SEED
    assert checked["unsupported"]["reason"] == "gpu_or_train_deps_unavailable"
    assert REFUSAL_SENTENCE in checked["unsupported"]["detail"]
    assert "before" not in checked
    assert "after" not in checked
    assert "train_proof" not in checked
    assert not _has_key(checked, "task_score")
    assert "adaptiveSandboxQualified" not in json.dumps(checked)


def test_runs_json_never_claims_weight_change_or_scores():
    reports = sorted((ROOT / "runs").rglob("*.json"))
    assert reports
    for path in reports:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["weight_change_claimed"] is False
        assert document["status"] in {"unsupported", "unscored"}
        assert "before" not in document
        assert "after" not in document
        assert "train_proof" not in document
        assert not _has_key(document, "task_score")
        assert "adaptiveSandboxQualified" not in json.dumps(document)


def test_blocked_loop_does_not_call_train_or_eval():
    def boom(**kwargs):
        raise AssertionError("measured step must not run")

    report = run_loop(
        seed=PHASE3_SEED,
        model_id=MODEL_ID,
        run_id=PHASE3_RUN_ID,
        eval_fn=boom,
        train_fn=boom,
    )
    assert report["status"] == "unsupported"
    assert "before" not in report
    assert report["weight_change_claimed"] is False


def test_execute_train_refuses_without_stack(tmp_path):
    with pytest.raises(RuntimeError, match="gpu_or_train_deps_unavailable"):
        execute_train(seed=0, proof_dir=tmp_path / "proof", adapter_dir=tmp_path / "adapter")
    assert list(tmp_path.rglob("*")) == []


def test_cli_entrypoints_fail_closed(capsys):
    assert train_main(["--model-id", MODEL_ID, "--seed", "0"]) == 0
    train_report = json.loads(capsys.readouterr().out)
    validate_report(train_report)
    assert train_report["status"] == "unsupported"
    assert "before" not in train_report
    assert not _has_key(train_report, "task_score")

    assert eval_main(["--model-id", MODEL_ID, "--seed", "0", "--split", "before"]) == 0
    before = json.loads(capsys.readouterr().out)
    assert eval_main(["--model-id", MODEL_ID, "--seed", "0", "--split", "after"]) == 0
    after = json.loads(capsys.readouterr().out)
    for report in (before, after):
        validate_report(report)
        assert report["status"] == "unsupported"
        assert report["seed"] == 0
        assert report["weight_change_claimed"] is False
        assert not _has_key(report, "task_score")
    assert "split before" in before["unsupported"]["detail"]
    assert "split after" in after["unsupported"]["detail"]

    assert loop_main(["--model-id", MODEL_ID, "--seed", "0"]) == 0
    loop_report = json.loads(capsys.readouterr().out)
    assert loop_report == json.loads(PHASE3.read_text(encoding="utf-8"))


def test_cli_rejects_other_model_and_bad_seed():
    assert train_main(["--model-id", "other/model", "--seed", "0"]) == 2
    assert eval_main(["--model-id", MODEL_ID, "--seed", "-1", "--split", "before"]) == 2
    assert loop_main(["--model-id", MODEL_ID, "--seed", "-1"]) == 2


def test_subprocess_loop_matches_checked_in_report():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sft_loop.loop",
            "--model-id",
            MODEL_ID,
            "--seed",
            "0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == json.loads(PHASE3.read_text(encoding="utf-8"))


def test_ready_stack_reporter_still_has_no_metrics(monkeypatch):
    monkeypatch.setattr(stack_mod, "module_available", lambda name: True)
    monkeypatch.setattr(stack_mod, "cuda_is_available", lambda: True)
    report = unsupported_report(run_id="ready-but-not-trained")
    validate_report(report)
    assert report["status"] == "unsupported"
    assert report["unsupported"]["reason"] == "train_not_run"
    assert report["weight_change_claimed"] is False
    assert "before" not in report
    assert "after" not in report


def test_weight_claim_true_is_assigned_only_in_report_builder():
    hits = []
    for path in SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if '"weight_change_claimed": True' in line:
                hits.append((path.name, lineno, line.strip()))
    assert hits == [("report.py", hits[0][1], hits[0][2])]
    assert "build_scored_report" in (SRC / "sft_loop" / "report.py").read_text(encoding="utf-8")


def test_scored_report_requires_same_seed_and_real_proof_shape(tmp_path):
    proof = _write_proof(tmp_path, seed=4)
    report = build_scored_report(
        run_id="unit-same-seed",
        seed=4,
        before=_slice(4, 0.0),
        after=_slice(4, 0.25),
        train_proof=proof,
        proof_dir=tmp_path,
    )
    validate_report(report, proof_dir=tmp_path)
    assert report["weight_change_claimed"] is True
    assert report["before"]["seed"] == report["after"]["seed"] == 4
    assert report["metric_role"] == "task_score_not_safety"
    assert report["task_id"] == TASK_ID
    assert "adaptiveSandboxQualified" not in report

    mismatched = dict(report)
    mismatched["before"] = _slice(5, 0.0)
    with pytest.raises(ValueError, match="seed"):
        validate_report(mismatched, proof_dir=tmp_path)

    missing = dict(proof)
    del missing["global_step"]
    broken = dict(report)
    broken["train_proof"] = missing
    with pytest.raises(ValueError):
        validate_report(broken, proof_dir=tmp_path)


def test_proof_rejects_missing_loss_and_non_lora(tmp_path):
    proof = _write_proof(tmp_path, history=[{"step": 2}])
    with pytest.raises(ValueError, match="loss"):
        build_scored_report(
            run_id="unit-no-loss",
            seed=0,
            before=_slice(0, 0.0),
            after=_slice(0, 0.0),
            train_proof=proof,
            proof_dir=tmp_path,
        )

    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["peft_type"] = "PROMPT_TUNING"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    proof["adapter_config_sha256"] = sha256_bytes(config_path.read_bytes())
    with pytest.raises(ValueError, match="LoRA"):
        build_scored_report(
            run_id="unit-not-lora",
            seed=0,
            before=_slice(0, 0.0),
            after=_slice(0, 0.0),
            train_proof=proof,
            proof_dir=tmp_path,
        )


def test_measured_path_uses_one_seed_and_discards_partial_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(stack_mod, "stack_block_reason", lambda cuda_available=None: None)
    calls = []

    def eval_fn(*, seed, split, adapter_dir):
        calls.append((split, seed, adapter_dir is None))
        return _slice(seed, 0.0 if split == "before" else 0.125)

    def train_fn(*, seed, proof_dir):
        calls.append(("train", seed, None))
        return _write_proof(proof_dir, seed=seed)

    report = run_loop(
        seed=3,
        model_id=MODEL_ID,
        run_id="unit-measured",
        eval_fn=eval_fn,
        train_fn=train_fn,
        proof_dir=tmp_path,
    )
    assert calls == [
        ("before", 3, True),
        ("train", 3, None),
        ("after", 3, False),
    ]
    assert report["status"] == "scored"
    assert report["before"]["seed"] == report["after"]["seed"] == 3
    assert report["weight_change_claimed"] is True
    assert not any(path.suffix == ".json" and "unit-measured" in path.name for path in (ROOT / "runs").rglob("*.json"))

    def drift_eval(*, seed, split, adapter_dir):
        return _slice(seed if split == "before" else seed + 1, 1.0)

    discarded = run_loop(
        seed=3,
        model_id=MODEL_ID,
        run_id="unit-drift",
        eval_fn=drift_eval,
        train_fn=train_fn,
        proof_dir=tmp_path,
    )
    assert discarded["status"] == "unsupported"
    assert discarded["weight_change_claimed"] is False
    assert "before" not in discarded
    assert not _has_key(discarded, "task_score")


def test_schema_forbids_claim_without_proof_and_soft_pass_status():
    from jsonschema import Draft202012Validator

    from sft_loop.report import load_schema

    validator = Draft202012Validator(load_schema())
    base = unsupported_report(run_id="fixture-unsupported-no-gpu")
    claimed = dict(base)
    claimed["weight_change_claimed"] = True
    assert list(validator.iter_errors(claimed))
    with_proof = dict(base)
    with_proof["train_proof"] = {"trainer": "trl.SFTTrainer"}
    assert list(validator.iter_errors(with_proof))
    soft = dict(base)
    soft["status"] = "soft-pass"
    assert list(validator.iter_errors(soft))


def test_eval_report_helper_has_no_score():
    report = build_eval_report(seed=0, model_id=MODEL_ID, split="after", run_id="phase3-eval")
    validate_report(report)
    assert report["status"] == "unsupported"
    assert not _has_key(report, "task_score")
