"""Verify-skill helpers: fail-closed all, plumbing smoke, venv recreate."""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".cursor" / "skills" / "verify-sft-loop" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_sft_loop as verify  # noqa: E402

from sft_loop.report import MODEL_ID, validate_report  # noqa: E402


@pytest.fixture(autouse=True)
def reset_run_config():
    verify.RUN = verify.RunConfig()
    yield
    verify.RUN = verify.RunConfig()


def test_all_exits_nonzero_when_smoke_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(verify, "cmd_install", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_doctor", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_test", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_unsupported", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_smoke", lambda *a, **k: 7)
    monkeypatch.setattr(verify, "cmd_cleanup", lambda *a, **k: 0)
    monkeypatch.setattr(
        verify,
        "hardware_probe",
        lambda *a, **k: {
            "cuda": False,
            "mps": False,
            "torch": None,
            "python": "3.12.3",
        },
    )
    monkeypatch.setattr(verify, "python_for", lambda root: sys.executable)
    monkeypatch.setattr(verify, "base_env", lambda root: os.environ.copy())
    monkeypatch.setattr(
        verify,
        "probe_imports",
        lambda *a, **k: {
            "bitsandbytes": {"ok": False, "error": "No module named bitsandbytes"},
            "flash_attn": {"ok": False, "error": "No module named flash_attn"},
            "unsloth": {"ok": False, "error": "No module named unsloth"},
        },
    )
    monkeypatch.setattr(verify, "run_id_from_env", lambda: "unit-all-smoke-fail")
    monkeypatch.setattr(verify, "evidence_dir", lambda root, run_id: tmp_path / "evidence")
    log = io.StringIO()
    rc = verify.cmd_all(tmp_path, log)
    assert rc == 7
    text = log.getvalue()
    assert "failed at step smoke" in text
    summary = json.loads((tmp_path / "evidence" / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_step"] == "smoke"
    assert summary["exit"] == 7


def test_all_exits_nonzero_when_quant_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(verify, "cmd_install", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_doctor", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_test", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_unsupported", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_smoke", lambda *a, **k: 0)
    monkeypatch.setattr(verify, "cmd_quant", lambda *a, **k: 9)
    monkeypatch.setattr(verify, "cmd_cleanup", lambda *a, **k: 0)
    monkeypatch.setattr(
        verify,
        "hardware_probe",
        lambda *a, **k: {
            "cuda": False,
            "mps": False,
            "torch": None,
            "python": "3.12.3",
        },
    )
    monkeypatch.setattr(verify, "python_for", lambda root: sys.executable)
    monkeypatch.setattr(verify, "base_env", lambda root: os.environ.copy())
    monkeypatch.setattr(
        verify,
        "probe_imports",
        lambda *a, **k: {
            "bitsandbytes": {"ok": False, "error": "No module named bitsandbytes"},
            "flash_attn": {"ok": False, "error": "No module named flash_attn"},
            "unsloth": {"ok": False, "error": "No module named unsloth"},
        },
    )
    monkeypatch.setattr(verify, "run_id_from_env", lambda: "unit-all-quant-fail")
    monkeypatch.setattr(verify, "evidence_dir", lambda root, run_id: tmp_path / "evidence")
    log = io.StringIO()
    rc = verify.cmd_all(tmp_path, log)
    assert rc == 9
    assert "failed at step quant" in log.getvalue()
    summary = json.loads((tmp_path / "evidence" / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_step"] == "quant"
    assert summary["exit"] == 9


def test_smoke_json_never_validates_as_scored(tmp_path):
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    plumbing = {
        "kind": "plumbing_smoke_not_quality",
        "status": "plumbing",
        "weight_change_claimed": False,
        "device": "cpu",
        "backend": "trl-peft",
        "stack": {"unsloth": "unused", "trl": "0.24.0", "method": "sft-lora"},
        "model_id": verify.SMOKE_MODEL_ID,
        "model_revision": verify.SMOKE_MODEL_REVISION,
        "before": {"seed": 0, "metric_name": "exact_match", "task_score": 0.0, "n": 8},
        "after": {"seed": 0, "metric_name": "exact_match", "task_score": 0.0, "n": 8},
    }
    (smoke / "plumbing.json").write_text(json.dumps(plumbing) + "\n", encoding="utf-8")
    (smoke / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "task_type": "CAUSAL_LM"}) + "\n",
        encoding="utf-8",
    )
    verify.assert_smoke_not_scored(smoke)

    scored = {
        "schema_version": "1",
        "run_id": "fake-scored",
        "model_id": MODEL_ID,
        "stack": {"unsloth": "unused", "trl": "0.24.0", "method": "sft-lora"},
        "seed": 0,
        "status": "scored",
        "metric_role": "task_score_not_safety",
        "weight_change_claimed": True,
        "task_id": "uppercase-word-v1",
        "before": {"seed": 0, "metric_name": "exact_match", "task_score": 0.0, "n": 8},
        "after": {"seed": 0, "metric_name": "exact_match", "task_score": 0.0, "n": 8},
        "train_proof": {
            "trainer": "trl.SFTTrainer",
            "global_step": 1,
            "train_loss": 1.0,
            "lora_r": 8,
            "max_steps": 2,
            "adapter_config_sha256": "0" * 64,
            "train_log_sha256": "0" * 64,
            "adapter_config_path": "adapter_config.json",
            "train_log_path": "train_log.txt",
        },
        "device": "mps",
        "backend": "trl-peft",
    }
    (smoke / "run_report.json").write_text(json.dumps(scored) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        verify.assert_smoke_not_scored(smoke)


def test_committed_smoke_evidence_is_not_scored():
    evidence_root = ROOT / ".cursor" / "skills" / "verify-sft-loop" / "evidence"
    dirs = sorted(p for p in evidence_root.glob("*/smoke") if p.is_dir())
    if not dirs:
        pytest.skip("no committed smoke evidence yet")
    for smoke in dirs:
        verify.assert_smoke_not_scored(smoke)
        for path in smoke.rglob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                continue
            try:
                validate_report(document)
            except (ValueError, Exception):
                continue
            assert document.get("status") != "scored"
            assert document.get("weight_change_claimed") is not True


def test_smoke_fails_when_global_step_lt_max_steps():
    with pytest.raises(RuntimeError, match="global_step 1 != requested max_steps 2"):
        verify.require_smoke_train(
            global_step=1,
            train_loss=1.25,
            max_steps=verify.MAX_STEPS,
            lora_b={"max_abs": 3e-4, "n_nonzero": 8, "n_tensors": 8},
        )


def test_smoke_fails_when_lora_b_is_zero(tmp_path):
    torch = pytest.importorskip("torch")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    torch.save(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 16),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 8),
        },
        adapter / "adapter_model.bin",
    )
    stats = verify.measure_lora_b(adapter)
    assert stats["n_tensors"] == 1
    assert stats["max_abs"] == 0.0
    assert stats["n_nonzero"] == 0
    with pytest.raises(RuntimeError, match="lora_B still at init"):
        verify.require_smoke_train(
            global_step=verify.MAX_STEPS,
            train_loss=1.25,
            max_steps=verify.MAX_STEPS,
            lora_b=stats,
        )


def test_require_smoke_train_accepts_nonzero_lora_b_and_full_steps(tmp_path):
    torch = pytest.importorskip("torch")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = torch.zeros(16, 8)
    weights[0, 0] = 3e-4
    torch.save(
        {"layers.0.self_attn.q_proj.lora_B.weight": weights},
        adapter / "adapter_model.bin",
    )
    stats = verify.measure_lora_b(adapter)
    assert stats["max_abs"] > 0.0
    verify.require_smoke_train(
        global_step=verify.MAX_STEPS,
        train_loss=11.93,
        max_steps=verify.MAX_STEPS,
        lora_b=stats,
    )


def test_smoke_script_uses_lora_b_and_full_step_gate():
    source = Path(verify.__file__).read_text(encoding="utf-8")
    assert "require_smoke_train(" in source
    assert "measure_lora_b(" in source
    assert "lora_B" in source
    assert "global_step < 1 or not math.isfinite(train_loss)" not in source


def test_committed_smoke_recorded_lora_b_and_full_steps():
    evidence_root = ROOT / ".cursor" / "skills" / "verify-sft-loop" / "evidence"
    paths = sorted(evidence_root.glob("*/smoke/plumbing.json"))
    if not paths:
        pytest.skip("no committed plumbing yet")
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("status") != "plumbing":
            continue
        train = document.get("train") or {}
        max_abs = train.get("lora_b_max_abs", document.get("lora_b_max_abs"))
        assert train.get("global_step") == verify.MAX_STEPS
        assert train.get("max_steps") == verify.MAX_STEPS
        assert max_abs is not None
        assert float(max_abs) > 0.0
        assert int(train.get("lora_b_nonzero") or 0) >= 1


def test_venv_usable_false_when_python_missing(tmp_path):
    assert verify.venv_usable(tmp_path) is False


def test_venv_usable_false_when_pip_missing(tmp_path):
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    assert verify.venv_usable(tmp_path) is False


def test_python_for_ignores_unusable_venv(tmp_path):
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    assert Path(verify.python_for(tmp_path)) == Path(sys.executable)


def test_install_skips_create_when_venv_usable(monkeypatch, tmp_path):
    created = []
    monkeypatch.setattr(verify, "venv_usable", lambda root: True)
    monkeypatch.setattr(verify, "_create_venv", lambda root, log: created.append(root) or 0)
    fake_py = tmp_path / ".venv" / "bin" / "python"
    fake_py.parent.mkdir(parents=True)
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(verify, "venv_python", lambda root: fake_py)

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(verify, "run_command", lambda *a, **k: Result())
    monkeypatch.setattr(verify, "write_json", lambda *a, **k: None)
    monkeypatch.setattr(verify, "run_id_from_env", lambda: "unit-venv-ok")
    monkeypatch.setattr(verify, "evidence_dir", lambda *a, **k: tmp_path)
    assert verify.cmd_install(tmp_path, None) == 0
    assert created == []


def test_install_recreates_unusable_venv(monkeypatch, tmp_path):
    created = []
    monkeypatch.setattr(verify, "venv_usable", lambda root: False)
    monkeypatch.setattr(verify, "_create_venv", lambda root, log: created.append(root) or 0)
    fake_py = tmp_path / ".venv" / "bin" / "python"
    fake_py.parent.mkdir(parents=True)
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(verify, "venv_python", lambda root: fake_py)

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(verify, "run_command", lambda *a, **k: Result())
    monkeypatch.setattr(verify, "write_json", lambda *a, **k: None)
    monkeypatch.setattr(verify, "run_id_from_env", lambda: "unit-venv")
    monkeypatch.setattr(verify, "evidence_dir", lambda *a, **k: tmp_path)
    assert verify.cmd_install(tmp_path, None) == 0
    assert created == [tmp_path]


def test_evidence_dir_ignores_stale_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIFY_EVIDENCE_DIR", str(tmp_path / "old-run"))
    path = verify.evidence_dir(tmp_path, "new-run")
    assert path.name == "new-run"
    assert "old-run" not in path.parts


def test_configure_run_ignores_stale_env_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIFY_EVIDENCE_DIR", str(tmp_path / "old-run"))
    monkeypatch.setenv("VERIFY_SCRATCH_DIR", str(tmp_path / "old-scratch"))
    args = argparse.Namespace(
        run_id="new-run",
        evidence_dir=None,
        scratch_dir=None,
        pinned_model=False,
    )
    verify.configure_run(args)
    assert verify.RUN.evidence.name == "new-run"
    assert "old-run" not in verify.RUN.evidence.parts
    assert verify.RUN.scratch.name == "sft-loop-verify-new-run"
    assert "old-scratch" not in verify.RUN.scratch.parts
    assert "ignoring VERIFY_EVIDENCE_DIR" in verify.RUN.evidence_warning
    assert "ignoring VERIFY_SCRATCH_DIR" in verify.RUN.evidence_warning


def test_surfaces_use_probe_errors(monkeypatch, tmp_path):
    hardware = {"cuda": False, "mps": False, "torch": "2.14.1+cpu", "python": "3.12.3"}
    probes = {
        "bitsandbytes": {"ok": False, "error": "No module named 'bitsandbytes'"},
        "flash_attn": {"ok": False, "error": "No module named 'flash_attn'"},
        "unsloth": {"ok": False, "error": "No module named 'unsloth'"},
    }
    surfaces = verify.build_surfaces(
        hardware,
        probes,
        tests_ok=True,
        unsupported_ok=True,
        smoke_ok=True,
    )
    by_id = {item.id: item for item in surfaces}
    assert by_id["bitsandbytes"].status == "unsupported"
    assert "No module named 'bitsandbytes'" in by_id["bitsandbytes"].detail
    assert by_id["flash-attn"].status == "unsupported"
    assert "No module named 'flash_attn'" in by_id["flash-attn"].detail
    assert by_id["train-mps"].status == "unsupported"
    assert by_id["train-cuda-unsloth"].status == "unsupported"
    assert by_id["cpu-trl-peft-smoke"].status == "supported"
    assert by_id["quant-comparison"].status == "unsupported"
    assert by_id["kaggle-quant-kernels"].status == "unsupported"
    assert "kaggle/quant_nf4" in by_id["kaggle-quant-kernels"].detail


def test_surfaces_record_hardware_probe_errors():
    hardware = {
        "cuda": False,
        "mps": False,
        "torch": None,
        "python": "3.12.3",
        "probe_error": "ImportError: boom",
    }
    probes = {
        "bitsandbytes": {"ok": False, "error": "TimeoutExpired: import bitsandbytes"},
        "flash_attn": {"ok": False, "error": "ImportError: flash_attn"},
        "unsloth": {"ok": False, "error": "ModuleNotFoundError: unsloth"},
    }
    surfaces = verify.build_surfaces(
        hardware,
        probes,
        tests_ok=True,
        unsupported_ok=True,
        smoke_ok=False,
    )
    by_id = {item.id: item for item in surfaces}
    assert by_id["train-mps"].status == "unsupported"
    assert "ImportError: boom" in by_id["train-mps"].detail
    assert by_id["bitsandbytes"].status == "unsupported"
    assert "TimeoutExpired" in by_id["bitsandbytes"].detail
    assert by_id["cpu-trl-peft-smoke"].status == "unsupported"
    assert by_id["quant-comparison"].status == "unsupported"
    assert by_id["kaggle-quant-kernels"].status == "unsupported"
    assert "kaggle/quant_nf4" in by_id["kaggle-quant-kernels"].detail
