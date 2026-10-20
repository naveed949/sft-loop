#!/usr/bin/env python3
"""Drive sft-loop the way a later agent must: install, doctor, test, CLI, CPU smoke, cleanup."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

SKILL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 0
SMOKE_RUN_ID = "verify-cpu-smoke-seed0"
REFUSAL_SENTENCE = "Soft-PASS is unused and is refused."
REASON_DEPS = "gpu_or_train_deps_unavailable"
REASON_TRAIN = "train_not_run"

TRAIN_EXTRAS = [
    "jsonschema>=4.18",
    "pytest>=8",
    "transformers>=4.56.1",
    "accelerate>=1.4.0",
    "datasets>=3.0.0",
    "peft>=0.8.0",
    "trl==0.24.0",
]


@dataclass
class Surface:
    id: str
    command: str
    status: str
    reason: str | None
    detail: str


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "schema" / "run_report.schema.json"
        ).is_file():
            return parent
    raise SystemExit("could not find repo root (pyproject.toml + schema/run_report.schema.json)")


def venv_python(root: Path) -> Path:
    return root / ".venv" / "bin" / "python"


def run_id_from_env() -> str:
    return os.environ.get("VERIFY_RUN_ID", "cpu-linux-seed0")


def evidence_dir(root: Path, run_id: str) -> Path:
    override = os.environ.get("VERIFY_EVIDENCE_DIR")
    if override:
        return Path(override)
    return SKILL_DIR / "evidence" / run_id


def scratch_dir(run_id: str) -> Path:
    override = os.environ.get("VERIFY_SCRATCH_DIR")
    if override:
        return Path(override)
    return Path("/tmp") / f"sft-loop-verify-{run_id}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_log(log: TextIO | None, text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()
    if log is not None:
        log.write(text)
        log.flush()


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: TextIO | None,
) -> subprocess.CompletedProcess[str]:
    append_log(log, f"$ {' '.join(argv)}\n")
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        append_log(log, completed.stdout if completed.stdout.endswith("\n") else completed.stdout + "\n")
    if completed.stderr:
        append_log(log, completed.stderr if completed.stderr.endswith("\n") else completed.stderr + "\n")
    append_log(log, f"exit {completed.returncode}\n")
    return completed


def python_for(root: Path) -> str:
    candidate = venv_python(root)
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def base_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("HF_HOME", str(root / "hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(root / "hf_cache" / "hub"))
    return env


def hardware_probe(py: str, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    code = (
        "import json, os, platform, sys\n"
        "cuda=False; mps=False; torch_v=None\n"
        "try:\n"
        " import torch\n"
        " torch_v=torch.__version__\n"
        " cuda=bool(torch.cuda.is_available())\n"
        " b=getattr(torch.backends,'mps',None)\n"
        " mps=bool(b.is_available()) if b is not None and hasattr(b,'is_available') else False\n"
        "except Exception:\n"
        " pass\n"
        "mem=None\n"
        "try:\n"
        " mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')\n"
        "except Exception:\n"
        " pass\n"
        "print(json.dumps({'python':sys.version.split()[0],'platform':platform.platform(),"
        "'machine':platform.machine(),'torch':torch_v,'cuda':cuda,'mps':mps,"
        "'cpu_count':os.cpu_count(),'ram_bytes':mem}))\n"
    )
    completed = subprocess.run(
        [py, "-c", code],
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": None,
            "cuda": False,
            "mps": False,
            "cpu_count": os.cpu_count(),
            "ram_bytes": None,
            "probe_error": completed.stderr.strip() or f"exit {completed.returncode}",
        }
    return json.loads(completed.stdout)


def _create_venv(root: Path, log: TextIO | None) -> int:
    target = root / ".venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(target)],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    append_log(log, created.stdout or "")
    append_log(log, created.stderr or "")
    if created.returncode == 0 and venv_python(root).is_file():
        return 0
    append_log(log, "stdlib venv failed. falling back to pip virtualenv.\n")
    shutil.rmtree(target, ignore_errors=True)
    bootstrap = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "virtualenv"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    append_log(log, bootstrap.stdout or "")
    append_log(log, bootstrap.stderr or "")
    if bootstrap.returncode != 0:
        return bootstrap.returncode
    created = subprocess.run(
        [sys.executable, "-m", "virtualenv", str(target)],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    append_log(log, created.stdout or "")
    append_log(log, created.stderr or "")
    if created.returncode != 0 or not venv_python(root).is_file():
        append_log(log, "virtualenv create failed\n")
        return created.returncode or 1
    return 0


def cmd_install(root: Path, log: TextIO | None) -> int:
    if not venv_python(root).is_file():
        code = _create_venv(root, log)
        if code != 0:
            return code
    py = str(venv_python(root))
    env = base_env(root)
    steps = [
        [py, "-m", "pip", "install", "-U", "pip"],
        [
            py,
            "-m",
            "pip",
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        [py, "-m", "pip", "install", *TRAIN_EXTRAS],
        [py, "-m", "pip", "install", "-e", str(root)],
    ]
    for argv in steps:
        completed = run_command(argv, cwd=root, env=env, log=log)
        if completed.returncode != 0:
            return completed.returncode
    freeze = run_command([py, "-m", "pip", "freeze"], cwd=root, env=env, log=log)
    write_json(
        evidence_dir(root, run_id_from_env()) / "pip-freeze.json",
        {"exit": freeze.returncode, "freeze": freeze.stdout.splitlines()},
    )
    return 0


def cmd_doctor(root: Path, log: TextIO | None) -> int:
    py = python_for(root)
    env = base_env(root)
    schema = root / "schema" / "run_report.schema.json"
    hardware = hardware_probe(py, root, env)
    checks = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        append_log(log, f"doctor {name}: {'ok' if ok else 'fail'} ({detail})\n")

    record("python>=3.10", tuple(map(int, hardware["python"].split(".")[:2])) >= (3, 10), hardware["python"])
    record("schema", schema.is_file(), str(schema))
    record("tests/", (root / "tests").is_dir(), str(root / "tests"))
    record("pyproject", (root / "pyproject.toml").is_file(), str(root / "pyproject.toml"))
    record("venv_python", Path(py).is_file(), py)
    imports = run_command(
        [
            py,
            "-c",
            "import jsonschema, pytest, sft_loop, sft_loop.train, sft_loop.eval, sft_loop.loop, sft_loop.report",
        ],
        cwd=root,
        env=env,
        log=log,
    )
    record("imports", imports.returncode == 0, imports.stderr.strip() or "jsonschema pytest sft_loop")
    record("cuda_absent_or_unused", hardware.get("cuda") is False, f"cuda={hardware.get('cuda')}")
    record("mps_absent_on_linux_cpu", hardware.get("mps") is False, f"mps={hardware.get('mps')}")
    payload = {
        "ok": all(item["ok"] for item in checks),
        "python": py,
        "hardware": hardware,
        "checks": checks,
        "evidence": str(evidence_dir(root, run_id_from_env())),
        "scratch": str(scratch_dir(run_id_from_env())),
    }
    out = evidence_dir(root, run_id_from_env()) / "doctor.json"
    write_json(out, payload)
    append_log(log, f"wrote {out}\n")
    return 0 if payload["ok"] else 1


def cmd_test(root: Path, log: TextIO | None) -> int:
    py = python_for(root)
    env = base_env(root)
    evidence = ensure_dir(evidence_dir(root, run_id_from_env()))
    completed = run_command(
        [py, "-m", "pytest", "-q", "--color=no"],
        cwd=root,
        env=env,
        log=log,
    )
    (evidence / "pytest.txt").write_text(
        completed.stdout + completed.stderr + f"exit {completed.returncode}\n",
        encoding="utf-8",
    )
    return completed.returncode


def validate_path(py: str, root: Path, env: dict[str, str], path: Path, proof_dir: Path | None) -> dict[str, Any]:
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from sft_loop.report import validate_report\n"
        "path = Path(sys.argv[1])\n"
        "proof = Path(sys.argv[2]) if sys.argv[2] else None\n"
        "report = json.loads(path.read_text())\n"
        "validate_report(report, proof_dir=proof)\n"
        "print('valid')\n"
    )
    argv = [py, "-c", code, str(path), str(proof_dir) if proof_dir else ""]
    completed = subprocess.run(argv, cwd=root, env=env, check=False, text=True, capture_output=True)
    return {
        "path": str(path),
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def cmd_unsupported(root: Path, log: TextIO | None) -> int:
    py = python_for(root)
    env = base_env(root)
    evidence = ensure_dir(evidence_dir(root, run_id_from_env()) / "reports")
    jobs = [
        ("cli-default.json", ["-m", "sft_loop", "--run-id", "verify-cli-default", "--out", str(evidence / "cli-default.json")]),
        (
            "loop-default.json",
            [
                "-m",
                "sft_loop.loop",
                "--model-id",
                MODEL_ID,
                "--seed",
                str(SEED),
                "--run-id",
                "verify-loop-default",
                "--out",
                str(evidence / "loop-default.json"),
            ],
        ),
        (
            "loop-mps.json",
            [
                "-m",
                "sft_loop.loop",
                "--model-id",
                MODEL_ID,
                "--seed",
                str(SEED),
                "--device",
                "mps",
                "--run-id",
                "verify-loop-mps",
                "--out",
                str(evidence / "loop-mps.json"),
            ],
        ),
        (
            "loop-cuda.json",
            [
                "-m",
                "sft_loop.loop",
                "--model-id",
                MODEL_ID,
                "--seed",
                str(SEED),
                "--device",
                "cuda",
                "--run-id",
                "verify-loop-cuda",
                "--out",
                str(evidence / "loop-cuda.json"),
            ],
        ),
        (
            "train-default.json",
            [
                "-m",
                "sft_loop.train",
                "--model-id",
                MODEL_ID,
                "--seed",
                str(SEED),
                "--run-id",
                "verify-train-default",
                "--out",
                str(evidence / "train-default.json"),
            ],
        ),
        (
            "eval-before.json",
            [
                "-m",
                "sft_loop.eval",
                "--model-id",
                MODEL_ID,
                "--seed",
                str(SEED),
                "--split",
                "before",
                "--run-id",
                "verify-eval-before",
                "--out",
                str(evidence / "eval-before.json"),
            ],
        ),
        (
            "eval-after.json",
            [
                "-m",
                "sft_loop.eval",
                "--model-id",
                MODEL_ID,
                "--seed",
                str(SEED),
                "--split",
                "after",
                "--run-id",
                "verify-eval-after",
                "--out",
                str(evidence / "eval-after.json"),
            ],
        ),
    ]
    validations = []
    failed = False
    for name, extra in jobs:
        argv = [py, *extra]
        completed = run_command(argv, cwd=root, env=env, log=log)
        path = evidence / name
        if completed.returncode != 0 or not path.is_file():
            failed = True
            validations.append({"path": str(path), "ok": False, "stderr": completed.stderr})
            continue
        result = validate_path(py, root, env, path, None)
        validations.append(result)
        if not result["ok"]:
            failed = True
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "unsupported" or report.get("weight_change_claimed") is not False:
            failed = True
            append_log(log, f"{path} is not a fail-closed unsupported report\n")
        if "before" in report or "after" in report or "task_score" in report:
            failed = True
            append_log(log, f"{path} invented metrics\n")
    write_json(evidence.parent / "unsupported-validation.json", validations)
    return 1 if failed else 0


def _unsupported_payload(reason: str, detail: str) -> dict[str, Any]:
    from sft_loop.report import unsupported_report, validate_report

    report = unsupported_report(
        run_id=SMOKE_RUN_ID,
        reason=reason,
        detail=detail,
        seed=SEED,
    )
    validate_report(report)
    return report


def cmd_smoke(root: Path, log: TextIO | None) -> int:
    py = python_for(root)
    env = base_env(root)
    run_id = run_id_from_env()
    evidence = ensure_dir(evidence_dir(root, run_id) / "smoke")
    scratch = ensure_dir(scratch_dir(run_id))
    adapter_dir = scratch / "adapters"
    proof_dir = evidence
    hardware = hardware_probe(py, root, env)
    plumbing = {
        "kind": "plumbing_smoke_not_quality",
        "model_id": MODEL_ID,
        "seed": SEED,
        "split": "held-out uppercase-word-v1 eval words, n=8, exact_match, greedy do_sample=False",
        "max_steps": 2,
        "lora_r": 8,
        "dtype": "float32",
        "attn_implementation": "eager",
        "device_requested": "cpu",
        "hardware": hardware,
        "note": (
            "before/after exact_match values from this run are plumbing checks that "
            "the TRL+PEFT train and eval functions ran on CPU. They are not quality "
            "results and are not an MPS or CUDA scored path. Schema device labels "
            "mps and cuda are omitted on purpose."
        ),
    }
    write_json(evidence / "plumbing.json", plumbing)
    if hardware.get("cuda") or hardware.get("mps"):
        append_log(
            log,
            "doctor expected a CPU-only VM. cuda or mps is visible. Refuse to label this smoke as those devices.\n",
        )
    code = r"""
import json, os, sys, traceback
from pathlib import Path

adapter_dir = Path(sys.argv[1])
proof_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
fail_path = Path(sys.argv[4])

from sft_loop.eval import _execute_eval_trl_peft
from sft_loop.report import build_scored_report, dump_report, validate_report
from sft_loop.train import _execute_train_trl_peft

SEED = 0
try:
    before = _execute_eval_trl_peft(seed=SEED, adapter_dir=None, device="cpu")
    proof = _execute_train_trl_peft(
        seed=SEED,
        proof_dir=proof_dir,
        adapter_dir=adapter_dir,
        device="cpu",
    )
    after = _execute_eval_trl_peft(seed=SEED, adapter_dir=adapter_dir, device="cpu")
    report = build_scored_report(
        run_id="verify-cpu-smoke-seed0",
        seed=SEED,
        before=before,
        after=after,
        train_proof=proof,
        proof_dir=proof_dir,
    )
    validate_report(report, proof_dir=proof_dir)
    out_path.write_text(dump_report(report), encoding="utf-8")
    print("smoke-scored")
except Exception as exc:
    fail_path.write_text(
        json.dumps({"type": type(exc).__name__, "error": str(exc), "trace": traceback.format_exc()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    raise
"""
    fail_path = evidence / "smoke-exception.json"
    out_path = evidence / "run_report.json"
    started = time.time()
    completed = run_command(
        [py, "-c", code, str(adapter_dir), str(proof_dir), str(out_path), str(fail_path)],
        cwd=root,
        env=env,
        log=log,
    )
    plumbing["elapsed_s"] = round(time.time() - started, 3)
    plumbing["exit"] = completed.returncode
    write_json(evidence / "plumbing.json", plumbing)
    if completed.returncode != 0:
        detail = (
            "Requested CPU plumbing smoke at seed 0 on the pinned model. "
            "The train step did not run. No before/after metrics were produced. "
            "Status is unsupported. " + REFUSAL_SENTENCE
        )
        if len(detail) > 500:
            detail = detail[:500]
        report = _unsupported_payload(REASON_TRAIN, detail)
        write_json(out_path, report)
        append_log(log, f"smoke failed closed. wrote unsupported report to {out_path}\n")
        return completed.returncode
    result = validate_path(py, root, env, out_path, proof_dir)
    write_json(evidence / "report-validation.json", result)
    if not result["ok"]:
        append_log(log, "smoke report failed schema validation\n")
        return 1
    append_log(log, f"smoke report valid at {out_path}\n")
    return 0


def static_surfaces(hardware: dict[str, Any], smoke_ok: bool, tests_ok: bool, unsupported_ok: bool) -> list[Surface]:
    cuda = bool(hardware.get("cuda"))
    mps = bool(hardware.get("mps"))
    torch_v = hardware.get("torch")
    return [
        Surface(
            id="pytest",
            command="python -m pytest -q",
            status="supported" if tests_ok else "unsupported",
            reason=None if tests_ok else REASON_TRAIN,
            detail="Full tests/ suite on this checkout.",
        ),
        Surface(
            id="cli-unsupported-report",
            command=f"python -m sft_loop --model-id {MODEL_ID}",
            status="supported" if unsupported_ok else "unsupported",
            reason=None if unsupported_ok else REASON_TRAIN,
            detail="Fail-closed unsupported JSON from python -m sft_loop.",
        ),
        Surface(
            id="loop-default",
            command=f"python -m sft_loop.loop --model-id {MODEL_ID} --seed {SEED}",
            status="supported" if unsupported_ok else "unsupported",
            reason=None if unsupported_ok else REASON_DEPS,
            detail="Default loop on a machine with no MPS and no CUDA Unsloth stack emits unsupported.",
        ),
        Surface(
            id="train-mps",
            command=f"python -m sft_loop.train --model-id {MODEL_ID} --seed {SEED} --device mps",
            status="unsupported",
            reason=REASON_DEPS,
            detail="MPS is unavailable on this Linux CPU VM. " + REFUSAL_SENTENCE,
        ),
        Surface(
            id="train-cuda-unsloth",
            command=f"python -m sft_loop.loop --model-id {MODEL_ID} --seed {SEED} --device cuda",
            status="unsupported",
            reason=REASON_DEPS,
            detail=(
                f"CUDA and the unsloth extra are not used here. cuda={cuda} torch={torch_v}. "
                + REFUSAL_SENTENCE
            ),
        ),
        Surface(
            id="eval-mps",
            command=f"python -m sft_loop.eval --model-id {MODEL_ID} --seed {SEED} --split before --device mps",
            status="unsupported",
            reason=REASON_DEPS,
            detail=f"Eval on MPS is blocked with the train stack. mps={mps}. " + REFUSAL_SENTENCE,
        ),
        Surface(
            id="bitsandbytes",
            command="not a scored-path dependency",
            status="unsupported",
            reason=REASON_DEPS,
            detail="bitsandbytes is not installed. The MPS TRL+PEFT path loads float32. " + REFUSAL_SENTENCE,
        ),
        Surface(
            id="flash-attn",
            command="not a scored-path dependency",
            status="unsupported",
            reason=REASON_DEPS,
            detail="flash-attn is not installed. Train uses attn_implementation=eager. " + REFUSAL_SENTENCE,
        ),
        Surface(
            id="fp16-bf16-gpu-kernels",
            command="SFTConfig bf16=False fp16=False",
            status="unsupported",
            reason=REASON_DEPS,
            detail="fp16/bf16 GPU kernels are not used. Train config sets bf16 and fp16 false. "
            + REFUSAL_SENTENCE,
        ),
        Surface(
            id="gated-huggingface-models",
            command=f"AutoModelForCausalLM.from_pretrained({MODEL_ID})",
            status="unsupported" if not smoke_ok else "supported",
            reason=REASON_DEPS if not smoke_ok else None,
            detail=(
                f"{MODEL_ID} is the pinned public Apache-2.0 model. Other gated models are not loaded."
            ),
        ),
        Surface(
            id="network-only-datasets",
            command="sft_loop.task train_examples / eval_examples",
            status="supported",
            reason=None,
            detail="Train and eval rows are in-code synthetic words. No Hub dataset is downloaded.",
        ),
        Surface(
            id="cpu-trl-peft-smoke",
            command="scripts/verify smoke (calls _execute_train_trl_peft and _execute_eval_trl_peft device=cpu)",
            status="supported" if smoke_ok else "unsupported",
            reason=None if smoke_ok else REASON_TRAIN,
            detail=(
                "CPU plumbing LoRA SFT, 2 steps, seed 0, pinned model. Metrics are plumbing checks, not quality."
            ),
        ),
    ]


def write_surfaces(
    root: Path,
    hardware: dict[str, Any],
    *,
    smoke_ok: bool,
    tests_ok: bool,
    unsupported_ok: bool,
) -> Path:
    surfaces = static_surfaces(hardware, smoke_ok, tests_ok, unsupported_ok)
    payload = {
        "hardware": hardware,
        "supported": [asdict(item) for item in surfaces if item.status == "supported"],
        "unsupported": [asdict(item) for item in surfaces if item.status == "unsupported"],
        "all": [asdict(item) for item in surfaces],
    }
    path = evidence_dir(root, run_id_from_env()) / "surfaces.json"
    write_json(path, payload)
    return path


def cmd_cleanup(root: Path, log: TextIO | None) -> int:
    run_id = run_id_from_env()
    scratch = scratch_dir(run_id)
    evidence = evidence_dir(root, run_id)
    if scratch.exists():
        shutil.rmtree(scratch)
        append_log(log, f"removed scratch {scratch}\n")
    else:
        append_log(log, f"no scratch at {scratch}\n")
    surviving = sorted(str(path.relative_to(evidence)) for path in evidence.rglob("*") if path.is_file()) if evidence.exists() else []
    marker = {
        "scratch_removed": str(scratch),
        "evidence_dir": str(evidence),
        "evidence_survived": bool(evidence.exists()),
        "files": surviving,
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "note": "cleanup never deletes evidence",
    }
    write_json(evidence / "cleanup.json", marker)
    append_log(log, f"evidence remains at {evidence} ({len(surviving)} files)\n")
    if not evidence.exists() or not surviving:
        append_log(log, "cleanup ate the proof\n")
        return 1
    return 0


def cmd_all(root: Path, log: TextIO | None) -> int:
    run_id = run_id_from_env()
    evidence = ensure_dir(evidence_dir(root, run_id))
    started = time.time()
    steps = [
        ("install", cmd_install),
        ("doctor", cmd_doctor),
        ("test", cmd_test),
        ("unsupported", cmd_unsupported),
        ("smoke", cmd_smoke),
        ("cleanup", cmd_cleanup),
    ]
    results = []
    overall = 0
    for name, fn in steps:
        append_log(log, f"\n== {name} ==\n")
        code = fn(root, log)
        results.append({"step": name, "exit": code})
        if code != 0 and name != "smoke":
            overall = code
            if name in {"install", "doctor", "test"}:
                break
        if name == "smoke" and code != 0:
            results[-1]["note"] = "smoke failed closed. official CLI unsupported path may still be valid."
    hardware = hardware_probe(python_for(root), root, base_env(root))
    tests_ok = all(item["exit"] == 0 for item in results if item["step"] == "test")
    unsupported_ok = all(item["exit"] == 0 for item in results if item["step"] == "unsupported")
    smoke_ok = all(item["exit"] == 0 for item in results if item["step"] == "smoke")
    surfaces_path = write_surfaces(
        root,
        hardware,
        smoke_ok=smoke_ok,
        tests_ok=tests_ok,
        unsupported_ok=unsupported_ok,
    )
    summary = {
        "run_id": run_id,
        "elapsed_s": round(time.time() - started, 3),
        "steps": results,
        "surfaces": str(surfaces_path),
        "evidence": str(evidence),
        "exit": overall if overall != 0 else (0 if unsupported_ok and tests_ok else 1),
    }
    write_json(evidence / "summary.json", summary)
    append_log(log, json.dumps(summary, indent=2) + "\n")
    return summary["exit"] if smoke_ok or unsupported_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify sft-loop on a CPU VM.")
    parser.add_argument(
        "command",
        choices=("install", "doctor", "test", "unsupported", "smoke", "cleanup", "all"),
    )
    parser.add_argument("--run-id", help="Evidence directory name. Default VERIFY_RUN_ID or cpu-linux-seed0.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_id:
        os.environ["VERIFY_RUN_ID"] = args.run_id
    root = repo_root()
    evidence = ensure_dir(evidence_dir(root, run_id_from_env()))
    log_path = evidence / "transcript.txt"
    commands = {
        "install": cmd_install,
        "doctor": cmd_doctor,
        "test": cmd_test,
        "unsupported": cmd_unsupported,
        "smoke": cmd_smoke,
        "cleanup": cmd_cleanup,
        "all": cmd_all,
    }
    with log_path.open("a", encoding="utf-8") as log:
        append_log(log, f"\n# verify {args.command} run_id={run_id_from_env()} cwd={root}\n")
        try:
            return commands[args.command](root, log)
        except Exception:
            append_log(log, traceback.format_exc())
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
