#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
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
PINNED_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
SMOKE_MODEL_REVISION = "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda"
SEED = 0
REFUSAL_SENTENCE = "Soft-PASS is unused and is refused."
REASON_DEPS = "gpu_or_train_deps_unavailable"
REASON_TRAIN = "train_not_run"
LORA_R = 8
LORA_ALPHA = 16
MAX_STEPS = 2
MAX_SEQ_LENGTH = 128
MAX_NEW_TOKENS = 8
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_RUN_ID = "cpu-linux-tiny"

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


@dataclass
class RunConfig:
    run_id: str = DEFAULT_RUN_ID
    evidence: Path | None = None
    scratch: Path | None = None
    pinned_model: bool = False
    evidence_warning: str = ""


RUN = RunConfig()


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
    return RUN.run_id


def smoke_model() -> tuple[str, str | None]:
    if RUN.pinned_model:
        return PINNED_MODEL_ID, None
    return SMOKE_MODEL_ID, SMOKE_MODEL_REVISION


def evidence_dir(root: Path, run_id: str) -> Path:
    if RUN.evidence is not None and RUN.run_id == run_id:
        return RUN.evidence
    override = os.environ.get("VERIFY_EVIDENCE_DIR")
    if override:
        path = Path(override)
        if path.name == run_id:
            return path
    return SKILL_DIR / "evidence" / run_id


def scratch_dir(run_id: str) -> Path:
    if RUN.scratch is not None and RUN.run_id == run_id:
        return RUN.scratch
    override = os.environ.get("VERIFY_SCRATCH_DIR")
    if override:
        path = Path(override)
        if path.name == run_id or path.name == f"sft-loop-verify-{run_id}":
            return path
    return Path("/tmp") / f"sft-loop-verify-{run_id}"


def configure_run(args: argparse.Namespace) -> None:
    run_id = args.run_id or DEFAULT_RUN_ID
    warning = ""
    evidence = None
    if getattr(args, "evidence_dir", None):
        evidence = Path(args.evidence_dir)
    elif os.environ.get("VERIFY_EVIDENCE_DIR"):
        env_path = Path(os.environ["VERIFY_EVIDENCE_DIR"])
        if env_path.name == run_id:
            evidence = env_path
        else:
            warning = (
                f"ignoring VERIFY_EVIDENCE_DIR={env_path} because it does not end with run-id {run_id}"
            )
            evidence = SKILL_DIR / "evidence" / run_id
    else:
        evidence = SKILL_DIR / "evidence" / run_id
    scratch = None
    if getattr(args, "scratch_dir", None):
        scratch = Path(args.scratch_dir)
    elif os.environ.get("VERIFY_SCRATCH_DIR"):
        env_path = Path(os.environ["VERIFY_SCRATCH_DIR"])
        if env_path.name == run_id or env_path.name == f"sft-loop-verify-{run_id}":
            scratch = env_path
        else:
            extra = (
                f"ignoring VERIFY_SCRATCH_DIR={env_path} because it does not match run-id {run_id}"
            )
            warning = f"{warning}; {extra}" if warning else extra
            scratch = Path("/tmp") / f"sft-loop-verify-{run_id}"
    else:
        scratch = Path("/tmp") / f"sft-loop-verify-{run_id}"
    RUN.run_id = run_id
    RUN.evidence = evidence
    RUN.scratch = scratch
    RUN.pinned_model = bool(getattr(args, "pinned_model", False))
    RUN.evidence_warning = warning


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
    if venv_usable(root):
        return str(venv_python(root))
    return sys.executable


def _tensor_max_abs(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().float().abs().max().item())
    import numpy as np

    return float(np.abs(value).max())


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, Any]:
    safe = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safe.is_file():
        from safetensors.torch import load_file

        return dict(load_file(str(safe), device="cpu"))
    if bin_path.is_file():
        import torch

        payload = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError("train_not_run: adapter_model.bin is not a state dict")
        return payload
    raise RuntimeError(f"train_not_run: no adapter weights under {adapter_dir}")


def measure_lora_b(adapter_dir: Path) -> dict[str, Any]:
    tensors = _load_adapter_tensors(adapter_dir)
    keys = [key for key in tensors if "lora_B" in key]
    max_abs = 0.0
    n_nonzero = 0
    for key in keys:
        value = _tensor_max_abs(tensors[key])
        if value > 0.0:
            n_nonzero += 1
        if value > max_abs:
            max_abs = value
    return {
        "max_abs": max_abs,
        "n_tensors": len(keys),
        "n_nonzero": n_nonzero,
        "keys": keys,
    }


def require_smoke_train(
    *,
    global_step: int,
    train_loss: float,
    max_steps: int,
    lora_b: dict[str, Any],
) -> None:
    if not math.isfinite(float(train_loss)):
        raise RuntimeError("train_not_run: train_loss is not finite")
    if int(global_step) != int(max_steps):
        raise RuntimeError(
            f"train_not_run: global_step {global_step} != requested max_steps {max_steps}"
        )
    max_abs = float(lora_b.get("max_abs") or 0.0)
    n_nonzero = int(lora_b.get("n_nonzero") or 0)
    if n_nonzero < 1 or max_abs == 0.0:
        raise RuntimeError(
            "train_not_run: LoRA lora_B still at init "
            f"(max_abs={max_abs}, n_nonzero={n_nonzero}, n_tensors={lora_b.get('n_tensors', 0)}). "
            "delta W = B A and B starts at 0, so a zero B means the adapter did not update."
        )


def venv_usable(root: Path) -> bool:
    py = venv_python(root)
    if not py.is_file():
        return False
    completed = subprocess.run(
        [str(py), "-m", "pip", "--version"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


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
        "cuda=False; mps=False; torch_v=None; err=None\n"
        "try:\n"
        " import torch\n"
        " torch_v=torch.__version__\n"
        " cuda=bool(torch.cuda.is_available())\n"
        " b=getattr(torch.backends,'mps',None)\n"
        " mps=bool(b.is_available()) if b is not None and hasattr(b,'is_available') else False\n"
        "except Exception as exc:\n"
        " err=f'{type(exc).__name__}: {exc}'\n"
        "mem=None\n"
        "try:\n"
        " mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')\n"
        "except Exception:\n"
        " pass\n"
        "print(json.dumps({'python':sys.version.split()[0],'platform':platform.platform(),"
        "'machine':platform.machine(),'torch':torch_v,'cuda':cuda,'mps':mps,"
        "'cpu_count':os.cpu_count(),'ram_bytes':mem,'probe_error':err}))\n"
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


def probe_imports(py: str, cwd: Path, env: dict[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in ("bitsandbytes", "flash_attn", "unsloth"):
        try:
            completed = subprocess.run(
                [py, "-c", f"import {name}"],
                cwd=cwd,
                env=env,
                check=False,
                text=True,
                capture_output=True,
                timeout=60,
            )
            if completed.returncode == 0:
                out[name] = {"ok": True, "error": None}
            else:
                err = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
                out[name] = {"ok": False, "error": err[:400]}
        except Exception as exc:
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def _create_venv(root: Path, log: TextIO | None) -> int:
    target = root / ".venv"
    if target.exists() and not venv_usable(root):
        append_log(log, "removing .venv with no working pip before recreate\n")
        shutil.rmtree(target, ignore_errors=True)
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(target)],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    append_log(log, created.stdout or "")
    append_log(log, created.stderr or "")
    if created.returncode == 0 and venv_usable(root):
        return 0
    append_log(log, "stdlib venv missing pip or failed. recreating with virtualenv.\n")
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
    if not venv_usable(root):
        append_log(log, "virtualenv create failed or pip is still missing\n")
        return created.returncode or 1
    return 0


def cmd_install(root: Path, log: TextIO | None) -> int:
    if not venv_usable(root):
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
    record("venv_pip", venv_usable(root), str(venv_python(root)))
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
                PINNED_MODEL_ID,
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
                PINNED_MODEL_ID,
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
                PINNED_MODEL_ID,
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
                PINNED_MODEL_ID,
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
                PINNED_MODEL_ID,
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
                PINNED_MODEL_ID,
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


def assert_smoke_not_scored(smoke_dir: Path) -> None:
    from sft_loop.report import validate_report

    if not smoke_dir.is_dir():
        raise AssertionError(f"missing smoke dir {smoke_dir}")
    json_files = list(smoke_dir.rglob("*.json"))
    if not json_files:
        raise AssertionError(f"no json under {smoke_dir}")
    for path in json_files:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        if document.get("status") == "scored":
            raise AssertionError(f"{path} has status scored")
        if document.get("weight_change_claimed") is True:
            raise AssertionError(f"{path} claims weight change")
        try:
            validate_report(document)
        except (ValueError, Exception):
            continue
        if document.get("status") == "scored":
            raise AssertionError(f"{path} validated as a scored report")


def cmd_smoke(root: Path, log: TextIO | None) -> int:
    py = python_for(root)
    env = base_env(root)
    run_id = run_id_from_env()
    evidence = ensure_dir(evidence_dir(root, run_id) / "smoke")
    scratch = ensure_dir(scratch_dir(run_id))
    adapter_dir = scratch / "adapters"
    hardware = hardware_probe(py, root, env)
    model_id, revision = smoke_model()
    plumbing: dict[str, Any] = {
        "kind": "plumbing_smoke_not_quality",
        "status": "plumbing",
        "weight_change_claimed": False,
        "device": "cpu",
        "backend": "trl-peft",
        "stack": {"unsloth": "unused", "trl": "0.24.0", "method": "sft-lora"},
        "model_id": model_id,
        "model_revision": revision,
        "seed": SEED,
        "split": "held-out uppercase-word-v1 eval words, n=8, exact_match, greedy do_sample=False",
        "max_steps": MAX_STEPS,
        "lora_r": LORA_R,
        "dtype": "float32",
        "attn_implementation": "eager",
        "lora_b_max_abs": None,
        "hardware": hardware,
        "note": (
            "CPU plumbing check only. Not a scored run report. "
            "device=cpu and backend=trl-peft are recorded here, not in schema/run_report.schema.json. "
            "stack.unsloth is unused because this path is TRL+PEFT. "
            "Smoke fails unless train_loss is finite, global_step == max_steps, and a saved lora_B tensor is nonzero."
        ),
    }
    write_json(evidence / "plumbing.json", plumbing)
    if hardware.get("cuda") or hardware.get("mps"):
        append_log(
            log,
            "cuda or mps is visible. Refuse to label this smoke as those devices.\n",
        )
    code = r"""
import json, math, shutil, sys, traceback
from pathlib import Path

adapter_dir = Path(sys.argv[1])
proof_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
fail_path = Path(sys.argv[4])
model_id = sys.argv[5]
revision = sys.argv[6] or None
sys.path.insert(0, sys.argv[7])
from verify_sft_loop import measure_lora_b, require_smoke_train

from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import torch

from sft_loop.task import METRIC_NAME, eval_examples, grade, prediction_token, train_examples

SEED = 0
LORA_R = 8
LORA_ALPHA = 16
MAX_STEPS = 2
MAX_SEQ_LENGTH = 128
MAX_NEW_TOKENS = 8
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
global_step = 0
train_loss = None
lora_b = None


def text_rows(tokenizer):
    rows = []
    for example in train_examples():
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = example["prompt"] + " " + example["completion"]
        rows.append({"text": text})
    return rows


def load_model():
    kwargs = {"torch_dtype": torch.float32, "attn_implementation": "eager", "low_cpu_mem_usage": True}
    tok_kwargs = {}
    if revision:
        kwargs["revision"] = revision
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model, tokenizer


def generate_scores(model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = eval_examples()
    predictions = []
    completions = []
    for example in examples:
        messages = [{"role": "user", "content": example["prompt"]}]
        if getattr(tokenizer, "chat_template", None):
            tokenized = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            tokenized = tokenizer(example["prompt"], return_tensors="pt").input_ids
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to("cpu")
        else:
            input_ids = tokenized.to("cpu")
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[0][input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(prediction_token(text))
        completions.append(example["completion"])
    return {
        "seed": SEED,
        "metric_name": METRIC_NAME,
        "task_score": grade(predictions, completions),
        "n": len(completions),
        "role": "plumbing_not_quality",
    }

try:
    torch.manual_seed(SEED)
    model, tokenizer = load_model()
    model.to("cpu")
    model.eval()
    before = generate_scores(model, tokenizer)
    del model
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    proof_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
        base_model_name_or_path=model_id,
    )
    model = get_peft_model(model, lora)
    model.to("cpu")
    args = SFTConfig(
        output_dir=str(adapter_dir),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(text_rows(tokenizer)),
        processing_class=tokenizer,
    )
    result = trainer.train()
    global_step = int(getattr(result, "global_step", 0) or 0)
    train_loss = float(getattr(result, "training_loss"))
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    lora_b = measure_lora_b(adapter_dir)
    require_smoke_train(
        global_step=global_step,
        train_loss=train_loss,
        max_steps=MAX_STEPS,
        lora_b=lora_b,
    )
    config_src = adapter_dir / "adapter_config.json"
    shutil.copyfile(config_src, proof_dir / "adapter_config.json")
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    log_lines = [
        "trainer=trl.SFTTrainer",
        f"model_id={model_id}",
        f"seed={SEED}",
        f"global_step={global_step}",
        f"train_loss={format(train_loss, '.8g')}",
        f"max_steps={MAX_STEPS}",
        f"lora_r={LORA_R}",
        f"lora_b_max_abs={format(lora_b['max_abs'], '.8g')}",
        f"lora_b_nonzero={lora_b['n_nonzero']}",
        "device=cpu",
        "backend=trl-peft",
        f"log_history={json.dumps(log_history, separators=(',', ':'), sort_keys=True)}",
    ]
    (proof_dir / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    del model
    model, tokenizer = load_model()
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to("cpu")
    model.eval()
    after = generate_scores(model, tokenizer)
    payload = {
        "kind": "plumbing_smoke_not_quality",
        "status": "plumbing",
        "weight_change_claimed": False,
        "device": "cpu",
        "backend": "trl-peft",
        "stack": {"unsloth": "unused", "trl": "0.24.0", "method": "sft-lora"},
        "model_id": model_id,
        "model_revision": revision,
        "seed": SEED,
        "before": before,
        "after": after,
        "lora_b_max_abs": lora_b["max_abs"],
        "train": {
            "trainer": "trl.SFTTrainer",
            "global_step": global_step,
            "train_loss": float(format(train_loss, ".8g")),
            "lora_r": LORA_R,
            "max_steps": MAX_STEPS,
            "device": "cpu",
            "backend": "trl-peft",
            "lora_b_max_abs": lora_b["max_abs"],
            "lora_b_tensors": lora_b["n_tensors"],
            "lora_b_nonzero": lora_b["n_nonzero"],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("smoke-plumbing")
except Exception as exc:
    fail_payload = {
        "type": type(exc).__name__,
        "error": str(exc),
        "trace": traceback.format_exc(),
        "global_step": global_step,
        "train_loss": train_loss,
        "lora_b": lora_b,
    }
    fail_path.write_text(json.dumps(fail_payload, indent=2) + "\n", encoding="utf-8")
    raise
"""
    fail_path = evidence / "smoke-exception.json"
    out_path = evidence / "plumbing.json"
    started = time.time()
    completed = run_command(
        [
            py,
            "-c",
            code,
            str(adapter_dir),
            str(evidence),
            str(out_path),
            str(fail_path),
            model_id,
            revision or "",
            str(SKILL_DIR / "scripts"),
        ],
        cwd=root,
        env=env,
        log=log,
    )
    plumbing["elapsed_s"] = round(time.time() - started, 3)
    plumbing["exit"] = completed.returncode
    if completed.returncode != 0:
        plumbing["status"] = "plumbing_failed"
        plumbing["weight_change_claimed"] = False
        plumbing["device"] = "cpu"
        plumbing["backend"] = "trl-peft"
        plumbing["stack"] = {"unsloth": "unused", "trl": "0.24.0", "method": "sft-lora"}
        if fail_path.is_file():
            plumbing["error"] = json.loads(fail_path.read_text(encoding="utf-8"))
            err = plumbing["error"]
            if isinstance(err, dict):
                plumbing["global_step"] = err.get("global_step")
                plumbing["train_loss"] = err.get("train_loss")
                lora_b = err.get("lora_b") or {}
                plumbing["lora_b_max_abs"] = lora_b.get("max_abs")
                plumbing["lora_b_tensors"] = lora_b.get("n_tensors")
                plumbing["lora_b_nonzero"] = lora_b.get("n_nonzero")
        write_json(out_path, plumbing)
        try:
            assert_smoke_not_scored(evidence)
        except AssertionError as exc:
            append_log(log, f"smoke produced a scored-shaped artifact: {exc}\n")
            return 1
        append_log(log, f"smoke failed closed. wrote plumbing artifact to {out_path}\n")
        return completed.returncode
    merged = json.loads(out_path.read_text(encoding="utf-8"))
    merged["elapsed_s"] = plumbing["elapsed_s"]
    merged["exit"] = 0
    merged["hardware"] = hardware
    merged["note"] = plumbing["note"]
    write_json(out_path, merged)
    try:
        assert_smoke_not_scored(evidence)
    except AssertionError as exc:
        append_log(log, f"smoke produced a scored-shaped artifact: {exc}\n")
        return 1
    append_log(log, f"smoke plumbing artifact at {out_path}\n")
    return 0


def build_surfaces(
    hardware: dict[str, Any],
    probes: dict[str, dict[str, Any]],
    *,
    tests_ok: bool,
    unsupported_ok: bool,
    smoke_ok: bool,
    quant_ok: bool = False,
) -> list[Surface]:
    cuda = hardware.get("cuda")
    mps = hardware.get("mps")
    mps_detail_extra = ""
    if hardware.get("probe_error"):
        mps_detail_extra = f" probe_error={hardware['probe_error']}"
    bnb = probes.get("bitsandbytes") or {"ok": False, "error": "probe missing"}
    flash = probes.get("flash_attn") or {"ok": False, "error": "probe missing"}
    unsloth = probes.get("unsloth") or {"ok": False, "error": "probe missing"}

    def blocked(detail: str) -> tuple[str, str, str]:
        return "unsupported", REASON_DEPS, detail

    cuda_status, cuda_reason, cuda_detail = (
        ("supported", None, f"torch.cuda.is_available()={cuda}")
        if cuda is True
        else blocked(f"torch.cuda.is_available()={cuda}.{mps_detail_extra} " + REFUSAL_SENTENCE)
    )
    mps_status, mps_reason, mps_detail = (
        ("supported", None, f"torch.backends.mps.is_available()={mps}")
        if mps is True
        else blocked(f"torch.backends.mps.is_available()={mps}.{mps_detail_extra} " + REFUSAL_SENTENCE)
    )
    bnb_status, bnb_reason, bnb_detail = (
        ("supported", None, "import bitsandbytes succeeded")
        if bnb.get("ok")
        else blocked(f"import bitsandbytes failed: {bnb.get('error')} " + REFUSAL_SENTENCE)
    )
    flash_status, flash_reason, flash_detail = (
        ("supported", None, "import flash_attn succeeded")
        if flash.get("ok")
        else blocked(f"import flash_attn failed: {flash.get('error')} " + REFUSAL_SENTENCE)
    )
    unsloth_status, unsloth_reason, unsloth_detail = (
        ("supported", None, "import unsloth succeeded")
        if unsloth.get("ok")
        else blocked(f"import unsloth failed: {unsloth.get('error')} " + REFUSAL_SENTENCE)
    )
    fp16_ok = bool(cuda) or bool(mps)
    fp16_status, fp16_reason, fp16_detail = (
        ("supported", None, "cuda or mps present so fp16/bf16 kernels may exist")
        if fp16_ok
        else blocked("no cuda or mps. SFTConfig sets bf16=False fp16=False. " + REFUSAL_SENTENCE)
    )
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
            command=f"python -m sft_loop --model-id {PINNED_MODEL_ID}",
            status="supported" if unsupported_ok else "unsupported",
            reason=None if unsupported_ok else REASON_TRAIN,
            detail="Fail-closed unsupported JSON from python -m sft_loop.",
        ),
        Surface(
            id="loop-default",
            command=f"python -m sft_loop.loop --model-id {PINNED_MODEL_ID} --seed {SEED}",
            status="supported" if unsupported_ok else "unsupported",
            reason=None if unsupported_ok else REASON_DEPS,
            detail="Default loop on a machine with no MPS and no CUDA Unsloth stack emits unsupported.",
        ),
        Surface(
            id="train-mps",
            command=f"python -m sft_loop.train --model-id {PINNED_MODEL_ID} --seed {SEED} --device mps",
            status=mps_status,
            reason=mps_reason,
            detail=mps_detail,
        ),
        Surface(
            id="train-cuda-unsloth",
            command=f"python -m sft_loop.loop --model-id {PINNED_MODEL_ID} --seed {SEED} --device cuda",
            status="unsupported" if not (cuda is True and unsloth.get("ok")) else "supported",
            reason=REASON_DEPS if not (cuda is True and unsloth.get("ok")) else None,
            detail=(
                f"cuda={cuda}; {unsloth_detail}"
            ),
        ),
        Surface(
            id="eval-mps",
            command=f"python -m sft_loop.eval --model-id {PINNED_MODEL_ID} --seed {SEED} --split before --device mps",
            status=mps_status,
            reason=mps_reason,
            detail=mps_detail,
        ),
        Surface(
            id="bitsandbytes",
            command="import bitsandbytes",
            status=bnb_status,
            reason=bnb_reason,
            detail=bnb_detail,
        ),
        Surface(
            id="flash-attn",
            command="import flash_attn",
            status=flash_status,
            reason=flash_reason,
            detail=flash_detail,
        ),
        Surface(
            id="unsloth",
            command="import unsloth",
            status=unsloth_status,
            reason=unsloth_reason,
            detail=unsloth_detail,
        ),
        Surface(
            id="fp16-bf16-gpu-kernels",
            command="SFTConfig bf16=False fp16=False",
            status=fp16_status,
            reason=fp16_reason,
            detail=fp16_detail,
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
            command="scripts/verify smoke",
            status="supported" if smoke_ok else "unsupported",
            reason=None if smoke_ok else REASON_TRAIN,
            detail=(
                f"CPU plumbing LoRA SFT on {SMOKE_MODEL_ID}@{SMOKE_MODEL_REVISION}. "
                "Not a scored report."
            ),
        ),
        Surface(
            id="quant-comparison",
            command="python -m sft_loop.quant validate && python -m sft_loop.quant probe-nf4",
            status="supported" if quant_ok else "unsupported",
            reason=None if quant_ok else REASON_TRAIN,
            detail=(
                "Schema-valid quant comparison plus fail-closed NF4. "
                "Not a scored SFT report."
            ),
        ),
        Surface(
            id="kaggle-quant-kernels",
            command="kaggle/quant_nf4 and kaggle/quant_onnx_int8",
            status="unsupported",
            reason=REASON_DEPS,
            detail=(
                "Local verify cannot run Kaggle kernels. "
                "GPU NF4+fp16 evidence is kaggle/quant_nf4. "
                "ONNX int8 evidence is kaggle/quant_onnx_int8. "
                + REFUSAL_SENTENCE
            ),
        ),
    ]


def write_surfaces(
    root: Path,
    hardware: dict[str, Any],
    probes: dict[str, dict[str, Any]],
    *,
    smoke_ok: bool,
    tests_ok: bool,
    unsupported_ok: bool,
    quant_ok: bool,
) -> Path:
    surfaces = build_surfaces(
        hardware,
        probes,
        tests_ok=tests_ok,
        unsupported_ok=unsupported_ok,
        smoke_ok=smoke_ok,
        quant_ok=quant_ok,
    )
    payload = {
        "hardware": hardware,
        "probes": probes,
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
    surviving = (
        sorted(str(path.relative_to(evidence)) for path in evidence.rglob("*") if path.is_file())
        if evidence.exists()
        else []
    )
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


def cmd_quant(root: Path, log: TextIO | None) -> int:
    py = python_for(root)
    env = base_env(root)
    evidence = ensure_dir(evidence_dir(root, run_id_from_env()) / "quant")
    dest = evidence / "nf4.json"
    probed = run_command(
        [py, "-m", "sft_loop.quant", "probe-nf4", "--dest", str(dest)],
        cwd=root,
        env=env,
        log=log,
    )
    if probed.returncode != 0:
        return probed.returncode
    document = json.loads(dest.read_text(encoding="utf-8"))
    if document.get("status") != "unsupported" or document.get("precision") != "nf4":
        append_log(log, "probe-nf4 did not write an unsupported nf4 arm\n")
        return 1
    if any(key in document for key in ("task_score", "items", "on_disk_bytes", "latency_p50_ms")):
        append_log(log, "probe-nf4 wrote metric keys\n")
        return 1
    if "Soft-PASS is unused and is refused." not in str(document.get("detail", "")):
        append_log(log, "probe-nf4 detail is missing the refusal sentence\n")
        return 1
    checked = run_command(
        [py, "-m", "sft_loop.quant", "validate"],
        cwd=root,
        env=env,
        log=log,
    )
    if checked.returncode != 0:
        return checked.returncode
    table = run_command(
        [py, "-m", "sft_loop.quant", "table", "--check", str(root / "README.md")],
        cwd=root,
        env=env,
        log=log,
    )
    if table.returncode != 0:
        return table.returncode
    for rel in ("kaggle/quant_nf4/kernel-metadata.json", "kaggle/quant_onnx_int8/kernel-metadata.json"):
        if not (root / rel).is_file():
            append_log(log, f"missing kernel evidence {rel}\n")
            return 1
    missing = run_command(
        [
            py,
            "-m",
            "sft_loop.quant",
            "ingest",
            "--precision",
            "nf4",
            "--from-dir",
            str(evidence / "absent-kernel"),
        ],
        cwd=root,
        env=env,
        log=log,
    )
    if missing.returncode == 0:
        append_log(log, "ingest of a missing kernel dir must fail closed\n")
        return 1
    try:
        ingested = json.loads(missing.stdout)
    except json.JSONDecodeError:
        append_log(log, "ingest of a missing kernel dir did not write JSON\n")
        return 1
    (evidence / "kernel-missing.json").write_text(missing.stdout, encoding="utf-8")
    if ingested.get("status") != "unsupported" or ingested.get("reason") != "kernel_output_missing":
        append_log(log, "ingest of a missing kernel dir is not kernel_output_missing\n")
        return 1
    if any(key in ingested for key in ("task_score", "items", "on_disk_bytes", "latency_p50_ms")):
        append_log(log, "missing kernel ingest wrote metric keys\n")
        return 1
    listed = subprocess.run(
        ["git", "ls-files", "-s", "runs"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        append_log(log, "git ls-files -s runs failed\n")
        return 1
    max_bytes = 5 * 1024 * 1024
    oversized = []
    for line in listed.stdout.splitlines():
        if "\t" not in line:
            continue
        meta, rel = line.split("\t", 1)
        parts = meta.split()
        if len(parts) < 2:
            continue
        blob = parts[1]
        sized = subprocess.run(
            ["git", "cat-file", "-s", blob],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if sized.returncode != 0:
            append_log(log, f"git cat-file -s failed for {rel}\n")
            return 1
        size = int(sized.stdout.strip())
        if size > max_bytes:
            oversized.append(f"{rel} {size} {blob}")
    if oversized:
        append_log(log, "tracked git blobs under runs/ exceed 5 MB:\n")
        for row in oversized:
            append_log(log, f"  {row}\n")
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
        ("quant", cmd_quant),
    ]
    results = []
    failed_step = None
    overall = 0
    for name, fn in steps:
        if failed_step is not None:
            results.append({"step": name, "exit": None, "skipped": True})
            continue
        append_log(log, f"\n== {name} ==\n")
        code = fn(root, log)
        results.append({"step": name, "exit": code})
        if code != 0:
            failed_step = name
            overall = code
            append_log(log, f"verify all failed at step {name} exit {code}\n")
    append_log(log, "\n== cleanup ==\n")
    cleanup_code = cmd_cleanup(root, log)
    results.append({"step": "cleanup", "exit": cleanup_code})
    if cleanup_code != 0 and failed_step is None:
        failed_step = "cleanup"
        overall = cleanup_code
        append_log(log, f"verify all failed at step cleanup exit {cleanup_code}\n")
    hardware = hardware_probe(python_for(root), root, base_env(root))
    probes = probe_imports(python_for(root), root, base_env(root))
    tests_ok = all(item.get("exit") == 0 for item in results if item["step"] == "test")
    unsupported_ok = all(item.get("exit") == 0 for item in results if item["step"] == "unsupported")
    smoke_ok = all(item.get("exit") == 0 for item in results if item["step"] == "smoke")
    quant_ok = all(item.get("exit") == 0 for item in results if item["step"] == "quant")
    surfaces_path = write_surfaces(
        root,
        hardware,
        probes,
        smoke_ok=smoke_ok,
        tests_ok=tests_ok,
        unsupported_ok=unsupported_ok,
        quant_ok=quant_ok,
    )
    summary = {
        "run_id": run_id,
        "elapsed_s": round(time.time() - started, 3),
        "steps": results,
        "failed_step": failed_step,
        "surfaces": str(surfaces_path),
        "evidence": str(evidence),
        "exit": overall,
        "pinned_model": RUN.pinned_model,
        "smoke_model_id": smoke_model()[0],
        "smoke_model_revision": smoke_model()[1],
    }
    write_json(evidence / "summary.json", summary)
    append_log(log, json.dumps(summary, indent=2) + "\n")
    return overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify sft-loop on a CPU VM.")
    parser.add_argument(
        "command",
        choices=("install", "doctor", "test", "unsupported", "smoke", "quant", "cleanup", "all"),
    )
    parser.add_argument(
        "--run-id",
        help="Evidence directory name. Default cpu-linux-tiny. Derives evidence and scratch paths.",
    )
    parser.add_argument(
        "--evidence-dir",
        help="Override evidence path for this invocation only. Do not reuse a path from another run-id.",
    )
    parser.add_argument(
        "--scratch-dir",
        help="Override scratch path for this invocation only.",
    )
    parser.add_argument(
        "--pinned-model",
        action="store_true",
        help="Smoke Qwen/Qwen2.5-1.5B-Instruct instead of the tiny plumbing model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_run(args)
    root = repo_root()
    evidence = ensure_dir(evidence_dir(root, run_id_from_env()))
    log_path = evidence / "transcript.txt"
    commands = {
        "install": cmd_install,
        "doctor": cmd_doctor,
        "test": cmd_test,
        "unsupported": cmd_unsupported,
        "smoke": cmd_smoke,
        "quant": cmd_quant,
        "cleanup": cmd_cleanup,
        "all": cmd_all,
    }
    with log_path.open("a", encoding="utf-8") as log:
        if RUN.evidence_warning:
            append_log(log, RUN.evidence_warning + "\n")
        append_log(
            log,
            f"\n# verify {args.command} run_id={run_id_from_env()} cwd={root} "
            f"evidence={evidence_dir(root, run_id_from_env())} "
            f"scratch={scratch_dir(run_id_from_env())}\n",
        )
        try:
            return commands[args.command](root, log)
        except Exception:
            append_log(log, traceback.format_exc())
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
