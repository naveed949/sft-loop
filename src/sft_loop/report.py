"""Schema-locked run reports.

``weight_change_claimed`` is true only inside ``build_scored_report``, and only
after the train-proof files hash-check. Every other builder leaves the claim
false and omits before/after scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sft_loop import stack as stack_mod
from sft_loop.task import EVAL_WORDS, METRIC_NAME, TASK_ID

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
UNSLOTH_VERSION = "2026.9.12"
TRL_VERSION = "0.24.0"
SCHEMA_VERSION = "1"
REFUSAL_SENTENCE = "Soft-PASS is unused and is refused."
TRAINER_NAME = "trl.SFTTrainer"
ADAPTER_CONFIG_NAME = "adapter_config.json"
TRAIN_LOG_NAME = "train_log.txt"

_REASONS = ("gpu_or_train_deps_unavailable", "train_not_run")

_DETAILS = {
    "gpu_or_train_deps_unavailable": (
        "GPU or train dependencies are unavailable. "
        "No train step ran. No before/after metrics were produced. "
        "Status is unsupported. "
        + REFUSAL_SENTENCE
    ),
    "train_not_run": (
        "The train step did not run. "
        "No before/after metrics were produced. "
        "Status is unsupported. "
        + REFUSAL_SENTENCE
    ),
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "schema" / "run_report.schema.json"
        if candidate.is_file():
            return parent
    raise FileNotFoundError("schema/run_report.schema.json")


def schema_path() -> Path:
    return repo_root() / "schema" / "run_report.schema.json"


def default_proof_dir() -> Path:
    return repo_root() / "runs" / "phase3" / "proof"


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def stack_block_reason(*args, **kwargs):
    return stack_mod.stack_block_reason(*args, **kwargs)


def _require_detail(text: str) -> str:
    if REFUSAL_SENTENCE not in text:
        raise ValueError("unsupported detail must include the refusal sentence")
    if not 1 <= len(text) <= 500:
        raise ValueError("unsupported detail length")
    return text


def unsupported_report(
    *,
    run_id: str = "scaffold-unsupported",
    reason: str | None = None,
    cuda_available=None,
    detail: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if reason is None:
        blocked = stack_mod.stack_block_reason(cuda_available)
        chosen = "train_not_run" if blocked is None else blocked
    else:
        chosen = reason
    if chosen not in _REASONS:
        raise ValueError(f"unknown unsupported reason: {chosen}")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "stack": {
            "unsloth": UNSLOTH_VERSION,
            "trl": TRL_VERSION,
            "method": "sft-lora",
        },
        "status": "unsupported",
        "metric_role": "task_score_not_safety",
        "weight_change_claimed": False,
        "unsupported": {
            "reason": chosen,
            "detail": _require_detail(detail if detail is not None else _DETAILS[chosen]),
        },
    }
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be an integer >= 0")
        report["seed"] = seed
    return report


def detail_for(reason: str, prefix: str) -> str:
    if reason not in _DETAILS:
        raise ValueError(f"unknown unsupported reason: {reason}")
    return _require_detail(prefix + _DETAILS[reason])


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def format_loss(value: float) -> str:
    return format(float(value), ".8g")


def render_train_log(
    *,
    seed: int,
    global_step: int,
    train_loss: float,
    max_steps: int,
    lora_r: int,
    log_history: list[dict[str, Any]],
) -> str:
    history = json.dumps(log_history, separators=(",", ":"), sort_keys=True)
    return (
        f"trainer={TRAINER_NAME}\n"
        f"model_id={MODEL_ID}\n"
        f"seed={seed}\n"
        f"global_step={global_step}\n"
        f"train_loss={format_loss(train_loss)}\n"
        f"max_steps={max_steps}\n"
        f"lora_r={lora_r}\n"
        f"log_history={history}\n"
    )


def _parse_log(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def _history_has_loss(raw: str) -> list[dict[str, Any]]:
    try:
        history = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("train log history is not JSON") from exc
    if not isinstance(history, list) or not history:
        raise ValueError("train log history is empty")
    found = False
    for item in history:
        if isinstance(item, dict) and isinstance(item.get("loss"), (int, float)):
            if isinstance(item["loss"], bool):
                continue
            found = True
    if not found:
        raise ValueError("train log history has no loss")
    return history


def _verify_train_proof(
    report: dict[str, Any],
    proof_dir: Path,
) -> None:
    proof = report["train_proof"]
    if proof["global_step"] > proof["max_steps"]:
        raise ValueError("global_step exceeds max_steps")
    config_path = proof_dir / proof["adapter_config_path"]
    log_path = proof_dir / proof["train_log_path"]
    if not config_path.is_file() or not log_path.is_file():
        raise ValueError("train proof files are missing")
    config_bytes = config_path.read_bytes()
    log_bytes = log_path.read_bytes()
    if sha256_bytes(config_bytes) != proof["adapter_config_sha256"]:
        raise ValueError("adapter config hash mismatch")
    if sha256_bytes(log_bytes) != proof["train_log_sha256"]:
        raise ValueError("train log hash mismatch")
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("adapter config is not JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("adapter config is not an object")
    if config.get("peft_type") != "LORA":
        raise ValueError("adapter config is not LoRA")
    if config.get("task_type") != "CAUSAL_LM":
        raise ValueError("adapter config task_type is not CAUSAL_LM")
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("adapter config base model is not the pinned model")
    if config.get("r") != proof["lora_r"]:
        raise ValueError("adapter rank does not match train proof")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise ValueError("adapter config has no target modules")
    fields = _parse_log(log_bytes.decode("utf-8"))
    required = (
        "trainer",
        "model_id",
        "seed",
        "global_step",
        "train_loss",
        "max_steps",
        "lora_r",
        "log_history",
    )
    if any(key not in fields for key in required):
        raise ValueError("train log is missing fields")
    if fields["trainer"] != TRAINER_NAME:
        raise ValueError("train log trainer is not TRL SFTTrainer")
    if fields["model_id"] != MODEL_ID:
        raise ValueError("train log model is not the pinned model")
    if int(fields["seed"]) != report["seed"]:
        raise ValueError("train log seed does not match the report seed")
    if int(fields["global_step"]) != proof["global_step"]:
        raise ValueError("train log global_step does not match train proof")
    if int(fields["max_steps"]) != proof["max_steps"]:
        raise ValueError("train log max_steps does not match train proof")
    if int(fields["lora_r"]) != proof["lora_r"]:
        raise ValueError("train log lora_r does not match train proof")
    if not math.isclose(float(fields["train_loss"]), float(proof["train_loss"]), rel_tol=0, abs_tol=1e-9):
        raise ValueError("train log loss does not match train proof")
    _history_has_loss(fields["log_history"])


def _check_same_seed(report: dict[str, Any]) -> None:
    if report.get("status") != "scored":
        return
    seed = report["seed"]
    before = report["before"]
    after = report["after"]
    if before["seed"] != seed or after["seed"] != seed:
        raise ValueError("before.seed and after.seed must equal seed")
    if before["metric_name"] != METRIC_NAME or after["metric_name"] != METRIC_NAME:
        raise ValueError("metric_name must be the synthetic task metric")
    if before["n"] != len(EVAL_WORDS) or after["n"] != len(EVAL_WORDS):
        raise ValueError("eval n must match the held-out set")
    if report.get("task_id") != TASK_ID:
        raise ValueError("task_id must be the synthetic task")
    for slice_ in (before, after):
        score = slice_["task_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("task_score must be a number")
        if not 0 <= float(score) <= 1:
            raise ValueError("task_score out of range")


def validate_report(
    report: dict[str, Any],
    *,
    proof_dir: Path | None = None,
) -> None:
    from jsonschema import Draft202012Validator

    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda err: [str(part) for part in err.path],
    )
    if errors:
        raise ValueError("; ".join(err.message for err in errors))
    _check_same_seed(report)
    if report.get("weight_change_claimed") is True:
        _verify_train_proof(report, proof_dir if proof_dir is not None else default_proof_dir())


def build_scored_report(
    *,
    run_id: str,
    seed: int,
    before: dict[str, Any],
    after: dict[str, Any],
    train_proof: dict[str, Any],
    proof_dir: Path,
) -> dict[str, Any]:
    """Assemble a scored report. The weight-change claim is checked against proof files."""

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "stack": {
            "unsloth": UNSLOTH_VERSION,
            "trl": TRL_VERSION,
            "method": "sft-lora",
        },
        "seed": seed,
        "status": "scored",
        "metric_role": "task_score_not_safety",
        "weight_change_claimed": True,
        "task_id": TASK_ID,
        "before": before,
        "after": after,
        "train_proof": train_proof,
    }
    validate_report(report, proof_dir=proof_dir)
    return report


def dump_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2) + "\n"


def emit_json(payload: dict[str, Any], out: Path | None) -> None:
    import sys

    text = dump_report(payload)
    if out is None:
        sys.stdout.write(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def require_model_id(model_id: str) -> None:
    if model_id != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID}")


def require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be an integer >= 0")
