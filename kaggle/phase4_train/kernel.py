#!/usr/bin/env python3
"""Self-contained Kaggle kernel for Phase 4 fp16 LoRA."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TASK_ID = "tenk-extract-v1"
SEED = 0
CEILING = 0.85
MIN_EVAL = 100
MAX_STEPS = 200
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
PACKAGES = [
    "transformers==4.56.1",
    "accelerate==1.10.1",
    "datasets==3.6.0",
    "peft==0.17.1",
    "trl==0.24.0",
]
WORK = Path(os.environ.get("PHASE4_WORK_DIR", "/kaggle/working"))
DEFAULT_INPUT_ROOT = Path("/kaggle/input")
EXPECTED_SPLIT_HASH = "3a298f9e0d83b00c9d3cfac99f90a160ebef3ce28c94d926b3f5efa266c95e33"
EXPECTED_EVAL_SHA256 = "09a6e33011761caf02a5eb3f732d172f778c234ed29d434af1298360d3e6a459"
EXPECTED_TRAIN_SHA256 = "c73756013ef53e615cc774e8bf5e354033346e0812a325d5c72ea1f262f06abc"
EXPECTED_CAPITALCITE_COMMIT = "34b42ee0b387738725705a0af682dd3f030a9944"
FP16_MAX = 65504
PROBE_GEN_ITEMS = 4
HOOK_NEEDLES = (
    "mlp",
    "self_attn",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
    "lm_head",
)


def write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class DatasetNotFound(Exception):
    reason = "dataset_not_found"


class DatasetHashMismatch(Exception):
    reason = "dataset_hash_mismatch"


class BaselineInvalid(Exception):
    reason = "baseline_invalid"


class AmpOverflow(Exception):
    reason = "fp16_amp_overflow"


def refuse_phase3(path: Path) -> Path:
    text = Path(path).as_posix().replace("\\", "/").lower()
    if "/runs/phase3" in text or "/adapters/phase3" in text:
        raise SystemExit(f"phase 4 refuses path under phase3: {path}")
    return Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_is_expected(directory: Path) -> bool:
    manifest_path = directory / "manifest.json"
    eval_path = directory / "eval.jsonl"
    train_path = directory / "train.jsonl"
    if not (manifest_path.is_file() and eval_path.is_file() and train_path.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("task_id") != TASK_ID:
        return False
    if manifest.get("capitalcite_commit") != EXPECTED_CAPITALCITE_COMMIT:
        return False
    if manifest.get("split_hash") != EXPECTED_SPLIT_HASH:
        return False
    if manifest.get("eval_sha256") != EXPECTED_EVAL_SHA256:
        return False
    if manifest.get("train_sha256") != EXPECTED_TRAIN_SHA256:
        return False
    return sha256_file(eval_path) == EXPECTED_EVAL_SHA256 and sha256_file(train_path) == EXPECTED_TRAIN_SHA256


def search_roots(search_root: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    override = os.environ.get("PHASE4_DATA_DIR")
    if override:
        roots.append(Path(override))
    roots.append(Path(search_root) if search_root is not None else DEFAULT_INPUT_ROOT)
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def find_data(search_root: Path | None = None) -> Path:
    seen: set[Path] = set()
    mismatches: list[str] = []
    searched: list[str] = []
    for base in search_roots(search_root):
        searched.append(base.as_posix())
        if not base.exists():
            continue
        if base.is_file():
            continue
        candidates = [base]
        candidates.extend(path.parent for path in sorted(base.rglob("manifest.json")))
        for directory in candidates:
            resolved = directory.resolve() if directory.exists() else directory
            if resolved in seen:
                continue
            seen.add(resolved)
            if not (directory / "manifest.json").is_file():
                continue
            if manifest_is_expected(directory):
                return directory
            if (directory / "eval.jsonl").is_file():
                mismatches.append(directory.as_posix())
    if mismatches:
        raise DatasetHashMismatch(
            "phase4 dataset hash mismatch; "
            f"expected split_hash={EXPECTED_SPLIT_HASH} "
            f"among {mismatches}"
        )
    raise DatasetNotFound(
        "phase4 dataset (manifest.json) not found under " + ", ".join(searched)
    )


def load_grade(data: Path):
    path = data / "grade.py"
    spec = importlib.util.spec_from_file_location("phase4_grade_kernel", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load grader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pip_install() -> dict:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *PACKAGES]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def hardware(torch) -> dict:
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory": int(props.total_memory),
                "major": int(props.major),
                "minor": int(props.minor),
            }
        )
    bf16 = "unsupported"
    bf16_reason = "Tesla T4 compute capability 7.5 has no bf16; fp16 only"
    if gpus and all(gpu["major"] >= 8 for gpu in gpus):
        bf16 = "available"
        bf16_reason = "compute capability >= 8.0"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
        "bf16": bf16,
        "bf16_reason": bf16_reason,
        "seed": SEED,
        "task_id": TASK_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_steps": MAX_STEPS,
        "per_device_batch": PER_DEVICE_BATCH,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lr": LEARNING_RATE,
        "lr_schedule": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_scale": LORA_ALPHA / LORA_R,
        "target_modules": TARGET_MODULES,
    }


def require_scaler(torch):
    if getattr(torch.version, "cuda", None) is None:
        raise RuntimeError("cpu torch wheel")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda unavailable")
    amp = getattr(torch, "amp", None)
    scaler_cls = getattr(amp, "GradScaler", None) if amp is not None else None
    autocast = getattr(amp, "autocast", None) if amp is not None else None
    if scaler_cls is None:
        scaler_cls = getattr(getattr(torch.cuda, "amp", None), "GradScaler", None)
    if autocast is None:
        autocast = getattr(getattr(torch.cuda, "amp", None), "autocast", None)
    if scaler_cls is None or autocast is None:
        raise RuntimeError("GradScaler missing")
    return scaler_cls, autocast


def format_text(tokenizer, prompt: str, completion: str | None = None) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        if completion is not None:
            messages.append({"role": "assistant", "content": completion})
            return tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if completion is None:
        return prompt
    return prompt + completion


def tokenize_row(tokenizer, prompt: str, completion: str, device):
    import torch

    text = format_text(tokenizer, prompt, completion)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    prompt_ids = tokenizer(
        format_text(tokenizer, prompt, None),
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=True,
    )["input_ids"]
    labels = encoded["input_ids"].clone()
    prompt_len = min(len(prompt_ids), labels.size(1))
    labels[:, :prompt_len] = -100
    labels[encoded["attention_mask"] == 0] = -100
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def cosine_lr(step: int, total: int) -> float:
    import math

    warmup = max(1, int(total * WARMUP_RATIO))
    if step < warmup:
        return LEARNING_RATE * float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def generations_sane(predictions: list[str]) -> tuple[bool, str]:
    if not predictions:
        return False, "no generations"
    stripped = [str(prediction).strip() for prediction in predictions]
    if any(not text for text in stripped):
        return False, "empty generation"
    if all(text and set(text) <= set("!?.,#*;: ") for text in stripped):
        return False, "generations are punctuation-only"
    if len(stripped) >= 3 and len(set(stripped)) == 1:
        return False, "all generations identical"
    return True, "ok"


def _hook_name(name: str) -> bool:
    return any(needle in name for needle in HOOK_NEEDLES)


def hidden_maxabs_probe(model, batch, torch, autocast, use_amp: bool) -> dict:
    stats: dict[str, dict] = {}
    overflow: list[str] = []

    def hook(name: str):
        def _fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor) or tensor.numel() == 0:
                return
            finite = bool(torch.isfinite(tensor).all().item())
            max_abs = float(tensor.detach().float().abs().max().item())
            stats[name] = {"max_abs": max_abs, "finite": finite, "dtype": str(tensor.dtype)}
            if (not finite) or max_abs > FP16_MAX:
                overflow.append(name)

        return _fn

    handles = []
    for name, module in model.named_modules():
        if _hook_name(name):
            handles.append(module.register_forward_hook(hook(name)))
    result = {
        "loss": None,
        "loss_finite": False,
        "modules": stats,
        "overflow_modules": overflow,
        "fp16_max": FP16_MAX,
        "amp": use_amp,
    }
    try:
        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss
        finite = bool(torch.isfinite(loss).item())
        result["loss_finite"] = finite
        result["loss"] = float(loss.detach().float().cpu()) if finite else None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not overflow:
            overflow.append("<forward_exception>")
    finally:
        for handle in handles:
            handle.remove()
    return result


def finite_nll(model, tokenizer, row, torch, device) -> dict:
    prompt = row.get("prompt") or row["question"]
    batch = tokenize_row(tokenizer, prompt, row["completion"], device)
    model.eval()
    with torch.no_grad():
        loss = model(**batch).loss
    finite = bool(torch.isfinite(loss).item())
    return {
        "nll": float(loss.detach().float().cpu()) if finite else None,
        "finite": finite,
    }


def worst_overflow(probe: dict) -> dict | None:
    modules = probe.get("modules") or {}
    overflow = list(probe.get("overflow_modules") or [])
    ranked = []
    for name in overflow:
        stat = modules.get(name) or {}
        ranked.append(
            {
                "name": name,
                "max_abs": stat.get("max_abs"),
                "finite": stat.get("finite"),
                "dtype": stat.get("dtype"),
            }
        )
    ranked.sort(key=lambda row: float(row["max_abs"] or 0.0), reverse=True)
    if ranked:
        return ranked[0]
    if probe.get("error"):
        return {"name": "<forward_exception>", "error": probe["error"]}
    return None


def evaluate(model, tokenizer, rows, grade, device) -> dict:
    import torch

    model.eval()
    items = []
    predictions = []
    with torch.no_grad():
        for row in rows:
            prompt = row.get("prompt") or grade.render_prompt(row["chunk"], row["question"])
            text = format_text(tokenizer, prompt, None)
            encoded = tokenizer(text, return_tensors="pt").to(device)
            output = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = output[0, encoded["input_ids"].shape[1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            hit = int(grade.grade_one(prediction, row["completion"]))
            predictions.append(prediction)
            items.append(
                {
                    "item_id": row["item_id"],
                    "doc_id": row["doc_id"],
                    "question": row["question"],
                    "completion": row["completion"],
                    "prediction": prediction,
                    "hit": hit,
                }
            )
    score = grade.grade(predictions, [row["completion"] for row in rows])
    return {
        "task_id": TASK_ID,
        "metric_name": "exact_match",
        "task_score": score,
        "n": len(rows),
        "seed": SEED,
        "items": items,
    }


def load_base(torch, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
        base_model_name_or_path=MODEL_ID,
    )
    model = get_peft_model(model, lora)
    model.to(device)
    dtypes = {str(param.dtype) for param in model.parameters()}
    if "torch.float16" in dtypes or "torch.bfloat16" in dtypes:
        raise RuntimeError(f"master weights must be fp32, found {sorted(dtypes)}")
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, device) -> str:
    import torch

    model.eval()
    text = format_text(tokenizer, prompt, None)
    encoded = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def probe_amp(model, tokenizer, rows, torch, autocast, device) -> dict:
    row = rows[0]
    prompt = row.get("prompt") or row["question"]
    batch = tokenize_row(tokenizer, prompt, row["completion"], device)
    model.train()
    amp = hidden_maxabs_probe(model, batch, torch, autocast, True)
    fp32 = hidden_maxabs_probe(model, batch, torch, autocast, False)
    sample_rows = rows[:PROBE_GEN_ITEMS]
    amp_preds = []
    try:
        with autocast("cuda", dtype=torch.float16):
            for item in sample_rows:
                amp_preds.append(
                    generate_one(model, tokenizer, item.get("prompt") or item["question"], device)
                )
    except Exception as exc:
        amp["generate_error"] = str(exc)
        amp_preds = []
    sane, reason = generations_sane(amp_preds)
    amp["predictions"] = amp_preds
    amp["generations_sane"] = sane
    amp["generations_reason"] = reason
    amp["ok"] = bool(amp["loss_finite"] and not amp["overflow_modules"] and sane)
    fp32["ok"] = bool(fp32["loss_finite"] and not fp32["overflow_modules"])
    return {"amp": amp, "fp32": fp32}


def train(model, tokenizer, rows, torch, scaler_cls, autocast, device, *, use_amp: bool) -> dict:
    import torch as torch_mod

    opt = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=LEARNING_RATE)
    scaler = None
    if use_amp:
        try:
            scaler = scaler_cls("cuda")
        except TypeError:
            scaler = scaler_cls()
    history = []
    global_step = 0
    cursor = 0
    model.train()
    while global_step < MAX_STEPS:
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _micro in range(GRAD_ACCUM):
            row = rows[cursor % len(rows)]
            cursor += 1
            prompt = row.get("prompt") or row["question"]
            batch = tokenize_row(tokenizer, prompt, row["completion"], device)
            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                loss = model(**batch).loss / GRAD_ACCUM
                loss.backward()
            step_loss += float(loss.detach().float().cpu())
        for group in opt.param_groups:
            group["lr"] = cosine_lr(global_step, MAX_STEPS)
        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        global_step += 1
        history.append({"step": global_step, "loss": step_loss, "lr": opt.param_groups[0]["lr"]})
        if not torch_mod.isfinite(torch_mod.tensor(step_loss)):
            raise RuntimeError(f"non-finite loss at step {global_step}")
    return {
        "global_step": global_step,
        "history": history,
        "train_loss": history[-1]["loss"] if history else None,
        "mixed_precision": "fp16_autocast" if use_amp else "fp32",
        "grad_scaler": bool(use_amp),
    }


def profiler_summary(model, tokenizer, rows, torch, autocast, device, *, use_amp: bool) -> dict:
    from torch.profiler import ProfilerActivity, profile

    batch = tokenize_row(tokenizer, rows[0].get("prompt") or rows[0]["question"], rows[0]["completion"], device)
    model.train()
    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "CUDA"):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        for _ in range(3):
            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss
            else:
                loss = model(**batch).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
    table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20)
    rows_out = []
    for item in prof.key_averages():
        rows_out.append(
            {
                "name": item.key,
                "cpu_time_total": float(item.cpu_time_total),
                "self_cpu_memory": int(getattr(item, "self_cpu_memory_usage", 0) or 0),
                "self_cuda_memory": int(getattr(item, "self_cuda_memory_usage", 0) or 0),
            }
        )
    rows_out.sort(key=lambda row: row["self_cuda_memory"], reverse=True)
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return {"top_ops": rows_out[:20], "table": table, "peak_memory_allocated": peak}


def checkpoint_compare(model, tokenizer, rows, torch, autocast, device, *, use_amp: bool) -> dict:
    batch = tokenize_row(tokenizer, rows[0].get("prompt") or rows[0]["question"], rows[0]["completion"], device)

    def peak_with(enabled: bool) -> int:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if enabled:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.train()
        model.zero_grad(set_to_none=True)
        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        return int(torch.cuda.max_memory_allocated())

    off = peak_with(False)
    on = peak_with(True)
    model.gradient_checkpointing_enable()
    return {
        "batch_size": PER_DEVICE_BATCH,
        "seq_length": MAX_SEQ_LENGTH,
        "checkpointing_off_peak": off,
        "checkpointing_on_peak": on,
        "delta_bytes": on - off,
    }


def try_ddp(torch) -> dict:
    if torch.cuda.device_count() < 2:
        return {"status": "unsupported", "reason": "fewer than 2 visible CUDA devices"}
    try:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        if dist.is_available() and not dist.is_initialized():
            return {
                "status": "unsupported",
                "reason": "optional DDP not launched via torchrun in this kernel",
            }
        return {"status": "unsupported", "reason": "DDP process group was not initialized"}
    except Exception as exc:
        return {"status": "unsupported", "reason": str(exc), "traceback": traceback.format_exc()}


def write_unsupported(reason: str, detail: str, *, exit_code: int = 0) -> int:
    write_json(
        refuse_phase3(WORK / "unsupported.json"),
        {
            "schema_version": "phase4-1",
            "document_kind": "phase4-tenk-lora",
            "status": "unsupported",
            "unsupported": {"reason": reason, "detail": detail},
            "adapter_written": False,
        },
    )
    return exit_code


def main() -> int:
    refuse_phase3(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    install = pip_install()
    write_json(WORK / "pip_install.json", install)
    if install["returncode"] != 0:
        return write_unsupported("gpu_or_train_deps_unavailable", "pip install of pinned deps failed")
    try:
        import torch
    except ImportError as exc:
        return write_unsupported("gpu_or_train_deps_unavailable", f"torch missing: {exc}")
    env = hardware(torch)
    write_json(WORK / "env.json", env)
    write_json(WORK / "bf16.json", {"status": env["bf16"], "reason": env["bf16_reason"]})
    write_json(WORK / "ddp.json", try_ddp(torch))
    try:
        scaler_cls, autocast = require_scaler(torch)
    except Exception as exc:
        return write_unsupported("gpu_or_train_deps_unavailable", str(exc))
    try:
        data = find_data()
    except (DatasetNotFound, DatasetHashMismatch) as exc:
        return write_unsupported(exc.reason, str(exc), exit_code=1)
    grade = load_grade(data)
    train_rows = load_jsonl(data / "train.jsonl")
    eval_rows = load_jsonl(data / "eval.jsonl")
    if len(eval_rows) < MIN_EVAL:
        return write_unsupported("train_not_run", f"eval n {len(eval_rows)} < {MIN_EVAL}")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    write_json(WORK / "split_manifest.json", manifest)
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, tokenizer = load_base(torch, device)
    probe = probe_amp(model, tokenizer, train_rows, torch, autocast, device)
    write_json(WORK / "fp16_probe.json", probe)
    use_amp = bool(probe["amp"]["ok"])
    if not use_amp:
        worst = worst_overflow(probe["amp"])
        write_json(
            WORK / "fp16.json",
            {
                "status": "unsupported",
                "reason": "fp16_amp_overflow",
                "detail": (
                    "fp16 max is 65504. Run 2 loaded the 1.5B weights as float16 "
                    "and GradScaler then saw a non-finite loss at step 1. "
                    "eval_before was 0/123 with every decode '!!!!!!!!!!!!!!!!'. "
                    "Masters are now fp32. This probe still found a non-finite "
                    "fp16 autocast forward or garbage generations, so mixed "
                    "precision is unsupported on this T4/Qwen pair."
                ),
                "overflow_modules": probe["amp"].get("overflow_modules", []),
                "worst_overflow": worst,
                "amp_loss_finite": probe["amp"].get("loss_finite"),
                "generations_reason": probe["amp"].get("generations_reason"),
                "fp16_max": FP16_MAX,
            },
        )
        if not probe["fp32"]["ok"]:
            return write_unsupported(
                "baseline_invalid",
                f"fp32 probe also non-finite: {probe['fp32']}",
                exit_code=1,
            )
    env["mixed_precision"] = "fp16_autocast" if use_amp else "fp32"
    env["weight_dtype"] = "float32"
    env["grad_scaler"] = bool(use_amp)
    write_json(WORK / "env.json", env)
    before = evaluate(model, tokenizer, eval_rows, grade, device)
    nll = finite_nll(model, tokenizer, eval_rows[0], torch, device)
    before["baseline_nll"] = nll
    write_json(WORK / "eval_before.json", before)
    sane, reason = generations_sane([item["prediction"] for item in before["items"]])
    if not sane:
        return write_unsupported(
            "baseline_invalid",
            f"eval_before generations are not a usable baseline: {reason}",
            exit_code=1,
        )
    if not nll["finite"]:
        return write_unsupported(
            "baseline_invalid",
            "eval_before NLL is not finite",
            exit_code=1,
        )
    if before["task_score"] >= CEILING:
        write_json(
            WORK / "refused_ceiling.json",
            {
                "schema_version": "phase4-1",
                "document_kind": "phase4-tenk-lora",
                "status": "refused_ceiling",
                "before": {"metric_name": "exact_match", "task_score": before["task_score"], "n": before["n"]},
                "adapter_written": False,
            },
        )
        return 0
    train_result = train(
        model, tokenizer, train_rows, torch, scaler_cls, autocast, device, use_amp=use_amp
    )
    adapter_dir = WORK / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    weight_files = sorted(adapter_dir.glob("*.safetensors")) + sorted(adapter_dir.glob("adapter_model.bin"))
    hashes = {path.name: sha256_file(path) for path in weight_files}
    (WORK / "adapter.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in hashes.items()) + ("\n" if hashes else ""),
        encoding="utf-8",
    )
    log_lines = [
        f"trainer=phase4.manual_fp16_lora",
        f"seed={SEED}",
        f"global_step={train_result['global_step']}",
        f"train_loss={train_result['train_loss']}",
        f"max_steps={MAX_STEPS}",
        f"lora_r={LORA_R}",
        f"device=cuda",
        f"backend=trl-peft",
        f"autocast={'fp16' if train_result['mixed_precision'] == 'fp16_autocast' else 'off'}",
        f"grad_scaler={int(train_result['grad_scaler'])}",
        f"weight_dtype=float32",
        f"mixed_precision={train_result['mixed_precision']}",
    ]
    (WORK / "trainer_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    write_json(WORK / "trainer_history.json", train_result)
    after = evaluate(model, tokenizer, eval_rows, grade, device)
    write_json(WORK / "eval_after.json", after)
    try:
        write_json(
            WORK / "profiler_summary.json",
            profiler_summary(model, tokenizer, train_rows, torch, autocast, device, use_amp=use_amp),
        )
    except Exception as exc:
        write_json(WORK / "profiler_summary.json", {"status": "unsupported", "reason": str(exc)})
    try:
        write_json(
            WORK / "memory_checkpoint.json",
            checkpoint_compare(model, tokenizer, train_rows, torch, autocast, device, use_amp=use_amp),
        )
    except Exception as exc:
        write_json(WORK / "memory_checkpoint.json", {"status": "unsupported", "reason": str(exc)})
    write_json(
        WORK / "run_summary.json",
        {
            "schema_version": "phase4-1",
            "document_kind": "phase4-tenk-lora",
            "status": "raw",
            "global_step": train_result["global_step"],
            "before_score": before["task_score"],
            "after_score": after["task_score"],
            "n": after["n"],
            "adapter_sha256": hashes,
            "adapter_written": True,
            "mixed_precision": train_result["mixed_precision"],
            "weight_dtype": "float32",
        },
    )
    return 0


def _work_dir() -> Path:
    return Path(os.environ.get("PHASE4_WORK_DIR", "/kaggle/working"))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DatasetNotFound, DatasetHashMismatch, BaselineInvalid, AmpOverflow) as exc:
        write_unsupported(exc.reason, str(exc), exit_code=1)
        raise SystemExit(1)
    except Exception:
        work = _work_dir()
        work.mkdir(parents=True, exist_ok=True)
        (work / "kernel_crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if not (work / "unsupported.json").is_file():
            write_json(
                work / "unsupported.json",
                {
                    "schema_version": "phase4-1",
                    "document_kind": "phase4-tenk-lora",
                    "status": "unsupported",
                    "unsupported": {"reason": "kernel_crash", "detail": traceback.format_exc()},
                    "adapter_written": False,
                },
            )
        raise
