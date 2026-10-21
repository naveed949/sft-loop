#!/usr/bin/env python3
"""Kaggle T4 kernel. fp32-on-T4 baseline, then fp16 and NF4.

Accelerator: NvidiaTeslaT4. Expected wall about 2 hours. Budget 2.5 hours.
Must finish in one 12 hour session and stay under 3 GPU hours.
No secrets. Downloads Qwen/Qwen2.5-1.5B-Instruct from the public Hub.
Pins transformers 5.20.2 to match the CPU fp32 arm.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path.cwd()
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ITEM_IDS = (
    "anchor",
    "barrel",
    "beacon",
    "biscuit",
    "blanket",
    "blossom",
    "border",
    "bottle",
    "breeze",
    "bronze",
    "bucket",
    "button",
    "cactus",
    "camera",
    "candle",
    "canyon",
    "carpet",
    "castle",
    "celery",
    "chapel",
    "cherry",
    "circle",
    "citrus",
    "cloak",
    "cobalt",
    "copper",
    "cradle",
    "cricket",
    "crystal",
    "curtain",
    "custom",
    "daisy",
    "desert",
    "donkey",
    "dragon",
    "drawer",
    "eagle",
    "ember",
    "engine",
    "fable",
    "feather",
    "fence",
    "field",
    "forest",
    "fountain",
    "frost",
    "galaxy",
    "garlic",
    "glacier",
    "goblet",
    "grape",
    "gravel",
    "guitar",
    "hammer",
    "hazel",
    "helmet",
    "honey",
    "horse",
    "ivory",
    "jacket",
    "jungle",
    "kettle",
    "knight",
    "ladder",
    "lantern",
    "lavender",
    "legend",
    "lizard",
    "lumber",
    "marble",
    "meadow",
    "melon",
    "mirror",
    "mossy",
    "needle",
    "nickel",
    "olive",
    "orange",
    "orchid",
    "otter",
    "packet",
    "palace",
    "pebble",
    "pepper",
    "pickle",
    "pillow",
    "planet",
    "plum",
    "pocket",
    "pollen",
    "prairie",
    "pumpkin",
    "quail",
    "quartz",
    "rabbit",
    "raven",
    "ribbon",
    "rocket",
    "saddle",
    "saffron",
    "sailor",
    "salmon",
    "scarlet",
    "shadow",
    "silver",
    "sparrow",
    "spice",
    "spider",
    "spruce",
    "squirrel",
    "stable",
    "storm",
    "saddlebag",
    "tablet",
    "temple",
    "thistle",
    "thunder",
    "timber",
    "turtle",
    "valley",
    "velvet",
    "violet",
    "walnut",
    "willow",
    "window",
    "winter",
    "wizard",
    "yellow",
    "zebra",
)
DECODE_NEW_TOKENS = 8
WARMUP = 2
REPEATS = 5
THREADS = 1
IGNORE_INDEX = -100
PROBE_ITEMS = ("anchor", "beacon")
NF4_QUANT_CONFIG_F32 = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float32",
}
NF4_QUANT_CONFIG_F16 = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
}


def log(message: str) -> None:
    line = message + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with (OUT / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_progress(payload: dict) -> None:
    write_json(OUT / "progress.json", payload)


def pip_install() -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "bitsandbytes==0.50.2",
        "transformers==5.20.2",
        "tokenizers==0.23.3",
        "accelerate==1.4.0",
    ]
    log("pip " + " ".join(cmd[4:]))
    subprocess.check_call(cmd)


def dump_env() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    env = {
        "kernel_url": None,
        "kernel_version": None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": {
            "cpu_model": platform.processor() or platform.machine() or "unknown",
            "logical_cpus": os.cpu_count() or 1,
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "cuda": True,
            "mps": False,
            "accelerator": "NvidiaTeslaT4",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": int(torch.cuda.device_count()),
            "gpu_memory_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
            "driver": getattr(torch.version, "cuda", "") or "unknown",
        },
    }
    try:
        nvidia = subprocess.check_output(["nvidia-smi"], text=True)
        (OUT / "nvidia-smi.txt").write_text(nvidia, encoding="utf-8")
        for line in nvidia.splitlines():
            if "Driver Version" in line:
                env["hardware"]["driver"] = line.split("Driver Version:")[1].split()[0]
                break
    except Exception as exc:
        log(f"nvidia-smi failed: {exc}")
    write_json(OUT / "env.json", env)
    return env


def prompt_for(word: str) -> str:
    return f"Uppercase the word: {word}"


def prediction_token(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split()[0]


def _round(value: float) -> float:
    return float(format(float(value), ".8g"))


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_prompt(torch, tokenizer, model, prompt: str):
    text = render_prompt(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt")
    return {key: value.to(model.device) for key, value in encoded.items()}


def generate_fn(torch, tokenizer, model, *, min_new_tokens: int | None = None):
    def generate(prompt: str):
        encoded = encode_prompt(torch, tokenizer, model, prompt)
        kwargs = {
            "max_new_tokens": DECODE_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if min_new_tokens is not None:
            kwargs["min_new_tokens"] = min_new_tokens
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**encoded, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded, int(new_tokens.shape[0]), elapsed

    return generate


def nll_fn(torch, tokenizer, model):
    def nll(prompt: str, target: str):
        prefix = render_prompt(tokenizer, prompt)
        prefix_len = tokenizer(prefix, return_tensors="pt")["input_ids"].shape[1]
        encoded = tokenizer(prefix + target, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        target_labels = labels.clone()
        target_labels[:, :prefix_len] = IGNORE_INDEX
        with torch.no_grad():
            logits = model(input_ids=encoded["input_ids"]).logits
        shift_logits = logits[:, :-1, :].contiguous()
        F = torch.nn.functional
        sequence_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=IGNORE_INDEX,
        )
        target_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_labels[:, 1:].contiguous().view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return float(target_loss.item()), float(sequence_loss.item())

    return nll


def maybe_float(value: float) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return _round(number)


def dump_prompt_hashes(tokenizer) -> None:
    rows = []
    for item_id in ITEM_IDS:
        text = render_prompt(tokenizer, prompt_for(item_id))
        rows.append(
            {
                "item_id": item_id,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "n_chars": len(text),
            }
        )
    write_json(OUT / "prompt_hashes.json", {"revision": REVISION, "items": rows})
    log(f"prompt hash anchor={rows[0]['sha256']} beacon={rows[2]['sha256']}")


def probe_numerics(torch, tokenizer) -> dict:
    from transformers import AutoModelForCausalLM

    report: dict = {"items": list(PROBE_ITEMS), "configs": {}}
    configs = (
        ("fp32_eager", torch.float32, "eager"),
        ("fp16_eager", torch.float16, "eager"),
        ("fp16_sdpa", torch.float16, "sdpa"),
    )
    for name, dtype, attn in configs:
        log(f"numerics probe {name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=REVISION,
                dtype=dtype,
                device_map={"": 0},
                attn_implementation=attn,
            )
            model.eval()
            rows = []
            for item_id in PROBE_ITEMS:
                encoded = encode_prompt(torch, tokenizer, model, prompt_for(item_id))
                with torch.no_grad():
                    out = model(**encoded, output_hidden_states=True)
                logits = out.logits.float()
                hidden = [
                    float(layer.detach().float().abs().max().item())
                    for layer in out.hidden_states
                ]
                row = {
                    "item_id": item_id,
                    "logits_finite": bool(torch.isfinite(out.logits).all().item()),
                    "logits_max_abs": maybe_float(float(logits.abs().max().item())),
                    "hidden_max_abs": hidden,
                    "hidden_finite": all(math.isfinite(value) for value in hidden),
                }
                rows.append(row)
                log(
                    f"numerics {name} {item_id} logits_finite={row['logits_finite']} "
                    f"logits_max_abs={row['logits_max_abs']}"
                )
            report["configs"][name] = {"ok": True, "items": rows}
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            report["configs"][name] = {
                "ok": False,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:200],
            }
            log(f"numerics {name} failed: {type(exc).__name__}: {exc}")
    write_json(OUT / "probe" / "numerics.json", report)
    return report


def repeat_generate(generate, prompt: str):
    latencies = []
    prediction = None
    tokens = 0
    for step in range(WARMUP + REPEATS):
        text, n_tokens, elapsed_ms = generate(prompt)
        if step < WARMUP:
            continue
        latencies.append(elapsed_ms)
        if prediction is None:
            prediction = text
            tokens = n_tokens
        elif text != prediction or n_tokens != tokens:
            raise RuntimeError("greedy prediction drifted across repeats")
    return prediction or "", int(tokens), _round(sum(latencies) / len(latencies))


def measure(precision: str, torch, tokenizer, model, *, on_disk_bytes: int, libraries: dict, extra: dict):
    import resource

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_ram = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split("kB")[0]) * 1024
    start = time.perf_counter()
    cpu0 = time.process_time()
    generate = generate_fn(torch, tokenizer, model)
    decode_generate = generate_fn(torch, tokenizer, model, min_new_tokens=DECODE_NEW_TOKENS)
    nll = nll_fn(torch, tokenizer, model)
    hashes_path = OUT / "prompt_hashes.json"
    if hashes_path.is_file():
        write_json(OUT / precision / "prompt_hashes.json", json.loads(hashes_path.read_text(encoding="utf-8")))
    write_json(
        OUT / precision / "generation_config.json",
        {
            "add_generation_prompt": True,
            "attn_implementation": "eager",
            "chat_template": "tokenizer.apply_chat_template",
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": DECODE_NEW_TOKENS,
            "min_new_tokens": None,
            "pad_token_id": tokenizer.pad_token_id,
            "stop_criteria": "eos_or_max_new_tokens",
            "temperature": None,
        },
    )
    items = []
    predictions = []
    for item_id in ITEM_IDS:
        prompt = prompt_for(item_id)
        prediction, tokens, latency_ms = repeat_generate(generate, prompt)
        _, decode_tokens, decode_latency_ms = repeat_generate(decode_generate, prompt)
        target_nll, sequence_nll = nll(prompt, item_id.upper())
        token = prediction_token(prediction)
        row = {
            "item_id": item_id,
            "correct": token == item_id.upper(),
            "latency_ms": latency_ms,
            "generated_tokens": int(tokens),
            "decode_latency_ms": decode_latency_ms,
            "decode_generated_tokens": int(decode_tokens),
        }
        predictions.append(
            {
                "item_id": item_id,
                "prediction": prediction,
                "prediction_token": token,
            }
        )
        target_out = maybe_float(target_nll)
        sequence_out = maybe_float(sequence_nll)
        if target_out is not None:
            row["nll"] = target_out
        if sequence_out is not None:
            row["sequence_nll"] = sequence_out
        items.append(row)
        log(f"{precision} {item_id} correct={row['correct']} p50={latency_ms} nll={row.get('nll')}")
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    peak_gpu = int(torch.cuda.max_memory_allocated())
    exact_match = sum(1 for row in items if row["correct"]) / max(len(items), 1)
    metrics = {
        "on_disk_bytes": int(on_disk_bytes),
        "peak_rss_bytes": max(rss, 1),
        "peak_gpu_bytes": max(peak_gpu, 1),
        "exact_match": _round(exact_match),
        "libraries": libraries,
        "limiter": {
            "kind": "gpu",
            "wall_s": _round(wall),
            "cpu_time_s": _round(cpu),
            "peak_rss_bytes": max(rss, 1),
            "available_ram_bytes_at_start": start_ram,
        },
        **extra,
    }
    write_json(OUT / precision / "items.json", items)
    write_json(OUT / precision / "metrics.json", metrics)
    write_json(OUT / precision / "predictions.json", predictions)
    return items, metrics


def baseline_is_healthy(items: list) -> bool:
    score = sum(1 for row in items if row["correct"])
    finite = all("nll" in row and "sequence_nll" in row for row in items)
    return score > 0 and finite


def load_causal(torch, *, dtype, attn: str = "eager"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=dtype,
        device_map={"": 0},
        attn_implementation=attn,
    )
    model.eval()
    return model


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def snapshot_bytes() -> int:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=["*.safetensors", "*.bin"])
    )
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def library_versions(kind: str) -> dict:
    from importlib.metadata import version

    import torch

    body = {
        "kind": kind,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
    }
    if kind == "bitsandbytes":
        body["bitsandbytes"] = version("bitsandbytes")
    return body


def run_t4_fp32(torch, tokenizer):
    write_progress({"status": "running", "stage": "t4-fp32", "arms": {}})
    model = load_causal(torch, dtype=torch.float32, attn="eager")
    items, metrics = measure(
        "t4-fp32",
        torch,
        tokenizer,
        model,
        on_disk_bytes=max(snapshot_bytes(), 1),
        libraries=library_versions("torch"),
        extra={"compute_dtype": "float32", "attn_implementation": "eager"},
    )
    del model
    torch.cuda.empty_cache()
    return items, metrics


def run_fp16(torch, tokenizer, *, finding: str | None):
    write_progress(
        {
            "status": "running",
            "stage": "fp16",
            "arms": {"t4-fp32": {"status": "measured"}},
        }
    )
    model = load_causal(torch, dtype=torch.float16, attn="eager")
    extra = {"compute_dtype": "float16", "attn_implementation": "eager"}
    if finding:
        extra["finding"] = finding
    items, metrics = measure(
        "fp16",
        torch,
        tokenizer,
        model,
        on_disk_bytes=max(snapshot_bytes(), 1),
        libraries=library_versions("torch"),
        extra=extra,
    )
    del model
    torch.cuda.empty_cache()
    return items, metrics


def run_nf4(torch, tokenizer, *, precision: str, compute_dtype, quant_config: dict, arms: dict):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    write_progress({"status": "running", "stage": precision, "arms": arms})
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        quantization_config=quant,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()
    work = OUT / f"{precision}-state.pt"
    torch.save(model.state_dict(), work)
    size = max(work.stat().st_size, 1)
    work.unlink(missing_ok=True)
    items, metrics = measure(
        precision,
        torch,
        tokenizer,
        model,
        on_disk_bytes=size,
        libraries=library_versions("bitsandbytes"),
        extra={
            "compute_dtype": "float32" if compute_dtype == torch.float32 else "float16",
            "quant_config": dict(quant_config),
            "attn_implementation": "eager",
        },
    )
    del model
    torch.cuda.empty_cache()
    return items, metrics


def fp16_overflow(report: dict) -> bool:
    eager = report.get("configs", {}).get("fp16_eager", {})
    if not eager.get("ok"):
        return False
    return any(not row.get("logits_finite", True) for row in eager.get("items", []))


def main() -> int:
    random.seed(0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    write_progress({"status": "running", "stage": "install", "arms": {}})
    try:
        pip_install()
        import torch

        torch.set_num_threads(THREADS)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this kernel")
        dump_env()
        tokenizer = load_tokenizer()
        dump_prompt_hashes(tokenizer)
        numerics = probe_numerics(torch, tokenizer)
        t4_items, _ = run_t4_fp32(torch, tokenizer)
        if not baseline_is_healthy(t4_items):
            score = sum(1 for row in t4_items if row["correct"])
            raise RuntimeError(
                f"t4-fp32 baseline scored {score}/129 or had non-finite NLL. "
                "Fail closed before fp16/NF4 pairing."
            )
        overflow = fp16_overflow(numerics)
        fp16_items, _ = run_fp16(torch, tokenizer, finding="overflow" if overflow else None)
        fp16_ok = baseline_is_healthy(fp16_items) and not overflow
        arms = {
            "t4-fp32": {"status": "measured"},
            "fp16": {"status": "measured"},
        }
        run_nf4(
            torch,
            tokenizer,
            precision="nf4",
            compute_dtype=torch.float32,
            quant_config=NF4_QUANT_CONFIG_F32,
            arms=arms,
        )
        arms["nf4"] = {"status": "measured"}
        if fp16_ok:
            run_nf4(
                torch,
                tokenizer,
                precision="nf4-fp16",
                compute_dtype=torch.float16,
                quant_config=NF4_QUANT_CONFIG_F16,
                arms=arms,
            )
            arms["nf4-fp16"] = {"status": "measured"}
        else:
            arms["nf4-fp16"] = {
                "status": "unsupported",
                "reason": "probed_failure",
                "detail": "fp16 is not a healthy compute dtype; skipped nf4-fp16",
            }
        write_progress({"status": "measured", "stage": "done", "arms": arms})
        log("done")
        return 0
    except Exception as exc:
        log(traceback.format_exc())
        write_progress(
            {
                "status": "failed",
                "reason": "probed_failure",
                "detail": f"{type(exc).__name__}: {exc}"[:400],
                "probe": {
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:200] or type(exc).__name__,
                    "cuda_available": None,
                },
                "arms": {},
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
