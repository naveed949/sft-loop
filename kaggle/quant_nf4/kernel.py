#!/usr/bin/env python3
"""Kaggle T4 kernel. Measures fp16 and NF4 with the sft-loop quant protocol.

Accelerator: NvidiaTeslaT4. Expected wall time is about 2 hours. Budget 4 hours.
No secrets. Downloads Qwen/Qwen2.5-1.5B-Instruct from the public Hub.
"""

from __future__ import annotations

import json
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
NF4_QUANT_CONFIG = {
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
        "transformers==4.56.2",
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


def generate_fn(torch, tokenizer, model, *, min_new_tokens: int | None = None):
    def generate(prompt: str):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
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
        messages = [{"role": "user", "content": prompt}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    items = []
    for item_id in ITEM_IDS:
        prompt = prompt_for(item_id)
        prediction, tokens, latency_ms = repeat_generate(generate, prompt)
        _, decode_tokens, decode_latency_ms = repeat_generate(decode_generate, prompt)
        target_nll, sequence_nll = nll(prompt, item_id.upper())
        items.append(
            {
                "item_id": item_id,
                "correct": prediction_token(prediction) == item_id.upper(),
                "latency_ms": latency_ms,
                "generated_tokens": int(tokens),
                "decode_latency_ms": decode_latency_ms,
                "decode_generated_tokens": int(decode_tokens),
                "nll": _round(target_nll),
                "sequence_nll": _round(sequence_nll),
            }
        )
        log(f"{precision} {item_id} correct={items[-1]['correct']} p50={latency_ms}")
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    peak_gpu = int(torch.cuda.max_memory_allocated())
    metrics = {
        "on_disk_bytes": int(on_disk_bytes),
        "peak_rss_bytes": max(rss, 1),
        "peak_gpu_bytes": max(peak_gpu, 1),
        "libraries": libraries,
        "limiter": {
            "kind": "gpu",
            "wall_s": _round(wall),
            "cpu_time_s": _round(cpu),
            "peak_rss_bytes": max(rss, 1),
            "available_ram_bytes_at_start": start_ram,
        },
        "compute_dtype": "float16",
        **extra,
    }
    write_json(OUT / precision / "items.json", items)
    write_json(OUT / precision / "metrics.json", metrics)
    return items, metrics


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


def run_fp16(tokenizer):
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForCausalLM

    write_progress({"status": "running", "stage": "fp16", "arms": {}})
    kwargs = {
        "revision": REVISION,
        "device_map": {"": 0},
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, **kwargs)
    model.eval()
    size = max(snapshot_bytes(), 1)
    items, metrics = measure(
        "fp16",
        torch,
        tokenizer,
        model,
        on_disk_bytes=size,
        libraries={
            "kind": "torch",
            "torch": torch.__version__,
            "transformers": version("transformers"),
        },
        extra={},
    )
    del model
    torch.cuda.empty_cache()
    return items, metrics


def run_nf4(tokenizer):
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    write_progress(
        {
            "status": "running",
            "stage": "nf4",
            "arms": {"fp16": {"status": "measured"}},
        }
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        quantization_config=quant,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()
    work = OUT / "nf4-state.pt"
    torch.save(model.state_dict(), work)
    size = max(work.stat().st_size, 1)
    work.unlink(missing_ok=True)
    items, metrics = measure(
        "nf4",
        torch,
        tokenizer,
        model,
        on_disk_bytes=size,
        libraries={
            "kind": "bitsandbytes",
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "bitsandbytes": version("bitsandbytes"),
        },
        extra={"quant_config": dict(NF4_QUANT_CONFIG)},
    )
    del model
    torch.cuda.empty_cache()
    return items, metrics


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
        run_fp16(tokenizer)
        run_nf4(tokenizer)
        write_progress(
            {
                "status": "measured",
                "stage": "done",
                "arms": {"fp16": {"status": "measured"}, "nf4": {"status": "measured"}},
            }
        )
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
