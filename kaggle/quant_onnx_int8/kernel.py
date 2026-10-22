#!/usr/bin/env python3
"""Kaggle CPU kernel. ONNX export plus onnxruntime dynamic int8.

No GPU. Expected wall time is about 4 hours. Budget 8 hours of one 12 hour session.
No secrets. Downloads Qwen/Qwen2.5-1.5B-Instruct from the public Hub.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
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


def peak_rss_bytes() -> int:
    import resource

    return max(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, 1)


def host_ram_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def record_export_mem(stage: str, **extra) -> dict:
    host = host_ram_bytes()
    rss = peak_rss_bytes()
    payload = {
        "stage": stage,
        "peak_rss_bytes": rss,
        "host_ram_bytes": host,
        "host_ram_gb": round(host / 1e9, 3),
        "fraction_of_host": round(rss / host, 4) if host else None,
    }
    payload.update(extra)
    write_json(OUT / "export_mem.json", payload)
    log(
        f"export_mem stage={stage} peak_rss_bytes={rss} "
        f"host_ram_bytes={host} fraction={payload['fraction_of_host']}"
    )
    return payload


def pip_install() -> None:
    # onnx 1.17.0 has no cp313 wheel. Kaggle is Python 3.13, so 1.17.0
    # built from sdist and failed. 1.18.0 is the oldest release with
    # cp313 manylinux wheels. onnxruntime 1.20.1 and onnxscript 0.2.0
    # already have cp313 / py3 wheels. onnxscript requires onnx>=1.16.
    # Do not add the onnx_ir package. torch 2.11's dynamo exporter
    # imports it; this kernel uses dynamo=False and TorchScript instead.
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "onnx==1.18.0",
        "onnxruntime==1.20.1",
        "onnxscript==0.2.0",
        "transformers==4.56.2",
    ]
    log("pip " + " ".join(cmd[4:]))
    subprocess.check_call(cmd)


def dump_env() -> None:
    env = {
        "kernel_url": None,
        "kernel_version": None,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": platform.processor() or platform.machine() or "unknown",
        "logical_cpus": os.cpu_count() or 1,
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    }
    write_json(OUT / "env.json", env)


def prompt_for(word: str) -> str:
    return f"Uppercase the word: {word}"


def prediction_token(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split()[0]


def apply_prompt_mask(labels: list[int], prefix_len: int) -> list[int]:
    return [IGNORE_INDEX] * prefix_len + list(labels[prefix_len:])


def _round(value: float) -> float:
    return float(format(float(value), ".8g"))


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


def load_cpu_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("cpu")
    model.eval()
    return torch, tokenizer, model


def additive_causal_mask(torch, attention_mask, dtype):
    batch, seqlen = attention_mask.shape
    causal = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=attention_mask.device))
    keep = attention_mask.to(torch.bool)[:, None, None, :] & causal
    fill = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, seqlen, seqlen), dtype=dtype, device=attention_mask.device)
    return mask.masked_fill(~keep, fill)


def export_and_quantize(torch, tokenizer, model):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            # transformers 4.56 create_causal_mask uses vmap on a 2D pad
            # mask. TorchScript cannot trace that. A 4D additive mask is
            # returned as-is and never enters that path.
            mask4 = additive_causal_mask(
                torch, attention_mask, next(self.inner.parameters()).dtype
            )
            return self.inner(
                input_ids=input_ids,
                attention_mask=mask4,
                use_cache=False,
            ).logits

    wrapper = _LogitsOnly(model)
    wrapper.eval()
    sample = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(ITEM_IDS[0])}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        return_tensors="pt",
    )
    work = Path(tempfile.mkdtemp(prefix="sft-quant-onnx-", dir=str(OUT)))
    fp_path = work / "model.onnx"
    int8_path = work / "model.int8.onnx"
    write_progress({"status": "running", "stage": "export", "arms": {}})
    record_export_mem("export_start")
    with torch.no_grad():
        # torch 2.11 defaults dynamo=True and imports onnx_ir via the
        # exporter internals. dynamo=False stays on TorchScript and does
        # not import that module. Verified on torch 2.11.0+cpu.
        torch.onnx.export(
            wrapper,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    data_path = Path(str(fp_path) + ".data")
    record_export_mem(
        "export_done",
        fp_onnx_bytes=fp_path.stat().st_size if fp_path.is_file() else 0,
        fp_onnx_data_bytes=data_path.stat().st_size if data_path.is_file() else 0,
    )
    del model, wrapper
    gc.collect()
    write_progress({"status": "running", "stage": "quantize", "arms": {}})
    quantize_dynamic(str(fp_path), str(int8_path), weight_type=QuantType.QInt8)
    import shutil

    fp_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    for leftover in list(work.iterdir()):
        if leftover.resolve() == int8_path.resolve():
            continue
        if leftover.is_dir():
            shutil.rmtree(leftover)
        else:
            leftover.unlink()
    record_export_mem(
        "quantize_done",
        int8_onnx_bytes=int8_path.stat().st_size if int8_path.is_file() else 0,
    )
    size = max(int8_path.stat().st_size, 1)
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    return np, sess, size, onnx, ort


def make_generate(np, sess, tokenizer, *, stop_on_eos: bool):
    def generate(prompt: str):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(text, return_tensors="np")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        start = time.perf_counter()
        new_tokens: list[int] = []
        for _ in range(DECODE_NEW_TOKENS):
            logits = sess.run(
                ["logits"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )[0]
            next_id = int(np.argmax(logits[0, -1]))
            new_tokens.append(next_id)
            input_ids = np.concatenate([input_ids, [[next_id]]], axis=1)
            attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
            if stop_on_eos and tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
        elapsed = (time.perf_counter() - start) * 1000.0
        return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens), elapsed

    return generate


def nll_fn(np, sess, tokenizer):
    def nll(prompt: str, target: str):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(prefix, return_tensors="np")["input_ids"][0].tolist()
        full = tokenizer(prefix + target, return_tensors="np")
        logits = sess.run(
            ["logits"],
            {"input_ids": full["input_ids"], "attention_mask": full["attention_mask"]},
        )[0]
        label_ids = full["input_ids"][0].tolist()
        target_labels = apply_prompt_mask(label_ids, len(prefix_ids))
        pred = logits[0, :-1]
        gold = label_ids[1:]
        masked = target_labels[1:]
        shifted = pred - pred.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        sequence = float(-log_probs[np.arange(len(gold)), gold].mean()) if gold else 0.0
        keep = [i for i, lab in enumerate(masked) if lab != IGNORE_INDEX]
        target_nll = float(-log_probs[keep, np.array(gold)[keep]].mean()) if keep else 0.0
        return target_nll, sequence

    return nll


def score(np, sess, tokenizer, size, torch, onnx, ort):
    import resource
    from importlib.metadata import version

    write_progress({"status": "running", "stage": "score", "arms": {}})
    start_ram = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split("kB")[0]) * 1024
    start = time.perf_counter()
    cpu0 = time.process_time()
    generate = make_generate(np, sess, tokenizer, stop_on_eos=True)
    decode_generate = make_generate(np, sess, tokenizer, stop_on_eos=False)
    nll = nll_fn(np, sess, tokenizer)
    items = []
    predictions = []
    for item_id in ITEM_IDS:
        prompt = prompt_for(item_id)
        prediction, tokens, latency_ms = repeat_generate(generate, prompt)
        _, decode_tokens, decode_latency_ms = repeat_generate(decode_generate, prompt)
        target_nll, sequence_nll = nll(prompt, item_id.upper())
        token = prediction_token(prediction)
        items.append(
            {
                "item_id": item_id,
                "correct": token == item_id.upper(),
                "latency_ms": latency_ms,
                "generated_tokens": int(tokens),
                "decode_latency_ms": decode_latency_ms,
                "decode_generated_tokens": int(decode_tokens),
                "nll": _round(target_nll),
                "sequence_nll": _round(sequence_nll),
            }
        )
        predictions.append(
            {
                "item_id": item_id,
                "prediction": prediction,
                "prediction_token": token,
            }
        )
        log(f"onnx-int8 {item_id} correct={items[-1]['correct']} p50={latency_ms}")
    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    exact_match = sum(1 for row in items if row["correct"]) / max(len(items), 1)
    metrics = {
        "on_disk_bytes": int(size),
        "peak_rss_bytes": max(rss, 1),
        "exact_match": _round(exact_match),
        "grader_case_sensitive": True,
        "libraries": {
            "kind": "onnx",
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "onnx": onnx.__version__,
            "onnxruntime": version("onnxruntime"),
        },
        "limiter": {
            "kind": "rss" if rss > start_ram * 0.8 else "cpu",
            "wall_s": _round(wall),
            "cpu_time_s": _round(cpu),
            "peak_rss_bytes": max(rss, 1),
            "available_ram_bytes_at_start": start_ram,
        },
    }
    write_json(OUT / "onnx-int8" / "items.json", items)
    write_json(OUT / "onnx-int8" / "metrics.json", metrics)
    write_json(OUT / "onnx-int8" / "predictions.json", predictions)
    write_progress(
        {
            "status": "measured",
            "stage": "done",
            "arms": {"onnx-int8": {"status": "measured"}},
        }
    )


def main() -> int:
    random.seed(0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    write_progress({"status": "running", "stage": "install", "arms": {}})
    try:
        pip_install()
        dump_env()
        torch, tokenizer, model = load_cpu_model()
        np, sess, size, onnx, ort = export_and_quantize(torch, tokenizer, model)
        score(np, sess, tokenizer, size, torch, onnx, ort)
        log("done")
        return 0
    except MemoryError as exc:
        log(traceback.format_exc())
        write_progress(
            {
                "status": "failed",
                "reason": "oom",
                "detail": f"onnx-int8 hit MemoryError. {exc}"[:400],
                "probe": {
                    "exception_type": "MemoryError",
                    "message": str(exc)[:200] or "MemoryError",
                    "cuda_available": False,
                },
                "arms": {"onnx-int8": {"status": "unsupported", "reason": "oom"}},
            }
        )
        return 1
    except Exception as exc:
        log(traceback.format_exc())
        text = str(exc)
        reason = "oom" if "out of memory" in text.lower() else "probed_failure"
        write_progress(
            {
                "status": "failed",
                "reason": reason,
                "detail": f"{type(exc).__name__}: {exc}"[:400],
                "probe": {
                    "exception_type": type(exc).__name__,
                    "message": text[:200] or type(exc).__name__,
                    "cuda_available": False,
                },
                "arms": {"onnx-int8": {"status": "unsupported", "reason": reason}},
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
