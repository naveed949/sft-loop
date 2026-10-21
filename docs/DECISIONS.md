# Decisions

This log records project choices that later agents should not rediscover. Each entry states what we chose and why.

## Why the model map is in the repo (2026-10-20)

Cloud agents read Cursor rules from this repository. They do not read a developer's home directory. The per-role map therefore lives at `.cursor/rules/pstack-models.mdc`.

`# budget: medium (high)` means each role starts at high reasoning. That trades some reasoning depth for lower cost compared with the xhigh skill defaults named in poteto-mode.

Naveed picks the budget. Do not change the map or the budget unless Naveed asks.

## Quant comparison model and metrics (2026-10-20)

The comparison uses `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. That is the same model id the SFT loop already pins. A revision SHA is required so every arm loads the same bytes.

The task score is first-token exact match on `evals/quant-heldout-v1.txt`, 129 lowercase words disjoint from `TRAIN_WORDS` and `EVAL_WORDS`. n is 129, which meets the n >= 100 bar. The grader is `sft_loop.task.prediction_token`. The scored pass is greedy, `max_new_tokens` 8, EOS allowed, seed 0. A second pass measures decode tokens/s with `min_new_tokens == max_new_tokens == 8`. Warmup is 2 iterations. Repeats per item are 5. Thread count is 1, set before torch imports.

When an arm exposes logits, `nll` is mean token NLL of the target word. The prompt tokens are masked with `labels[:, :prefix_len] = -100`. `sequence_nll` is full-sequence CE including the chat template. `evals/quant-perplexity-v1.txt` is the reserved Alice extract. An arm without logits omits both NLL fields. The paired difference is int8 minus fp32 on the same item ids. Parse recomputes it and rejects a stored block that disagrees. The published int8 arm is `default_dynamic_qconfig` (per-tensor symmetric) on every `nn.Linear`, including `lm_head`.

`schema/quant_comparison.schema.json` is a sibling of the SFT run report. Quant JSON has no `weight_change_claimed` field. NF4 cannot be a measured arm in this comparison. A 1.5B arm that cannot load on this VM is `unsupported` with the probed failure. A smaller-model rerun, if added, cannot occupy `runs/quant/comparison.json`.

On this CPU VM the pinned 1.5B model loaded. fp32 and torch dynamic int8 finished the 129-item protocol, including the fixed-length decode pass. ONNX export wrote a 5.8 GB `model.onnx.data` then `quantize_dynamic` was SIGKILL 137, so that arm is `oom`. NF4 is `missing_cuda` on this host. The README table is generated from those files. Do not type the scores by hand.

## Quant GPU pairing and Kaggle kernels (2026-10-21)

NF4 can be a measured arm when a Kaggle T4 kernel output is ingested. The same-hardware baseline is fp16 on that T4, not CPU fp32. `outcomes` owns two pairings: `int8-minus-fp32` on the CPU host, and `nf4-minus-fp16` on one `hardware.accelerator`. A GPU number is never subtracted from a CPU number. A cross-hardware view is `kind: not_paired` with reason `cross_hardware`.

GPU measured arms require `hardware.accelerator`, `peak_gpu_bytes`, a kernel URL/version placeholder, and fp16 compute. NF4 also records BitsAndBytesConfig `load_in_4bit`, `bnb_4bit_quant_type=nf4`, double quant on, and fp16 compute.

Local `python -m sft_loop.quant probe-nf4` stays unsupported on this CPU VM. The only measured GPU path is `python -m sft_loop.quant ingest`. A missing kernel directory is `unsupported` / `kernel_output_missing` with no metric keys.

Kernels live at `kaggle/quant_nf4` (NvidiaTeslaT4) and `kaggle/quant_onnx_int8` (CPU only). They pin deps, download the public Hub revision, and write env, logs, progress, and per-arm items. Local verify cannot run them. Downloaded outputs are committed under `runs/quant/kernel/`. The table is regenerated from ingest. `python -m sft_loop.quant run` reuses measured kernel arms so it cannot overwrite them with a local probe.

## Quant T4 ingest and ONNX cp313 pin (2026-10-21)

Kaggle kernel `https://www.kaggle.com/code/inaveed101/sft-loop-quant-nf4` version 1 ran both T4 arms. Wall time was about 27 minutes. Ingest wrote measured `fp16` and `nf4` rows. The pairing is `nf4-minus-fp16` on `NvidiaTeslaT4`.

Trust the memory and decode-throughput numbers. Do not treat the exact-match of 0.0 as a quality result. Both GPU arms scored 0/129 and emitted 8 tokens on the scored pass (CPU fp32 mean generated tokens is 3.25). Kernel `nll` / `sequence_nll` were non-finite; ingest dropped those keys and stored `nll_delta` as null. The paired score delta is 0.0 with a degenerate CI because every item pair is 0 − 0, not because the task still works.

Peak GPU bytes are 3122830336 (fp16) and 1189611520 (NF4). That matches 1.54B × 2 bytes plus a small activation tail versus 4-bit storage plus embeddings / quant state. NF4 decode is 17.860264 tok/s against fp16 24.543959 tok/s on the fixed-length pass. NF4 is slower. That is the expected T4 outcome: dequant to fp16 for GEMM, so the win is memory, not compute.

The ONNX kernel failed during pip. `onnx==1.17.0` has no cp313 wheel; the image is Python 3.13.15. `onnxruntime==1.20.1` and `onnxscript==0.2.0` already had cp313 / py3 wheels. The pin is now `onnx==1.18.0`, the oldest release with cp313 manylinux wheels. onnxscript requires `onnx>=1.16`. The ONNX arm stays `unsupported` / `probed_failure` until that kernel is rerun.
