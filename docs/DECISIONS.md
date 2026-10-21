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

NF4 can be a measured arm when a Kaggle T4 kernel output is ingested. The same-hardware baseline is fp32 on that T4, not CPU fp32 and not T4 fp16. `outcomes` owns `int8-minus-fp32` on the CPU host and `nf4-minus-t4-fp32` / `nf4-fp16-minus-t4-fp32` on one `hardware.accelerator`. A GPU number is never subtracted from a CPU number. A cross-hardware view is `kind: not_paired` with reason `cross_hardware`.

GPU measured arms require `hardware.accelerator`, `peak_gpu_bytes`, a kernel URL/version placeholder, and fp16 compute. NF4 also records BitsAndBytesConfig `load_in_4bit`, `bnb_4bit_quant_type=nf4`, double quant on, and fp16 compute.

Local `python -m sft_loop.quant probe-nf4` stays unsupported on this CPU VM. The only measured GPU path is `python -m sft_loop.quant ingest`. A missing kernel directory is `unsupported` / `kernel_output_missing` with no metric keys.

Kernels live at `kaggle/quant_nf4` (NvidiaTeslaT4) and `kaggle/quant_onnx_int8` (CPU only). They pin deps, download the public Hub revision, and write env, logs, progress, and per-arm items. Local verify cannot run them. Downloaded outputs are committed under `runs/quant/kernel/`. The table is regenerated from ingest. `python -m sft_loop.quant run` reuses measured kernel arms so it cannot overwrite them with a local probe.

## Quant T4 ingest and ONNX cp313 pin (2026-10-21)

Kaggle kernel `https://www.kaggle.com/code/inaveed101/sft-loop-quant-nf4` version 1 is withdrawn. Both T4 arms scored 0/129, every item emitted 8 tokens, and NLL was non-finite. CPU fp32 on the same model, items, and grader scored 47/129. That is a harness or numerics bug, not a pairing. The v1 numbers stay under `runs/quant/kernel/quant_nf4/` as evidence. They are not a baseline.

The replacement kernel runs fp32-on-T4 first (`t4-fp32`, about 6.2 GB weights, fits a 15 GB T4). It fail-closes if that baseline scores 0 or has non-finite NLL. Ingest refuses the same state unless `finding` is `collapse`. fp16 stays its own arm. If it overflows, it is a measured `finding: overflow`, not a pairing baseline. NF4 compute dtype is float32; nf4-fp16 compute runs only if fp16 is healthy. Pairing is NF4 versus `t4-fp32`.

The kernel pins `transformers==5.20.2` and `tokenizers==0.23.3` to match the CPU arm. It writes `prompt_hashes.json` and a two-item numerics probe (`fp32_eager`, `fp16_eager`, `fp16_sdpa`) with `torch.isfinite` on logits and per-layer hidden max abs.

Expected T4 wall is about 2 hours. Budget 2.5 hours. Stay under 3 GPU hours.

The ONNX kernel failed during pip. `onnx==1.17.0` has no cp313 wheel; the image is Python 3.13.15. The pin is now `onnx==1.18.0`. That kernel is being rerun separately. The ONNX arm stays `unsupported` / `probed_failure` until those outputs land.
