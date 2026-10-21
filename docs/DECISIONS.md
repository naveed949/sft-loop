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

PR #15's Phase 4 T4 kernel hit a non-finite training loss at step 1 in fp16 on the same accelerator. That is a training-loop observation, not this eval harness, and it is consistent with the fp16 overflow hypothesis.

The ONNX kernel failed during pip. `onnx==1.17.0` has no cp313 wheel; the image is Python 3.13.15. The pin is now `onnx==1.18.0`. That kernel is being rerun separately. The ONNX arm stays `unsupported` / `probed_failure` until those outputs land.

## Quant ONNX torch 2.11 exporter: dynamo=False, not onnx-ir (2026-10-21)

ONNX Kaggle rerun v2 at `2eede76` installed `onnx==1.18.0`, `onnxruntime==1.20.1`, `onnxscript==0.2.0`, `transformers==4.56.2` and then failed in `torch.onnx.export`:

`ModuleNotFoundError: No module named 'onnx_ir'`

The host was 4 CPUs and `ram_bytes=33658318848` (33.658 GB). Python 3.13.15, torch 2.11.0. v2 never reached graph export; peak RSS during export is unmeasured.

Chose **(b) `dynamo=False`**, not **(a) pin `onnx-ir` plus a dynamo-compatible onnxscript**.

torch 2.11's `torch.onnx.export` defaults `dynamo=True`. That branch is the only one that does `from torch.onnx._internal.exporter import _compat`. `_compat` → `_core` → `_schemas` → `_lazy_import("onnx_ir")`. The TorchScript branch (`else`) imports `torch.onnx._internal.torchscript_exporter.utils` and never touches `onnx_ir`.

Rejected (a) because v2 already had `onnxscript==0.2.0` and still died on `onnx_ir`. The dynamo path needs both, and the versions have to match torch 2.11's private API (`onnxscript._framework_apis.torch_2_11`). pytorch/pytorch#179280 on 2.11.0 needed `onnx-ir` plus onnxscript 0.6.2, not 0.2.0. Pinning that pair is a second moving dependency after the 1.17.0 wheel miss. `dynamic_axes` / `do_constant_folding` are the TorchScript flags.

Local proof, same torch 2.11.0+cpu, pins `onnx==1.18.0` and `onnxscript==0.2.0`, no `onnx-ir`. This VM has Python 3.12.3, not 3.13; the wheel's `torch/onnx/__init__.py` and `_schemas.py` match the Kaggle traceback line for line.

```
python -c "import torch.onnx._internal.exporter._core"
# ModuleNotFoundError: No module named 'onnx_ir'
```

`torch.onnx.export(..., dynamo=False)` on a one-add module wrote a 184-byte ONNX file. After that call, `onnx_ir` and `torch.onnx._internal.exporter._compat` were still absent from `sys.modules`. `dynamo=True` still raised `No module named 'onnxscript'` or `onnx_ir` depending on whether 0.2.0 was installed.

Export peak RAM vs 33.6 GB: v2 died at import, so there is no Kaggle export peak to report. The earlier local 1.5B export on this 16.781 GB host wrote a 5.8 GB `model.onnx.data` and then took SIGKILL 137 in `quantize_dynamic`. Export of the same model already fit in 16.781 GB, so 33.658 GB (2.0×) has headroom for export. The remaining RAM risk is quantize, not `torch.onnx.export`. The kernel writes `export_mem.json` at `export_start`, `export_done` (file sizes included), and `quantize_done` with `peak_rss_bytes`, `host_ram_bytes`, and `fraction_of_host`. Compare the next run's `export_done` peak against 33658318848.

The ONNX arm stays `unsupported` / `probed_failure` until that rerun lands. v2 logs are under `runs/quant/kernel/quant_onnx_int8/`.
