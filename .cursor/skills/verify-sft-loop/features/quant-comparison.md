# Quant comparison

CPU and GPU quantization arms on one pinned model share one seed, one held-out split, and one grader. The comparison document owns named pairings. NF4 is paired only with fp32 on the same T4. T4 fp16 is not a pairing baseline. GPU rows are not paired with CPU rows. Local verify cannot run the Kaggle kernels.

## Sub-features

- `quant-fp32` is the CPU float32 baseline for `Qwen/Qwen2.5-1.5B-Instruct`.
- `quant-int8` is per-tensor symmetric dynamic PTQ (`default_dynamic_qconfig`) on every `nn.Linear`, including `lm_head`.
- `quant-onnx-int8` is ONNX export plus onnxruntime dynamic int8. The local 16 GB attempt is `unsupported` / `oom`. The first Kaggle CPU retry is `unsupported` / `probed_failure`. A CPU retry lives at `kaggle/quant_onnx_int8`.
- `quant-t4-fp32` is the T4 fp32 same-hardware baseline. Ingest refuses a 0-score or non-finite-NLL baseline unless it is marked `finding: collapse`.
- `quant-nf4` is bitsandbytes NF4 with float32 compute. Local `probe-nf4` stays `unsupported`. Measured numbers come only from ingested `kaggle/quant_nf4` output.
- `quant-nf4-fp16` is NF4 with fp16 compute. It runs only if T4 fp16 is healthy.
- `quant-fp16` is T4 fp16. It is not a pairing baseline. Overflow is a measured finding, not a quality score.
- `quant-table` is the README region generated from `runs/quant/comparison.json`.
- `kaggle-quant-kernels` is a local-unsupported surface. The scripts live at `kaggle/quant_nf4` and `kaggle/quant_onnx_int8`. Downloaded outputs live at `runs/quant/kernel/`. Local verify does not run those scripts.

## How to get to it (user POV)

- `python -m sft_loop.quant run`
- `python -m sft_loop.quant validate`
- `python -m sft_loop.quant table --check README.md`
- `python -m sft_loop.quant probe-nf4`
- `python -m sft_loop.quant ingest --precision nf4 --from-dir <kernel-output> --write`
- Inspect `runs/quant/<precision>.json` and `runs/quant/comparison.json`.
- Inspect `kaggle/quant_nf4`, `kaggle/quant_onnx_int8`, and `runs/quant/kernel/`.

## Driving it with verify-sft-loop

Preconditions:

- Doctor is green.
- `schema/quant_comparison.schema.json` is present.
- `runs/quant/comparison.json` is present.

- **NF4 probe.** Run `scripts/verify quant`. Evidence `quant/nf4.json` has `status` `unsupported`, no `task_score`, and the refusal sentence.
- **Missing kernel.** The same command ingests a missing kernel dir. Evidence `quant/kernel-missing.json` is `unsupported` / `kernel_output_missing` with no metric keys. The CLI exits 2.
- **Portfolio.** The same command runs `python -m sft_loop.quant validate`. Exit 0 means the comparison and its projections match.
- **Table.** The same command runs `table --check README.md`. Exit 0 means the marked README region equals `render_table`.
- **Kernel evidence.** `kaggle/quant_nf4/kernel-metadata.json` and `kaggle/quant_onnx_int8/kernel-metadata.json` exist. Downloaded outputs, when present, live under `runs/quant/kernel/`. Local verify does not execute those scripts.
- **Official SFT CLIs.** Leave `scripts/verify unsupported` as the seven fail-closed SFT reports. Do not add quant scores there.

## Gotchas

- `python -m sft_loop.loop --device cpu` still exits 2. Quant has no `--device` flag.
- A quant file is not a run report. `validate_report` must reject it.
- NF4 numbers are never estimated on this VM. Missing CUDA is `unsupported`. Missing kernel output is `kernel_output_missing`.
- Do not pair NF4 or T4 fp16 against CPU fp32. That view is `not_paired`. T4 fp16 is not the GPU baseline.
- Torch 2.14 ONNX export needs `onnxscript`. A missing module is `dependency_missing`. Export that dies after `model.onnx.data` is written is `oom`.
- Do not write quant output into `runs/phase3/`.
- `verify quant` does not reload the 1.5B weights. That is `python -m sft_loop.quant run`.
- scored p50/p95 allow EOS. decode tokens/s is a separate fixed-length pass (`min_new_tokens == max_new_tokens == 8`). Do not treat scored emitted tokens / scored wall as matched throughput.
- Kaggle kernels need Naveed's token on another machine. This VM has no credentials and must not invent GPU numbers.
