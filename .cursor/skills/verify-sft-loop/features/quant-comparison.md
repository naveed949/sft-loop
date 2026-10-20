# Quant comparison

Four CPU quantization arms on one pinned model share one seed, one held-out split, and one grader. The comparison document owns pairing. NF4 on this VM is an unsupported report with a probed reason and no numbers.

## Sub-features

- `quant-fp32` is the CPU float32 baseline for `Qwen/Qwen2.5-1.5B-Instruct`.
- `quant-int8` is post-training dynamic int8 on Linear layers.
- `quant-onnx-int8` is ONNX export plus onnxruntime dynamic int8.
- `quant-nf4` is bitsandbytes NF4. On this CPU VM it is `unsupported`.
- `quant-table` is the README region generated from `runs/quant/comparison.json`.

## How to get to it (user POV)

- `python -m sft_loop.quant run`
- `python -m sft_loop.quant validate`
- `python -m sft_loop.quant table --check README.md`
- `python -m sft_loop.quant probe-nf4`
- Inspect `runs/quant/<precision>.json` and `runs/quant/comparison.json`.

## Driving it with verify-sft-loop

Preconditions:

- Doctor is green.
- `schema/quant_comparison.schema.json` is present.
- `runs/quant/comparison.json` is present.

- **NF4 probe.** Run `scripts/verify quant`. Evidence `quant/nf4.json` has `status` `unsupported`, no `task_score`, and the refusal sentence.
- **Portfolio.** The same command runs `python -m sft_loop.quant validate`. Exit 0 means the comparison and its projections match.
- **Table.** The same command runs `table --check README.md`. Exit 0 means the marked README region equals `render_table`.
- **Official SFT CLIs.** Leave `scripts/verify unsupported` as the seven fail-closed SFT reports. Do not add quant scores there.

## Gotchas

- `python -m sft_loop.loop --device cpu` still exits 2. Quant has no `--device` flag.
- A quant file is not a run report. `validate_report` must reject it.
- NF4 numbers are never estimated. Missing CUDA is `unsupported`.
- Torch 2.14 ONNX export needs `onnxscript`. A missing module is `dependency_missing`. Export that dies after `model.onnx.data` is written is `oom`.
- Do not write quant output into `runs/phase3/`.
- `verify quant` does not reload the 1.5B weights. That is `python -m sft_loop.quant run`.
