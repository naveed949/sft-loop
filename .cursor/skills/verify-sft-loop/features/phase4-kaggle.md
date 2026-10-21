# Phase 4 Kaggle GPU LoRA

A scored fp16 LoRA run on capitalcite 10-K extraction. The local CPU verify cannot start that kernel. On this VM the surface is `unsupported`. Kernel evidence is produced later on Naveed's T4 and linked from `kaggle/phase4_train/`.

## Sub-features

- `phase4-dataset` is the issuer-split extraction set in `kaggle/phase4_data/` (`eval` n >= 100, contamination report `ok`).
- `phase4-kernel` is the private GPU script in `kaggle/phase4_train/` (`enable_gpu` true, no secrets).
- `phase4-local-unsupported` is the CPU probe: no CUDA `GradScaler` path, no adapter, no after-eval.
- `phase4-no-scored-json` is the Phase 1 honesty lock: `runs/phase4/scored.json` does not exist yet.

## How to get to it (user POV)

- Inspect `kaggle/phase4_data/manifest.json` and `kaggle/phase4_data/contamination.json`.
- Inspect `kaggle/phase4_train/kernel-metadata.json` and `kaggle/phase4_train/kernel.py`.
- On this VM run `python -m phase4 probe --out /tmp/phase4-probe.json`. Expect `status` `unsupported`.
- After Naveed returns kernel outputs, Phase 2 writes `runs/phase4/scored.json` with `phase4.assemble`. That file is not present in Phase 1.

## Driving it with verify-sft-loop

Preconditions:

- `scripts/verify doctor` exits 0.
- This VM has no CUDA device and no Kaggle credentials.
- Do not run `kaggle kernels push` and do not execute `kaggle/phase4_train/kernel.py` here.

- **Local map status.** Mark this feature `unsupported` with `gpu_or_train_deps_unavailable`. The CPU plumbing smoke is a different surface and still must pass.
- **Dataset and fence.** Run `scripts/verify test`. `tests/test_phase4.py` requires eval n >= 100, a clean contamination report, an AST fence (no `sft_loop` or `unsloth` in Phase 4 files), and `runs/phase4/scored.json` absent.
- **Kernel evidence (not local).** The script that will write per-item `eval_before.json` / `eval_after.json`, `fp16_probe.json`, `trainer_log.txt`, `profiler_summary.json`, `memory_checkpoint.json`, `env.json`, and `adapter.sha256` is `kaggle/phase4_train/kernel.py`. Those files appear only after the T4 run. Link that directory as the kernel evidence path. Do not invent scores.
- **Official SFT CLIs.** Leave `scripts/verify unsupported` and `scripts/verify smoke` on the uppercase fixture. Phase 4 must not change those reports.

## Gotchas

- A T4 (compute capability 7.5) cannot run bf16. Record bf16 as `unsupported` with that reason.
- Do not load the 1.5B weights as `torch_dtype=float16` and then call `GradScaler`. Masters must be fp32. Autocast is the compute dtype. Run 2 at `ce57285` proved the wrong recipe: `eval_before` 0/123 of `!!!!!!!!!!!!!!!!`, then `non-finite loss at step 1`, no adapter.
- The kernel writes `fp16_probe.json` first. If AMP overflows (hidden max-abs > 65504 or non-finite loss), `fp16.json` is `unsupported` and training continues in fp32, labeled `fp32`.
- Refuse a garbage baseline: punctuation-only or non-finite NLL writes `unsupported.json` with `baseline_invalid` and does not train.
- Do not write `runs/phase4/scored.json` on this VM. Phase 2 assembles it from the downloaded per-item JSON.
- Do not point Phase 4 writers at `runs/phase3` or `adapters/phase3`.
- If baseline exact match is at least 0.85, the kernel must stop and write `refused_ceiling` instead of an adapter.
- `verify all` must stay green on the CPU smoke path. Failure of the Kaggle kernel is not a local verify failure.
