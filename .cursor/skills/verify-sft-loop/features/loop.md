# Same-seed loop

The loop evals before, runs LoRA SFT, evals after, and builds one report. Both splits request the same `--seed`. On this CPU VM the official loop does not invoke train or eval.

## Sub-features

- `loop-blocked` returns `unsupported` without calling train or eval when the stack is missing.
- `loop-mps` is `--device mps` on Cap's M1 scored path.
- `loop-cuda` is `--device cuda` with optional Unsloth.
- `loop-smoke-equivalent` is `scripts/verify smoke`, which is the CPU stand-in for before/train/after.

## How to get to it (user POV)

- Run `python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0`.
- Add `--device mps` or `--device cuda`.
- Add `--out PATH` to write the report.

## Driving it with verify-sft-loop

Preconditions:

- Doctor is green.
- Do not pass `--out runs/phase3/unsupported.json` from verify. That file is the checked-in probe.

- **Default loop.** Run `scripts/verify unsupported`. `reports/loop-default.json` matches the fail-closed shape of `runs/phase3/unsupported.json` (same reason, same refusal sentence, seed 0). `run_id` may differ because verify uses `verify-loop-default`.
- **No measured steps.** Confirm `reports/loop-default.json` has no `before`, `after`, or `train_proof`.
- **CPU before/train/after.** Run `scripts/verify smoke`. That is the plumbing equivalent, not `python -m sft_loop.loop --device cpu` (argparse exits 2 for that flag).
- **Proof.** `surfaces.json` lists `loop-default` as supported (the fail-closed CLI ran) and `train-mps` / `train-cuda-unsloth` as unsupported from live probes.

## Gotchas

- Checked-in `runs/phase3/unsupported.json` uses `run_id` `phase3-unsupported-no-gpu`. Verify writes a different `run_id` on purpose so it does not overwrite the portfolio file.
- A live `run_loop(...)` with default args on this VM must still equal the checked-in unsupported report. `pytest` covers that. Do not "fix" the checked-in file from a CPU smoke.
- If train extras are installed, the loop still blocks unless MPS or CUDA Unsloth is actually usable. `resolve_train_target` does not treat CPU as ready.
- Smoke must not emit `status: scored`. The loop's scored path is MPS or CUDA only.
