# Fail-closed unsupported CLI

`python -m sft_loop` prints one schema-valid JSON report with `status` `unsupported` when this machine cannot train. It does not train and does not write before/after metrics.

## Sub-features

- `cli-stdout` prints the unsupported report on stdout when `--out` is omitted.
- `cli-out` writes the same shape to `--out`.
- `cli-seed-loop` is the loop, train, and eval CLIs on a CPU VM, each fail-closed.
- `cli-reject-model` exits 2 for a model id other than `Qwen/Qwen2.5-1.5B-Instruct`.
- `cli-reject-cpu-device` is product argparse: `--device cpu` exits 2 because choices are `{mps,cuda}`.

## How to get to it (user POV)

- Run `python -m sft_loop` in the repo.
- Run `python -m sft_loop --out runs/scratch/unsupported.json`.
- Run `python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0`.
- Run the same loop command with `--device mps` or `--device cuda`.
- Run the same loop command with `--device cpu` and watch argparse reject it.

## Driving it with verify-sft-loop

Preconditions:

- `scripts/verify doctor` exits 0.
- This VM has no MPS and no CUDA Unsloth stack.
- Evidence dir is `.cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/`.

- **Default CLI.** Emit the scaffold report. Run `.cursor/skills/verify-sft-loop/scripts/verify unsupported --run-id "$VERIFY_RUN_ID"`. Exit code `0`. `reports/cli-default.json` has `status` `unsupported`, `weight_change_claimed` false, and no `before` or `after`.
- **Loop without device.** Read `reports/loop-default.json`. `unsupported.reason` is `gpu_or_train_deps_unavailable`. Detail includes seed 0 for both eval splits. Soft-PASS is unused and is refused.
- **Loop MPS.** Read `reports/loop-mps.json`. Same reason. Detail mentions `mps`. `device` is absent.
- **Loop CUDA.** Read `reports/loop-cuda.json`. Same reason. `device` is absent.
- **Bad model.** Run `.venv/bin/python -m sft_loop.loop --model-id other/model --seed 0`. Exit code `2`. stdout is empty.
- **CPU device flag.** Run `.venv/bin/python -m sft_loop.loop --model-id Qwen/Qwen2.5-1.5B-Instruct --seed 0 --device cpu`. Process exit is `2`. stderr contains `invalid choice: 'cpu'`. The same exit applies to `.train` and `.eval`.
- **Proof.** Open `unsupported-validation.json`. Every listed report has `"ok": true`.

## Gotchas

- Exit code 0 on this path still means blocked. Read `status`, not the process code.
- An unsupported report that sets `device` to `mps` or `cuda` is a product bug. Fail the run.
- Do not treat this JSON as a scored baseline. There are no task scores here.
- `runs/scratch/*.json` is gitignored. Prefer the evidence `--out` path so the file survives.
- `--device cpu` is argparse, not `resolve_train_target`. The process exits 2 before any report is printed.
