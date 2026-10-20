# sft-loop verification map

This directory is the maintained source for verifying the user-facing behavior of sft-loop. Read the index before driving the CLI, then use the matching feature file as the recipe.

## Baseline preconditions

- Work from the repo root of `sft-loop`.
- Create a disposable run with `VERIFY_RUN_ID` (default `cpu-linux-seed0`).
- Set `VERIFY_SCRATCH_DIR=/tmp/sft-loop-verify-$VERIFY_RUN_ID` so concurrent runs do not share adapters.
- Put evidence in `.cursor/skills/verify-sft-loop/evidence/$VERIFY_RUN_ID/`.
- Run `.cursor/skills/verify-sft-loop/scripts/verify install` then `scripts/verify doctor`.
- Require doctor `cuda=false` and `mps=false` on this CPU VM.
- Never write verify output into `runs/phase3/`.
- Never install the `unsloth` extra for this skill.

## Driving conventions

- Start every recipe from the baseline unless its preconditions say otherwise.
- Treat every command as literal. Keep model id, seed, and flags unchanged.
- Drive through `scripts/verify` or `.venv/bin/python -m sft_loop...`.
- Restore nothing in `runs/phase3/`. Do not remove proof artifacts during cleanup.

## Proof and skip reporting

- Capture the command, stdout, stderr, exit code, and the JSON file written.
- Mutation proof includes a second read of the report on disk and `validate_report`.
- Record the feature ID with every artifact.
- A blocked GPU path is `unsupported` with `gpu_or_train_deps_unavailable` or `train_not_run`. It is not a skip and not a pass.
- Do not report a skipped MPS entry point as verified through the CPU smoke.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with verify-sft-loop` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable flags, required state, commands, and observable proof.

## Features

- [Fail-closed unsupported CLI](./cli-unsupported-report.md) covers `python -m sft_loop` on a machine that cannot train.
- [LoRA train](./train-lora.md) covers the official train CLI and the CPU plumbing smoke.
- [Same-seed eval](./eval-same-seed.md) covers before/after splits that must share seed, task, and grader.
- [Same-seed loop](./loop.md) covers `python -m sft_loop.loop` before, train, after.
- [Run report schema](./report-schema.md) covers `schema/run_report.schema.json` validation and honesty locks.
